import os
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import teradatasql
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.preprocessing import LabelEncoder

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

HOSTNAME = os.environ["TD_HOST"]
USERNAME = os.environ["TD_USERNAME"]
PASSWORD = os.environ["TD_PASSWORD"]
DATABASE_NAME = os.environ["TD_DATABASE"]
VIEW_NAME = os.environ["TD_VIEW"]

ID_COLUMN = "Shipment_Number"
DATE_COLUMN = "Deadline_Departure_Time"
SECOND_DATE_COLUMN = "Planned_Ship_Timestamp"
THIRD_DATE_COLUMN = "Load_Create_Time"
TARGET_COLUMN = "has_slack"
FILTER_COLUMN = "has_unplanned"

EXCLUDE_COLUMNS = ["has_unplanned"]

CONTINUOUS_COLUMNS = [
    "Weight_Utilization",
    "Volume_Utilization",
    "Route_Miles",
    "Load_Volume",
    "Load_Gross_Weight",
    "load_create_lead_days",
]

BASE_CATEGORICAL_COLUMNS = [
    "Carrier_Number",
    "Responsibility_Code",
    "Origin_State",
    "Destination_State",
    "Transp_Respon_Hier",
    "transp_equipment_type_descr",
    "Material_Business_Unit",
    "Material_Business_Group",
    "Material_Type",
    "Protein_Type",
]

CHUNK_SIZE = 100000
NUM_THREADS = 4

@dataclass
class FTTransformerHyperparameters:
    """Central place to tune all FT-Transformer hyperparameters."""

    # Transformer architecture
    n_layers: int = 3
    n_heads: int = 8
    attn_dropout: float = 0.2
    ffn_dropout: float = 0.3
    ffn_multiplier: float = 4 / 3  # FFN hidden dim = embedding_dim * multiplier

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 512
    epochs: int = 50
    early_stopping_patience: int = 7

    # Categorical embedding sizing
    # Formula: min(max_embed_dim, base_factor * cardinality ** exponent)
    embed_base_factor: float = 1.6
    embed_exponent: float = 0.56
    embed_max_dim: int = 128
    embed_min_dim: int = 8

    # Global token dimension (continuous features projected to this)
    # Set to 0 to auto-calculate as median of categorical embedding dims
    token_dim: int = 0

    # Misc
    random_state: int = 42
    n_cv_folds: int = 5

    def compute_embedding_dim(self, cardinality: int) -> int:
        """Auto-size embedding dim based on cardinality."""
        dim = int(round(self.embed_base_factor * (cardinality ** self.embed_exponent)))
        return max(self.embed_min_dim, min(self.embed_max_dim, dim))
    
    
def get_teradata_connection():
    return teradatasql.connect(
        host=HOSTNAME, user=USERNAME, password=PASSWORD, database=DATABASE_NAME,
    )

def get_row_count() -> int:
    query = f"SELECT COUNT(*) FROM {DATABASE_NAME}.{VIEW_NAME} WHERE {FILTER_COLUMN} = '1'"
    with get_teradata_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            count = cursor.fetchone()[0]
    print(f"Total rows: {count:,}")
    return count

def fetch_chunk(offset: int, chunk_size: int) -> pd.DataFrame:
    query = f"""
        SELECT *
        FROM {DATABASE_NAME}.{VIEW_NAME}
        WHERE {FILTER_COLUMN} = '1'
        QUALIFY ROW_NUMBER() OVER (ORDER BY {ID_COLUMN} ASC)
            BETWEEN {offset + 1} AND {offset + chunk_size}
    """
    with get_teradata_connection() as conn:
        return pd.read_sql(query, conn)

def load_data() -> pd.DataFrame:
    total_rows = get_row_count()
    offsets = list(range(0, total_rows, CHUNK_SIZE))
    print(f"Loading in {len(offsets)} chunks using {NUM_THREADS} threads...")
    chunk_dict = {}
    completed = 0
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        future_to_offset = {
            executor.submit(fetch_chunk, offset, CHUNK_SIZE): offset
            for offset in offsets
        }
        for future in as_completed(future_to_offset):
            offset = future_to_offset[future]
            chunk_df = future.result()
            chunk_dict[offset] = chunk_df
            completed += 1
            print(f"  Chunk {completed}/{len(offsets)} ({len(chunk_df):,} rows)")
    ordered = [chunk_dict[o] for o in sorted(chunk_dict.keys())]
    df = pd.concat(ordered, ignore_index=True)
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype(np.int32)
    print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df

def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    deadline = pd.to_datetime(df[DATE_COLUMN])
    planned = pd.to_datetime(df[SECOND_DATE_COLUMN])
    df["has_slack"] = ((deadline.dt.normalize() - planned.dt.normalize()).dt.days >= 1).astype(np.int32)
    print(f"  has_slack — {df['has_slack'].mean()*100:.1f}% of shipments")
    return df

def engineer_load_create_lead(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    planned = pd.to_datetime(df[SECOND_DATE_COLUMN])
    created = pd.to_datetime(df[THIRD_DATE_COLUMN])
    df["load_create_lead_days"] = (
        (planned.dt.normalize() - created.dt.normalize()).dt.days.abs().astype(np.float32)
    )
    print(f"  load_create_lead_days — median {df['load_create_lead_days'].median():.0f}, "
          f"mean {df['load_create_lead_days'].mean():.1f}")
    return df

def report_late_shipments_2024():
    query = f"""
        SELECT COUNT(*) AS total_shipments,
               SUM(CASE
                   WHEN CAST({DATE_COLUMN} AS DATE) - CAST({SECOND_DATE_COLUMN} AS DATE) >= 1
                   THEN 1 ELSE 0
               END) AS late_shipments
        FROM {DATABASE_NAME}.{VIEW_NAME}
        WHERE EXTRACT(YEAR FROM {DATE_COLUMN}) = 2024
    """
    with get_teradata_connection() as conn:
        result = pd.read_sql(query, conn)
    total = result["total_shipments"].iloc[0]
    late = result["late_shipments"].iloc[0]
    pct = (late / total * 100) if total > 0 else 0
    print("=" * 60)
    print("2024 LATE SHIPMENT SUMMARY (ALL RAW DATA)")
    print("=" * 60)
    print(f"  Total shipments:  {total:,}")
    print(f"  Late shipments:   {late:,}")
    print(f"  Late rate:        {pct:.1f}%")
    print()
    return total, late


class CategoricalEncoder:
    """Fit label encoders per column, track cardinalities, handle unseen."""

    def __init__(self):
        self.encoders: Dict[str, LabelEncoder] = {}
        self.cardinalities: Dict[str, int] = {}

    def fit(self, df: pd.DataFrame, cat_cols: List[str]) -> "CategoricalEncoder":
        for col in cat_cols:
            le = LabelEncoder()
            vals = df[col].astype(str).fillna("__missing__")
            le.fit(list(vals.unique()) + ["__unseen__"])
            self.encoders[col] = le
            self.cardinalities[col] = len(le.classes_)
        return self

    def transform(self, df: pd.DataFrame, cat_cols: List[str]) -> np.ndarray:
        encoded = np.zeros((len(df), len(cat_cols)), dtype=np.int64)
        for i, col in enumerate(cat_cols):
            le = self.encoders[col]
            vals = df[col].astype(str).fillna("__missing__")
            # Map unseen categories to the __unseen__ index
            unseen_idx = np.where(le.classes_ == "__unseen__")[0][0]
            encoded[:, i] = np.array([
                le.transform([v])[0] if v in le.classes_ else unseen_idx
                for v in vals
            ])
        return encoded


class ShipmentDataset(Dataset):
    def __init__(self, cat_data: np.ndarray, cont_data: np.ndarray, targets: np.ndarray):
        self.cat = torch.tensor(cat_data, dtype=torch.long)
        self.cont = torch.tensor(cont_data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.cat[idx], self.cont[idx], self.targets[idx]
    
    
class FeatureTokenizer(nn.Module):
    """Convert each feature (categorical or continuous) into a token of uniform dimension."""

    def __init__(
        self,
        cat_cardinalities: List[int],
        cat_embed_dims: List[int],
        n_continuous: int,
        token_dim: int,
    ):
        super().__init__()
        self.token_dim = token_dim

        # Categorical: embedding table per feature → linear projection to token_dim
        self.cat_embeddings = nn.ModuleList()
        self.cat_projections = nn.ModuleList()
        for card, edim in zip(cat_cardinalities, cat_embed_dims):
            self.cat_embeddings.append(nn.Embedding(card, edim))
            self.cat_projections.append(nn.Linear(edim, token_dim))

        # Continuous: each scalar → linear projection to token_dim
        self.cont_projections = nn.ModuleList([
            nn.Linear(1, token_dim) for _ in range(n_continuous)
        ])

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim))

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        batch_size = x_cat.size(0)
        tokens = []

        # Categorical tokens
        for i, (emb, proj) in enumerate(zip(self.cat_embeddings, self.cat_projections)):
            tokens.append(proj(emb(x_cat[:, i])))

        # Continuous tokens
        for i, proj in enumerate(self.cont_projections):
            tokens.append(proj(x_cont[:, i : i + 1]))

        # Stack: (batch, n_features, token_dim)
        tokens = torch.stack(tokens, dim=1)

        # Prepend [CLS] token
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        return tokens


class FTTransformer(nn.Module):
    def __init__(
        self,
        cat_cardinalities: List[int],
        cat_embed_dims: List[int],
        n_continuous: int,
        token_dim: int,
        n_layers: int,
        n_heads: int,
        ffn_multiplier: float,
        attn_dropout: float,
        ffn_dropout: float,
    ):
        super().__init__()

        self.tokenizer = FeatureTokenizer(
            cat_cardinalities, cat_embed_dims, n_continuous, token_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=int(token_dim * ffn_multiplier),
            dropout=ffn_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.attn_dropout = nn.Dropout(attn_dropout)

        # Classification head on [CLS] token
        self.head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, 1),
        )

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x_cat, x_cont)
        tokens = self.attn_dropout(tokens)
        encoded = self.transformer(tokens)
        cls_out = encoded[:, 0]  # [CLS] token
        return self.head(cls_out).squeeze(-1)
    
    
def get_device():
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"  CUDA available: {n_gpus} GPU(s)")
        for i in range(n_gpus):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
        return torch.device("cuda"), n_gpus
    print("  No CUDA — using CPU")
    return torch.device("cpu"), 0


def build_model(hp: FTTransformerHyperparameters, cat_encoder: CategoricalEncoder,
                cat_cols: List[str], device: torch.device, n_gpus: int) -> nn.Module:
    """Build FT-Transformer with auto-sized embeddings."""
    cardinalities = [cat_encoder.cardinalities[c] for c in cat_cols]
    embed_dims = [hp.compute_embedding_dim(card) for card in cardinalities]

    # Auto-calculate token_dim if not set
    token_dim = hp.token_dim if hp.token_dim > 0 else int(np.median(embed_dims))
    # Round up to nearest multiple of n_heads
    token_dim = ((token_dim + hp.n_heads - 1) // hp.n_heads) * hp.n_heads

    print(f"  Categorical embeddings:")
    for col, card, edim in zip(cat_cols, cardinalities, embed_dims):
        print(f"    {col}: cardinality={card}, embed_dim={edim}")
    print(f"  Token dim: {token_dim}")
    print(f"  Continuous features: {len(CONTINUOUS_COLUMNS)}")

    model = FTTransformer(
        cat_cardinalities=cardinalities,
        cat_embed_dims=embed_dims,
        n_continuous=len(CONTINUOUS_COLUMNS),
        token_dim=token_dim,
        n_layers=hp.n_layers,
        n_heads=hp.n_heads,
        ffn_multiplier=hp.ffn_multiplier,
        attn_dropout=hp.attn_dropout,
        ffn_dropout=hp.ffn_dropout,
    )

    if n_gpus > 1:
        print(f"  Using DataParallel across {n_gpus} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    return model


def compute_pos_weight(y: np.ndarray, device: torch.device) -> torch.Tensor:
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    weight = neg / pos if pos > 0 else 1.0
    print(f"  pos_weight = {weight:.3f}  (neg={neg:,}, pos={pos:,})")
    return torch.tensor([weight], dtype=torch.float32, device=device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    n_batches = 0
    for cat, cont, targets in loader:
        cat, cont, targets = cat.to(device), cont.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(cat, cont)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    n_batches = 0
    all_proba = []
    all_targets = []
    for cat, cont, targets in loader:
        cat, cont, targets = cat.to(device), cont.to(device), targets.to(device)
        logits = model(cat, cont)
        loss = criterion(logits, targets)
        total_loss += loss.item()
        n_batches += 1
        all_proba.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    avg_loss = total_loss / n_batches
    proba = np.concatenate(all_proba)
    targets = np.concatenate(all_targets)
    return avg_loss, proba, targets

def prepare_features(df: pd.DataFrame, label: str = "Train"):
    drop_cols = [
        col for col in
        EXCLUDE_COLUMNS + [ID_COLUMN, DATE_COLUMN, SECOND_DATE_COLUMN,
                           TARGET_COLUMN, THIRD_DATE_COLUMN]
        if col in df.columns
    ]
    X = df.drop(columns=drop_cols)
    y = df[TARGET_COLUMN].values.astype(np.float32)

    cat_cols = [c for c in BASE_CATEGORICAL_COLUMNS if c in X.columns]
    cont_cols = [c for c in CONTINUOUS_COLUMNS if c in X.columns]

    print(f"  [{label}] Features: {len(cat_cols)} categorical, {len(cont_cols)} continuous")
    print(f"  [{label}] Target rate: {y.mean()*100:.1f}% positive")
    return X, y, cat_cols, cont_cols


def expanding_window_split(df, n_splits=5, min_val_size=100):
    dates = pd.to_datetime(df[DATE_COLUMN])
    sorted_idx = dates.argsort().values
    n = len(sorted_idx)
    initial_train_size = n // (n_splits + 1)
    remaining = n - initial_train_size
    val_size = remaining // n_splits
    splits = []
    for i in range(n_splits):
        train_end = initial_train_size + val_size * i
        val_start = train_end
        val_end = train_end + val_size
        if i == n_splits - 1:
            val_end = n
        train_idx = sorted_idx[:train_end]
        val_idx = sorted_idx[val_start:val_end]
        if len(val_idx) < min_val_size:
            print(f"  Fold {i+1}: SKIPPED — only {len(val_idx)} val samples")
            continue
        splits.append((train_idx, val_idx))
        print(f"  Fold {i+1}: train {len(train_idx):,} | val {len(val_idx):,} | "
              f"val dates {dates.iloc[val_idx].min().date()} → {dates.iloc[val_idx].max().date()}")
    return splits


def train_fold(hp, cat_encoder, cat_cols, cont_cols,
               X_train, y_train, X_val, y_val, device, n_gpus, fold_label=""):
    """Train one model, return best val probabilities and metrics."""

    cat_train = cat_encoder.transform(X_train, cat_cols)
    cont_train = X_train[cont_cols].fillna(0).values.astype(np.float32)

    cat_val = cat_encoder.transform(X_val, cat_cols)
    cont_val = X_val[cont_cols].fillna(0).values.astype(np.float32)

    train_ds = ShipmentDataset(cat_train, cont_train, y_train)
    val_ds = ShipmentDataset(cat_val, cont_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=hp.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=hp.batch_size * 2, shuffle=False,
                            num_workers=2, pin_memory=True)

    model = build_model(hp, cat_encoder, cat_cols, device, n_gpus)
    pos_weight = compute_pos_weight(y_train, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.learning_rate,
                                  weight_decay=hp.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    best_val_loss = float("inf")
    best_proba = None
    patience_counter = 0
    best_state = None

    for epoch in range(1, hp.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_proba, val_targets = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        roc = roc_auc_score(val_targets, val_proba)
        lr_now = optimizer.param_groups[0]["lr"]

        if epoch % 5 == 0 or epoch == 1:
            print(f"  {fold_label}Epoch {epoch:3d} | "
                  f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"ROC={roc:.4f} lr={lr_now:.1e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_proba = val_proba
            patience_counter = 0
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            best_state = {k: v.cpu().clone() for k, v in raw_model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= hp.early_stopping_patience:
                print(f"  {fold_label}Early stopping at epoch {epoch}")
                break

    return best_proba, val_targets, best_val_loss, best_state, model


def run_pipeline(hp: FTTransformerHyperparameters):
    report_late_shipments_2024()

    print("=" * 60)
    print("DEVICE SETUP")
    print("=" * 60)
    device, n_gpus = get_device()
    torch.manual_seed(hp.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(hp.random_state)

    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    df = load_data()
    df = engineer_time_features(df)
    df = engineer_load_create_lead(df)

    print("\n" + "=" * 60)
    print("HOLDOUT SPLIT")
    print("=" * 60)
    dates = pd.to_datetime(df[DATE_COLUMN])
    cutoff = dates.max() - pd.DateOffset(months=2)
    df_train = df.loc[dates < cutoff].reset_index(drop=True)
    df_test = df.loc[dates >= cutoff].reset_index(drop=True)
    print(f"  Train: {len(df_train):,} rows  (before {cutoff.date()})")
    print(f"  Test:  {len(df_test):,} rows  (from {cutoff.date()} onward)")

    print("\n" + "=" * 60)
    print("PREPARING FEATURES")
    print("=" * 60)
    X_train_raw, y_train_full, cat_cols, cont_cols = prepare_features(df_train, "Train")
    X_test_raw, y_test, _, _ = prepare_features(df_test, "Holdout")

    # Fit categorical encoder on training data only
    cat_encoder = CategoricalEncoder().fit(X_train_raw, cat_cols)

    # ── Expanding-window CV ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPANDING-WINDOW CROSS VALIDATION")
    print("=" * 60)
    splits = expanding_window_split(df_train, n_splits=hp.n_cv_folds)

    cv_results = []
    for fold_i, (tr_idx, va_idx) in enumerate(splits, 1):
        print(f"\n--- Fold {fold_i} ---")
        X_tr = X_train_raw.iloc[tr_idx]
        y_tr = y_train_full[tr_idx]
        X_va = X_train_raw.iloc[va_idx]
        y_va = y_train_full[va_idx]

        # Fit encoder on this fold's training data
        fold_encoder = CategoricalEncoder().fit(X_tr, cat_cols)

        proba, targets, val_loss, _, _ = train_fold(
            hp, fold_encoder, cat_cols, cont_cols,
            X_tr, y_tr, X_va, y_va, device, n_gpus, fold_label=f"[F{fold_i}] ",
        )
        roc = roc_auc_score(targets, proba)
        pr = average_precision_score(targets, proba)
        print(f"  Fold {fold_i} FINAL — ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f}")
        cv_results.append(dict(fold=fold_i, roc_auc=roc, pr_auc=pr))

    cv_df = pd.DataFrame(cv_results)
    print(f"\nCV Summary:")
    print(f"  ROC-AUC  mean={cv_df['roc_auc'].mean():.4f}  std={cv_df['roc_auc'].std():.4f}")
    print(f"  PR-AUC   mean={cv_df['pr_auc'].mean():.4f}  std={cv_df['pr_auc'].std():.4f}")

    # ── Final model on full training set ─────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL MODEL — HOLDOUT EVALUATION")
    print("=" * 60)
    y_proba, y_true, _, best_state, final_model = train_fold(
        hp, cat_encoder, cat_cols, cont_cols,
        X_train_raw, y_train_full, X_test_raw, y_test,
        device, n_gpus, fold_label="[Final] ",
    )
    y_pred = (y_proba >= 0.5).astype(int)

    # ── Save results ─────────────────────────────────────────────────
    output_dir = Path.cwd() / "holdout_results"
    output_dir.mkdir(exist_ok=True)

    print("\nClassification Report (holdout):")
    report_text = classification_report(y_test, y_pred, digits=4)
    print(report_text)
    (output_dir / "classification_report.txt").write_text(report_text)

    predictions_df = df_test[[ID_COLUMN, DATE_COLUMN, SECOND_DATE_COLUMN]].copy()
    predictions_df["y_true"] = y_test
    predictions_df["y_proba"] = y_proba
    predictions_df["y_pred"] = y_pred
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    print(f"Saved predictions.csv — {predictions_df.shape[0]:,} rows")

    # ── Plots ────────────────────────────────────────────────────────
    # Confusion matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["No Slack", "Has Slack"]).plot(ax=ax)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ROC curve
    fig, ax = plt.subplots(figsize=(7, 6))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_score = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, label=f"AUC = {roc_score:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set(xlabel="FPR", ylabel="TPR", title="ROC Curve")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    # Precision-Recall curve
    fig, ax = plt.subplots(figsize=(7, 6))
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    pr_score = average_precision_score(y_test, y_proba)
    ax.plot(rec, prec, label=f"AP = {pr_score:.4f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "precision_recall_curve.png", dpi=150)
    plt.close(fig)

    # Feature importance (via embedding norm as proxy)
    fig, ax = plt.subplots(figsize=(10, 8))
    raw_model = final_model.module if isinstance(final_model, nn.DataParallel) else final_model
    importance = {}
    for i, col in enumerate(cat_cols):
        weight = raw_model.tokenizer.cat_embeddings[i].weight.detach().cpu().numpy()
        importance[col] = np.linalg.norm(weight, axis=1).mean()
    for i, col in enumerate(cont_cols):
        weight = raw_model.tokenizer.cont_projections[i].weight.detach().cpu().numpy()
        importance[col] = np.linalg.norm(weight)
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
    names, vals = zip(*sorted_imp)
    ax.barh(list(reversed(names)), list(reversed(vals)))
    ax.set_title("Top 20 Feature Importance (Embedding Weight Norm)")
    plt.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close(fig)

    # Save model weights
    torch.save(best_state, output_dir / "ft_transformer_weights.pt")

    print(f"\nAll results saved to: {output_dir}")
    return final_model, predictions_df, cv_df, cat_encoder


if __name__ == "__main__":
    hp = FTTransformerHyperparameters()
    model, preds, cv, encoder = run_pipeline(hp)
