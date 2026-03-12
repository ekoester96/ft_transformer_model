To ensure this README perfectly fits your needs, I have a few clarifying questions before we finalize it:

1. **Cluster Execution:** Since this runs on an NVIDIA GPU cluster, do you need a section detailing the specific job submission commands (e.g., Slurm `sbatch` commands) used to run the script?
2. **Business Context:** The model predicts `has_slack` based on shipment times. Would you like to add a specific summary of *why* the University of Arkansas is researching this (e.g., supply chain optimization, cost reduction)?
3. **Licensing/Citation:** Will this research be published? If so, do you want to include a citation section for the FT-Transformer architecture or an open-source license?

In the meantime, based on the code and configuration files you provided, here is a comprehensive draft for your `README.md` that you can use right away:

```markdown
# FT Transformer Model for Shipment Slack Prediction

## Overview
A PyTorch-based implementation of the Feature Tokenizer Transformer (FT-Transformer) architecture, used by the University of Arkansas for research running on an NVIDIA GPU cluster. 

This repository contains a machine learning pipeline designed to predict schedule "slack" (`has_slack`) for freight shipments. It connects directly to a Teradata database to pull millions of shipment records, automatically engineers time and lead-day features, and trains a deep learning model to classify whether a shipment will have schedule slack.

## Features
* **FT-Transformer Architecture:** Implements a custom PyTorch model that embeds both categorical and continuous features into uniform tokens before passing them through a Transformer encoder.
* **Teradata Integration:** Uses multi-threaded SQL queries to efficiently fetch large datasets in chunks.
* **Expanding-Window Cross Validation:** Built-in time-series validation to prevent data leakage and evaluate model stability over time.
* **Automated Evaluation:** Automatically generates ROC curves, Precision-Recall curves, Confusion Matrices, and Feature Importance plots.
* **Multi-GPU Support:** Automatically detects and utilizes multiple GPUs via `nn.DataParallel` when available on the cluster.

## Prerequisites & Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/ekoester96/ft_transformer_model.git](https://github.com/ekoester96/ft_transformer_model.git)
   cd ft_transformer_model

```

2. Install the required Python dependencies:
```bash
pip install -r requirements.txt

```


*Core dependencies include: `torch`, `pandas`, `teradatasql`, and `scikit-learn`.*

## Configuration

Before running the model, you must configure your Teradata database credentials.

1. Copy the example environment file:
```bash
cp .env.example .env

```


2. Open `.env` and fill in your connection details:
```env
export TD_HOST=your_host
export TD_USERNAME=your_username
export TD_PASSWORD=your_password
export TD_DATABASE=your_database
export TD_VIEW=your_view

```



## Usage

To run the full end-to-end pipeline (data loading, feature engineering, cross-validation, and final holdout evaluation), execute the main script:

```bash
python ft_transformer.py

```

### Pipeline Workflow:

1. **Data Loading:** Pulls 2024 shipment data using the configured Teradata connection.
2. **Preprocessing:** Handles missing values, scales continuous variables, and encodes categorical variables (with handling for unseen categories).
3. **Training:** Trains the FT-Transformer using expanding-window CV (default 5 folds).
4. **Holdout Evaluation:** Tests the final model on the most recent 2 months of data.
5. **Output Generation:** All artifacts are saved to the `holdout_results/` directory.

## Outputs

After a successful run, the script will generate a `holdout_results/` directory containing:

* `ft_transformer_weights.pt`: The trained model weights.
* `predictions.csv`: The raw predictions and probabilities for the holdout dataset.
* `classification_report.txt`: Precision, recall, and F1-scores.
* `roc_curve.png` & `precision_recall_curve.png`: Model performance visualizations.
* `confusion_matrix.png`: True positive/negative breakdown.
* `feature_importance.png`: Top 20 features based on embedding weight norms.

## Project Structure

* `ft_transformer.py`: The core pipeline script containing data loading, PyTorch model definitions, and the training loop.
* `requirements.txt`: Python package dependencies.
* `.env.example`: Template for required environment variables.
* `.gitignore`: Configured to ignore generated files, `.env` secrets, and `holdout_results/`.

```

```
