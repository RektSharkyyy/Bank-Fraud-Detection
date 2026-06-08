# Bank Fraud Detection

A comprehensive Machine Learning project designed to detect fraudulent bank transactions. This project includes complete data pipelines, model training workflows with XGBoost, evaluation metrics tracking using MLflow, and inference pipelines.

## 🚀 Features

- **Data Ingestion & Preprocessing:** Robust handling of raw data, feature scaling, and handling imbalanced datasets (using SMOTE).
- **Model Training:** Automated training pipelines using XGBoost with hyperparameter configurations.
- **Experiment Tracking:** Integrated with **MLflow** for tracking metrics (Accuracy, Precision, Recall, F1 Score) and model artifacts.
- **Inference Pipeline:** Pipeline to run predictions on test datasets and generate classification reports.
- **Modular Architecture:** Clean code structure separating source code (`src`), pipelines, and utilities.

## 🛠️ Tech Stack

- **Language:** Python 3
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn, XGBoost, LightGBM
- **Handling Imbalance:** Imbalanced-Learn (SMOTE)
- **Experiment Tracking:** MLflow
- **API & Deployment:** FastAPI, Uvicorn
- **Automation:** Makefile

## 📂 Project Structure

```text
.
├── artifacts/           # Stored datasets, processed data, and models
├── config/              # Configuration files
├── data/                # Raw dataset files
├── mlruns/              # MLflow tracking directory
├── pipelines/           # Execution pipelines
│   ├── data_pipeline.py       # Data processing and preparation
│   ├── model_training.py      # Training XGBoost models
│   └── inference_pipeline.py  # Model evaluation and prediction
├── src/                 # Core source code components
│   ├── data_ingestion.py
│   ├── feature_scaling.py
│   ├── handle_imbalance.py
│   └── model_building.py
├── utils/               # Helper utilities
├── .env                 # Environment variables
├── Makefile             # Automation scripts for setup and execution
├── mlflow.db            # MLflow SQLite backend database
└── requirements.txt     # Python dependencies
```

## 🏁 Getting Started

### Prerequisites

- Python 3.8 or higher installed
- `make` installed (optional, but recommended for running Makefile commands)

### Installation

1. **Clone the repository** (if applicable)
   ```bash
   git clone <repository-url>
   cd "Bank Fraud Detection"
   ```

2. **Setup Environment and Install Dependencies**
   You can use the provided Makefile to set up the virtual environment and install dependencies:
   ```bash
   make install
   ```

3. **Activate the Virtual Environment**
   * On Windows (Powershell):
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   * On Windows (CMD):
     ```cmd
     .\.venv\Scripts\activate.bat
     ```

## 💻 Usage

### 1. Run Data Pipeline
Prepare, scale, and balance the data for training:
```bash
python pipelines/data_pipeline.py
```
*(Or use `make data-pipeline`)*

### 2. Run Model Training
Train the XGBoost model and track experiments in MLflow:
```bash
python pipelines/model_training.py
```

### 3. Run Inference Pipeline
Test the trained model and view the final fraud detection classification report:
```bash
python pipelines/inference_pipeline.py
```

## 📊 MLflow Tracking

This project uses MLflow with an SQLite backend (`mlflow.db`) to track experiments.

To view the MLflow UI, run:
```bash
make mlflow-ui
```
Then, open your browser and navigate to: `http://localhost:5000`

To stop the MLflow server:
```bash
make stop-all
```

## 🧹 Cleaning Up

To clean up generated artifacts and cached runs:
```bash
make clean
```
