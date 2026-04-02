import os
import sys
import pandas as pd
import logging
import mlflow
import mlflow.xgboost
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Path setup
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from model_building import XGBoostModelBuilder

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_training_pipeline():
    X_train = pd.read_csv('artifacts/data/processed/X_train.csv')
    y_train = pd.read_csv('artifacts/data/processed/y_train.csv').values.ravel()
    X_test = pd.read_csv('artifacts/data/processed/X_test.csv')
    y_test = pd.read_csv('artifacts/data/processed/y_test.csv').values.ravel()

    # MLflow Experiment
    mlflow.set_experiment("Credit_Card_Fraud_Detection")

    with mlflow.start_run(run_name="XGBoost_Training"):
        logger.info("Starting Model Training...")

        # Build the model
        builder = XGBoostModelBuilder(n_estimators=100, max_depth=6)
        model = builder.build_model()

        # Train model (Fitting)
        model.fit(X_train, y_train)
        logger.info("Model Training Completed!")

        # Predictions (Testing)
        y_pred = model.predict(X_test)

        # Metrics 
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }

        # save on MLflow  (Logging)
        mlflow.log_params({"n_estimators": 100, "max_depth": 6})
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "model")

        logger.info(f"Training Metrics: {metrics}")
        logger.info("Model and Metrics logged to MLflow successfully!")

if __name__ == "__main__":
    run_training_pipeline()