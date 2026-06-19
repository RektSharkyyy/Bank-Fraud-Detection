import os
import sys
import pandas as pd
import logging
import mlflow.xgboost
from sklearn.metrics import classification_report

# Path setup
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from feature_scaling import RobustScalingStrategy

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_inference(test_data_path: str):
    try:
        X_test = pd.read_csv('artifacts/data/processed/X_test.csv')
        y_test = pd.read_csv('artifacts/data/processed/y_test.csv').values.ravel()
        logger.info("Loading trained model from MLflow...")
        
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("Credit_Card_Fraud_Detection")
        
        runs = mlflow.search_runs(order_by=["start_time desc"], max_results=1)
        if runs.empty:
            raise Exception("No MLflow runs found. Please train a model first.")
        
        latest_run_id = runs.iloc[0].run_id
        model_uri = f"runs:/{latest_run_id}/model"
        logger.info(f"Using model from run: {latest_run_id}")
        model = mlflow.xgboost.load_model(model_uri)

        # Prediction
        logger.info("Making predictions on test data...")
        y_pred = model.predict(X_test)

        # Classification Report
        report = classification_report(y_test, y_pred)
        print("\n--- Final Fraud Detection Report ---")
        print(report)

        # Fraud 
        fraud_indices = [i for i, val in enumerate(y_pred) if val == 1]
        logger.info(f"Detected {len(fraud_indices)} potential fraud cases out of {len(y_pred)} transactions.")

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise e

if __name__ == "__main__":
    run_inference('artifacts/data/processed/X_test.csv')