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
        
        model_uri = f"runs:/05ef91976f5e4612815450e45da2a5ac/model"
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