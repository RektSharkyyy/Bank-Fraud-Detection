import os
import sys
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from data_ingestion import DataIngestorCSV
from handle_imbalance import SMOTEHandler
from feature_scaling import RobustScalingStrategy

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_data_pipeline(data_path: str):
    try:
        ingestor = DataIngestorCSV()
        df = ingestor.ingest(data_path)
        
        X = df.drop(columns=['Class', 'Time']) 
        y = df['Class']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        logger.info(f"Initial split done. Train: {len(X_train)}, Test: {len(X_test)}")
        
        scaler_strategy=RobustScalingStrategy()

        logger.info("scaling 'Amount' coloumn...")

        X_train = scaler_strategy.scale(X_train, ['Amount'])
        X_test = scaler_strategy.scale(X_test, ['Amount'])

        handler = SMOTEHandler()
        X_train_resampled, y_train_resampled = handler.handle(X_train, y_train)
        
        os.makedirs('artifacts/data/processed', exist_ok=True)
        X_train_resampled.to_csv('artifacts/data/processed/X_train.csv', index=False)
        X_test.to_csv('artifacts/data/processed/X_test.csv', index=False)
        y_train_resampled.to_csv('artifacts/data/processed/y_train.csv', index=False)
        y_test.to_csv('artifacts/data/processed/y_test.csv', index=False)
        
        logger.info("Data Pipeline completed successfully! Cleaned data saved in artifacts.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise e

if __name__ == "__main__":
    raw_data_path = "data/raw/creditcard.csv"
    run_data_pipeline(raw_data_path)