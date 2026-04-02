import pandas as pd
import logging
from imblearn.over_sampling import SMOTE
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class ImbalanceHandler(ABC):
    @abstractmethod
    def handle(self, X: pd.DataFrame, y: pd.Series):
        pass

class SMOTEHandler(ImbalanceHandler):
    def __init__(self, sampling_strategy='auto', random_state=42):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.smote = SMOTE(sampling_strategy=self.sampling_strategy, random_state=self.random_state)

    def handle(self, X: pd.DataFrame, y: pd.Series):
        try:
            logger.info("Handling class imbalance using SMOTE...")
            
            logger.info(f"Before SMOTE: \n{y.value_counts()}")
            
            X_resampled, y_resampled = self.smote.fit_resample(X, y)
            
            logger.info(f"After SMOTE: \n{y_resampled.value_counts()}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.error(f"Error while handling imbalance: {e}")
            raise e