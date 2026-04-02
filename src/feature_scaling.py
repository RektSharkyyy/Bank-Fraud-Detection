import pandas as pd
import logging
from sklearn.preprocessing import RobustScaler
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class FeatureScaler(ABC):
    @abstractmethod
    def scale(self, df: pd.DataFrame, columns: list):
        pass

class RobustScalingStrategy(FeatureScaler):
    def __init__(self):
        self.scaler = RobustScaler()

    def scale(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        try:
            logger.info(f"Scaling columns: {columns}")
            df_scaled = df.copy()
            df_scaled[columns] = self.scaler.fit_transform(df[columns])
            return df_scaled
        except Exception as e:
            logger.error(f"Error during scaling: {e}")
            raise e