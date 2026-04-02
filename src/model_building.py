import logging
from xgboost import XGBClassifier
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class ModelBuilder(ABC):
    @abstractmethod
    def build_model(self):
        pass

class XGBoostModelBuilder(ModelBuilder):
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': random_state,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }

    def build_model(self):
        try:
            logger.info(f"Building XGBoost model with params: {self.params}")
            model = XGBClassifier(**self.params)
            return model
        except Exception as e:
            logger.error(f"Error building model: {e}")
            raise e