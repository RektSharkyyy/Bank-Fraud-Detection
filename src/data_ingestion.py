import os
import pandas as pd
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        pass

class DataIngestorCSV(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        try:
            logger.info(f"Ingesting data from CSV: {file_path}")
            df = pd.read_csv(file_path)
            # Fraud (1) සහ Legit (0) කීයක් තියෙනවාද කියලා මෙතනින්ම බලමු
            logger.info(f"Loaded {len(df)} rows. Class counts: \n{df['Class'].value_counts()}")
            return df
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            raise e

class DataIngesterExcel(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        try:
            logger.info(f"Ingesting data from Excel: {file_path}")
            return pd.read_excel(file_path)
        except Exception as e:
            logger.error(f"Error reading Excel: {e}")
            raise e