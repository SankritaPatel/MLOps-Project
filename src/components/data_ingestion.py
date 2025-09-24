import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging

def load_sample_data() -> pd.DataFrame:
    """Load and prepare the dataset"""
    try:
        df = pd.DataFrame({
            'short_summary': [
                'machine learning deployment cloud',
                'financial analysis banking',
                'healthcare AI diagnostics',
                'retail customer analytics',
                'manufacturing IoT'
            ],
            'industry': ['tech', 'finance', 'healthcare', 'retail', 'manufacturing'],
            'year': [2022, 2021, 2023, 2022, 2023],
            'company': ['Company A', 'Company B', 'Company C', 'Company D', 'Company E']
        })
        logging.info(f"Loaded sample data with {len(df)} records.")
        return df
    except Exception as e:
        logging.exception("Failed to load data.")
        raise CustomException(e, sys)
