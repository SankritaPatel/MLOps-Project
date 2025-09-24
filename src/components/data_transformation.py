import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from src.utils import clean_text
import logging

logger = logging.getLogger(__name__)

class DataTransformer:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        self.feature_names = []
        self.class_names = []

    def transform(self, df: pd.DataFrame):
        if df.empty:
            logger.error("DataFrame is empty.")
            raise ValueError("Empty dataframe received for transformation.")

        logger.info("Starting transformation.")
        df = df.copy()
        df['short_summary'] = df['short_summary'].fillna("").astype(str).apply(clean_text)
        df['industry'] = df['industry'].fillna("unknown").astype(str)

        y = self.label_encoder.fit_transform(df['industry'])
        self.class_names = list(self.label_encoder.classes_)
        logger.info(f"Encoded target variable: {self.class_names}")

        X_text = self.vectorizer.fit_transform(df['short_summary'])
        text_features = list(self.vectorizer.get_feature_names_out())

        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if num_cols:
            df[num_cols] = df[num_cols].fillna(0)
            scaler = StandardScaler()
            X_num = scaler.fit_transform(df[num_cols])
            X_combined = hstack([X_text, X_num])
            self.feature_names = text_features + num_cols
        else:
            X_combined = X_text
            self.feature_names = text_features

        logger.info(f"Feature matrix shape: {X_combined.shape}")
        return X_combined, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        logger.info("Splitting data into train and test sets.")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
