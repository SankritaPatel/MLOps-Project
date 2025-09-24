import sys
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from src.exception import CustomException
from src.logger import logging

class ModelTrainer:
    def __init__(self, model=None):
        self.model = model if model else LogisticRegression(max_iter=500)

    def train(self, X_train, y_train):
        try:
            logging.info("Starting model training...")
            self.model.fit(X_train, y_train)
            logging.info("Model training completed.")
            return self.model
        except Exception as e:
            logging.exception("Model training failed.")
            raise CustomException(e, sys)

    def evaluate(self, X_test, y_test):
        try:
            logging.info("Evaluating model performance...")
            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            logging.info(f"Model Accuracy: {acc:.4f}")
            return acc, report
        except Exception as e:
            logging.exception("Model evaluation failed.")
            raise CustomException(e, sys)

    def log_to_mlflow(self, acc, report):
        mlflow.log_metric("accuracy", acc)
        for label, metrics in report.items():
            if isinstance(metrics, dict):  # skip avg values
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric_name}", value)
        mlflow.sklearn.log_model(self.model, "model")
