import mlflow
from mlflow.tracking import MlflowClient
from src.logger import logging

class MLflowManager:
    def __init__(self, experiment_name="default_experiment", tracking_uri="http://127.0.0.1:5000"):
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
        logging.info(f"MLflow initialized for experiment: {experiment_name}")

    def start_run(self, run_name="ml_pipeline"):
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict):
        for k, v in params.items():
            mlflow.log_param(k, v)

    def log_metrics(self, metrics: dict):
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

    def log_artifact(self, path, artifact_path=None):
        mlflow.log_artifact(path, artifact_path=artifact_path)

    def log_logs_as_artifact(self, logging):
        if hasattr(logging, "_log_path"):
            mlflow.log_artifact(logging._log_path, artifact_path="logs")
            logging.info(f"Log file logged to MLflow: {logging._log_path}")
