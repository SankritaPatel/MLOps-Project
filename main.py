import mlflow
from src.components.data_ingestion import load_sample_data
from src.components.data_transformation import DataTransformer
from src.logger import logging

def main():
    logging.info("Pipeline execution started.")

    with mlflow.start_run(run_name="data_pipeline") as run:
        logging.info(f"MLflow run started with ID: {run.info.run_id}")

        df = load_sample_data()
        transformer = DataTransformer()
        X, y = transformer.transform(df)
        X_train, X_test, y_train, y_test = transformer.split_data(X, y)

        logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        logging.info(f"Target classes: {transformer.class_names}")

        mlflow.log_param("num_classes", len(transformer.class_names))
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])

        
        if hasattr(logging, "_log_path"):
            mlflow.log_artifact(logging._log_path, artifact_path="logs")
            logging.info(f"Log file logged to MLflow: {logging._log_path}")

    logging.info("Pipeline execution completed.")

if __name__ == "__main__":
    main()
