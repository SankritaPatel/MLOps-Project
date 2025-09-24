import mlflow
from src.components.data_ingestion import load_sample_data
from src.components.data_transformation import DataTransformer
from src.components.model_trainer import ModelTrainer
from src.mlflow_manager import MLflowManager
from src.logger import logging

def main():
    logging.info("Pipeline execution started.")
    ml_manager = MLflowManager(experiment_name="mlops_pipeline")
    with ml_manager.start_run(run_name="data_pipeline") as run:
        logging.info(f"MLflow run started with ID: {run.info.run_id}")
        # Data Pipeline
        df = load_sample_data()
        transformer = DataTransformer()
        X, y = transformer.transform(df)
        X_train, X_test, y_train, y_test = transformer.split_data(X, y)

        logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        logging.info(f"Target classes: {transformer.class_names}")

        ml_manager.log_params({"num_classes": len(transformer.class_names)})
        ml_manager.log_params({"train_samples": X_train.shape[0]})
        ml_manager.log_params({"test_samples": X_test.shape[0]})


        # Model Training and Evaluation
        trainer = ModelTrainer()
        model = trainer.train(X_train, y_train)
        acc, report = trainer.evaluate(X_test, y_test)

        logging.info(f"Model Accuracy: {acc:.4f}")
        logging.info(f"Evaluation Report: {report}")

        trainer.log_to_mlflow(acc, report)

        ml_manager.log_logs_as_artifact(logging)

    logging.info("Pipeline execution completed.")

if __name__ == "__main__":
    main()
