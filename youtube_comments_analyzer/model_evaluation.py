import json
import mlflow
import dagshub
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Tuple
from pathlib import Path
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix
from mlflow.models import infer_signature

from youtube_comments_analyzer.config import (
    MODELS_DIR,
    MODEL_FILE_NAME,
    VECTORIZER_FILE_NAME,
    PROCESSED_DATA_DIR,
    X_TEST_FILE_NAME,
    Y_TEST_FILE_NAME,
    FIGURES_DIR,
    params,
    MODEL_PATH_FOR_MLFLOW,
    EXPERIMENT_INFO_FILE_NAME,
)

# dagshub.init(
#     repo_owner="SushrutGaikwad",
#     repo_name="youtube-comments-analyzer",
#     mlflow=True,
# )
# mlflow.set_tracking_uri(
#     "https://dagshub.com/SushrutGaikwad/youtube-comments-analyzer.mlflow"
# )


class ModelEvaluation:
    def __init__(
        self,
        model_path: Path,
        pipeline_params: dict,
        vectorizer_path: Path,
        processed_data_dir: Path,
        experiment_name: str,
    ) -> None:
        """Initializes a `ModelEvaluation` object.

        Args:
            model_path (Path): Path of the trained model.
            pipeline_params (dict): Pipeline parameters.
            vectorizer_path (Path): Path of the vectorizer.
            processed_data_dir (Path): Path of the directory with the processed
                data.
            experiment_name (str): Name of the MLFlow experiment.
        """
        self.model_path = model_path
        self.pipeline_params = pipeline_params
        self.vectorizer_path = vectorizer_path
        self.processed_data_dir = processed_data_dir
        self.experiment_name = experiment_name
        self.model = None
        self.vectorizer = None
        self.X_test = None
        self.y_test = None

    def load_trained_model_and_vectorizer(self) -> None:
        """Loads the trained model and the vectorizer.

        Raises:
            RuntimeError: If an error occurs while loading.
        """
        try:
            logger.info("Loading trained model and vectorizer...")
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            with open(self.vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            logger.success("Model and vectorizer loaded.")
        except Exception as e:
            logger.exception("Failed to load model/vectorizer.")
            raise RuntimeError(f"Loading model/vectorizer failed: {e}")

    def load_test_data(self) -> None:
        """Loads the test inputs (features) and outputs (labels).

        Raises:
            RuntimeError: If an error occurs while loading.
        """
        try:
            logger.info("Loading test data...")
            self.X_test = pd.read_csv(self.processed_data_dir / X_TEST_FILE_NAME)
            self.y_test = pd.read_csv(self.processed_data_dir / Y_TEST_FILE_NAME)
            logger.success(
                f"Test data loaded. X_test shape: {self.X_test.shape}, y_test shape: {self.y_test.shape}"
            )
        except Exception as e:
            logger.exception("Failed to load test data.")
            raise RuntimeError(f"Loading test data failed: {e}")

    def evaluate_model(self) -> Tuple[dict, np.ndarray]:
        """Evaluates the model on the test data and gives the classification report
        and the confusion matrix.

        Raises:
            RuntimeError: If an error occurs while evaluation.

        Returns:
            Tuple[dict, np.ndarray]: Classification report and the confusion matrix.
        """
        try:
            logger.info("Evaluating model...")
            y_pred = self.model.predict(self.X_test)
            report = classification_report(
                y_true=self.y_test, y_pred=y_pred, output_dict=True
            )
            cm = confusion_matrix(y_true=self.y_test, y_pred=y_pred)
            logger.success("Model evaluation complete.")
            return report, cm
        except Exception as e:
            logger.exception("Evaluation failed.")
            raise RuntimeError(f"Model evaluation failed: {e}")

    def log_confusion_matrix(self, cm: np.ndarray, dataset_name: str) -> None:
        """Locally logs the confusion matrix in the form of a plot.

        Args:
            cm (np.ndarray): Confusion matrix.
            dataset_name (str): Name of the data (training or test) on which the
                confusion matrix is evaluated.
        """
        try:
            logger.info("Logging confusion matrix...")
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix - {dataset_name}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            filename = f"Confusion_matrix_{dataset_name}.png"
            file_path = FIGURES_DIR / filename
            plt.savefig(file_path)
            mlflow.log_artifact(file_path)
            plt.close()
            logger.success("Confusion matrix logged.")
        except Exception as e:
            logger.exception("Failed to log confusion matrix.")
            raise

    def save_model_info(
        self,
        run_id: str,
        model_path: str,
        file_path: str,
    ) -> None:
        """Saves the run_id and model_path of the model in a JSON file.

        Args:
            run_id (str): Run ID.
            model_path (str): Model path.
            file_path (str): Path of the JSON file to save.
        """
        try:
            logger.info("Saving model info...")
            info = {"run_id": run_id, "model_path": model_path}
            with open(file_path, "w") as f:
                json.dump(info, f, indent=4)
            logger.success(f"Model info saved to {file_path}")
        except Exception as e:
            logger.exception("Failed to save model info.")
            raise

    def run_model_evaluation(self) -> None:
        """Orchestrates model evaluation."""
        try:
            logger.info("Starting model evaluation...")
            mlflow.set_experiment(self.experiment_name)

            with mlflow.start_run() as run:
                for k, v in self.pipeline_params.items():
                    mlflow.log_param(k, v)

                self.load_test_data()
                self.load_trained_model_and_vectorizer()

                input_example = self.X_test.iloc[:5].copy()
                signature = infer_signature(
                    input_example,
                    self.model.predict(self.X_test[:5]),
                )

                mlflow.sklearn.log_model(
                    self.model,
                    MODEL_PATH_FOR_MLFLOW,
                    signature=signature,
                    input_example=input_example,
                )

                mlflow.log_artifact(self.vectorizer_path)
                self.save_model_info(
                    run.info.run_id,
                    MODEL_PATH_FOR_MLFLOW,
                    EXPERIMENT_INFO_FILE_NAME,
                )

                report, cm = self.evaluate_model()

                for label, metrics in report.items():
                    if isinstance(metrics, dict):
                        for metric, value in metrics.items():
                            mlflow.log_metric(f"{label}: {metric} - test", value)

                self.log_confusion_matrix(cm, "test")
                mlflow.set_tags(
                    {
                        "model_type": "LightGBM",
                        "task": "sentiment analysis",
                        "dataset": "reddit comments",
                    }
                )
                logger.success("Model evaluation completed successfully.")
        except Exception as e:
            logger.error(f"Model evaluation pipeline failed: {e}")
            raise


if __name__ == "__main__":
    try:
        dagshub.init(
            repo_owner="SushrutGaikwad",
            repo_name="youtube-comments-analyzer",
            mlflow=True,
        )
        mlflow.set_tracking_uri(
            "https://dagshub.com/SushrutGaikwad/youtube-comments-analyzer.mlflow"
        )

        model_evaluation = ModelEvaluation(
            model_path=MODELS_DIR / MODEL_FILE_NAME,
            pipeline_params=params,
            vectorizer_path=MODELS_DIR / VECTORIZER_FILE_NAME,
            processed_data_dir=PROCESSED_DATA_DIR,
            experiment_name="DVC pipeline runs",
        )
        model_evaluation.run_model_evaluation()
    except Exception as e:
        logger.error(f"Execution failed: {e}")
