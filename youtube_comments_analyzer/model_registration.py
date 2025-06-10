import json
import mlflow
import dagshub

from pathlib import Path
from loguru import logger
from mlflow.tracking import MlflowClient

from youtube_comments_analyzer.config import EXPERIMENT_INFO_FILE_NAME

dagshub.init(
    repo_owner="SushrutGaikwad",
    repo_name="youtube-comments-analyzer",
    mlflow=True,
)


class ModelRegistration:
    def __init__(self, model_info_file_path: Path, model_name: str) -> None:
        self.model_info_file_path = model_info_file_path
        self.model_name = model_name
        self.model_info = dict()

    def load_model_info(self) -> None:
        try:
            logger.info("Loading model info...")
            with open(self.model_info_file_path, "r") as f:
                self.model_info = json.load(f)
            logger.success("Model info successfully loaded.")
        except FileNotFoundError:
            logger.exception(f"File not found: {self.model_info_path}")
            raise
        except Exception as e:
            logger.exception("Unexpected error while loading model info.")
            raise RuntimeError(f"Failed to load model info: {e}")

    def register_and_stage_model(self) -> None:
        try:
            logger.info("Registering model in MLFlow model registry...")
            model_uri = (
                f"runs:/{self.model_info['run_id']}/{self.model_info['model_path']}"
            )
            model_version = mlflow.register_model(model_uri, self.model_name)

            logger.info("Transitioning model to 'Staging' stage...")
            client = MlflowClient()
            client.transition_model_version_stage(
                name=self.model_name,
                version=model_version.version,
                stage="Staging",
            )
            logger.success(
                f"Model '{self.model_name}', version {model_version.version} registered and transitioned to 'Staging'."
            )
        except Exception as e:
            logger.exception("Model registration or transition failed.")
            raise RuntimeError(f"Model registration failed: {e}")

    def run_model_registration(self) -> None:
        try:
            logger.info("Starting model registration...")
            self.load_model_info()
            self.register_and_stage_model()
            logger.success("Model registration completed successfully.")
        except Exception as e:
            logger.error(f"Model registration pipeline failed: {e}")
            raise


if __name__ == "__main__":
    try:
        model_registration = ModelRegistration(
            model_info_file_path=Path(EXPERIMENT_INFO_FILE_NAME),
            model_name="yt_chrome_plugin_model",
        )
        model_registration.run_model_registration()
    except Exception as e:
        logger.error(f"Execution failed: {e}")
