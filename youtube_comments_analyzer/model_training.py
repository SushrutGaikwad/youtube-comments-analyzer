import pickle

import pandas as pd

from pathlib import Path
from loguru import logger
from lightgbm import LGBMClassifier
from typing import Tuple, Any

from youtube_comments_analyzer.config import (
    MODEL_PARAMS,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    X_TRAIN_FILE_NAME,
    Y_TRAIN_FILE_NAME,
    MODEL_FILE_NAME,
)


class ModelTraining:
    def __init__(
        self,
        processed_data_dir: Path,
        models_dir: Path,
        model_params: dict[str, Any],
    ) -> None:
        """Initializes a `ModelTraining` object.

        Args:
            processed_data_dir (Path): Path to the directory containing the processed
                training data.
            models_dir (Path): Path to the directory to save the trained model.
            model_params (dict): Model parameters.
        """
        self.processed_data_dir = processed_data_dir
        self.models_dir = models_dir
        self.model_params = model_params
        self.model = LGBMClassifier(**self.model_params)

    def load_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads the training inputs (features) and outputs (labels).

        Raises:
            RuntimeError: If an error occurs during loading.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (X_train, y_train).
        """
        try:
            logger.info("Loading training data...")
            X_train = pd.read_csv(self.processed_data_dir / X_TRAIN_FILE_NAME)
            y_train = pd.read_csv(self.processed_data_dir / Y_TRAIN_FILE_NAME).squeeze()
            logger.success(
                f"Training data loaded: X shape {X_train.shape}, y shape {y_train.shape}"
            )
            return X_train, y_train
        except Exception as e:
            logger.exception("Failed to load training data.")
            raise RuntimeError(f"Data loading failed: {e}")

    def train_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """Trains the LightGBM model.

        Args:
            X_train (pd.DataFrame): Training inputs (features).
            y_train (pd.DataFrame): Training outputs (labels).

        Raises:
            RuntimeError: If an error occurs during training.
        """
        try:
            logger.debug(f"Training the model with parameters {self.model_params}")
            self.model.fit(X_train, y_train)
            logger.success("Model training completed.")
        except Exception as e:
            logger.exception("Model training failed.")
            raise RuntimeError(f"Training failed: {e}")

    def save_model(self) -> None:
        """Saves the trained model to disk.

        Raises:
            RuntimeError: If an error occurs during saving.
        """
        try:
            logger.info("Saving the model to disk...")
            self.models_dir.mkdir(parents=True, exist_ok=True)
            model_file_path = self.models_dir / MODEL_FILE_NAME

            with open(model_file_path, "wb") as f:
                pickle.dump(self.model, f)
            logger.success(f"Model successfully saved to {model_file_path}")
        except Exception as e:
            logger.exception("Failed to save the model.")
            raise RuntimeError(f"Saving model failed: {e}")

    def run_model_training(self) -> None:
        """Orchestrates model training."""
        try:
            logger.info("Starting model training...")
            X_train, y_train = self.load_training_data()
            self.train_model(X_train, y_train)
            self.save_model()
            logger.success("Model training completed successfully.")
        except Exception as e:
            logger.error(f"Model training pipeline failed: {e}")
            raise


if __name__ == "__main__":
    try:
        model_training = ModelTraining(
            processed_data_dir=PROCESSED_DATA_DIR,
            models_dir=MODELS_DIR,
            model_params=MODEL_PARAMS,
        )
        model_training.run_model_training()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
