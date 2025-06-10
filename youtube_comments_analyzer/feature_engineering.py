import pickle

import pandas as pd

from pathlib import Path
from loguru import logger
from typing import Tuple
from sklearn.feature_extraction.text import CountVectorizer

from youtube_comments_analyzer.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    NGRAM_RANGE,
    MAX_FEATURES,
    TRAIN_FILE_NAME,
    TEST_FILE_NAME,
    LABEL_MAPPING,
    X_TRAIN_FILE_NAME,
    Y_TRAIN_FILE_NAME,
    X_TEST_FILE_NAME,
    Y_TEST_FILE_NAME,
    VECTORIZER_FILE_NAME,
)


class FeatureEngineering:
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        models_dir: Path,
        label_mapping: dict,
        max_features: int,
        ngram_range: Tuple[int, int],
    ) -> None:
        """Initializes a `FeatureEngineering` object.

        Args:
            input_dir (Path): Path to the directory containing interim training
                and test data.
            output_dir (Path): Directory path to save the processed training and
                test data.
            models_dir (Path): Path to saved the fitted vectorizer.
            label_mapping (dict): Mapping of original labels to transformed labels.
            max_features (int): Maximum number of features for the `CountVectorizer`.
            ngram_range (Tuple[int, int]): N-gram range to use.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.models_dir = models_dir
        self.label_mapping = label_mapping
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
        )

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads the interim training and test data.

        Raises:
            RuntimeError: If an error occurs while loading.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and test DataFrames.
        """
        try:
            logger.info("Loading the interim training and test data...")
            train_df = pd.read_csv(self.input_dir / TRAIN_FILE_NAME)
            test_df = pd.read_csv(self.input_dir / TEST_FILE_NAME)
            logger.info(
                f"Loaded train shape: {train_df.shape}, loaded test shape: {test_df.shape}"
            )
            return train_df, test_df
        except Exception as e:
            logger.exception("Failed to load interim processed data.")
            raise RuntimeError(f"Data loading failed: {e}")

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies label mapping and removes rows with missing labels.

        Args:
            df (pd.DataFrame): DataFrame to process.

        Raises:
            RuntimeError: Cleaned DataFrame.

        Returns:
            pd.DataFrame: If an error occurs while processing.
        """
        try:
            logger.info("Remapping labels and removing invalid entries...")
            df["category"] = df["category"].map(self.label_mapping)
            df.dropna(subset=["clean_comment", "category"], inplace=True)
            logger.success(f"Done. Data shape after: {df.shape}")
            return df
        except Exception as e:
            logger.exception("Error during label remapping or filtering.")
            raise RuntimeError(f"Preprocessing failed: {e}")

    def input_output_split(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Splits the data into inputs (features) and outputs (labels) for training and testing.

        Args:
            train_df (pd.DataFrame): Training data.
            test_df (pd.DataFrame): Test data.

        Raises:
            RuntimeError: If an error occurs while splitting.

        Returns:
            Tuple: (X_train, y_train, X_test, y_test)
        """
        try:
            logger.info("Splitting into inputs (features) and outputs (labels)...")
            X_train = train_df["clean_comment"]
            y_train = train_df["category"]
            X_test = test_df["clean_comment"]
            y_test = test_df["category"]
            logger.success("Split completed.")
            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.exception("Failed during input-output split.")
            raise RuntimeError(f"Input-output split failed: {e}")

    def vectorize_inputs(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Vectorizes the input data using `CountVectorizer`.

        Args:
            X_train (pd.DataFrame): Training inputs.
            X_test (pd.DataFrame): Test inputs.

        Raises:
            RuntimeError: If an error occurs during vectorization.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Vectorized training and test inputs.
        """
        try:
            logger.info("Vectorizing comments...")
            X_train = self.vectorizer.fit_transform(X_train)
            X_test = self.vectorizer.transform(X_test)
            logger.success(
                f"Vectorization complete. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}"
            )
            return pd.DataFrame(X_train.toarray()), pd.DataFrame(X_test.toarray())
        except Exception as e:
            logger.exception("Vectorization failed.")
            raise RuntimeError(f"Vectorization failed: {e}")

    def save_vectorized_data(
        self,
        X_train_vectorized: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test_vectorized: pd.DataFrame,
        y_test: pd.DataFrame,
    ) -> None:
        """Saves the vectorized inputs and outputs to disk.

        Args:
            X_train_vectorized (pd.DataFrame): Vectorized training inputs.
            y_train (pd.DataFrame): Training outputs.
            X_test_vectorized (pd.DataFrame): Vectorized test inputs.
            y_test (pd.DataFrame): Test outputs.

        Raises:
            RuntimeError: If an error occurs while saving.
        """
        try:
            logger.info("Saving processed data to disk...")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            X_train_vectorized_file_path = self.output_dir / X_TRAIN_FILE_NAME
            y_train_file_path = self.output_dir / Y_TRAIN_FILE_NAME
            X_test_vectorized_file_path = self.output_dir / X_TEST_FILE_NAME
            y_test_file_path = self.output_dir / Y_TEST_FILE_NAME

            X_train_vectorized.to_csv(X_train_vectorized_file_path, index=False)
            y_train.to_csv(y_train_file_path, index=False)
            X_test_vectorized.to_csv(X_test_vectorized_file_path, index=False)
            y_test.to_csv(y_test_file_path, index=False)
            logger.success("Processed data saved successfully.")
        except Exception as e:
            logger.exception("Failed to save processed data.")
            raise RuntimeError(f"Saving vectorized data failed: {e}")

    def save_vectorizer(self) -> None:
        """Saves the fitted vectorizer to disk.

        Raises:
            RuntimeError: If an error occurs while saving.
        """
        try:
            logger.info("Saving vectorizer to disk...")
            self.models_dir.mkdir(parents=True, exist_ok=True)
            vectorizer_file_path = self.models_dir / VECTORIZER_FILE_NAME

            with open(vectorizer_file_path, "wb") as f:
                pickle.dump(self.vectorizer, f)
            logger.success(f"Vectorizer saved to {vectorizer_file_path}")
        except Exception as e:
            logger.exception("Failed to save vectorizer.")
            raise RuntimeError(f"Saving vectorizer failed: {e}")

    def run_feature_engineering(self) -> None:
        """Orchestrates feature engineering.

        Raises:
            RuntimeError: If any step in feature engineering fails.
        """
        try:
            logger.info("Starting feature engineering...")
            train_df, test_df = self.load_data()
            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)
            X_train, y_train, X_test, y_test = self.input_output_split(
                train_df, test_df
            )
            X_train, X_test = self.vectorize_inputs(X_train, X_test)
            self.save_vectorized_data(X_train, y_train, X_test, y_test)
            self.save_vectorizer()
            logger.success("Feature engineering completed successfully.")
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise RuntimeError("run_feature_engineering() failed.") from e


if __name__ == "__main__":
    try:
        feature_engineering = FeatureEngineering(
            input_dir=INTERIM_DATA_DIR,
            output_dir=PROCESSED_DATA_DIR,
            models_dir=MODELS_DIR,
            label_mapping=LABEL_MAPPING,
            max_features=MAX_FEATURES,
            ngram_range=NGRAM_RANGE,
        )
        feature_engineering.run_feature_engineering()
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
