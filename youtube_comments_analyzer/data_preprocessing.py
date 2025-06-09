import re
import nltk

import pandas as pd

from loguru import logger
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Tuple

from youtube_comments_analyzer.config import (
    STOP_WORDS_TO_KEEP,
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
    TRAIN_FILE_NAME,
    TEST_FILE_NAME,
)

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


class DataPreprocessing:
    def __init__(self, input_dir: Path, output_dir: Path) -> None:
        """Initializes a `DataPreprocessing` object.

        Args:
            input_dir (Path): Directory containing the raw train and test data.
            output_dir (Path): Directory where processed train and test data will
                be saved.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.stop_words = set(stopwords.words("english")) - STOP_WORDS_TO_KEEP
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_comment(self, comment: str) -> str:
        """Preprocesses a single comment.

            1. Converts to lowercase,
            2. Removes leading and trailing whitespaces,
            3. Replaces newline character with a single whitespace,
            4. Removes all non-alphanumeric characters, except punctuations,
            5. Removes stop words except a few,
            6. Lemmatizes the comment.

        Args:
            comment (str): Raw comment string.

        Raises:
            RuntimeError: If an error occurs during preprocessing.

        Returns:
            str: Preprocessed comment.
        """
        try:
            comment = comment.lower().strip()
            comment = re.sub(r"\n", " ", comment)
            comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)
            tokens = []
            for word in comment.split():
                if word not in self.stop_words:
                    tokens.append(word)
            lemmatized = []
            for word in tokens:
                lemmatized.append(self.lemmatizer.lemmatize(word))
            return " ".join(lemmatized)
        except Exception as e:
            logger.error(f"Error preprocessing comment: {e}")
            raise RuntimeError(f"Failed to preprocess comment: {e}")

    def preprocess_comments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocesses all comments in the data.

        Args:
            df (pd.DataFrame): Data.

        Raises:
            RuntimeError: If an error occurs while preprocessing.

        Returns:
            pd.DataFrame: Updated data with all comments preprocessed.
        """
        try:
            logger.info("Preprocessing all comments...")
            df["clean_comment"] = df["clean_comment"].apply(self.preprocess_comment)
            logger.info("Preprocessing complete.")
            return df
        except Exception as e:
            logger.exception("Error during text normalization.")
            raise RuntimeError(f"Text normalization failed: {e}")

    def save_data(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
    ) -> None:
        """Saves processed train and test data to the specified output directory.

        Args:
            train_data (pd.DataFrame): Preprocessed training data.
            test_data (pd.DataFrame): Preprocessed test data.

        Raises:
            RuntimeError: If an error occurs while saving.
        """
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            train_data_path = self.output_dir / TRAIN_FILE_NAME
            test_data_path = self.output_dir / TEST_FILE_NAME

            train_data.to_csv(train_data_path, index=False)
            logger.info(f"Train data saved to {train_data_path}")
            test_data.to_csv(test_data_path, index=False)
            logger.info(f"Test data saved to {test_data_path}")
        except Exception as e:
            logger.exception("Failed to save processed datasets.")
            raise RuntimeError(f"Data saving failed: {e}")

    def run_data_preprocessing(self) -> None:
        """Orchestrates the data preprocessing.

            1. Loads raw training and test data,
            2. Preprocesses the comments:
                1. Converts to lowercase,
                2. Removes leading and trailing whitespaces,
                3. Replaces newline character with a single whitespace,
                4. Removes all non-alphanumeric characters, except punctuations,
                5. Removes stop words except a few,
                6. Lemmatizes the comment.
            3. Saves the preprocessed training and test data.

        Raises:
            RuntimeError: If any step in data preprocessing fails.
        """
        try:
            logger.info("Starting data preprocessing...")

            train_df = pd.read_csv(self.input_dir / TRAIN_FILE_NAME)
            test_df = pd.read_csv(self.input_dir / TEST_FILE_NAME)
            logger.info("Raw data loaded.")

            train_processed = self.preprocess_comments(train_df)
            test_processed = self.preprocess_comments(test_df)

            self.save_data(train_processed, test_processed)
            logger.success("Data preprocessing completed successfully.")
        except Exception as e:
            logger.error(f"Preprocessing pipeline failed: {e}")
            raise RuntimeError("run_data_preprocessing() failed.") from e


if __name__ == "__main__":
    try:
        data_preprocessing = DataPreprocessing(
            input_dir=RAW_DATA_DIR, output_dir=INTERIM_DATA_DIR
        )
        data_preprocessing.run_data_preprocessing()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
