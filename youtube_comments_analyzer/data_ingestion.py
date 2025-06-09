import boto3

import pandas as pd

from pathlib import Path
from loguru import logger
from typing import Optional
from sklearn.model_selection import train_test_split
from botocore.exceptions import BotoCoreError, ClientError

from youtube_comments_analyzer.config import (
    RAW_DATA_DIR,
    RAW_DATA_S3_BUCKET_NAME,
    RAW_DATA_S3_KEY,
    TRAIN_FILE_NAME,
    TEST_FILE_NAME,
    TEST_SIZE,
    RANDOM_STATE,
)


class DataIngestion:
    def __init__(self, bucket_name: str, s3_key: str) -> None:
        """Initializes a `DataIngestion` object.

        Args:
            bucket_name (str): Name of the S3 bucket containing the raw data.
            s3_key (str): Path to the raw data file in the bucket.
        """
        self.bucket_name = bucket_name
        self.s3_key = s3_key

    def save_raw_data_if_missing(
        self,
        filename: str = "raw_data.csv",
    ) -> Path:
        """
        Downloads and saves the file from S3 only if it does not already exist.

        Args:
            filename (str, optional): name to use when saving locally. Defaults
                to "raw_data.csv".

        Raises:
            RuntimeError: If the download from S3 fails.
            RuntimeError: If an unexpected error occurs.

        Returns:
            Path: Path to the saved (or existing) file.
        """
        local_path = RAW_DATA_DIR / filename
        if local_path.exists():
            logger.info(f"Raw data already exists at {local_path}")
            return local_path

        try:
            logger.info(f"Downloading raw data from S3 bucket...")
            s3 = boto3.client("s3")
            response = s3.get_object(Bucket=self.bucket_name, Key=self.s3_key)

            RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(response["Body"].read())

            logger.success(f"Saved raw data to {local_path}")
            return local_path

        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to download file from S3: {e}")
            raise RuntimeError(f"S3 download failed: {e}")
        except Exception as e:
            logger.exception("Unexpected error during file download.")
            raise RuntimeError(f"Unexpected error: {e}")

    def read_data(
        self,
        filename: str = "raw_data.csv",
    ) -> pd.DataFrame:
        """
        Ensures the raw file exists (downloads it if missing), then loads it into
        a DataFrame.

        Args:
            filename (str, optional): Name of the file to load. Defaults to
                "raw_data.csv".

        Raises:
            RuntimeError: If an error occurs while parsing the file.
            RuntimeError: If an unexpected error occurs.

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        local_path = self.save_raw_data_if_missing(filename=filename)

        try:
            logger.info(f"Reading data from {local_path}")
            df = pd.read_csv(local_path)
            logger.success(f"Loaded data with shape {df.shape}")
            return df
        except pd.errors.ParserError as e:
            logger.error(f"Parsing error in file {local_path}: {e}")
            raise RuntimeError(f"CSV parsing failed: {e}")
        except Exception as e:
            logger.exception("Unexpected error while reading data.")
            raise RuntimeError(f"Unexpected error: {e}")

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the data by dropping missing values, duplicates, and rows
        with empty comments.

        Args:
            df (pd.DataFrame): Raw DataFrame to be cleaned.

        Raises:
            RuntimeError: If 'clean_comment' column is not present.
            RuntimeError: If an unexpected error occurs.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        try:
            logger.info("Dropping missing values, duplicates, and empty comments...")
            initial_shape = df.shape

            df.dropna(inplace=True)
            df.drop_duplicates(inplace=True)
            df = df[df["clean_comment"].str.strip() != ""]
            logger.success(
                f"Task successful. Shape changed from {initial_shape} to {df.shape}"
            )
            return df
        except KeyError as e:
            logger.error(f"Missing 'clean_comment' column in DataFrame: {e}")
            raise RuntimeError("Required column 'clean_comment' not found.")
        except Exception as e:
            logger.exception("Unexpected error during preprocessing.")
            raise RuntimeError(f"Unexpected error during preprocessing: {e}")

    def save_train_test_data(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        raw_data_dir: Optional[Path] = RAW_DATA_DIR,
    ) -> None:
        """Saves the train and test datasets as CSV files.

        Args:
            train_data (pd.DataFrame): Training dataset to save.
            test_data (pd.DataFrame): Test dataset to save.
            raw_data_dir (Path, optional): Directory where data should be saved.
                Defaults to RAW_DATA_DIR.

        Raises:
            RuntimeError: If saving to disk fails.
        """
        try:
            logger.info("Saving train and test datasets...")
            raw_data_dir.mkdir(parents=True, exist_ok=True)

            train_data_path = raw_data_dir / TRAIN_FILE_NAME
            test_data_path = raw_data_dir / TEST_FILE_NAME

            train_data.to_csv(train_data_path, index=False)
            logger.success(f"Train data saved to {train_data_path}")

            test_data.to_csv(test_data_path, index=False)
            logger.success(f"Test data saved to {test_data_path}")
        except Exception as e:
            logger.exception("Failed to save train/test data.")
            raise RuntimeError(f"Error saving train/test data: {e}")

    def run_data_ingestion(self) -> None:
        """Orchestrates data ingestion.

            1. Loads raw data from S3 (only if not already saved)
            2. Does simple preprocessing.
            3. Splits the data into train and test sets.
            4. Saves the train and test sets locally.

        Raises:
            RuntimeError: If any step in data ingestion fails.
        """
        try:
            logger.info("Starting data ingestion...")

            # Loading the raw data
            df = self.read_data()

            # Simple preprocessing
            df = self.preprocess_data(df)

            # Train-test split
            logger.info(
                f"Splitting data into train and test sets (test_size={TEST_SIZE}, random_state={RANDOM_STATE})..."
            )
            train_df, test_df = train_test_split(
                df,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=df["category"],
            )
            logger.success(
                f"Split complete. Train shape: {train_df.shape}, Test shape: {test_df.shape}"
            )

            # Saving training and test sets
            self.save_train_test_data(train_df, test_df)
            logger.success("Data ingestion completed successfully.")
        except Exception as e:
            logger.error(f"Data ingestion pipeline failed: {e}")
            raise RuntimeError("run_data_ingestion() failed.") from e


if __name__ == "__main__":
    try:
        data_ingestion = DataIngestion(
            bucket_name=RAW_DATA_S3_BUCKET_NAME, s3_key=RAW_DATA_S3_KEY
        )
        data_ingestion.run_data_ingestion()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
