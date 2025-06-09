import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from youtube_comments_analyzer.utils.common import load_params

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Parameter file content
PARAMS_FILE_PATH = PROJ_ROOT / "params.yaml"
params = load_params(PARAMS_FILE_PATH)

# Data ingestion variables
RAW_DATA_S3_BUCKET_NAME = os.getenv("RAW_DATA_S3_BUCKET_NAME")
RAW_DATA_S3_KEY = os.getenv("RAW_DATA_S3_KEY")
TEST_SIZE = params["data_ingestion"]["test_size"]
RANDOM_STATE = params["data_ingestion"]["random_state"]
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
