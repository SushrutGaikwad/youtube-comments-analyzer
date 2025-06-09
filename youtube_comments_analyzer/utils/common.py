"""
common.py

This module provides common utility functions that will be used across this
project.
"""

import yaml

from loguru import logger
from pathlib import Path


def load_params(file_path: Path) -> dict:
    """Loads parameters from a parameter file.

    Args:
        file_path (Path): Path of the parameter file.

    Raises:
        e: If the file is not found.
        e: If error occurs during YAML parsing.
        e: If an unexpected error occurs.

    Returns:
        dict: Content of the parameter file.
    """
    try:
        logger.info(f"Attempting to load parameters from {file_path}")
        with open(file_path, "r") as f:
            params = yaml.safe_load(f)
        logger.success("Parameters successfully loaded.")
        return params
    except FileNotFoundError as e:
        logger.error(f"Parameter file not found: {file_path}")
        raise e
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in file {file_path}: {e}")
        raise e
    except Exception as e:
        logger.exception(
            f"An unexpected error occurred while loading params from {file_path}"
        )
        raise e
