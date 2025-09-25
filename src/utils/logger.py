"""
Logging utilities for data operations
"""

import logging

import pandas as pd

from config.logging import get_logger


def log_data_info(data: pd.DataFrame, data_name: str, logger: logging.Logger = None):
    """Log information about a DataFrame"""
    if logger is None:
        logger = get_logger("data_ingestion")

    logger.info("%s - Shape: %s", data_name, data.shape)
    logger.info(
        "%s - Memory usage: %.2f MB",
        data_name,
        data.memory_usage(deep=True).sum() / 1024**2,
    )
    logger.info("%s - Columns: %s", data_name, list(data.columns))

    # Log missing data
    missing_data = data.isnull().sum()
    if missing_data.sum() > 0:
        logger.warning(
            "%s - Missing data: %s", data_name, missing_data[missing_data > 0].to_dict()
        )
