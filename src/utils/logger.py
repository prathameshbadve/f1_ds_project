"""
Logging utilities for data operations
"""

import functools
import logging
import time
from typing import Callable

import pandas as pd

from config.logging import get_logger


def log_execution_time(func: Callable) -> Callable:
    """Decorator to log function execution time"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()

        logger.info("Starting %s", func.__name__)

        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info("Completed %s in %.2f seconds", func.__name__, execution_time)
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                "Failed %s after %.2f seconds: %s",
                func.__name__,
                execution_time,
                str(e),
            )
            raise

    return wrapper


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
