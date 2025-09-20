"""
Logging utilities for data operations
"""

import logging
import functools
import time
import pandas as pd
from typing import Callable


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the specified module"""
    return logging.getLogger(name)


def log_execution_time(func: Callable) -> Callable:
    """Decorator to log function execution time"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()

        logger.info(f"Starting {func.__name__}")

        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Completed {func.__name__} in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Failed {func.__name__} after {execution_time:.2f} seconds: {str(e)}"
            )
            raise

    return wrapper


def log_data_info(data: pd.DataFrame, data_name: str, logger: logging.Logger = None):
    """Log information about a DataFrame"""
    if logger is None:
        logger = logging.getLogger("data_ingestion")

    logger.info(f"{data_name} - Shape: {data.shape}")
    logger.info(
        f"{data_name} - Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    )
    logger.info(f"{data_name} - Columns: {list(data.columns)}")

    # Log missing data
    missing_data = data.isnull().sum()
    if missing_data.sum() > 0:
        logger.warning(
            f"{data_name} - Missing data: {missing_data[missing_data > 0].to_dict()}"
        )
