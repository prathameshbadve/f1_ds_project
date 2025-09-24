"""
Decorators for measuring time, retrying functions and logging
"""

import functools
import time
from typing import Any, Callable

import pandas as pd

from config.logging import get_logger


def measure_time(func: Callable) -> Callable:
    """
    Simple time measurement decorator.

    Purpose: Just measure and log execution time - nothing else.
    Use case: Performance monitoring, optimization analysis.

    Example:
        @measure_time
        def load_data():
            # ... code

    Output: "load_data executed in 2.34 seconds"
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time

        logger = get_logger(f"performance.{func.__module__}")
        logger.info("%s executed in %.2f seconds", func.__name__, execution_time)

        return result

    return wrapper


def log_operation(func: Callable) -> Callable:
    """
    Full operation logging decorator (renamed from log_execution_time).

    Purpose: Complete operation tracking with start/success/failure logging.
    Use case: Important business operations, data ingestion, critical functions.

    Example:
        @log_operation
        def ingest_session_data(year, gp, session):
            # ... code

    Output:
    - "Starting ingest_session_data"
    - "Completed ingest_session_data in 45.67 seconds" (success)
    - "Failed ingest_session_data after 12.34 seconds: Connection error" (failure)
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()

        # Log operation start
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


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 1.0):
    """
    Retry decorator with exponential backoff.

    Purpose: Handle transient failures (network issues, API rate limits).
    Use case: API calls, file operations, network requests.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each attempt

    Example:
        @retry(max_attempts=3, delay=2.0, backoff=1.5)
        def api_call():
            # ... code that might fail
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = get_logger(func.__module__)

            current_delay = delay
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        logger.error(
                            "Final attempt failed for %s: %s", func.__name__, str(e)
                        )
                        raise
                    else:
                        logger.warning(
                            "Attempt %s/%s failed for %s: %s. Retrying in %.1f...",
                            attempt + 1,
                            max_attempts,
                            func.__name__,
                            str(e),
                            current_delay,
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff

        return wrapper

    return decorator


def log_data_operation(data_name: str = None):
    """
    Specialized decorator for data operations.

    Purpose: Log data-specific information (shapes, memory usage, etc.).
    Use case: Data processing functions that work with DataFrames.

    Args:
        data_name: Optional name for the data being processed

    Example:
        @log_data_operation("Session Laps")
        def process_lap_data(laps_df):
            # ... process data
            return processed_df

    Output:
    - Function timing
    - Input data shape and memory
    - Output data shape and memory
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger("data_processing")
            start_time = time.time()

            for i, arg in enumerate(args):
                if isinstance(arg, pd.DataFrame):
                    name = data_name or f"Input DataFrame {i}"
                    logger.debug(
                        "%s - Input shape: %s, Memory: %.2f MB",
                        name,
                        arg.shape,
                        arg.memory_usage(deep=True).sum() / 1024**2,
                    )

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Log output data info
                if isinstance(result, pd.DataFrame):
                    output_name = data_name or "Output DataFrame"
                    logger.debug(
                        "%s - Output shape: %s, Memory: %.2f MB",
                        output_name,
                        result.shape,
                        result.memory_usage(deep=True).sum() / 1024**2,
                    )

                logger.info(
                    "Data operation %s completed in %.2f seconds",
                    func.__name__,
                    execution_time,
                )
                return result

            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    "Data operation %s failed after %.2f seconds: %s",
                    func.__name__,
                    execution_time,
                    str(e),
                )
                raise

        return wrapper

    return decorator
