"""
Useful decorators for data ingestion
"""

import functools
import time
import logging
from typing import Callable, Any


def retry(max_attempts: int = 3, delay: float = 1.0):
    """Retry decorator for handling transient failures"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = logging.getLogger(func.__module__)

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
                            "Attempt %d failed for %s: %s. Retrying in %.2fs...",
                            attempt + 1,
                            func.__name__,
                            str(e),
                            delay,
                        )
                        time.sleep(delay)

        return wrapper

    return decorator


def measure_time(func: Callable) -> Callable:
    """Measure and log execution time"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time

        logger = logging.getLogger(func.__module__)
        logger.info("%s executed in %.2f seconds", func.__name__, execution_time)

        return result

    return wrapper
