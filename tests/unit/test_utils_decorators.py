import logging
from time import sleep

from config.logging import setup_logging, get_logger
from src.utils.logger import log_execution_time

setup_logging()


def test_log_execution_time(caplog):
    """Function to test the log execution time decorator"""

    test_logger = get_logger("test")

    @log_execution_time
    def slow_func(x, y):
        sleep(1)
        test_logger.info("Performing operations of slow_func...")
        return x + y

    with caplog.at_level(logging.DEBUG):
        result = slow_func(2, 3)

    # Check the function result is returned correctly
    assert result == 5

    # Ensure exactly one log entry was created
    assert len(caplog.records) == 3

    # Check log message content
    log_message = caplog.records[-1].message
    assert "Completed slow_func in" in log_message
    assert "seconds" in log_message
