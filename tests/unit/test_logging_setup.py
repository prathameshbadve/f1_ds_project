"""Test to check if logging works as per the configured settings"""

import logging

from config.logging import setup_logging, get_logger

setup_logging()


def test_logger_level(caplog):
    """Function to test the root logger"""
    logger = get_logger("test")

    # Check that the log level is set to DEBUG as per the config dict
    assert logger.level == logging.DEBUG

    # Check that there is at least one handler attached
    assert logger.handlers

    # Capture log messages with caplog
    with caplog.at_level(logging.DEBUG):
        logger.debug("This is a debug message.")
        logger.info("This is an info message.")
        logger.warning("This is a warning message.")
        logger.error("This is a error message.")
        logger.critical("This is a critical message.")

    # Ensure that the messages are logged at the correct levels.
    levels = [rec.levelno for rec in caplog.records]
    messages = [rec.message for rec in caplog.records]

    assert logging.DEBUG in levels
    assert logging.INFO in levels
    assert logging.WARNING in levels
    assert logging.ERROR in levels
    assert logging.CRITICAL in levels

    assert "This is a debug message." in messages
    assert "This is an info message." in messages
    assert "This is a warning message." in messages
    assert "This is a error message." in messages
    assert "This is a critical message." in messages
