import logging
import logging.config
import os
from pathlib import Path


def setup_logging():
    """Setup application logging"""

    # Create logs directory
    log_dir = Path("monitoring/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Log configuration
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "simple": {"format": "%(levelname)s - %(message)s"},
            "json": {
                "format": '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "function": "%(funcName)s", "line": %(lineno)d, "message": "%(message)s"}',
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": os.getenv("LOG_LEVEL", "INFO"),
                "formatter": "detailed",
                "stream": "ext://sys.stdout",
            },
            "file_info": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "detailed",
                "filename": log_dir / "app.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
            },
            "file_error": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": log_dir / "error.log",
                "maxBytes": 10485760,
                "backupCount": 5,
            },
            "data_ingestion": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "json",
                "filename": log_dir / "data_ingestion.log",
                "maxBytes": 52428800,  # 50MB
                "backupCount": 10,
            },
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console", "file_info", "file_error"],
                "level": "DEBUG",
                "propagate": False,
            },
            "data_ingestion": {
                "handlers": ["console", "data_ingestion"],
                "level": "DEBUG",
                "propagate": False,
            },
            "fastf1": {
                "handlers": ["console", "file_info"],
                "level": os.getenv("FASTF1_LOG_LEVEL", "INFO"),
                "propagate": False,
            },
        },
    }

    logging.config.dictConfig(log_config)

    # Log startup info
    logger = logging.getLogger(__name__)
    logger.info("Logging configured. Log directory: %s", log_dir.absolute())
    logger.info("Environment: %s", os.getenv("ENVIRONMENT", "development"))


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(name)
