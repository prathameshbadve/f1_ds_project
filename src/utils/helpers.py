"""
Helper functions for file operations and data handling
"""

from pathlib import Path
import pandas as pd
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


def ensure_directory(path: Path) -> Path:
    """Ensure directory exists, create if not"""
    path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Directory ensured: {path}")
    return path


def save_data(
    data: pd.DataFrame,
    filepath: Path,
    file_format: str = "parquet",
    compression: str = "snappy",
) -> bool:
    """Save DataFrame to file with specified format"""
    try:
        ensure_directory(filepath.parent)

        if file_format.lower() == "parquet":
            data.to_parquet(filepath, compression=compression, index=False)
        elif file_format.lower() == "csv":
            data.to_csv(filepath, index=False)
        elif file_format.lower() == "json":
            data.to_json(filepath, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        logger.info(f"Data saved: {filepath} ({len(data)} rows, {file_format})")
        return True

    except Exception as e:
        logger.error(f"Failed to save data to {filepath}: {str(e)}")
        return False


def load_data(filepath: Path, file_format: str = "parquet") -> Optional[pd.DataFrame]:
    """Load DataFrame from file"""
    try:
        if not filepath.exists():
            logger.warning(f"File does not exist: {filepath}")
            return None

        if file_format.lower() == "parquet":
            data = pd.read_parquet(filepath)
        elif file_format.lower() == "csv":
            data = pd.read_csv(filepath)
        elif file_format.lower() == "json":
            data = pd.read_json(filepath, orient="records")
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        logger.info(f"Data loaded: {filepath} ({len(data)} rows)")
        return data

    except Exception as e:
        logger.error(f"Failed to load data from {filepath}: {str(e)}")
        return None
