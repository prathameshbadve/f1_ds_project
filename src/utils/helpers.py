"""
Helper functions for file operations and data handling
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def ensure_directory(path: Path) -> Path:
    """Ensure directory exists, create if not"""
    path.mkdir(parents=True, exist_ok=True)
    logger.debug("Directory ensured: %s", path)
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

        logger.info("Data saved: %s (%s rows, %s)", filepath, len(data), file_format)
        return True

    except Exception as e:
        logger.error("Failed to save data to %s: %s", filepath, str(e))
        return False


def load_data(filepath: Path, file_format: str = "parquet") -> Optional[pd.DataFrame]:
    """Load DataFrame from file"""
    try:
        if not filepath.exists():
            logger.warning("File does not exist: %s", filepath)
            return None

        if file_format.lower() == "parquet":
            data = pd.read_parquet(filepath)
        elif file_format.lower() == "csv":
            data = pd.read_csv(filepath)
        elif file_format.lower() == "json":
            data = pd.read_json(filepath, typ="series").to_frame().T
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        logger.info("Data loaded: %s (%s rows)", filepath, len(data))
        return data

    except Exception as e:
        logger.error("Failed to load data from %s: %s", filepath, str(e))
        return None
