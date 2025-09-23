"""Settings configuration"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv
import fastf1
from fastf1 import Cache

# Load environment variables
load_dotenv()


class Config:
    """Base configuration class"""

    def __init__(self):
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.project_root = self._get_project_root()

    def _get_project_root(self) -> Path:
        """Get project root directory"""
        current = Path(__file__).parent.parent
        indicators = ["pyproject.toml", "requirements.txt", ".git"]

        for parent in [current] + list(current.parents):
            if any((parent / indicator).exists() for indicator in indicators):
                return parent
        return current


class FastF1Config(Config):
    """FastF1-specific configuration"""

    def __init__(self):
        super().__init__()

        # Cache settings
        cache_dir = os.getenv("FASTF1_CACHE_DIR", "data/raw/cache")
        self.cache_dir = (
            self.project_root / cache_dir
            if not Path(cache_dir).is_absolute()
            else Path(cache_dir)
        )
        self.cache_enabled = os.getenv("FASTF1_CACHE_ENABLED", "true").lower() == "true"
        self.force_renew = os.getenv("FASTF1_FORCE_RENEW", "false").lower() == "true"

        # Season settings
        self.default_season = int(os.getenv("DEFAULT_SEASON", "2022"))
        self.earliest_season = int(os.getenv("EARLIEST_SEASON", "2022"))

        # Feature flags
        self.enable_telemetry = (
            os.getenv("ENABLE_TELEMETRY_PROCESSING", "true").lower() == "true"
        )
        self.enable_weather = os.getenv("ENABLE_WEATHER_DATA", "true").lower() == "true"
        self.include_testing = (
            os.getenv("INCLUDE_TESTING_SESSIONS", "true").lower() == "true"
        )
        self.race_control_messages = (
            os.getenv("ENABLE_RACE_CONTROL_MESSAGES", "true").lower() == "true"
        )

        # Session types to process
        self.session_types = self._get_session_types()

    def _get_session_types(self) -> List[str]:
        """Get session types based on configuration"""

        types = []
        if os.getenv("PROCESS_PRACTICE_SESSIONS", "true").lower() == "true":
            types.extend(["FP1", "FP2", "FP3"])
        if os.getenv("PROCESS_QUALIFYING_SESSIONS", "true").lower() == "true":
            types.append("Q")
        if os.getenv("PROCESS_RACE_SESSIONS", "true").lower() == "true":
            types.append("R")
        if os.getenv("PROCESS_SPRINT_SESSIONS", "true").lower() == "true":
            types.extend(["SQ", "S"])
        return types

    def setup_fastf1(self):
        """Initialize FastF1 with configuration"""

        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            Cache.enable_cache(
                cache_dir=str(self.cache_dir), force_renew=self.force_renew
            )
        else:
            Cache.set_disabled()

        log_level = os.getenv("FASTF1_LOG_LEVEL", "INFO")
        fastf1.set_log_level(log_level)


class DataConfig(Config):
    """Data storage configuration"""

    def __init__(self):
        super().__init__()

        # Data paths
        raw_path = os.getenv("RAW_DATA_PATH", "data/raw")
        processed_path = os.getenv("PROCESSED_DATA_PATH", "data/processed")

        self.raw_data_path = self.project_root / raw_path
        self.processed_data_path = self.project_root / processed_path

        # File settings
        self.file_format = os.getenv("DATA_FORMAT", "parquet")
        self.compression = os.getenv("COMPRESSION", "snappy")

        # Batch settings
        self.batch_size = int(os.getenv("BATCH_SIZE", "1000"))
        self.parallel_workers = int(os.getenv("PARALLEL_WORKERS", "4"))


# Global config instances
fastf1_config = FastF1Config()
data_config = DataConfig()

# Initialize FastF1
fastf1_config.setup_fastf1()
