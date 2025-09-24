"""
Schedule and event data loader
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from config.logging import get_logger
from config.settings import data_config, fastf1_config

from src.utils.decorators import log_operation, measure_time
from src.utils.helpers import ensure_directory, load_data, save_data
from src.utils.logger import log_data_info
from src.data_ingestion.fastf1_client import FastF1Client


class ScheduleLoader:
    """Loads F1 season schedules and event information"""

    def __init__(self, client: Optional[FastF1Client] = None):
        self.client = client or FastF1Client()
        self.config = fastf1_config
        self.data_config = data_config
        self.logger = get_logger("data_ingestion.schedule_loader")

    def _get_schedule_file_path(self, year: int) -> Path:
        """Get the file path for a season schedule"""
        return (
            self.data_config.raw_data_path
            / str(year)
            / f"season_schedule.{self.data_config.file_format}"
        )

    def _is_schedule_file_valid(self, schedule_path: Path, year: int) -> bool:
        """Check if existing schedule file is valid and up-to-date"""

        if not schedule_path.exists():
            self.logger.debug("Schedule file does not exist: %s", schedule_path)
            return False

        try:
            # Check file age
            file_age = datetime.now() - datetime.fromtimestamp(
                schedule_path.stat().st_mtime
            )

            # For current year, refresh if file is older than 1 day
            # For past years, file is valid for longer (7 days)
            current_year = datetime.now().year

            if year == current_year:
                max_age = timedelta(days=1)  # Refresh daily for current season
            elif year == current_year - 1:
                max_age = timedelta(days=7)  # Weekly for last year
            else:
                max_age = timedelta(days=30)  # Monthly for older years

            if file_age > max_age:
                self.logger.info(
                    "Schedule file is outdated (age: %s days): %s",
                    file_age.days,
                    schedule_path,
                )
                return False

            # Try to load and validate basic structure
            schedule_data = load_data(schedule_path, self.data_config.file_format)
            if schedule_data is None or schedule_data.empty:
                self.logger.warning(
                    "Schedule file exists but is empty/invalid: %s", schedule_path
                )
                return False

            # Validate required columns
            required_columns = [
                "EventName",
                "Location",
                "Country",
                "EventFormat",
                "Season",
                "RoundNumber",
                "EventDate",
            ]
            missing_columns = [
                col for col in required_columns if col not in schedule_data.columns
            ]

            if missing_columns:
                self.logger.warning(
                    "Schedule file missing required columns %s: %s",
                    missing_columns,
                    schedule_path,
                )
                return False

            self.logger.debug("Schedule file is valid: %s", schedule_path)
            return True

        except Exception as e:
            self.logger.warning(
                "Error validating schedule file %s: %s", schedule_path, str(e)
            )
            return False

    def _load_schedule_from_file(self, year: int) -> Optional[pd.DataFrame]:
        """Load schedule from local file if it exists and is valid"""

        schedule_path = self._get_schedule_file_path(year)

        if not self._is_schedule_file_valid(schedule_path, year):
            return None

        try:
            schedule = load_data(schedule_path, self.data_config.file_format)
            if schedule is not None:
                self.logger.info(
                    "Loaded schedule from local file: %s (%s events)",
                    schedule_path,
                    len(schedule),
                )
                log_data_info(schedule, f"Cached Season Schedule {year}", self.logger)
            return schedule

        except Exception as e:
            self.logger.error(
                "Failed to load schedule from file %s: %s", schedule_path, str(e)
            )
            return None

    @log_operation
    def load_season_schedule(
        self, year: int, save_to_file: bool = True, force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Load complete season schedule with intelligent caching

        Args:
            year: Season year
            save_to_file: Whether to save to local file
            force_refresh: Force refresh from API even if local file exists
        """

        self.logger.info(
            "Loading season schedule for %d (force_refresh=%s)", year, force_refresh
        )

        # Try to load from local file first (unless force_refresh is True)
        if not force_refresh:
            cached_schedule = self._load_schedule_from_file(year)
            if cached_schedule is not None:
                return cached_schedule

        # Load from API if no valid local file or force_refresh is True
        self.logger.info("Loading schedule from API for %d", year)

        try:
            schedule = self.client.get_season_schedule(year)

            # Add computed columns
            schedule = self._enhance_schedule_data(schedule, year)

            if save_to_file:
                self._save_schedule(schedule, year)

            log_data_info(schedule, f"API Season Schedule {year}", self.logger)
            return schedule

        except Exception as e:
            self.logger.error("Failed to load season schedule for %d: %s", year, str(e))
            raise

    @measure_time
    def get_events_for_ingestion(
        self, year: int, force_refresh: bool = False
    ) -> List[str]:
        """
        Get list of events that should be ingested based on configuration

        Args:
            year: Season year
            force_refresh: Force refresh schedule from API

        Returns:
            List of event names to ingest
        """

        self.logger.info(
            "Getting events for ingestion: %d (force_refresh=%s)", year, force_refresh
        )

        try:
            # Check for cached schedule first, then load from API if needed
            schedule_path = self._get_schedule_file_path(year)

            if not force_refresh and self._is_schedule_file_valid(schedule_path, year):
                # Load from local file
                self.logger.debug("Using cached schedule for %d", year)
                schedule = self._load_schedule_from_file(year)
                if schedule is None:
                    # Fallback to API if file load fails
                    self.logger.warning(
                        "Cached schedule load failed, falling back to API for %d", year
                    )
                    schedule = self.load_season_schedule(
                        year, save_to_file=True, force_refresh=True
                    )
            else:
                # Load from API
                self.logger.debug("Loading fresh schedule from API for %d", year)
                schedule = self.load_season_schedule(
                    year, save_to_file=True, force_refresh=force_refresh
                )

            # Filter events based on configuration
            events_to_ingest = []

            for _, event in schedule.iterrows():
                event_name = event.get("EventName")

                if not event_name or pd.isna(event_name):
                    self.logger.debug("Skipping event with missing name: %s", event)
                    continue

                # Skip testing events if not configured to include them
                if not self.config.include_testing and "Test" in str(event_name):
                    self.logger.debug("Skipping testing event: %s", event_name)
                    continue

                # Skip events without expected sessions (if enhanced data available)
                expected_sessions = event.get(
                    "ExpectedSessions", 1
                )  # Default to 1 if not available
                if expected_sessions == 0:
                    self.logger.debug(
                        "Skipping event with no expected sessions: %s", event_name
                    )
                    continue

                events_to_ingest.append(event_name)

            self.logger.info("Found %s events for ingestion", len(events_to_ingest))

            # Log sample events for verification
            if events_to_ingest:
                sample_events = events_to_ingest[:3]
                self.logger.debug("Sample events: %s", sample_events)

            return events_to_ingest

        except Exception as e:
            self.logger.error("Failed to get events for ingestion %d: %s", year, str(e))
            raise

    def refresh_schedule(self, year: int) -> pd.DataFrame:
        """
        Force refresh schedule from API and update local cache

        Args:
            year: Season year

        Returns:
            Fresh schedule data
        """

        self.logger.info("Force refreshing schedule for %d", year)
        return self.load_season_schedule(year, save_to_file=True, force_refresh=True)

    def get_raw_data_status(self, year: int) -> Dict[str, Any]:
        """
        Get information about the raw schedule files data

        Args:
            year: Season year

        Returns:
            Dictionary with cache status information
        """

        schedule_path = self._get_schedule_file_path(year)

        raw_data_status = {
            "year": year,
            "file_exists": schedule_path.exists(),
            "file_path": str(schedule_path),
            "is_valid": False,
            "file_age_days": None,
            "file_size_mb": None,
            "last_modified": None,
        }

        if raw_data_status["file_exists"]:
            try:
                stat = schedule_path.stat()
                raw_data_status["file_size_mb"] = stat.st_size / (1024 * 1024)
                raw_data_status["last_modified"] = datetime.fromtimestamp(stat.st_mtime)
                raw_data_status["file_age_days"] = (
                    datetime.now() - raw_data_status["last_modified"]
                ).days
                raw_data_status["is_valid"] = self._is_schedule_file_valid(
                    schedule_path, year
                )
            except Exception as e:
                self.logger.warning(
                    "Error getting cache status for %d: %s", year, str(e)
                )

        return raw_data_status

    def _enhance_schedule_data(self, schedule: pd.DataFrame, year: int) -> pd.DataFrame:
        """Enhance schedule data with additional information"""

        self.logger.debug("Enhancing schedule data")

        enhanced_schedule = schedule.copy()

        # Add year column
        enhanced_schedule["Season"] = year

        return enhanced_schedule

    def _save_schedule(self, schedule: pd.DataFrame, year: int):
        """Save schedule to file"""

        schedule_path = (
            self.data_config.raw_data_path
            / str(year)
            / f"season_schedule.{self.data_config.file_format}"
        )
        ensure_directory(schedule_path.parent)

        save_data(
            schedule,
            schedule_path,
            self.data_config.file_format,
            self.data_config.compression,
        )
        self.logger.info("Season schedule saved: %s", schedule_path)
