"""
Session-specific data loader for F1 sessions
"""

import json
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

import pandas as pd

from config.settings import data_config, fastf1_config
from src.data_ingestion.fastf1_client import FastF1Client
from src.utils.decorators import log_operation, measure_time
from src.utils.helpers import ensure_directory, save_data, load_data
from src.utils.logger import log_data_info


class SessionLoader:
    """Loads and processes F1 session data"""

    def __init__(self, client: Optional[FastF1Client] = None):
        self.client = client or FastF1Client()
        self.config = fastf1_config
        self.data_config = data_config
        self.logger = logging.getLogger("data_ingestion.session_loader")
        self.file_formats = {
            "session_info": "json",
            "laps": self.data_config.file_format,
            "results": self.data_config.file_format,
            "weather": self.data_config.file_format,
        }

    def _get_session_base_path(self, year: int, gp: str, session: str) -> Path:
        """Get the base path of where the data for the session should be saved."""
        base_path = self.data_config.raw_data_path / str(year) / gp / session
        return base_path

    def _get_file_path(self, year: int, gp: str, session: str, file_type: str) -> Path:
        """Get the complete file path for a specific data type"""
        base_path = self._get_session_base_path(year, gp, session)
        file_format = self.file_formats[file_type]
        return base_path / f"{file_type}.{file_format}"

    def _is_session_file_valid(
        self, year: int, gp: str, session: str, file_type: str
    ) -> bool:
        """
        Checks if there are already existing session files.
        If yes it also checks if they are valid.
        """
        filepath = self._get_file_path(year, gp, session, file_type)

        if not filepath.exists():
            self.logger.debug("File does not exist: %s", filepath)
            return False

        # Basic validation - check if file is not empty
        try:
            if filepath.stat().st_size == 0:
                self.logger.warning("File is empty: %s", filepath)
                return False

            # # Additional validation for specific file types
            # if file_type in ["laps", "results", "weather"]:
            #     # Try to load a small sample to verify file integrity
            #     if self.file_formats[file_type] == "parquet":
            #         test_data = pd.read_parquet(filepath, nrows=1)
            #         if test_data.empty:
            #             self.logger.warning("Parquet file appears empty: %s", filepath)
            #             return False

        except Exception as e:
            self.logger.warning("File validation failed for %s: %s", filepath, str(e))
            return False

        self.logger.debug("File validation passed: %s", filepath)
        return True

    def _load_session_data_from_file(
        self, year: int, gp: str, session: str, file_type: str
    ):
        """Load session data from local file if available and valid"""

        if not self._is_session_file_valid(year, gp, session, file_type):
            self.logger.debug(
                "No valid cached data for %d %s %s %s", year, gp, session, file_type
            )
            return None

        filepath = self._get_file_path(year, gp, session, file_type)

        try:
            if self.file_formats[file_type] == "json":
                with open(filepath, "r", encoding="utf-8") as f:
                    session_data = json.load(f)
                    self.logger.info("Data loaded: %s", filepath)
            else:
                session_data = load_data(filepath, self.file_formats[file_type])

            if session_data is not None:
                self.logger.info(
                    "✅ Loaded cached %s data: %d %s %s",
                    file_type,
                    year,
                    gp,
                    session,
                )

                # Log data info for DataFrame types
                if isinstance(session_data, pd.DataFrame):
                    log_data_info(session_data, f"Cached {file_type} data", self.logger)

                return session_data
            else:
                self.logger.warning("Cached file exists but data is None: %s", filepath)
                return None

        except Exception as e:
            self.logger.error(
                "Failed to load cached %s from %s: %s", file_type, filepath, str(e)
            )
            return None

    def _check_complete_session_cache(
        self, year: int, gp: str, session: str
    ) -> Dict[str, Any]:
        """Check if complete session data is available in cache"""

        self.logger.debug(
            "Checking cache for complete session: %d %s %s", year, gp, session
        )

        cached_data = {}
        missing_data_types = []

        # Check each data type
        for data_type in self.file_formats:
            cached_item = self._load_session_data_from_file(
                year, gp, session, data_type
            )

            if cached_item is not None:
                cached_data[data_type] = cached_item
                self.logger.debug(
                    "✅ Found cached %d %s %s %s",
                    year,
                    gp,
                    session,
                    data_type,
                )
            else:
                cached_data[data_type] = None
                missing_data_types.append(data_type)
                self.logger.debug(
                    "❌ Missing cached %s for %d %s %s", data_type, year, gp, session
                )

        if missing_data_types:
            self.logger.info(
                "Partial cache for %d %s %s. Missing: %s",
                year,
                gp,
                session,
                missing_data_types,
            )
        else:
            self.logger.info("✅ Complete cache found for %d %s %s", year, gp, session)

        return cached_data, missing_data_types

    @log_operation
    def load_session_data(
        self, year: int, gp: str, session: str, force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Load comprehensive session data with intelligent caching

        Args:
            year: Season year
            gp: Grand Prix name
            session: Session identifier (Q, R, FP1, etc.)
            force_refresh: Force refresh from API even if cached data exists

        Returns:
            Dictionary containing all session data types
        """

        self.logger.info(
            "Loading session data: %d %s %s (force_refresh=%s)",
            year,
            gp,
            session,
            force_refresh,
        )

        # Step 1: Check cache first (unless force_refresh)
        if not force_refresh:
            cached_data, missing_data_types = self._check_complete_session_cache(
                year, gp, session
            )

            # If we have complete cache, return it
            if not missing_data_types:
                self.logger.info(
                    "Using complete cached data for %d %s %s", year, gp, session
                )
                return cached_data

            # If we have partial cache, we'll need to load missing data from API
            if missing_data_types:
                self.logger.info(
                    "Loading missing data from API: %s", missing_data_types
                )
                return self._load_partial_session_data(
                    year, gp, session, cached_data, missing_data_types
                )

        # Step 2: Load complete session from API
        self.logger.info(
            "Loading complete session from API: %d %s %s", year, gp, session
        )
        return self._load_complete_session_from_api(year, gp, session)

    def _load_partial_session_data(
        self,
        year: int,
        gp: str,
        session: str,
        cached_data: Dict[str, Any],
        missing_data_types: List[str],
    ) -> Dict[str, Any]:
        """Load only missing data types from API and merge with cached data"""

        try:
            # Get session object from FastF1
            session_obj = self.client.get_session(year, gp, session)

            # Mapping of data types to extraction methods
            extraction_methods = {
                "session_info": self._extract_session_info,
                "laps": self._extract_lap_data,
                "results": self._extract_session_results,
                "weather": self._extract_weather_data,
            }

            # Load only missing data
            newly_loaded_data = {}
            for data_type in missing_data_types:
                if data_type in extraction_methods:
                    self.logger.info(
                        "Loading %s from API for %d %s %s",
                        data_type,
                        year,
                        gp,
                        session,
                    )
                    newly_loaded_data[data_type] = extraction_methods[data_type](
                        session_obj
                    )
                else:
                    self.logger.warning("Unknown data type: %s", data_type)
                    newly_loaded_data[data_type] = None

            # Save newly loaded data
            if newly_loaded_data:
                self._save_session_data(newly_loaded_data, year, gp, session)

            # Merge cached and newly loaded data
            final_data = cached_data.copy()
            final_data.update(newly_loaded_data)

            self.logger.info(
                "Successfully merged cached and API data for %d %s %s",
                year,
                gp,
                session,
            )
            return final_data

        except Exception as e:
            self.logger.error(
                "Failed to load partial session data %d %s %s: %s",
                year,
                gp,
                session,
                str(e),
            )
            raise

    def _load_complete_session_from_api(
        self, year: int, gp: str, session: str
    ) -> Dict[str, Any]:
        """Load complete session data from API"""

        try:
            # Get session object from FastF1
            session_obj = self.client.get_session(year, gp, session)

            # Extract all data types
            session_data = {
                "session_info": self._extract_session_info(session_obj),
                "laps": self._extract_lap_data(session_obj),
                "results": self._extract_session_results(session_obj),
                "weather": self._extract_weather_data(session_obj),
            }

            # Save all data
            self._save_session_data(session_data, year, gp, session)

            self.logger.info(
                "Successfully loaded complete session data: %d %s %s", year, gp, session
            )
            return session_data

        except Exception as e:
            self.logger.error(
                "Failed to load complete session data %d %s %s: %s",
                year,
                gp,
                session,
                str(e),
            )
            raise

    def _extract_session_info(self, session_obj) -> Dict[str, Any]:
        """Extract basic session information"""

        self.logger.debug("Extracting session info")

        session_info = {
            "event_name": session_obj.event.EventName
            if hasattr(session_obj, "event")
            else None,
            "location": session_obj.event.Location
            if hasattr(session_obj, "event")
            else None,
            "country": session_obj.event.Country
            if hasattr(session_obj, "event")
            else None,
            "session_name": session_obj.name if hasattr(session_obj, "name") else None,
            "session_date": session_obj.date if hasattr(session_obj, "date") else None,
        }

        # Add event metadata if available
        if hasattr(session_obj, "event"):
            event = session_obj.event
            additional_info = {
                "event_format": getattr(event, "EventFormat", None),
                "round_number": getattr(event, "RoundNumber", None),
                "official_event_name": getattr(event, "OfficialEventName", None),
            }
            session_info.update(additional_info)

        self.logger.debug("Extracted session info: %s", session_info)
        return session_info

    def _extract_lap_data(self, session_obj) -> Optional[pd.DataFrame]:
        """Extract lap timing data"""

        self.logger.debug("Extracting lap data")

        if not hasattr(session_obj, "laps") or session_obj.laps.empty:
            self.logger.warning("No lap data available")
            return None

        laps = session_obj.laps.copy()

        # Add computed columns
        if "LapTime" in laps.columns:
            laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()

        # Add session metadata to each lap
        laps["EventName"] = (
            session_obj.event.EventName if hasattr(session_obj, "event") else None
        )
        laps["SessionName"] = session_obj.name if hasattr(session_obj, "name") else None
        laps["SessionDate"] = session_obj.date if hasattr(session_obj, "date") else None

        log_data_info(laps, "Lap Data", self.logger)
        return laps

    def _extract_session_results(self, session_obj) -> Optional[pd.DataFrame]:
        """Extract session results"""

        self.logger.debug("Extracting session results")

        if not hasattr(session_obj, "results") or session_obj.results.empty:
            self.logger.warning("No results data available")
            return None

        results = session_obj.results.copy()

        # Add session metadata
        results["EventName"] = (
            session_obj.event.EventName if hasattr(session_obj, "event") else None
        )
        results["SessionName"] = (
            session_obj.name if hasattr(session_obj, "name") else None
        )
        results["SessionDate"] = (
            session_obj.date if hasattr(session_obj, "date") else None
        )

        log_data_info(results, "Session Results", self.logger)
        return results

    def _extract_weather_data(self, session_obj) -> Optional[pd.DataFrame]:
        """Extract weather data"""

        if not self.config.enable_weather:
            self.logger.debug("Weather data disabled by configuration")
            return None

        self.logger.debug("Extracting weather data")

        if not hasattr(session_obj, "weather_data") or session_obj.weather_data.empty:
            self.logger.warning("No weather data available")
            return None

        weather = session_obj.weather_data.copy()

        # Add session metadata
        weather["EventName"] = (
            session_obj.event.EventName if hasattr(session_obj, "event") else None
        )
        weather["SessionName"] = (
            session_obj.name if hasattr(session_obj, "name") else None
        )
        weather["SessionDate"] = (
            session_obj.date if hasattr(session_obj, "date") else None
        )

        log_data_info(weather, "Weather Data", self.logger)
        return weather

    def _save_session_data(
        self, session_data: Dict[str, Any], year: int, gp: str, session: str
    ):
        """Save session data to files"""

        self.logger.debug("Saving session data: %d %s %s", year, gp, session)

        # Create directory structure
        base_path = self.data_config.raw_data_path / str(year) / gp / session
        ensure_directory(base_path)

        # Save each data component
        for data_type, data in session_data.items():
            if data is None:
                self.logger.debug("Skipping None data for %s", data_type)
                continue

            try:
                if isinstance(data, pd.DataFrame):
                    filepath = self._get_file_path(year, gp, session, data_type)
                    save_data(
                        data,
                        filepath,
                        self.file_formats[data_type],
                        self.data_config.compression,
                    )
                    self.logger.debug("Saved DataFrame %s: %s", data_type, filepath)

                elif isinstance(data, dict):
                    filepath = self._get_file_path(year, gp, session, data_type)
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, default=str)
                    self.logger.debug("Saved dict %s: %s", data_type, filepath)

                else:
                    self.logger.warning(
                        "Unknown data type for %s: %s", data_type, type(data)
                    )

            except Exception as e:
                self.logger.error("Failed to save %s: %s", data_type, str(e))

    def is_session_cached(self, year: int, gp: str, session: str) -> Dict[str, bool]:
        """Check which data types are available in cache for a session"""
        cache_status = {}

        for data_type in self.file_formats:
            cache_status[data_type] = self._is_session_file_valid(
                year, gp, session, data_type
            )

        return cache_status

    def get_session_cache_summary(
        self, year: int, gp: str, session: str
    ) -> Dict[str, Any]:
        """Get detailed cache status for a session"""
        base_path = self._get_session_base_path(year, gp, session)
        cache_status = self.is_session_cached(year, gp, session)

        summary = {
            "session_id": f"{year}_{gp}_{session}",
            "base_path": str(base_path),
            "cache_exists": base_path.exists(),
            "cached_data_types": [dt for dt, cached in cache_status.items() if cached],
            "missing_data_types": [
                dt for dt, cached in cache_status.items() if not cached
            ],
            "cache_complete": all(cache_status.values()),
            "cache_partial": any(cache_status.values())
            and not all(cache_status.values()),
            "file_details": {},
        }

        # Get file size information
        for data_type, is_cached in cache_status.items():
            if is_cached:
                filepath = self._get_file_path(year, gp, session, data_type)
                if filepath.exists():
                    summary["file_details"][data_type] = {
                        "size_mb": filepath.stat().st_size / (1024 * 1024),
                        "path": str(filepath),
                    }

        return summary

    @measure_time
    def load_multiple_sessions(
        self, year: int, events: List[str], session_types: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Load multiple sessions efficiently"""

        if session_types is None:
            session_types = self.config.session_types

        self.logger.info(
            "Loading multiple sessions: %d, %d events, %d session types",
            year,
            len(events),
            len(session_types),
        )

        all_data = {}

        for event in events:
            self.logger.info("Processing event: %s", event)
            all_data[event] = {}

            for session_type in session_types:
                try:
                    session_data = self.load_session_data(year, event, session_type)
                    all_data[event][session_type] = session_data
                    self.logger.debug("✅ Loaded %d %s %s", year, event, session_type)

                except Exception as e:
                    self.logger.warning(
                        "Failed to load %d %s %s: %s", year, event, session_type, str(e)
                    )
                    all_data[event][session_type] = None

        self.logger.info("Completed loading multiple sessions for %d", year)
        return all_data

    @log_operation
    def get_session_summary(self, year: int, gp: str, session: str) -> Dict[str, Any]:
        """Get a summary of session without loading full data"""

        self.logger.debug("Getting session summary: %d %s %s", year, gp, session)

        try:
            session_obj = self.client.get_session(year, gp, session)

            summary = {
                "event_name": session_obj.event.EventName
                if hasattr(session_obj, "event")
                else None,
                "session_name": session_obj.name
                if hasattr(session_obj, "name")
                else None,
                "session_date": session_obj.date
                if hasattr(session_obj, "date")
                else None,
                "has_laps": hasattr(session_obj, "laps") and not session_obj.laps.empty,
                "has_results": hasattr(session_obj, "results")
                and not session_obj.results.empty,
                "has_weather": hasattr(session_obj, "weather_data")
                and not session_obj.weather_data.empty,
                "lap_count": len(session_obj.laps)
                if hasattr(session_obj, "laps")
                else 0,
                "driver_count": len(session_obj.results)
                if hasattr(session_obj, "results")
                else 0,
            }

            # Add cache information
            summary["cache_info"] = self.get_session_cache_summary(year, gp, session)

            return summary

        except Exception as e:
            self.logger.error(
                "Failed to get session summary %d %s %s: %s", year, gp, session, str(e)
            )
            return {"error": str(e)}
