"""
Loader for telemetry data for all laps for an F1 sessions
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from config.logging import get_logger
from config.settings import data_config, fastf1_config
from src.data_ingestion.fastf1_client import FastF1Client
from src.utils.decorators import log_operation, measure_time
from src.utils.helpers import ensure_directory, load_data, save_data
from src.utils.logger import log_data_info


class TelemetryLoader:
    """Loads and processes F1 telemetry data"""

    def __init__(self, client: Optional[FastF1Client] = None):
        self.client = client or FastF1Client()
        self.config = fastf1_config
        self.data_config = data_config
        self.logger = get_logger("data_ingestion.telemetry_loader")
        self.file_formats = {
            "car_data": self.data_config.file_format,
            "pos_data": self.data_config.file_format,
        }

    def _get_session_telemetry_base_path(
        self, year: int, gp: str, session: str
    ) -> Path:
        """Get the base path of the session storage location"""
        base_path = (
            self.data_config.raw_data_path / str(year) / gp / session / "telemetry"
        )
        return base_path

    def _get_file_path(self, year: int, gp: str, session: str, file: str) -> Path:
        """Get the complete file path for a specific file type of the session"""
        base_path = self._get_session_telemetry_base_path(year, gp, session)
        file_format = self.file_formats[file]
        return base_path / f"{file}.{file_format}"

    def _is_telemetry_file_valid(
        self, year: int, gp: str, session: str, file: str
    ) -> bool:
        """
        Checks if session files exist already.
        If yes it also checks if they are valid.
        """

        filepath = self._get_file_path(year, gp, session, file)

        if not filepath.exists():
            self.logger.debug("File does not exist: %s", filepath)
            return False

        # Basic validation - check if file is not empty
        try:
            if filepath.stat().st_size == 0:
                self.logger.warning("File is empty: %s", filepath)
                return False

        except Exception as e:
            self.logger.warning("File validation failed for %s: %s", filepath, str(e))
            return False

        self.logger.debug("File validation passed: %s", filepath)
        return True

    def _load_telemetry_data_from_file(
        self, year: int, gp: str, session: str, file: str
    ):
        """Load telemetry data from local file if available and valid"""

        if not self._is_telemetry_file_valid(year, gp, session, file):
            self.logger.debug(
                "No valid cached data for %d %s %s %s", year, gp, session, file
            )
            return None

        filepath = self._get_file_path(year, gp, session, file)

        try:
            if self.file_formats[file] == "json":
                with open(filepath, "r", encoding="utf-8") as f:
                    tele_data = json.load(f)
                    self.logger.info("Data loaded from: %s", filepath)
            else:
                tele_data = load_data(filepath, self.file_formats[file])

            if tele_data is not None:
                self.logger.info(
                    "Loaded cached %s data for: %d %s %s",
                    file,
                    year,
                    gp,
                    session,
                )

                if isinstance(tele_data, pd.DataFrame):
                    log_data_info(
                        tele_data,
                        f"Loaded cached {file} dataframe for {year} {gp} {session}",
                        self.logger,
                    )

                return tele_data

            self.logger.warning("Cached file exists bu data is None: %s", filepath)
            return None

        except Exception as e:
            self.logger.error(
                "Failed to load cached %s from %s: %s",
                file,
                filepath,
                str(e),
            )
            return None

    @log_operation
    def load_telemetry_data(
        self, year: int, gp: str, session: str, force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Load comprehensive telemetry data

        Args:
            year
            gp
            session
            force_refresh

        Returns:
            Dictionary containing car_data and pos_data
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
            cached_data, missing_files = self._check_complete_telemetry_cache(
                year, gp, session
            )

            # If complete cache is available, return it
            if not missing_files:
                self.logger.info(
                    "Using complete cached data for %d %s %s", year, gp, session
                )
                return cached_data

            # If we have partial cache, return available files and load the missing data
            if missing_files:
                self.logger.info("Loading missing data from API: %s", missing_files)
                return self._load_partial_telemetry_data(
                    year, gp, session, cached_data, missing_files
                )

        # Step 2: Load complete telemetry from API
        self.logger.info(
            "Loading complete telemetry data from API: %d %s %s", year, gp, session
        )
        return self._load_complete_telemetry_from_api(year, gp, session)

    def _check_complete_telemetry_cache(
        self, year: int, gp: str, session: str
    ) -> Dict[str, Any]:
        """Check if telemetry data available in cache"""

        self.logger.debug(
            "Checking cache for telemetry data files: %d %s %s", year, gp, session
        )

        cached_data = {}
        missing_files = []

        # Check each file type
        for file in self.file_formats:
            cached_item = self._load_telemetry_data_from_file(year, gp, session, file)

            if cached_item is not None:
                cached_data[file] = cached_item
                self.logger.debug(
                    "Found cached file for %d %s %s %s",
                    year,
                    gp,
                    session,
                    file,
                )
            else:
                cached_data[file] = None
                missing_files.append(file)
                self.logger.debug(
                    "Missing cached file %s for %d %s %s",
                    file,
                    year,
                    gp,
                    session,
                )

        if missing_files:
            self.logger.info(
                "Partial telemetry cache for %d %s %s. Missing %s",
                year,
                gp,
                session,
                missing_files,
            )
        else:
            self.logger.info(
                "Complete cache found for telemetry for %d %s %s",
                year,
                gp,
                session,
            )

        return cached_data, missing_files

    def _load_partial_telemetry_data(
        self,
        year: int,
        gp: str,
        session: str,
        cached_data: Dict[str, Any],
        missing_files: List[str],
    ) -> Dict[str, Any]:
        """Load only the missing files from API and merge with cached data files"""

        try:
            # Get session object from FastF1
            session_obj = self.client.get_session(year, gp, session)

            if not hasattr(session_obj, "laps") or session_obj.laps.empty:
                self.logger.warning(
                    "No lap data available, as a result no telemetry data can be loaded"
                )
                return None

            # Mapping of files to extraction methods
            extraction_methods = {
                "car_data": self._extract_car_data,
                "pos_data": self._extract_pos_data,
            }

            # Load missing data
            newly_loaded_data = {}
            for file in missing_files:
                if file in extraction_methods:
                    self.logger.info(
                        "Loading %s from API for %d %s %s",
                        file,
                        year,
                        gp,
                        session,
                    )
                    newly_loaded_data[file] = extraction_methods[file](session_obj)
                else:
                    self.logger.warning("Unknown file type: %s", file)
                    newly_loaded_data[file] = None

            # Save newly loaded data
            if newly_loaded_data:
                self._save_telemetry_data(newly_loaded_data, year, gp, session)

            # Merge cached and new loaded data
            final_data = cached_data.copy()
            final_data.update(newly_loaded_data)

            self.logger.info(
                "Successfully merged cached and API data %s for %d %s %s",
                missing_files,
                year,
                gp,
                session,
            )

            return final_data

        except Exception as e:
            self.logger.error(
                "Failed to load partial telemetry data for %d %s %s: %s",
                year,
                gp,
                session,
                e,
            )
            raise

    def _load_complete_telemetry_from_api(
        self, year: int, gp: str, session: str
    ) -> Dict[str, Any]:
        """Load complete telemetry data from API"""

        try:
            # Get session object from FastF1
            session_obj = self.client.get_session(year, gp, session)

            if not hasattr(session_obj, "laps") or session_obj.laps.empty:
                self.logger.warning(
                    "No lap data available, as a result no telemetry data can be loaded"
                )
                return None

            # Extract all data types
            telemetry_data = {
                "car_data": self._extract_car_data(session_obj),
                "pos_data": self._extract_pos_data(session_obj),
            }

            # Save all data
            self._save_telemetry_data(telemetry_data, year, gp, session)

            self.logger.info(
                "Successfully loaded complete telemetry data for %d %s %s",
                year,
                gp,
                session,
            )
            return telemetry_data

        except Exception as e:
            self.logger.error(
                "Failed to load complete telemetry data for %d %s %s: %s",
                year,
                gp,
                session,
                str(e),
            )
            raise

    def _extract_car_data(self, session_obj) -> Optional[pd.DataFrame]:
        """Extract the car data"""

        self.logger.debug("Extracting telemtry car data")

        laps = session_obj.laps.copy()
        car_data_dfs = []

        for index, lap in laps.iterlaps():
            self.logger.info("Getting telemetry car data %d/%d", index, laps.shape[0])
            car_data = lap.get_car_data().copy()

            # Add lap metadata to the car data table
            car_data["DriverNumber"] = lap["DriverNumber"]
            car_data["LapNumber"] = lap["LapNumber"]

            # Append the dataframe to the car_data_dfs list
            car_data_dfs.append(car_data)

        all_car_data = pd.concat(car_data_dfs, ignore_index=True)

        log_data_info(all_car_data, "Telemetry Car Data", self.logger)
        return all_car_data

    def _extract_pos_data(self, session_obj) -> Optional[pd.DataFrame]:
        """Extract the pos data"""

        self.logger.debug("Extract telemetry pos data")

        laps = session_obj.laps.copy()
        pos_data_dfs = []

        for index, lap in laps.iterlaps():
            self.logger.info("Getting telemetry pos data %d/%d", index, laps.shape[0])
            pos_data = lap.get_pos_data().copy()

            # Add lap metadata to the pos data table
            pos_data["DriverNumber"] = lap["DriverNumber"]
            pos_data["LapNumber"] = lap["LapNumber"]

            # Append the dataframe to the pos_data_dfs list
            pos_data_dfs.append(pos_data)

        all_pos_data = pd.concat(pos_data_dfs, ignore_index=True)

        log_data_info(all_pos_data, "Telemetry Pos Data", self.logger)
        return all_pos_data

    def _save_telemetry_data(
        self, telemetry_data: Dict[str, Any], year: int, gp: str, session: str
    ):
        """Save telemetry data to files"""

        self.logger.debug("Saving telemetry data to cache: %d %s %s", year, gp, session)

        # Create directory structure
        base_path = self.data_config.raw_data_path / str(year) / gp / session
        ensure_directory(base_path)

        # Save each telemetry data component
        for file, data in telemetry_data.items():
            if data is None:
                self.logger.debug("Skipping None data for %s", file)
                continue

            try:
                if isinstance(data, pd.DataFrame):
                    filepath = self._get_file_path(year, gp, session, file)
                    save_data(
                        data,
                        filepath,
                        self.file_formats[file],
                        self.data_config.compression,
                    )
                    self.logger.debug("Save DataFrame %s: %s", file, filepath)

                elif isinstance(data, dict):
                    filepath = self._get_file_path(year, gp, session, file)
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, default=str)
                    self.logger.debug("Saved dict %s: %s", file, filepath)

                else:
                    self.logger.warning(
                        "Unknown data type for %s: %s", file, type(data)
                    )

            except Exception as e:
                self.logger.error("Failed to save %s: %s", file, str(e))

    def is_telemetry_cached(self, year: int, gp: str, session: str) -> Dict[str, bool]:
        """Check which telemetry files are available in cache for a session"""
        cache_status = {}

        for file in self.file_formats:
            cache_status[file] = self._is_telemetry_file_valid(year, gp, session, file)

        return cache_status

    def get_telemetry_cache_summary(
        self, year: int, gp: str, session: str
    ) -> Dict[str, Any]:
        """Get detailed cache summary for a session's telemetry"""

        base_path = self._get_session_telemetry_base_path(year, gp, session)
        cache_status = self.is_telemetry_cached(year, gp, session)

        summary = {
            "session_id": f"{year}_{gp}_{session}",
            "base_path": str(base_path),
            "cache_exits": base_path.exists(),
            "cached_files": [dt for dt, cached in cache_status.items() if cached],
            "missing_files": [dt for dt, cached in cache_status.items() if not cached],
            "cache_complete": all(cache_status.values()),
            "cache_partial": any(cache_status.values())
            and not all(cache_status.values()),
            "file_details": {},
        }

        # Get file size information
        for file, is_cached in cache_status.items():
            if is_cached:
                filepath = self._get_file_path(year, gp, session, file)
                if filepath.exists():
                    summary["file_details"][file] = {
                        "size_mb": filepath.stat().st_size / (1024 * 1024),
                        "path": str(filepath),
                    }

        return summary

    @measure_time
    def load_multiple_sessions_telemetry(
        self, year: int, events: List[str], session_types: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Load telemetry of multiple sessions efficiently"""

        if session_types is None:
            session_types = self.config.session_types

        self.logger.info(
            "Loading telemetry of multiple sessions: %d, %d events, %d session types",
            year,
            len(events),
            len(session_types),
        )

        all_data = {}

        for event in events:
            self.logger.info("Processing telemetry for event: %s", event)
            all_data[event] = {}

            for session_type in session_types:
                try:
                    session_data = self.load_telemetry_data(year, event, session_type)
                    all_data[event][session_type] = session_data
                    self.logger.debug("Loaded %d %s %s", year, event, session_type)

                except Exception as e:
                    self.logger.warning(
                        "Failed to load %d %s %s: %s", year, event, session_type, str(e)
                    )
                    all_data[event][session_type] = None

        self.logger.info(
            "Completed loading telemetry for multiple sessions for %d", year
        )
        return all_data
