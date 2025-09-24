"""
Session-specific data loader for F1 sessions
"""

import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from config.settings import data_config, fastf1_config
from src.data_ingestion.fastf1_client import FastF1Client
from src.utils.decorators import log_operation, measure_time
from src.utils.helpers import ensure_directory, save_data
from src.utils.logger import log_data_info


class SessionLoader:
    """Loads and processes F1 session data"""

    def __init__(self, client: Optional[FastF1Client] = None):
        self.client = client or FastF1Client()
        self.config = fastf1_config
        self.data_config = data_config
        self.logger = logging.getLogger("data_ingestion.session_loader")

    @log_operation
    def load_session_data(self, year: int, gp: str, session: str) -> Dict[str, Any]:
        """Load comprehensive session data"""

        self.logger.info("Loading session data: %d %s %s", year, gp, session)

        try:
            # Get session from FastF1
            session_obj = self.client.get_session(year, gp, session)

            # Extract different types of data
            session_data = {
                "session_info": self._extract_session_info(session_obj),
                "laps": self._extract_lap_data(session_obj),
                "results": self._extract_session_results(session_obj),
                "weather": self._extract_weather_data(session_obj),
                "telemetry": self._extract_telemetry_summary(session_obj),
            }

            # Save data
            self._save_session_data(session_data, year, gp, session)

            self.logger.info(
                "Successfully loaded session data: %d %s %s", year, gp, session
            )
            return session_data

        except Exception as e:
            self.logger.error(
                "Failed to load session data %d %s %s: %s", year, gp, session, str(e)
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

    def _extract_telemetry_summary(self, session_obj) -> Dict[str, Any]:
        """Extract telemetry summary information"""

        if not self.config.enable_telemetry:
            self.logger.debug("Telemetry processing disabled by configuration")
            return {}

        self.logger.debug("Extracting telemetry summary")

        telemetry_summary = {
            "telemetry_available": False,
            "fastest_lap_telemetry": None,
            "total_telemetry_points": 0,
        }

        try:
            if hasattr(session_obj, "laps") and not session_obj.laps.empty:
                # Get fastest lap
                fastest_lap = session_obj.laps.pick_fastest()
                if fastest_lap is not None and hasattr(fastest_lap, "get_telemetry"):
                    telemetry = fastest_lap.get_telemetry()
                    if telemetry is not None and not telemetry.empty:
                        telemetry_summary["telemetry_available"] = True
                        telemetry_summary["total_telemetry_points"] = len(telemetry)
                        telemetry_summary["fastest_lap_driver"] = fastest_lap.get(
                            "Driver", "Unknown"
                        )
                        telemetry_summary["fastest_lap_time"] = fastest_lap.get(
                            "LapTime"
                        )

                        # Basic telemetry stats
                        if "Speed" in telemetry.columns:
                            telemetry_summary["max_speed"] = telemetry["Speed"].max()
                            telemetry_summary["avg_speed"] = telemetry["Speed"].mean()

                        if "RPM" in telemetry.columns:
                            telemetry_summary["max_rpm"] = telemetry["RPM"].max()

        except Exception as e:
            self.logger.warning("Error extracting telemetry summary: %s", str(e))

        self.logger.debug("Telemetry summary: %s", telemetry_summary)
        return telemetry_summary

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
                continue

            filepath = base_path / f"{data_type}.{self.data_config.file_format}"

            if isinstance(data, pd.DataFrame):
                save_data(
                    data,
                    filepath,
                    self.data_config.file_format,
                    self.data_config.compression,
                )
            elif isinstance(data, dict):
                with open(filepath.with_suffix(".json"), "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)
                self.logger.debug("Saved dict data: %s", filepath.with_suffix(".json"))

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

            return summary

        except Exception as e:
            self.logger.error(
                "Failed to get session summary %d %s %s: %s", year, gp, session, str(e)
            )
            return {}
