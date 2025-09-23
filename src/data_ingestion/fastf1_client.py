"""
FastF1 client that uses environment configuration
"""

from typing import List, Optional

import fastf1
import pandas as pd
from fastf1 import Cache

from config.logging import get_logger
from config.settings import fastf1_config

logger = get_logger(__name__)


class FastF1Client:
    """Extending the FastF1Client with data ingestion methods"""

    def __init__(self):
        self.config = fastf1_config

    def get_session(self, year: str, gp: str, session: str):
        """Get a session from the FastF1 API"""

        try:
            # Use FastF1's get_session function
            session_obj = fastf1.get_session(year, gp, session)

            # Configure session based on environment settings
            if hasattr(session_obj, "load"):
                # Load with telemetry based on configuration
                load_telemetry = self.config.enable_telemetry
                load_weather = self.config.enable_weather
                load_race_control_messages = self.config.enable_race_control_messages

                logger.info("Loading session %d %s %s", year, gp, session)
                logger.info("Telemetry: %s, Weather: %s", load_telemetry, load_weather)

                session_obj.load(
                    telemetry=load_telemetry,
                    weather=load_weather,
                    messages=load_race_control_messages,
                )

            return session_obj

        except Exception as e:
            logger.error("Error loading session %d %s %s: %s", year, gp, session, e)
            raise

    def get_season_schedule(self, year: int) -> pd.DataFrame:
        """Get season schedule with configuration"""
        try:
            schedule = fastf1.get_event_schedule(
                year=year, include_testing=self.config.include_testing
            )
            return schedule
        except Exception as e:
            logger.error("Error loading schedule for %d: %s", year, e)
            raise

    def get_multiple_sessions(
        self, year: int, events: List[str], session_types: Optional[List[str]] = None
    ) -> dict:
        """Get multiple sessions efficiently"""

        if session_types is None:
            session_types = self.config.session_types

        sessions = {}

        for event in events:
            sessions[event] = {}
            for session_type in session_types:
                try:
                    session_obj = self.get_session(year, event, session_type)
                    sessions[event][session_type] = session_obj
                    logger.info("Loaded %d %s %s", year, event, session_type)
                except Exception as e:
                    logger.warning(
                        "Could not load %d %s %s: %s", year, event, session_type, e
                    )
                    sessions[event][session_type] = None

        return sessions

    def cache_info(self) -> tuple:
        """Get cache information"""
        return Cache.get_cache_info()

    def clear_cache(self, deep: bool = False):
        """Clear FastF1 cache"""
        Cache.clear_cache(deep=deep)
        logger.info("Cache cleared (deep=%s)", deep)
