"""
Main data fetcher orchestrator
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from config.settings import data_config, fastf1_config
from src.data_ingestion.fastf1_client import FastF1Client
from src.data_ingestion.schedule_loader import ScheduleLoader
from src.data_ingestion.session_loader import SessionLoader
from src.data_ingestion.telemetry_loader import TelemetryLoader
from src.utils.decorators import log_operation, measure_time
from src.utils.helpers import ensure_directory, load_data


class DataFetcher:
    """Main orchestrator for F1 data fetching operations"""

    def __init__(self):
        self.client = FastF1Client()
        self.session_loader = SessionLoader(self.client)
        self.schedule_loader = ScheduleLoader(self.client)
        self.telemetry_loader = TelemetryLoader(self.client)
        self.config = fastf1_config
        self.data_config = data_config
        self.logger = logging.getLogger("data_ingestion.data_fetcher")

        self.logger.info("DataFetcher initialized")

    @log_operation
    def ingest_full_season(
        self,
        year: int,
        events: Optional[List[str]] = None,
        session_types: Optional[List[str]] = None,
        parallel: bool = True,
    ) -> Dict[str, Any]:
        """Ingest data for a full F1 season"""

        self.logger.info("Starting full season ingestion for %d", year)

        # Get season schedule if events not provided
        if events is None:
            schedule = self.schedule_loader.load_season_schedule(year)
            events = schedule["EventName"].tolist()

        if session_types is None:
            session_types = self.config.session_types

        self.logger.info(
            "Ingesting %d events with %d session types. Total %d sessions.",
            len(events),
            len(session_types),
            len(events) * len(session_types),
        )

        start_time = time.time()  # To track how long it takes to ingest all the data
        ingestion_summary = {
            "year": year,
            "events_processed": 0,
            "sessions_processed": 0,
            "sessions_failed": 0,
            "start_time": datetime.now(),
            "events": {},
        }

        try:
            if parallel and len(events) > 1:
                # Parallel processing
                ingestion_summary = self._ingest_season_parallel(
                    year, events, session_types, ingestion_summary
                )
            else:
                # Sequential processing
                ingestion_summary = self._ingest_season_sequential(
                    year, events, session_types, ingestion_summary
                )

            ingestion_summary["end_time"] = datetime.now()
            ingestion_summary["total_duration"] = time.time() - start_time

            self.logger.info(
                "Season ingestion completed: %d events, %d sessions successful, %d sessions failed",
                ingestion_summary["events_processed"],
                ingestion_summary["sessions_processed"],
                ingestion_summary["sessions_failed"],
            )

            # Save ingestion summary
            self._save_ingestion_summary(ingestion_summary, year)

            return ingestion_summary

        except Exception as e:
            self.logger.error("Season ingestion failed for %d: %s", year, str(e))
            raise

    def _ingest_season_sequential(
        self,
        year: int,
        events: List[str],
        session_types: List[str],
        summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Ingest season data sequentially"""

        self.logger.info("Using sequential processing")

        for event in events:
            self.logger.info("Processing event: %s", event)
            event_summary = {
                "sessions": {},
                "success_count": 0,
                "fail_count": 0,
            }

            for session_type in session_types:
                try:
                    session_data = self.session_loader.load_session_data(
                        year, event, session_type
                    )  # Returns a dict {session_info: dict, laps: df, results: df, weather: df}

                    # Updating the event processing summary
                    event_summary["sessions"][session_type] = "success"
                    event_summary["success_count"] += 1

                    # Updating the ingestion summary
                    summary["sessions_processed"] += 1

                    self.logger.info(
                        "✅ Successfully processed %d %s %s", year, event, session_type
                    )

                except Exception as e:
                    # Updating event processing summary
                    event_summary["sessions"][session_type] = f"Failed: {str(e)}"
                    event_summary["fail_count"] += 1

                    # Updating the ingestion summary
                    summary["sessions_failed"] += 1

                    self.logger.error(
                        "❌ Error while processing %d %s %s: %s",
                        year,
                        event,
                        session_type,
                        str(e),
                    )

                # Small delay to avoid overwhelming the API
                time.sleep(1)

            # Updating the ingestion summary with the event summary and the number of events processed count
            summary["events"][event] = event_summary
            summary["events_processed"] += 1

            self.logger.info(
                "Event %s completed: %.2f sessions successful",
                event,
                event_summary["success_count"] / len(session_types),
            )

        return summary

    def _ingest_season_parallel(
        self,
        year: int,
        events: List[str],
        session_types: List[str],
        summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Ingest season data with parallel processing"""

        self.logger.info("Using parallel processing")
        max_workers = min(self.data_config.parallel_workers, len(events))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks for each event
            future_to_event = {
                executor.submit(self._process_event, year, event, session_types): event
                for event in events
            }

            # Process completed tasks
            for future in as_completed(future_to_event):
                event = future_to_event[future]
                try:
                    event_summary = future.result()
                    summary["events"][event] = event_summary
                    summary["events_processed"] += 1
                    summary["sessions_processed"] += event_summary["success_count"]
                    summary["sessions_failed"] += event_summary["fail_count"]

                    self.logger.info(
                        "Event %s completed: %.2f sessions successful",
                        event,
                        event_summary["success_count"] / len(session_types),
                    )

                except Exception as e:
                    self.logger.error("Event %s processing failed: %s", event, str(e))
                    summary["events"][event] = {
                        "error": str(e),
                        "success_count": 0,
                        "fail_count": len(session_types),
                    }
                    summary["sessions_failed"] += len(session_types)

        return summary

    def _process_event(
        self, year: int, event: str, session_types: List[str]
    ) -> Dict[str, Any]:
        """Process a single event (used in parallel processing)"""

        event_summary = {"sessions": {}, "success_count": 0, "fail_count": 0}

        for session_type in session_types:
            try:
                session_data = self.session_loader.load_session_data(
                    year, event, session_type
                )
                event_summary["sessions"][session_type] = "success"
                event_summary["success_count"] += 1

            except Exception as e:
                event_summary["sessions"][session_type] = f"failed: {str(e)}"
                event_summary["fail_count"] += 1

        return event_summary

    def ingest_event_data(
        self,
        year: int,
        event: str,
        session_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Ingest data for a single event"""

        if session_types is None:
            session_types = self.config.session_types

        self.logger.info("Ingesting event data: %d %s", year, event)

        return self.session_loader.load_multiple_sessions(year, [event], session_types)

    def get_yearly_schedule(self, year: int) -> pd.DataFrame:
        """Get event schedule for a year"""
        return self.schedule_loader.load_season_schedule(year)

    def _save_ingestion_summary(self, summary: Dict[str, Any], year: int):
        """Save ingestion summary to file"""

        summary_path = (
            self.data_config.raw_data_path / str(year) / "ingestion_summary.json"
        )
        ensure_directory(summary_path.parent)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        self.logger.info("Ingestion summary saved: %s", summary_path)

    @measure_time
    def validate_ingested_data(self, year: int) -> Dict[str, Any]:
        """Validate ingested data for a year"""

        self.logger.info("Validating ingested data for %d", year)

        validation_results = {
            "year": year,
            "events_found": 0,
            "sessions_found": 0,
            "total_laps": 0,
            "data_quality_issues": [],
            "missing_sessions": [],
        }

        year_path = self.data_config.raw_data_path / str(year)

        if not year_path.exists():
            validation_results["data_quality_issues"].append(
                f"No data directory found for {year}"
            )
            return validation_results

        # Check each event directory
        for event_dir in year_path.iterdir():
            if event_dir.is_dir() and event_dir.name != "ingestion_summary.json":
                validation_results["events_found"] += 1

                # Check sessions in each event
                for session_dir in event_dir.iterdir():
                    if session_dir.is_dir():
                        validation_results["sessions_found"] += 1

                        # Check for lap data
                        lap_file = session_dir / f"laps.{self.data_config.file_format}"
                        if lap_file.exists():
                            try:
                                laps = load_data(lap_file, self.data_config.file_format)
                                if laps is not None:
                                    validation_results["total_laps"] += len(laps)
                            except Exception as e:
                                validation_results["data_quality_issues"].append(
                                    f"Failed to read {lap_file}: {str(e)}"
                                )

        self.logger.info(
            "Validation completed: %d events, %d sessions, %d total laps",
            validation_results["events_found"],
            validation_results["sessions_found"],
            validation_results["total_laps"],
        )

        return validation_results
