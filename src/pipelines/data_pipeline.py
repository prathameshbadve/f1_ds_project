"""
Production Data Ingestion Pipeline
This orchestrates the complete data ingestion process for F1 seasons
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.logging import get_logger
from config.settings import data_config
from src.data_ingestion.data_fetcher import DataFetcher
from src.utils.decorators import log_operation, retry
from src.utils.helpers import ensure_directory


class PipelineStatus(Enum):
    """Pipeline execution status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineConfig:
    """Configuration for data ingestion pipeline"""

    # Season configuration
    seasons: List[int]
    session_types: List[str]

    # Performance settings
    parallel_seasons: bool = False  # Process seasons in parallel
    parallel_events: bool = True  # Process events within season in parallel
    max_workers: int = 3  # Conservative default for FastF1 API

    # Retry settings
    max_retries: int = 2
    retry_delay: float = 5.0

    # Data quality settings
    validate_data: bool = True
    min_events_per_season: int = 15  # F1 season typically has 20+ races

    # Resume settings
    resume_on_failure: bool = True
    skip_existing: bool = True

    # Reporting
    progress_interval: int = 5  # Report progress every N events

    @classmethod
    def default_config(cls) -> "PipelineConfig":
        """Create default pipeline configuration"""
        return cls(
            seasons=list(range(2022, 2025)),
            session_types=[
                "FP1",
                "FP2",
                "FP3",
                "Q",
                "R",
            ],  # All main sessions
            parallel_seasons=False,
            parallel_events=True,
            max_workers=3,
            max_retries=2,
            retry_delay=5.0,
            validate_data=True,
            min_events_per_season=15,
            resume_on_failure=True,
            skip_existing=True,
            progress_interval=5,
        )

    @classmethod
    def conservative_config(cls) -> "PipelineConfig":
        """Conservative configuration for reliable ingestion"""
        return cls(
            seasons=list(range(2022, 2025)),  # Recent seasons only
            session_types=["Q", "R"],  # Essential sessions only
            parallel_seasons=False,
            parallel_events=False,  # Sequential for reliability
            max_workers=1,
            max_retries=3,
            retry_delay=10.0,
            validate_data=True,
            min_events_per_season=15,
            resume_on_failure=True,
            skip_existing=True,
            progress_interval=3,
        )

    @classmethod
    def performance_config(cls) -> "PipelineConfig":
        """High-performance configuration (use with caution)"""
        return cls(
            seasons=list(range(2018, 2025)),  # All available seasons
            session_types=[
                "FP1",
                "FP2",
                "FP3",
                "Q",
                "R",
                "S",
                "SQ",
                "SS",
            ],  # All sessions
            parallel_seasons=True,  # ⚠️ High API load
            parallel_events=True,
            max_workers=4,
            max_retries=1,
            retry_delay=2.0,
            validate_data=False,  # Skip validation for speed
            min_events_per_season=10,
            resume_on_failure=True,
            skip_existing=True,
            progress_interval=10,
        )


@dataclass
class IngestionProgress:
    """Track pipeline progress"""

    total_seasons: int
    completed_seasons: int
    total_events: int
    completed_events: int
    total_sessions: int
    completed_sessions: int
    failed_sessions: int
    start_time: datetime
    current_season: Optional[int] = None
    current_event: Optional[str] = None
    estimated_completion: Optional[datetime] = None

    @property
    def progress_percent(self) -> float:
        """Calculate overall progress percentage"""
        if self.total_sessions == 0:
            return 0.0
        return (self.completed_sessions / self.total_sessions) * 100

    @property
    def elapsed_time(self) -> timedelta:
        """Time elapsed since start"""
        return datetime.now() - self.start_time

    def estimate_completion(self) -> Optional[datetime]:
        """Estimate completion time based on current progress"""
        if self.completed_sessions == 0:
            return None

        avg_time_per_session = (
            self.elapsed_time.total_seconds() / self.completed_sessions
        )
        remaining_sessions = self.total_sessions - self.completed_sessions
        remaining_time = timedelta(seconds=avg_time_per_session * remaining_sessions)

        return datetime.now() + remaining_time


class DataIngestionPipeline:
    """
    Production data ingestion pipeline for F1 data

    Design Choices Explained:
    1. Configuration-driven: Easy to adjust behavior without code changes
    2. Progress tracking: Monitor long-running operations
    3. Resume capability: Handle failures gracefully
    4. Parallel processing: Configurable performance vs reliability
    5. Comprehensive logging: Track every operation for debugging
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig.default_config()
        self.logger = get_logger("data_pipeline")
        self.data_fetcher = DataFetcher()

        self.status = PipelineStatus.PENDING
        self.progress: Optional[IngestionProgress] = None
        self.results: Dict[str, Any] = {}

        # Create pipeline output directory
        self.pipeline_output_dir = data_config.raw_data_path / "pipeline_runs"
        ensure_directory(self.pipeline_output_dir)

        self.logger.info("Pipeline initialized with config: %s", self.config.seasons)

    @log_operation
    def run_full_ingestion(self) -> Dict[str, Any]:
        """
        Run complete data ingestion pipeline

        Returns:
            Dictionary with ingestion results and statistics
        """

        self.logger.info("Starting full data ingestion pipeline")
        self.logger.info("Seasons: %s", self.config.seasons)
        self.logger.info("Session types: %s", self.config.session_types)
        self.logger.info(
            "Parallel processing: seasons=%s, events=%s",
            self.config.parallel_seasons,
            self.config.parallel_events,
        )

        try:
            self.status = PipelineStatus.RUNNING

            # Initialize progress tracking
            self.progress = self._initialize_progress()

            # Create run directory for this execution
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = self.pipeline_output_dir / f"run_{run_id}"
            ensure_directory(run_dir)

            # Save configuration
            self._save_run_config(run_dir)

            # Execute pipeline
            if self.config.parallel_seasons:
                self.results = self._run_seasons_parallel()
            else:
                self.results = self._run_seasons_sequential()

            # Finalize results
            self.status = PipelineStatus.COMPLETED
            self.results["pipeline_status"] = "completed"
            self.results["run_id"] = run_id
            self.results["final_progress"] = asdict(self.progress)

            # Save results
            self._save_run_results(run_dir)

            # Generate summary report
            self._generate_summary_report(run_dir)

            self.logger.info("Pipeline completed successfully. Run ID: %s", run_id)
            return self.results

        except Exception as e:
            self.status = PipelineStatus.FAILED
            self.logger.error("Pipeline failed: %s", str(e))
            raise

    def _initialize_progress(self) -> IngestionProgress:
        """Initialize progress tracking"""

        self.logger.info("Calculating pipeline scope...")

        total_seasons = len(self.config.seasons)
        total_events = 0
        total_sessions = 0

        # Calculate total work by checking season schedules
        for year in self.config.seasons:
            try:
                schedule = self.data_fetcher.get_yearly_schedule(year)
                events_for_ingestion = (
                    self.data_fetcher.schedule_loader.get_events_for_ingestion(year)
                )

                year_events = len(events_for_ingestion)
                year_sessions = year_events * len(self.config.session_types)

                total_events += year_events
                total_sessions += year_sessions

                self.logger.debug(
                    "Year %d: %d events, %d sessions",
                    year,
                    year_events,
                    year_sessions,
                )

            except Exception as e:
                self.logger.warning(
                    "Could not calculate scope for %d: %s", year, str(e)
                )
                # Use estimates if API call fails
                total_events += 20  # Typical F1 season
                total_sessions += 20 * len(self.config.session_types)

        progress = IngestionProgress(
            total_seasons=total_seasons,
            completed_seasons=0,
            total_events=total_events,
            completed_events=0,
            total_sessions=total_sessions,
            completed_sessions=0,
            failed_sessions=0,
            start_time=datetime.now(),
        )

        self.logger.info(
            "Pipeline scope: %d seasons, %d events, %d sessions",
            total_seasons,
            total_events,
            total_sessions,
        )
        return progress

    def _run_seasons_sequential(self) -> Dict[str, Any]:
        """Run seasons sequentially (safer, more reliable)"""

        self.logger.info("Running seasons sequentially")

        all_results = {
            "seasons": {},
            "summary": {
                "total_seasons": len(self.config.seasons),
                "successful_seasons": 0,
                "failed_seasons": 0,
                "total_events_processed": 0,
                "total_sessions_processed": 0,
                "total_sessions_failed": 0,
            },
        }

        for year in self.config.seasons:
            self.progress.current_season = year
            self.logger.info("Processing season %d", year)

            try:
                season_results = self._ingest_season(year)
                all_results["seasons"][year] = season_results

                # Update summary
                if season_results.get("status") == "completed":
                    all_results["summary"]["successful_seasons"] += 1
                else:
                    all_results["summary"]["failed_seasons"] += 1

                all_results["summary"]["total_events_processed"] += season_results.get(
                    "events_processed", 0
                )
                all_results["summary"]["total_sessions_processed"] += (
                    season_results.get("sessions_processed", 0)
                )
                all_results["summary"]["total_sessions_failed"] += season_results.get(
                    "sessions_failed", 0
                )

                self.progress.completed_seasons += 1

                # Log progress
                self._log_progress()

            except Exception as e:
                self.logger.error("Season %d failed: %s", year, str(e))
                all_results["seasons"][year] = {
                    "status": "failed",
                    "error": str(e),
                    "events_processed": 0,
                    "sessions_processed": 0,
                    "sessions_failed": 0,
                }
                all_results["summary"]["failed_seasons"] += 1

                # Continue with next season if resume_on_failure is True
                if not self.config.resume_on_failure:
                    raise

        return all_results

    def _run_seasons_parallel(self) -> Dict[str, Any]:
        """Run seasons in parallel (faster but higher API load)"""

        self.logger.info("Running seasons in parallel")
        self.logger.warning("⚠️ Parallel season processing may exceed API rate limits")

        all_results = {
            "seasons": {},
            "summary": {
                "total_seasons": len(self.config.seasons),
                "successful_seasons": 0,
                "failed_seasons": 0,
                "total_events_processed": 0,
                "total_sessions_processed": 0,
                "total_sessions_failed": 0,
            },
        }

        # Use smaller thread pool for seasons to avoid API overload
        max_season_workers = min(2, self.config.max_workers)

        with ThreadPoolExecutor(max_workers=max_season_workers) as executor:
            # Submit all season tasks
            future_to_year = {
                executor.submit(self._ingest_season, year): year
                for year in self.config.seasons
            }

            # Process completed seasons
            for future in as_completed(future_to_year):
                year = future_to_year[future]

                try:
                    season_results = future.result()
                    all_results["seasons"][year] = season_results

                    # Update summary (thread-safe for this use case)
                    if season_results.get("status") == "completed":
                        all_results["summary"]["successful_seasons"] += 1
                    else:
                        all_results["summary"]["failed_seasons"] += 1

                    all_results["summary"]["total_events_processed"] += (
                        season_results.get("events_processed", 0)
                    )
                    all_results["summary"]["total_sessions_processed"] += (
                        season_results.get("sessions_processed", 0)
                    )
                    all_results["summary"]["total_sessions_failed"] += (
                        season_results.get("sessions_failed", 0)
                    )

                    self.progress.completed_seasons += 1
                    self.logger.info("✅ Season %d completed", year)

                except Exception as e:
                    self.logger.error("❌ Season %d failed: %s", year, str(e))
                    all_results["seasons"][year] = {
                        "status": "failed",
                        "error": str(e),
                        "events_processed": 0,
                        "sessions_processed": 0,
                        "sessions_failed": 0,
                    }
                    all_results["summary"]["failed_seasons"] += 1

        return all_results

    @retry(max_attempts=2, delay=10.0)
    def _ingest_season(self, year: int) -> Dict[str, Any]:
        """
        Ingest data for a single season

        Args:
            year: Season year to ingest

        Returns:
            Dictionary with season ingestion results
        """

        self.logger.info("Starting season ingestion: %d", year)

        season_start_time = time.time()
        season_results = {
            "year": year,
            "status": "running",
            "start_time": datetime.now(),
            "events": {},
            "events_processed": 0,
            "sessions_processed": 0,
            "sessions_failed": 0,
            "duration_seconds": 0,
        }

        try:
            # Get events for this season
            events = self.data_fetcher.schedule_loader.get_events_for_ingestion(year)

            if len(events) < self.config.min_events_per_season:
                self.logger.warning(
                    "Season %d has only %d events (expected minimum: %d)",
                    year,
                    len(events),
                    self.config.min_events_per_season,
                )

            self.logger.info("Season %d: Processing %d events", year, len(events))

            # Process events (parallel or sequential based on config)
            if self.config.parallel_events:
                season_results["events"] = self._process_events_parallel(year, events)
            else:
                season_results["events"] = self._process_events_sequential(year, events)

            # Calculate season statistics
            for event_name, event_result in season_results["events"].items():
                if event_result.get("status") == "completed":
                    season_results["events_processed"] += 1
                    season_results["sessions_processed"] += event_result.get(
                        "sessions_processed", 0
                    )

                season_results["sessions_failed"] += event_result.get(
                    "sessions_failed", 0
                )

            season_results["status"] = "completed"
            season_results["duration_seconds"] = time.time() - season_start_time
            season_results["end_time"] = datetime.now()

            self.logger.info(
                "Season %d completed: {%d}/{%d} events, %d sessions successful, %d sessions failed",
                year,
                season_results["events_processed"],
                season_results["events_processed"],
                season_results["sessions_processed"],
                season_results["sessions_failed"],
            )

            return season_results

        except Exception as e:
            season_results["status"] = "failed"
            season_results["error"] = str(e)
            season_results["duration_seconds"] = time.time() - season_start_time
            season_results["end_time"] = datetime.now()

            self.logger.error("Season %d failed: %s", year, str(e))
            raise

    def _process_events_sequential(
        self, year: int, events: List[str]
    ) -> Dict[str, Any]:
        """Process events sequentially within a season"""

        self.logger.debug("Processing %d events sequentially for %d", len(events), year)

        event_results = {}

        for i, event in enumerate(events, 1):
            self.progress.current_event = event

            try:
                self.logger.info(
                    "Processing event %d/%d: %d %s", i, len(events), year, event
                )

                event_result = self._process_single_event(year, event)
                event_results[event] = event_result

                self.progress.completed_events += 1

                # Report progress periodically
                if i % self.config.progress_interval == 0:
                    self._log_progress()

                # Small delay to be respectful to FastF1 API
                time.sleep(1)

            except Exception as e:
                self.logger.error("Event %d %s failed: %s", year, event, str(e))
                event_results[event] = {
                    "status": "failed",
                    "error": str(e),
                    "sessions_processed": 0,
                    "sessions_failed": len(self.config.session_types),
                }

                if not self.config.resume_on_failure:
                    raise

        return event_results

    def _process_events_parallel(self, year: int, events: List[str]) -> Dict[str, Any]:
        """Process events in parallel within a season"""

        self.logger.debug("Processing %d events in parallel for %d", len(events), year)

        event_results = {}

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all event tasks
            future_to_event = {
                executor.submit(self._process_single_event, year, event): event
                for event in events
            }

            # Process completed events
            completed_events = 0
            for future in as_completed(future_to_event):
                event = future_to_event[future]
                completed_events += 1

                try:
                    event_result = future.result()
                    event_results[event] = event_result

                    self.progress.completed_events += 1
                    self.logger.info(
                        "✅ Event completed (%d/%d): %d %s",
                        completed_events,
                        len(events),
                        year,
                        event,
                    )

                    # Report progress periodically
                    if completed_events % self.config.progress_interval == 0:
                        self._log_progress()

                except Exception as e:
                    self.logger.error(
                        "❌ Event failed: %d %s - %s", year, event, str(e)
                    )
                    event_results[event] = {
                        "status": "failed",
                        "error": str(e),
                        "sessions_processed": 0,
                        "sessions_failed": len(self.config.session_types),
                    }

        return event_results

    @retry(max_attempts=3, delay=5.0)
    def _process_single_event(self, year: int, event: str) -> Dict[str, Any]:
        """
        Process a single event (all configured sessions)

        Args:
            year: Season year
            event: Event name

        Returns:
            Dictionary with event processing results
        """

        event_start_time = time.time()

        # Check if we should skip existing data
        if self.config.skip_existing and self._event_already_processed(year, event):
            self.logger.info("Skipping existing event: %d %s", year, event)
            return {
                "status": "skipped",
                "reason": "already_processed",
                "sessions_processed": len(self.config.session_types),
                "sessions_failed": 0,
                "duration_seconds": 0,
            }

        try:
            # Use DataFetcher to ingest event data
            event_data = self.data_fetcher.ingest_event_data(
                year=year, event=event, session_types=self.config.session_types
            )

            # Count successful and failed sessions
            sessions_processed = 0
            sessions_failed = 0

            for event_name, sessions in event_data.items():
                for session_type, session_data in sessions.items():
                    if session_data is not None:
                        sessions_processed += 1
                        self.progress.completed_sessions += 1
                    else:
                        sessions_failed += 1
                        self.progress.failed_sessions += 1

            # Validate data if configured
            if self.config.validate_data:
                validation_results = self._validate_event_data(year, event, event_data)
                if not validation_results["valid"]:
                    self.logger.warning(
                        "Data validation issues for %d %s: %s",
                        year,
                        event,
                        validation_results["issues"],
                    )

            duration = time.time() - event_start_time

            return {
                "status": "completed",
                "sessions_processed": sessions_processed,
                "sessions_failed": sessions_failed,
                "duration_seconds": duration,
                "data_size_info": self._get_data_size_info(year, event)
                if self.config.validate_data
                else None,
            }

        except Exception as e:
            duration = time.time() - event_start_time

            # Update progress for failed sessions
            self.progress.failed_sessions += len(self.config.session_types)

            return {
                "status": "failed",
                "error": str(e),
                "sessions_processed": 0,
                "sessions_failed": len(self.config.session_types),
                "duration_seconds": duration,
            }

    def _event_already_processed(self, year: int, event: str) -> bool:
        """Check if event data already exists"""

        event_dir = data_config.raw_data_path / str(year) / event

        if not event_dir.exists():
            return False

        # Check if all configured session types have data
        for session_type in self.config.session_types:
            session_dir = event_dir / session_type
            if not session_dir.exists():
                return False

            # Check for key data files
            key_files = [
                session_dir / f"laps.{data_config.file_format}",
                session_dir / "session_info.json",
            ]

            if not all(file.exists() for file in key_files):
                return False

        return True

    def _validate_event_data(
        self, year: int, event: str, event_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate ingested event data"""

        validation_results = {"valid": True, "issues": []}

        try:
            for event_name, sessions in event_data.items():
                for session_type, session_data in sessions.items():
                    if session_data is None:
                        validation_results["issues"].append(
                            f"No data for {session_type}"
                        )
                        continue

                    # Check for lap data in race sessions
                    if (
                        session_type in ["R", "Q"]
                        and session_data.get("laps") is not None
                    ):
                        laps = session_data["laps"]
                        if len(laps) < 10:  # Expect reasonable number of laps
                            validation_results["issues"].append(
                                f"{session_type}: Only {len(laps)} laps"
                            )

        except Exception as e:
            validation_results["issues"].append(f"Validation error: {str(e)}")

        validation_results["valid"] = len(validation_results["issues"]) == 0
        return validation_results

    def _get_data_size_info(self, year: int, event: str) -> Dict[str, Any]:
        """Get information about data size for an event"""

        event_dir = data_config.raw_data_path / str(year) / event

        if not event_dir.exists():
            return {"total_size_mb": 0, "file_count": 0}

        total_size = 0
        file_count = 0

        for file_path in event_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1

        return {"total_size_mb": total_size / (1024 * 1024), "file_count": file_count}

    def _log_progress(self):
        """Log current pipeline progress"""

        if not self.progress:
            return

        # Update estimated completion
        self.progress.estimated_completion = self.progress.estimate_completion()

        self.logger.info(
            "Progress: %.1f%% (%d/%d sessions)",
            self.progress.progress_percent,
            self.progress.completed_sessions,
            self.progress.total_sessions,
        )

        if self.progress.current_season:
            self.logger.info("Current: Season %s", self.progress.current_season)

        if self.progress.estimated_completion:
            self.logger.info(
                "Estimated completion: %s",
                self.progress.estimated_completion.strftime("%Y-%m-%d %H:%M:%S"),
            )

        if self.progress.failed_sessions > 0:
            self.logger.warning("Failed sessions: %s", self.progress.failed_sessions)

    def _save_run_config(self, run_dir: Path):
        """Save pipeline configuration for this run"""

        config_file = run_dir / "pipeline_config.json"

        config_dict = asdict(self.config)
        config_dict["pipeline_version"] = "1.0.0"
        config_dict["run_timestamp"] = datetime.now().isoformat()

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

        self.logger.info("Pipeline config saved: %s", config_file)

    def _save_run_results(self, run_dir: Path):
        """Save pipeline results for this run"""

        results_file = run_dir / "pipeline_results.json"

        # Prepare results for JSON serialization
        json_results = self._prepare_results_for_json(self.results)

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, default=str)

        self.logger.info("Pipeline results saved: %s", results_file)

    def _prepare_results_for_json(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare results dictionary for JSON serialization"""

        def convert_value(value):
            if isinstance(value, datetime):
                return value.isoformat()
            elif isinstance(value, timedelta):
                return value.total_seconds()
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert_value(v) for v in value]
            else:
                return value

        return convert_value(results)

    def _generate_summary_report(self, run_dir: Path):
        """Generate human-readable summary report"""

        report_file = run_dir / "pipeline_summary.txt"

        summary = self.results.get("summary", {})

        report_lines = [
            "F1 DATA INGESTION PIPELINE - SUMMARY REPORT",
            "=" * 50,
            "",
            f"Run ID: {self.results.get('run_id', 'unknown')}",
            f"Execution Time: {self.progress.elapsed_time}",
            f"Status: {self.status.value}",
            "",
            "CONFIGURATION:",
            f"  Seasons: {self.config.seasons}",
            f"  Session Types: {self.config.session_types}",
            f"  Parallel Processing: seasons={self.config.parallel_seasons}, events={self.config.parallel_events}",
            f"  Max Workers: {self.config.max_workers}",
            "",
            "RESULTS:",
            f"  Total Seasons: {summary.get('total_seasons', 0)}",
            f"  Successful Seasons: {summary.get('successful_seasons', 0)}",
            f"  Failed Seasons: {summary.get('failed_seasons', 0)}",
            f"  Total Events Processed: {summary.get('total_events_processed', 0)}",
            f"  Total Sessions Processed: {summary.get('total_sessions_processed', 0)}",
            f"  Total Sessions Failed: {summary.get('total_sessions_failed', 0)}",
            "",
            f"Success Rate: {(summary.get('total_sessions_processed', 0) / max(1, summary.get('total_sessions_processed', 0) + summary.get('total_sessions_failed', 0))) * 100:.1f}%",
            "",
            "DETAILED RESULTS BY SEASON:",
        ]

        # Add season-by-season breakdown
        for year, season_data in self.results.get("seasons", {}).items():
            report_lines.extend(
                [
                    f"  {year}: {season_data.get('status', 'unknown')} - "
                    f"{season_data.get('events_processed', 0)} events, "
                    f"{season_data.get('sessions_processed', 0)} sessions successful, "
                    f"{season_data.get('sessions_failed', 0)} sessions failed"
                ]
            )

        report_lines.extend(
            [
                "",
                "DATA LOCATION:",
                f"  Raw Data Directory: {data_config.raw_data_path}",
                f"  Pipeline Output: {run_dir}",
                "",
                "=" * 50,
            ]
        )

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        self.logger.info("Summary report generated: %s", report_file)

        # Also log key metrics
        self.logger.info("PIPELINE COMPLETED - SUMMARY:")
        self.logger.info("  Duration: %s", self.progress.elapsed_time)
        self.logger.info(
            "  Sessions: %d successful, %d failed",
            summary.get("total_sessions_processed", 0),
            summary.get("total_sessions_failed", 0),
        )
        self.logger.info(
            "  Success Rate: %.1f%%",
            (
                summary.get("total_sessions_processed", 0)
                / max(
                    1,
                    summary.get("total_sessions_processed", 0)
                    + summary.get("total_sessions_failed", 0),
                )
            )
            * 100,
        )
