"""
Pipeline configuration management
"""

import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config.logging import get_logger
from config.settings import data_config
from src.data_processing.base.processing_context import ProcessingContext
from src.data_processing.core.lap_processor import LapProcessor
from src.data_processing.core.session_processor import SessionProcessor
from src.utils.helpers import ensure_directory, save_data


class ErrorMode(Enum):
    """Error handling modes for pipeline execution"""

    STRICT = "strict"  # Stop on first error
    CONTINUE = "continue"  # Log error, continue processing
    PARTIAL = "partial"  # Save partial results, continue


class ProcessingMode(Enum):
    """Processing execution modes"""

    SEQUENTIAL = "sequential"  # Process one at a time
    BATCH = "batch"  # Process in batches
    PARALLEL = "parallel"  # Process in parallel (future)


@dataclass
class PipelineConfig:
    """Configuration for data processing pipeline"""

    # Processor configuration
    processors: List[str] = field(default_factory=lambda: ["session", "lap"])
    processor_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Execution configuration
    processing_mode: ProcessingMode = ProcessingMode.SEQUENTIAL
    error_mode: ErrorMode = ErrorMode.CONTINUE
    max_retries: int = 2

    # Data configuration
    skip_existing: bool = True
    save_intermediate: bool = True
    validate_outputs: bool = True

    # Input/output paths
    raw_data_path: Path = field(default_factory=lambda: data_config.raw_data_path)
    processed_data_path: Path = field(
        default_factory=lambda: data_config.processed_data_path
    )

    # Performance configuration
    batch_size: int = 10
    max_workers: int = 3

    # Logging and monitoring
    verbose: bool = True
    log_level: str = "INFO"
    generate_reports: bool = True

    @classmethod
    def default_config(cls) -> "PipelineConfig":
        """Create default pipeline configuration"""
        return cls(
            processors=["session", "lap"],
            processing_mode=ProcessingMode.SEQUENTIAL,
            error_mode=ErrorMode.CONTINUE,
            skip_existing=True,
            save_intermediate=True,
            validate_outputs=True,
        )

    @classmethod
    def quick_config(cls) -> "PipelineConfig":
        """Quick processing configuration (skip validation for speed)"""
        return cls(
            processors=["session", "lap"],
            processing_mode=ProcessingMode.SEQUENTIAL,
            error_mode=ErrorMode.CONTINUE,
            skip_existing=True,
            save_intermediate=False,
            validate_outputs=False,
        )

    @classmethod
    def strict_config(cls) -> "PipelineConfig":
        """Strict configuration (fail on any error)"""
        return cls(
            processors=["session", "lap"],
            processing_mode=ProcessingMode.SEQUENTIAL,
            error_mode=ErrorMode.STRICT,
            skip_existing=False,
            save_intermediate=True,
            validate_outputs=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""

        return {
            "processors": self.processors,
            "processor_configs": self.processor_configs,
            "processing_mode": self.processing_mode.value,
            "error_mode": self.error_mode.value,
            "max_retries": self.max_retries,
            "skip_existing": self.skip_existing,
            "save_intermediate": self.save_intermediate,
            "validate_outputs": self.validate_outputs,
            "raw_data_path": str(self.raw_data_path),
            "processed_data_path": str(self.processed_data_path),
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
            "verbose": self.verbose,
            "log_level": self.log_level,
            "generate_reports": self.generate_reports,
        }


class PipelineException(Exception):
    """Custom exception for pipeline errors"""

    def __init__(
        self, message: str, pipeline_name: str = "", context: Dict[str, Any] = {}
    ):
        super().__init__(message)
        self.pipeline_name = pipeline_name
        self.context = context or {}


class PipelineResult:
    """Result of pipeline execution"""

    def __init__(self, pipeline_id: str):
        self.pipeline_id = pipeline_id
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.duration_seconds: Optional[float] = None

        self.status = "running"
        self.total_items = 0
        self.processed_items = 0
        self.failed_items = 0
        self.skipped_items = 0

        self.processor_results: Dict[str, List[Dict]] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []

        self.metadata: Dict[str, Any] = {}

    def complete(self, status: str = "completed"):
        """Mark pipeline as complete"""
        self.end_time = datetime.now()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.status = status

    def add_processor_result(self, processor_name: str, result: Dict):
        """Add result from a processor"""
        if processor_name not in self.processor_results:
            self.processor_results[processor_name] = []
        self.processor_results[processor_name].append(result)

    def add_error(self, error_message: str):
        """Add an error"""
        self.errors.append(error_message)

    def add_warning(self, warning_message: str):
        """Add a warning"""
        self.warnings.append(warning_message)

    def get_summary(self) -> Dict[str, Any]:
        """Get pipeline execution summary"""
        return {
            "pipeline_id": self.pipeline_id,
            "status": self.status,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "failed_items": self.failed_items,
            "skipped_items": self.skipped_items,
            "success_rate": (self.processed_items / max(self.total_items, 1)) * 100,
            "errors_count": len(self.errors),
            "warnings_count": len(self.warnings),
            "processors_run": list(self.processor_results.keys()),
            "metadata": self.metadata,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            **self.get_summary(),
            "processor_results": self.processor_results,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class BasePipeline(ABC):
    """
    Abstract base class for data processing pipelines

    Provides common functionality:
    - Pipeline lifecycle management
    - Progress tracking
    - Error handling
    - Result aggregation
    - Metadata tracking
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.pipeline_id = str(uuid.uuid4())[:8]
        self.logger = get_logger(f"pipeline.{name}")

        self.result: Optional[PipelineResult] = None

    @abstractmethod
    def _execute_pipeline(self, **kwargs) -> PipelineResult:
        """
        Execute the pipeline - must be implemented by subclasses

        Returns:
            PipelineResult with execution details
        """
        pass

    def run(self, **kwargs) -> PipelineResult:
        """
        Run the pipeline with full error handling and monitoring

        Returns:
            PipelineResult with execution summary
        """

        # Create pipeline result tracker
        run_id = (
            f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.pipeline_id}"
        )
        self.result = PipelineResult(run_id)

        self.logger.info("Starting pipeline: %s (ID: %s)", self.name, run_id)

        try:
            # Execute pipeline implementation
            self.result = self._execute_pipeline(**kwargs)

            # Mark as complete
            if self.result.status == "running":
                self.result.complete("completed")

            self.logger.info("Pipeline completed: %s", self.result.status)
            self.logger.info(
                "Processed: %s",
                self.result.processed_items / self.result.total_items,
            )

            if self.result.failed_items > 0:
                self.logger.warning("Failed items: %s", self.result.failed_items)

            return self.result

        except Exception as e:
            self.result.complete("failed")
            self.result.add_error(f"Pipeline execution failed: {str(e)}")

            self.logger.error("Pipeline failed: %s", str(e))
            raise PipelineException(
                f"Pipeline {self.name} failed", self.name, self.result.to_dict()
            ) from e

    def save_result(self, output_dir: Path):
        """Save pipeline result to file"""
        if not self.result:
            self.logger.warning("No result to save")
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result_file = output_dir / f"{self.result.pipeline_id}_result.json"

        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(self.result.to_dict(), f, indent=2, default=str)

        self.logger.info("Pipeline result saved: %s", result_file)

    def generate_report(self, output_dir: Path):
        """Generate human-readable pipeline report"""
        if not self.result:
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_file = output_dir / f"{self.result.pipeline_id}_report.txt"

        report_lines = [
            "=" * 80,
            f"PIPELINE EXECUTION REPORT: {self.name}",
            "=" * 80,
            "",
            f"Pipeline ID: {self.result.pipeline_id}",
            f"Status: {self.result.status}",
            f"Duration: {self.result.duration_seconds:.2f} seconds",
            "",
            "SUMMARY:",
            f"  Total Items: {self.result.total_items}",
            f"  Processed: {self.result.processed_items}",
            f"  Failed: {self.result.failed_items}",
            f"  Skipped: {self.result.skipped_items}",
            f"  Success Rate: {(self.result.processed_items / max(self.result.total_items, 1)) * 100:.1f}%",
            "",
            "PROCESSORS:",
        ]

        for processor_name, results in self.result.processor_results.items():
            report_lines.append(f"  {processor_name}: {len(results)} executions")

        if self.result.errors:
            report_lines.extend(
                [
                    "",
                    "ERRORS:",
                ]
            )
            for error in self.result.errors[:10]:  # Show first 10
                report_lines.append(f"  - {error}")

        if self.result.warnings:
            report_lines.extend(
                [
                    "",
                    "WARNINGS:",
                ]
            )
            for warning in self.result.warnings[:10]:  # Show first 10
                report_lines.append(f"  - {warning}")

        report_lines.append("")
        report_lines.append("=" * 80)

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        self.logger.info("Pipeline report saved: %s", report_file)

    def __str__(self) -> str:
        return f"{self.name} v{self.version}"

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}(name='{self.name}', version='{self.version}')>"
        )


class ProcessingPipeline(BasePipeline):
    """
    F1 Data Processing Pipeline

    Orchestrates the transformation of raw F1 data into analysis-ready formats

    Pipeline Flow:
    1. Load raw session data
    2. SessionProcessor → Clean session metadata
    3. LapProcessor → Enrich lap timing data
    4. [Future] ResultsProcessor → Process race results
    5. Save processed data to structured directories
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        super().__init__("f1_processing_pipeline", "1.0.0")

        self.config = config or PipelineConfig.default_config()

        # Initialize processors based on config
        self.processors = self._initialize_processors()

        # Setup output directories
        self._setup_output_directories()

        self.logger.info(
            "Pipeline initialized with %s processors", len(self.processors)
        )
        self.logger.info("Processors: %s", list(self.processors.keys()))

    def _initialize_processors(self) -> Dict[str, Any]:
        """Initialize configured processors"""
        processors = {}

        for processor_name in self.config.processors:
            if processor_name == "session":
                processors["session"] = SessionProcessor()
            elif processor_name == "lap":
                processors["lap"] = LapProcessor()
            # Add more processors as they're built:
            # elif processor_name == 'results':
            #     processors['results'] = ResultsProcessor()
            else:
                self.logger.warning("Unknown processor: %s", processor_name)

        return processors

    def _setup_output_directories(self):
        """Create output directory structure"""
        base_path = self.config.processed_data_path

        self.output_dirs = {
            "sessions": base_path / "sessions",
            "laps": base_path / "laps",
            "results": base_path / "results",
            "metadata": base_path / "pipeline_metadata",
        }

        for dir_path in self.output_dirs.values():
            ensure_directory(dir_path)

    def _execute_pipeline(self, **kwargs) -> PipelineResult:
        """Execute the processing pipeline"""

        # Determine what to process
        if "year" in kwargs and "event" in kwargs and "session_type" in kwargs:
            # Process single session
            return self._process_single_session(
                kwargs["year"], kwargs["event"], kwargs["session_type"]
            )
        elif "year" in kwargs:
            # Process entire year
            return self._process_year(
                kwargs["year"], session_types=kwargs.get("session_types", ["Q", "R"])
            )
        elif "sessions" in kwargs:
            # Process list of sessions
            return self._process_session_list(kwargs["sessions"])
        else:
            raise ValueError(
                "Must provide either (year, event, session_type) or (year) or (sessions)"
            )

    def process_session(
        self, year: int, event: str, session_type: str
    ) -> Dict[str, Any]:
        """
        Process a single F1 session through the pipeline

        Args:
            year: Season year
            event: Event name (e.g., 'Monaco')
            session_type: Session type (e.g., 'Q', 'R')

        Returns:
            Dictionary with processed outputs from each processor
        """

        result = self._process_single_session(year, event, session_type)
        return result.processor_results

    def _process_single_session(
        self, year: int, event: str, session_type: str
    ) -> PipelineResult:
        """Process a single session through all processors"""

        session_id = f"{year}_{event}_{session_type}"
        self.logger.info("Processing session: %s", session_id)

        self.result.total_items = 1

        try:
            # Check if already processed
            if self.config.skip_existing and self._is_session_processed(
                year, event, session_type
            ):
                self.logger.info("Session already processed, skipping: %s", session_id)
                self.result.skipped_items += 1
                return self.result

            # Load raw session data
            raw_data = self._load_raw_session_data(year, event, session_type)

            # Create processing context
            context = ProcessingContext(
                year=year, event_name=event, session_type=session_type
            )

            # Track input sources
            session_dir = self.config.raw_data_path / str(year) / event / session_type
            for file in session_dir.glob("*"):
                context.add_source_file(str(file))

            # Run through each processor
            processor_outputs = {}

            for processor_name, processor in self.processors.items():
                try:
                    self.logger.info("Running %s processor...", processor_name)
                    start_time = time.time()

                    # Process data
                    output, updated_context = processor.process(raw_data, context)

                    duration = time.time() - start_time

                    # Store output
                    processor_outputs[processor_name] = output

                    # Save intermediate output if configured
                    if self.config.save_intermediate:
                        self._save_processor_output(
                            processor_name, output, year, event, session_type
                        )

                    # Update context for next processor
                    context = updated_context

                    # Record success
                    self.result.add_processor_result(
                        processor_name,
                        {
                            "session": session_id,
                            "status": "success",
                            "duration": duration,
                            "records": len(output)
                            if isinstance(output, pd.DataFrame)
                            else 1,
                        },
                    )

                    self.logger.info(
                        "✅ %s completed in %.2fs", processor_name, duration
                    )

                except Exception as e:
                    error_msg = f"{processor_name} failed for {session_id}: {str(e)}"
                    self.logger.error(error_msg)

                    self.result.add_processor_result(
                        processor_name,
                        {"session": session_id, "status": "failed", "error": str(e)},
                    )

                    if self.config.error_mode == ErrorMode.STRICT:
                        raise PipelineException(error_msg, self.name)
                    elif self.config.error_mode == ErrorMode.CONTINUE:
                        self.result.add_warning(error_msg)
                        continue
                    elif self.config.error_mode == ErrorMode.PARTIAL:
                        self.result.add_warning(error_msg)
                        # Save what we have so far
                        break

            self.result.processed_items += 1
            self.logger.info("✅ Session processing complete: %s", session_id)

        except Exception as e:
            self.result.failed_items += 1
            error_msg = f"Failed to process {session_id}: {str(e)}"
            self.result.add_error(error_msg)
            self.logger.error(error_msg)

            if self.config.error_mode == ErrorMode.STRICT:
                raise

        return self.result

    def _process_year(self, year: int, session_types: List[str]) -> PipelineResult:
        """Process all sessions for a year"""

        self.logger.info(
            "Processing year %d with session types: %s", year, session_type
        )

        # Get all events for the year
        events = self._get_year_events(year)

        self.result.total_items = len(events) * len(session_types)
        self.logger.info("Total sessions to process: %s", self.result.total_items)

        # Process each event/session combination
        for event in events:
            for session_type in session_types:
                try:
                    self._process_single_session(year, event, session_type)
                except Exception as e:
                    if self.config.error_mode == ErrorMode.STRICT:
                        raise
                    # Continue with next session if in CONTINUE or PARTIAL mode
                    continue

        return self.result

    def _load_raw_session_data(
        self, year: int, event: str, session_type: str
    ) -> Dict[str, Any]:
        """Load raw session data from ingestion output"""

        session_dir = self.config.raw_data_path / str(year) / event / session_type

        if not session_dir.exists():
            raise FileNotFoundError(f"Session data directory not found: {session_dir}")

        self.logger.debug("Loading raw data from: %s", session_dir)

        # Load all components of session data
        session_data = {
            "session_info": None,
            "laps": None,
            "results": None,
            "weather": None,
            "telemetry": None,
        }

        # Load session info (JSON)
        session_info_file = session_dir / "session_info.json"
        if session_info_file.exists():
            with open(session_info_file, "r", encoding="utf-8") as f:
                session_data["session_info"] = json.load(f)

        # Load laps (Parquet)
        laps_file = (
            session_dir
            / f"laps.{self.config.raw_data_path.name.split('.')[-1] or 'parquet'}"
        )
        if not laps_file.exists():
            laps_file = session_dir / "laps.parquet"  # Fallback

        if laps_file.exists():
            session_data["laps"] = pd.read_parquet(laps_file)

        # Load results (Parquet)
        results_file = session_dir / "results.parquet"
        if results_file.exists():
            session_data["results"] = pd.read_parquet(results_file)

        # Load weather (Parquet)
        weather_file = session_dir / "weather.parquet"
        if weather_file.exists():
            session_data["weather"] = pd.read_parquet(weather_file)

        # Load telemetry (JSON)
        telemetry_file = session_dir / "telemetry.json"
        if telemetry_file.exists():
            with open(telemetry_file, "r", encoding="utf-8") as f:
                session_data["telemetry"] = json.load(f)

        for k, v in session_data.items():
            if v is not None:
                self.logger.debug("Loaded data components: %s", k)

        return session_data

    def _save_processor_output(
        self, processor_name: str, output: Any, year: int, event: str, session_type: str
    ):
        """Save processor output to file"""

        session_id = f"{year}_{event}_{session_type}"

        # Determine output directory based on processor
        if processor_name == "session":
            output_dir = self.output_dirs["sessions"]
            filename = f"{session_id}.parquet"
        elif processor_name == "lap":
            output_dir = self.output_dirs["laps"]
            filename = f"{session_id}_laps.parquet"
        else:
            output_dir = self.output_dirs.get(
                processor_name, self.config.processed_data_path
            )
            filename = f"{session_id}_{processor_name}.parquet"

        output_file = output_dir / filename

        try:
            if isinstance(output, pd.DataFrame):
                save_data(output, output_file, "parquet", "snappy")
                self.logger.debug("Saved %s output: %s", processor_name, output_file)
            else:
                self.logger.warning(
                    "Cannot save non-DataFrame output from %s", processor_name
                )

        except Exception as e:
            self.logger.error("Failed to save %s output: %s", processor_name, str(e))

    def _is_session_processed(self, year: int, event: str, session_type: str) -> bool:
        """Check if session has already been processed"""

        session_id = f"{year}_{event}_{session_type}"

        # Check if output files exist for all configured processors
        for processor_name in self.processors:
            if processor_name == "session":
                output_file = self.output_dirs["sessions"] / f"{session_id}.parquet"
            elif processor_name == "lap":
                output_file = self.output_dirs["laps"] / f"{session_id}_laps.parquet"
            else:
                output_file = (
                    self.config.processed_data_path
                    / processor_name
                    / f"{session_id}_{processor_name}.parquet"
                )

            if not output_file.exists():
                return False

        return True

    def _get_year_events(self, year: int) -> List[str]:
        """Get list of events for a year from raw data"""

        year_dir = self.config.raw_data_path / str(year)

        if not year_dir.exists():
            raise FileNotFoundError(f"Year directory not found: {year_dir}")

        # Get all event directories
        events = []
        for event_dir in year_dir.iterdir():
            if event_dir.is_dir() and event_dir.name != "pipeline_runs":
                events.append(event_dir.name)

        events.sort()  # Sort alphabetically

        self.logger.info("Found %d events for %d", len(events), year)
        return events

    def _process_session_list(
        self, sessions: List[Tuple[int, str, str]]
    ) -> PipelineResult:
        """Process a list of sessions"""

        self.result.total_items = len(sessions)

        for year, event, session_type in sessions:
            try:
                self._process_single_session(year, event, session_type)
            except Exception as e:
                if self.config.error_mode == ErrorMode.STRICT:
                    raise
                continue

        return self.result

    def process_year(
        self, year: int, session_types: Optional[List[str]] = None
    ) -> PipelineResult:
        """
        Process all sessions for a specific year

        Args:
            year: Season year to process
            session_types: List of session types to process (default: ['Q', 'R'])

        Returns:
            PipelineResult with execution summary
        """

        session_types = session_types or ["Q", "R"]
        return self.run(year=year, session_types=session_types)
