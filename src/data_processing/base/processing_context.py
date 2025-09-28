"""
Processing context and metadata management
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid

from config.logging import get_logger


@dataclass
class ProcessingMetadata:
    """Metadata about processing operation"""

    # Processing identification
    processing_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    processor_name: str = ""
    processor_version: str = "1.0.0"

    # Timing information
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    # Data information
    input_source: Optional[str] = None
    output_destination: Optional[str] = None
    records_processed: Optional[int] = None

    # Quality metrics
    validation_passed: bool = True
    warnings_count: int = 0
    errors_count: int = 0

    # Custom metrics (processor-specific)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def start_processing(self):
        """Mark start of processing"""
        self.start_time = datetime.now()

    def end_processing(self):
        """Mark end of processing and calculate duration"""
        self.end_time = datetime.now()
        if self.start_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()

    def add_custom_metric(self, key: str, value: Any):
        """Add a custom metric"""
        self.custom_metrics[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "processing_id": self.processing_id,
            "processor_name": self.processor_name,
            "processor_version": self.processor_version,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "input_source": self.input_source,
            "output_destination": self.output_destination,
            "records_processed": self.records_processed,
            "validation_passed": self.validation_passed,
            "warnings_count": self.warnings_count,
            "errors_count": self.errors_count,
            "custom_metrics": self.custom_metrics,
        }


@dataclass
class ProcessingContext:
    """
    Context object that carries information through processing pipeline
    Contains metadata, configuration, and state information
    """

    # Processing metadata
    metadata: ProcessingMetadata = field(default_factory=ProcessingMetadata)

    # Data lineage
    source_files: List[str] = field(default_factory=list)
    intermediate_files: List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)

    # Processing configuration
    config: Dict[str, Any] = field(default_factory=dict)

    # Session/event context for F1 data
    year: Optional[int] = None
    event_name: Optional[str] = None
    session_type: Optional[str] = None

    # Processing state
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checkpoints: Dict[str, Any] = field(
        default_factory=dict
    )  # For saving intermediate state

    def __post_init__(self):
        self.logger = get_logger("processing_context")

    def add_source_file(self, file_path: str):
        """Add a source file to the lineage"""
        self.source_files.append(file_path)
        self.logger.debug("Added source file: %s", file_path)

    def add_intermediate_file(self, file_path: str):
        """Add an intermediate file to the lineage"""
        self.intermediate_files.append(file_path)
        self.logger.debug("Added intermediate file: %s", file_path)

    def add_output_file(self, file_path: str):
        """Add an output file to the lineage"""
        self.output_files.append(file_path)
        self.logger.debug("Added output file: %s", file_path)

    def add_error(self, error_message: str):
        """Add an error to the context"""
        self.errors.append(error_message)
        self.metadata.errors_count = len(self.errors)
        self.logger.error(error_message)

    def add_warning(self, warning_message: str):
        """Add a warning to the context"""
        self.warnings.append(warning_message)
        self.metadata.warnings_count = len(self.warnings)
        self.logger.warning(warning_message)

    def set_checkpoint(self, name: str, data: Any):
        """Set a processing checkpoint"""
        self.checkpoints[name] = data
        self.logger.debug("Checkpoint set: %s", name)

    def get_checkpoint(self, name: str) -> Any:
        """Get a processing checkpoint"""
        return self.checkpoints.get(name)

    def create_child_context(self, processor_name: str) -> "ProcessingContext":
        """
        Create a child context for sub-processing
        Inherits parent context but has its own metadata
        """
        child_metadata = ProcessingMetadata(
            processor_name=processor_name, processing_id=str(uuid.uuid4())
        )

        child_context = ProcessingContext(
            metadata=child_metadata,
            source_files=self.source_files.copy(),
            config=self.config.copy(),
            year=self.year,
            event_name=self.event_name,
            session_type=self.session_type,
        )

        return child_context

    def merge_child_context(self, child_context: "ProcessingContext"):
        """Merge child context results back into parent"""
        # Merge file lineage
        self.intermediate_files.extend(child_context.intermediate_files)
        self.output_files.extend(child_context.output_files)

        # Merge errors and warnings
        self.errors.extend(child_context.errors)
        self.warnings.extend(child_context.warnings)

        # Update metadata counts
        self.metadata.errors_count = len(self.errors)
        self.metadata.warnings_count = len(self.warnings)

        # Merge custom metrics
        for key, value in child_context.metadata.custom_metrics.items():
            self.metadata.custom_metrics[
                f"{child_context.metadata.processor_name}_{key}"
            ] = value

    def get_session_identifier(self) -> str:
        """Get a unique identifier for the current session being processed"""
        if all([self.year, self.event_name, self.session_type]):
            return f"{self.year}_{self.event_name}_{self.session_type}"
        return f"unknown_session_{self.metadata.processing_id}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization"""
        return {
            "metadata": self.metadata.to_dict(),
            "source_files": self.source_files,
            "intermediate_files": self.intermediate_files,
            "output_files": self.output_files,
            "config": self.config,
            "year": self.year,
            "event_name": self.event_name,
            "session_type": self.session_type,
            "errors": self.errors,
            "warnings": self.warnings,
            "session_identifier": self.get_session_identifier(),
        }
