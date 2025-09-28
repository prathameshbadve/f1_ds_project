"""
Base classes and foundational components for data processing
"""

from .base_processor import BaseProcessor
from .data_contracts import (
    DataContract,
    InputContract,
    OutputContract,
    ValidationResult,
)
from .processing_context import ProcessingContext, ProcessingMetadata

__all__ = [
    "BaseProcessor",
    "DataContract",
    "InputContract",
    "OutputContract",
    "ValidationResult",
    "ProcessingContext",
    "ProcessingMetadata",
]
