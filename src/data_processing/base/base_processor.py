"""
Abstract base class for all data processors
Defines the interface and common functionality
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from config.logging import get_logger
from .data_contracts import InputContract, OutputContract, ValidationResult
from .processing_context import ProcessingContext


class ProcessorException(Exception):
    """Custom exception for processor errors"""

    def __init__(
        self,
        message: str,
        processor_name: str = "",
        context: ProcessingContext = None,
    ):
        super().__init__(message)
        self.processor_name = processor_name
        self.context = context


class BaseProcessor(ABC):
    """
    Abstract base class for all data processors

    Provides common functionality:
    - Input/output validation
    - Logging and error handling
    - Processing context management
    - Metadata tracking
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.logger = get_logger(f"data_processing.{name}")

        # Contracts - to be defined by subclasses
        self.input_contract: Optional[InputContract] = None
        self.output_contract: Optional[OutputContract] = None

        # Processing statistics
        self.total_processed = 0
        self.total_errors = 0
        self.total_warnings = 0

    @abstractmethod
    def _process_data(self, data: Any, context: ProcessingContext) -> Any:
        """
        Core processing logic - must be implemented by subclasses

        Args:
            data: Input data to process
            context: Processing context with metadata and configuration

        Returns:
            Processed data
        """
        pass

    def process(
        self, data: Any, context: Optional[ProcessingContext] = None
    ) -> Tuple[Any, ProcessingContext]:
        """
        Main processing method with full validation and error handling

        Args:
            data: Input data to process
            context: Optional processing context (creates new one if None)

        Returns:
            Tuple of (processed_data, updated_context)
        """
        # Initialize context if not provided
        if context is None:
            context = ProcessingContext()

        # Set up metadata
        context.metadata.processor_name = self.name
        context.metadata.processor_version = self.version
        context.metadata.start_processing()

        try:
            self.logger.info("Starting processing with %s", self.name)

            # Input validation
            if self.input_contract:
                self.logger.debug("Validating input data")
                input_validation = self.input_contract.validate(data)
                self._handle_validation_result(input_validation, context, "input")

                if not input_validation.is_valid:
                    raise ProcessorException(
                        f"Input validation failed: {input_validation.get_error_summary()}",
                        self.name,
                        context,
                    )

            # Core processing
            self.logger.debug("Executing core processing logic")
            processed_data = self._process_data(data, context)

            # Track processing statistics
            if isinstance(processed_data, pd.DataFrame):
                context.metadata.records_processed = len(processed_data)
            elif isinstance(processed_data, (list, dict)):
                context.metadata.records_processed = len(processed_data)

            # Output validation
            if self.output_contract:
                self.logger.debug("Validating output data")
                output_validation = self.output_contract.validate(processed_data)
                self._handle_validation_result(output_validation, context, "output")

                if not output_validation.is_valid:
                    raise ProcessorException(
                        f"Output validation failed: {output_validation.get_error_summary()}",
                        self.name,
                        context,
                    )

            # Success - update metadata
            context.metadata.end_processing()
            context.metadata.validation_passed = True

            self.total_processed += 1

            self.logger.info(
                "Processing completed successfully in %.2f s",
                context.metadata.duration_seconds,
            )

            return processed_data, context

        except ProcessorException:
            # Re-raise processor exceptions
            context.metadata.end_processing()
            context.metadata.validation_passed = False
            self.total_errors += 1
            raise

        except Exception as e:
            # Wrap other exceptions
            context.metadata.end_processing()
            context.metadata.validation_passed = False
            self.total_errors += 1

            error_message = f"Processing failed in {self.name}: {str(e)}"
            context.add_error(error_message)

            raise ProcessorException(error_message, self.name, context) from e

    def _handle_validation_result(
        self, result: ValidationResult, context: ProcessingContext, stage: str
    ):
        """Handle validation results by updating context"""

        # Add errors to context
        for error in result.errors:
            context.add_error(f"{stage} validation - {error.field}: {error.message}")

        # Add warnings to context
        for warning in result.warnings:
            context.add_warning(
                f"{stage} validation - {warning.field}: {warning.message}"
            )

        # Add validation metadata
        context.metadata.add_custom_metric(
            f"{stage}_validation_metadata", result.metadata
        )

    def set_input_contract(self, contract: InputContract):
        """Set the input validation contract"""
        self.input_contract = contract
        self.logger.debug("Input contract set: %s", contract.name)

    def set_output_contract(self, contract: OutputContract):
        """Set the output validation contract"""
        self.output_contract = contract
        self.logger.debug("Output contract set: %s", contract.name)

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "processor_name": self.name,
            "processor_version": self.version,
            "total_processed": self.total_processed,
            "total_errors": self.total_errors,
            "total_warnings": self.total_warnings,
            "success_rate": (self.total_processed - self.total_errors)
            / max(self.total_processed, 1)
            * 100,
        }

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate processor configuration
        Override in subclasses for specific validation needs
        """
        # Base implementation - override in subclasses
        return True

    def get_required_config_keys(self) -> List[str]:
        """
        Get list of required configuration keys
        Override in subclasses
        """
        return []

    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for this processor
        Override in subclasses
        """
        return {}

    def __str__(self) -> str:
        return f"{self.name} v{self.version}"

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}(name='{self.name}', version='{self.version}')>"
        )
