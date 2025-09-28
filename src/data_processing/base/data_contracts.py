"""
Data contracts for input/output validation and schema definition
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Type

import pandas as pd

from config.logging import get_logger


class ValidationLevel(Enum):
    """Validation strictness level"""

    STRICT = "strict"  # Fail on any validation error
    WARNING = "warning"  # Log warnings but continue
    PERMISSIVE = "permissive"  # Log info only, always continue


@dataclass
class ValidationError:
    """Individual validation error details"""

    field: str
    error_type: str
    message: str
    severity: ValidationLevel
    actual_value: Any = None
    expected_value: Any = None


@dataclass
class ValidationResult:
    """Result of data validation"""

    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    metadata: Dict[str, Any]

    @property
    def has_errors(self) -> bool:
        """Check if validation has any errors"""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if validation has any warnings"""
        return len(self.warnings) > 0

    def get_error_summary(self) -> str:
        """Get summary of validation errors"""
        if not self.has_errors:
            return "No validation errors"

        error_counts = {}
        for error in self.errors:
            error_counts[error.error_type] = error_counts.get(error.error_type, 0) + 1

        return f"Validation errors: {dict(error_counts)}"


class DataContract(ABC):
    """
    Abstract base class for data contracts
    Defines the interface for validating data structures
    """

    def __init__(
        self, name: str, validation_level: ValidationLevel = ValidationLevel.STRICT
    ):
        self.name = name
        self.validation_level = validation_level
        self.logger = get_logger(f"data_contracts.{name}")

    @abstractmethod
    def validate_schema(self, data: Any) -> ValidationResult:
        """Validate data schema (columns, types, structure)"""
        pass

    @abstractmethod
    def validate_content(self, data: Any) -> ValidationResult:
        """Validate data content (business rules, ranges, relationships)"""
        pass

    def validate(self, data: Any) -> ValidationResult:
        """
        Complete validation: schema + content

        Args:
            data: Data to validate

        Returns:
            ValidationResult with combined schema and content validation
        """
        self.logger.debug("Starting validation for %s", self.name)

        # Schema validation first
        schema_result = self.validate_schema(data)

        # Content validation only if schema passes or validation level allows
        content_result = None
        if schema_result.is_valid or self.validation_level != ValidationLevel.STRICT:
            content_result = self.validate_content(data)

        # Combine results
        all_errors = schema_result.errors.copy()
        all_warnings = schema_result.warnings.copy()

        if content_result:
            all_errors.extend(content_result.errors)
            all_warnings.extend(content_result.warnings)

        # Determine final validity based on validation level
        is_valid = self._determine_validity(all_errors, all_warnings)

        combined_metadata = {
            "schema_validation": schema_result.metadata,
            "content_validation": content_result.metadata if content_result else {},
            "validation_level": self.validation_level.value,
            "data_type": type(data).__name__,
        }

        result = ValidationResult(
            is_valid=is_valid,
            errors=all_errors,
            warnings=all_warnings,
            metadata=combined_metadata,
        )

        self._log_validation_result(result)
        return result

    def _determine_validity(
        self, errors: List[ValidationError], warnings: List[ValidationError]
    ) -> bool:
        """Determine if data is valid based on validation level"""
        if self.validation_level == ValidationLevel.STRICT:
            return len(errors) == 0
        elif self.validation_level == ValidationLevel.WARNING:
            # Valid if no critical errors (you can customize this logic)
            critical_errors = [
                e for e in errors if e.severity == ValidationLevel.STRICT
            ]
            return len(critical_errors) == 0

        return True

    def _log_validation_result(self, result: ValidationResult):
        """Log validation results"""
        if result.is_valid:
            if result.has_warnings:
                self.logger.warning(
                    "%s validation passed with %d warnings",
                    self.name,
                    len(result.warnings),
                )
            else:
                self.logger.debug("%s validation passed", self.name)
        else:
            self.logger.error(
                "%s validation failed: %s", self.name, result.get_error_summary()
            )

    def create_validation_error(
        self,
        field: str,
        error_type: str,
        message: str,
        severity: ValidationLevel = None,
        actual_value: Any = None,
        expected_value: Any = None,
    ) -> ValidationError:
        """Helper method to create validation errors"""
        return ValidationError(
            field=field,
            error_type=error_type,
            message=message,
            severity=severity or self.validation_level,
            actual_value=actual_value,
            expected_value=expected_value,
        )


class InputContract(DataContract):
    """
    Contract for validating processor input data
    Ensures data meets requirements before processing
    """

    def __init__(
        self,
        name: str,
        required_columns: List[str] = None,
        optional_columns: List[str] = None,
        column_types: Dict[str, Type] = None,
        validation_level: ValidationLevel = ValidationLevel.STRICT,
    ):
        super().__init__(name, validation_level)
        self.required_columns = required_columns or []
        self.optional_columns = optional_columns or []
        self.column_types = column_types or {}

    def validate_schema(self, data: Any) -> ValidationResult:
        """Validate input data schema"""
        errors = []
        warnings = []
        metadata = {}

        # To Do: Implement based on your data structure
        # Example structure - you'll customize this:

        if isinstance(data, pd.DataFrame):
            metadata["row_count"] = len(data)
            metadata["column_count"] = len(data.columns)
            metadata["columns"] = list(data.columns)

            # Check required columns
            missing_columns = [
                col for col in self.required_columns if col not in data.columns
            ]
            for col in missing_columns:
                errors.append(
                    self.create_validation_error(
                        field=col,
                        error_type="missing_column",
                        message=f"Required column '{col}' is missing",
                        severity=ValidationLevel.STRICT,
                    )
                )

            # Check column types
            for col, expected_type in self.column_types.items():
                if col in data.columns:
                    actual_type = data[col].dtype
                    # To Do: Add type checking logic based on your needs
                    # This is a placeholder - implement type validation
                    pass

        elif isinstance(data, dict):
            metadata["keys"] = list(data.keys())

            # Check required keys
            missing_keys = [key for key in self.required_columns if key not in data]
            for key in missing_keys:
                errors.append(
                    self.create_validation_error(
                        field=key,
                        error_type="missing_key",
                        message=f"Required key '{key}' is missing from dictionary",
                        severity=ValidationLevel.STRICT,
                    )
                )

        else:
            errors.append(
                self.create_validation_error(
                    field="data_type",
                    error_type="invalid_type",
                    message=f"Expected DataFrame or dict, got {type(data)}",
                    severity=ValidationLevel.STRICT,
                )
            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )

    def validate_content(self, data: Any) -> ValidationResult:
        """Validate input data content"""
        errors = []
        warnings = []
        metadata = {}

        # To Do: Implement business rule validation
        # Examples of what you might validate:
        # - Lap times are reasonable (not negative, not too fast/slow)
        # - Driver names are consistent
        # - Session dates are valid
        # - Position numbers are sequential

        # Placeholder structure:
        if isinstance(data, pd.DataFrame):
            # Example: Check for completely empty DataFrame
            if len(data) == 0:
                warnings.append(
                    self.create_validation_error(
                        field="data",
                        error_type="empty_data",
                        message="DataFrame is empty",
                        severity=ValidationLevel.WARNING,
                    )
                )

            metadata["empty_rows"] = data.isnull().all(axis=1).sum()
            metadata["duplicate_rows"] = data.duplicated().sum()

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )


class OutputContract(DataContract):
    """
    Contract for validating processor output data
    Ensures processed data meets quality standards
    """

    def __init__(
        self,
        name: str,
        expected_columns: List[str] = None,
        column_constraints: Dict[str, Dict] = None,
        validation_level: ValidationLevel = ValidationLevel.STRICT,
    ):
        super().__init__(name, validation_level)
        self.expected_columns = expected_columns or []
        self.column_constraints = column_constraints or {}
        # column_constraints format: {'column_name': {'min': 0, 'max': 100, 'not_null': True}}

    def validate_schema(self, data: Any) -> ValidationResult:
        """Validate output data schema"""
        errors = []
        warnings = []
        metadata = {}

        # To Do: Implement output schema validation
        # Similar to InputContract but focused on output requirements

        if isinstance(data, pd.DataFrame):
            metadata["output_row_count"] = len(data)
            metadata["output_columns"] = list(data.columns)

            # Check expected columns are present
            missing_columns = [
                col for col in self.expected_columns if col not in data.columns
            ]
            for col in missing_columns:
                errors.append(
                    self.create_validation_error(
                        field=col,
                        error_type="missing_output_column",
                        message=f"Expected output column '{col}' is missing",
                        severity=ValidationLevel.STRICT,
                    )
                )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )

    def validate_content(self, data: Any) -> ValidationResult:
        """Validate output data content quality"""
        errors = []
        warnings = []
        metadata = {}

        # To Do: Implement output quality validation
        # Examples:
        # - No null values in critical columns
        # - Calculated fields are within expected ranges
        # - Aggregations sum up correctly
        # - No duplicate keys where uniqueness is expected

        if isinstance(data, pd.DataFrame):
            # Check column constraints
            for col, constraints in self.column_constraints.items():
                if col in data.columns:
                    # Check not_null constraint
                    if constraints.get("not_null", False):
                        null_count = data[col].isnull().sum()
                        if null_count > 0:
                            errors.append(
                                self.create_validation_error(
                                    field=col,
                                    error_type="null_values",
                                    message=f"Column '{col}' has {null_count} null values but should not have any",
                                    severity=ValidationLevel.STRICT,
                                )
                            )

                    # Check min/max constraints for numeric columns
                    if "min" in constraints and pd.api.types.is_numeric_dtype(
                        data[col]
                    ):
                        min_val = data[col].min()
                        expected_min = constraints["min"]
                        if min_val < expected_min:
                            errors.append(
                                self.create_validation_error(
                                    field=col,
                                    error_type="value_below_minimum",
                                    message=f"Column '{col}' has minimum value {min_val}, expected >= {expected_min}",
                                    severity=ValidationLevel.STRICT,
                                    actual_value=min_val,
                                    expected_value=expected_min,
                                )
                            )

                    if "max" in constraints and pd.api.types.is_numeric_dtype(
                        data[col]
                    ):
                        max_val = data[col].max()
                        expected_max = constraints["max"]
                        if max_val > expected_max:
                            errors.append(
                                self.create_validation_error(
                                    field=col,
                                    error_type="value_above_maximum",
                                    message=f"Column '{col}' has maximum value {max_val}, expected <= {expected_max}",
                                    severity=ValidationLevel.STRICT,
                                    actual_value=max_val,
                                    expected_value=expected_max,
                                )
                            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )
