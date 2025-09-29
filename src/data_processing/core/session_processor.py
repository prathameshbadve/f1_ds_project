"""
Processor for session level data. We want to have a fact table for every session.
The following fields are essential for any session.

    - event_name
    - round_number
    - session_name
    - location
    - country
    - session_date

    - circuit
"""

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from ..base.base_processor import BaseProcessor
from ..base.data_contracts import (
    InputContract,
    OutputContract,
    ValidationLevel,
    ValidationResult,
)
from ..base.processing_context import ProcessingContext


class SessionInputContract(InputContract):
    """Custom input contract for F1 session data"""

    def __init__(self):
        super().__init__(
            name="f1_session_input",
            required_columns=[
                "event_name",
                "location",
                "country",
                "session_name",
                "session_date",
                "round_number",
            ],
            validation_level=ValidationLevel.WARNING,
        )

    def validate_schema(self, data: Any) -> ValidationResult:
        """Validate F1 session input data"""

        errors = []
        warnings = []
        metadata = {}

        if isinstance(data, dict):
            metadata["input_type"] = "session_dict"
            metadata["keys"] = list(data.keys())

            # Check if all required columns are present in the input data and are not empty
            expected_keys = self.required_columns

            for key in expected_keys:
                if key not in data or data[key] is None:
                    warnings.append(
                        self.create_validation_error(
                            field=key,
                            error_type="missing_session_component",
                            message=f"Missing required field {key}",
                            severity=ValidationLevel.STRICT,
                        )
                    )

        elif isinstance(data, pd.DataFrame):
            # Handle case where someone passes DataFrame directly
            errors.append(
                self.create_validation_error(
                    field="data_type",
                    error_type="unexpected_type",
                    message="Expected session dictionary, got DataFrame",
                    severity=ValidationLevel.STRICT,
                )
            )

        else:
            errors.append(
                self.create_validation_error(
                    field="data_type",
                    error_type="invalid_type",
                    message=f"Expected dict with session data, got {type(data)}",
                    severity=ValidationLevel.STRICT,
                )
            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )

    def validate_content(self, data):
        """Validate F1 session data quality"""

        errors = []
        warnings = []
        metadata = {}

        if isinstance(data, dict):
            # First check if the data has a session_date and is reasonable
            if "session_date" in data and data["session_date"]:
                session_date = data["session_date"]

                try:
                    if isinstance(session_date, str):
                        # Try to parse different date formats
                        parsed_date = pd.to_datetime(session_date)
                        year = parsed_date.year
                    else:
                        year = (
                            session_date.year if hasattr(session_date, "year") else None
                        )

                    if year and (year < 1950 or year > datetime.now().year + 1):
                        errors.append(
                            self.create_validation_error(
                                field="session_date",
                                error_type="invalid_date_range",
                                message=f"Session year {year} is outside valid F1 range (1950-{datetime.now().year + 1})",
                                severity=ValidationLevel.STRICT,
                                actual_value=year,
                            )
                        )

                    metadata["parsed_session_year"] = year

                except Exception as e:
                    warnings.append(
                        self.create_validation_error(
                            field="session_date",
                            error_type="date_parsing_error",
                            message=f"Could not parse session date: {str(e)}",
                            severity=ValidationLevel.WARNING,
                        )
                    )

            # Then check if the session name is a valid F1 session
            if "session_name" in data and data["session_name"]:
                session_name = data["session_name"]
                valid_session_types = [
                    "Practice 1",
                    "Practice 2",
                    "Practice 3",
                    "Qualifying",
                    "Race",
                    "Sprint",
                    "Sprint Qualifying",
                    "Sprint Shootout",
                ]

                if session_name not in valid_session_types:
                    warnings.append(
                        self.create_validation_error(
                            field="session_name",
                            error_type="unknown_session_type",
                            message=f"Session type '{session_name}' not in standard F1 sessions",
                            severity=ValidationLevel.WARNING,
                            actual_value=session_name,
                        )
                    )

            # Validate location/country consistency
            if all(key in data for key in ["location", "country"]):
                location = data["location"]
                country = data["country"]

                # Basic consistency check - location and country shouldn't be the same
                if location == country:
                    warnings.append(
                        self.create_validation_error(
                            field="location_country",
                            error_type="location_country_same",
                            message=f"Location and country are identical: {location}",
                            severity=ValidationLevel.WARNING,
                        )
                    )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )


class SessionOutputContract(OutputContract):
    """Custom output contract for F1 session output"""

    def __init__(self):
        super().__init__(
            name="f1_session_ouput",
            expected_columns=[
                "session_id",
                "event_name_clean",
                "session_name_clean",
                "session_type",
                "session_date_clean",
                "year",
                "location_clean",
                "country_clean",
                "weekend_format",
                "round_number",
                "processed_at",
            ],
            column_constraints={
                "year": {"min": 1950, "max": datetime.now().year + 1, "not_null": True},
                "session_id": {"not_null": True},
            },
            validation_level=ValidationLevel.WARNING,
        )

    def validate_schema(self, data: Any) -> ValidationResult:
        """Validate processed session output schema"""

        errors = []
        warnings = []
        metadata = {}

        if isinstance(data, pd.DataFrame):
            metadata["output_row_count"] = len(data)
            metadata["output_columns"] = list(data.columns)

            # Check expected columns
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

            # Check for unexpected columns (informational)
            extra_columns = [
                col for col in data.columns if col not in self.expected_columns
            ]
            if extra_columns:
                warnings.append(
                    self.create_validation_error(
                        field="extra_columns",
                        error_type="unexpected_columns",
                        message=f"Output contains unexpected columns: {extra_columns}",
                        severity=ValidationLevel.WARNING,
                        actual_value=extra_columns,
                    )
                )

        else:
            errors.append(
                self.create_validation_error(
                    field="data_type",
                    error_type="invalid_output_type",
                    message=f"Expected DataFrame output, got {type(data)}",
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
        """Validate processed session output content quality"""

        errors = []
        warnings = []
        metadata = {}

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
                                    actual_value=null_count,
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

            # Check for duplicate session IDs
            if "session_id" in data.columns:
                duplicate_count = data["session_id"].duplicated().sum()
                if duplicate_count > 0:
                    errors.append(
                        self.create_validation_error(
                            field="session_id",
                            error_type="duplicate_ids",
                            message=f"Found {duplicate_count} duplicate session IDs",
                            severity=ValidationLevel.STRICT,
                            actual_value=duplicate_count,
                        )
                    )

                metadata["unique_session_count"] = data["session_id"].nunique()

            # Quality metrics
            metadata["total_sessions_processed"] = len(data)
            if "year" in data.columns:
                metadata["years_covered"] = sorted(data["year"].unique().tolist())
            if "session_type" in data.columns:
                metadata["session_types_processed"] = (
                    data["session_type"].value_counts().to_dict()
                )

            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                metadata=metadata,
            )


class SessionProcessor(BaseProcessor):
    """
    F1 Session Processor

    Processes F1 session metadata to create clean, standardized session information

    Input: Dictionary with session_info from your data ingestion
    Output: Clean DataFrame with standardized session metadata

    Processing steps:
    1. Extract session metadata from ingested data
    2. Clean and standardize event/session names
    3. Parse and validate dates
    4. Create derived fields (year, session_type, etc.)
    5. Generate unique session identifiers
    """

    def __init__(self):
        super().__init__("session_processor", "1.0.0")

        # Set up validation contracts
        self.set_input_contract(SessionInputContract())
        self.set_output_contract(SessionOutputContract())

        # Session type mapping for standardization
        self.session_type_mapping = {
            "Practice 1": "FP1",
            "Practice 2": "FP2",
            "Practice 3": "FP3",
            "Qualifying": "Q",
            "Race": "R",
            "Sprint": "S",
            "Sprint Qualifying": "SQ",
            "Sprint Shootout": "SS",
        }

        # Event name cleaning patterns (you can expand this)
        self.event_name_cleaning = {
            "FORMULA 1": "",
            "GRAND PRIX": "GP",
            "EMIRATES": "",  # Sponsor names
            "ARAMCO": "",  # Sponsor names
            "ROLEX": "",
        }

    def _process_data(self, data: Any, context: ProcessingContext) -> pd.DataFrame:
        """
        Core session processing logic

        Args:
            data: Dictionary containing session_info and other components
            context: Processing context with metadata

        Returns:
            Clean DataFrame with session metadata
        """

        self.logger.info("Processing F1 session metadata")

        # Extract session info from the input data
        session_info = data.copy()

        # Create base session record
        processed_session = self._create_base_session_record(session_info, context)

        # Clean and standardize session data
        processed_session = self._clean_session_data(processed_session, context)

        # Add derived fields
        processed_session = self._add_derived_fields(processed_session, context)

        # Create DataFrame
        session_df = pd.DataFrame([processed_session])

        # Final cleanup and validation
        session_df = self._final_cleanup(session_df, context)

        self.logger.info(
            "Successfully processed session: %s",
            processed_session.get("session_id", "unknown"),
        )

        # Add processing metrics to context
        context.metadata.add_custom_metric("sessions_processed", 1)
        context.metadata.add_custom_metric(
            "session_year", processed_session.get("year")
        )
        context.metadata.add_custom_metric(
            "session_type", processed_session.get("session_type")
        )

        return session_df

    def _create_base_session_record(
        self, session_info: Dict[str, Any], context: ProcessingContext
    ) -> Dict[str, Any]:
        """Create base session record from raw session info"""

        self.logger.debug("Creating base session record")

        # Extract basic fields with defaults
        base_record = {
            "event_name": session_info.get("event_name", "Unknown Event"),
            "location": session_info.get("location", "Unknown Location"),
            "country": session_info.get("country", "Unknown Country"),
            "session_name": session_info.get("session_name", "Unknown Session"),
            "session_date": session_info.get("session_date"),
            "official_event_name": session_info.get("official_event_name"),
            "event_format": session_info.get("event_format"),
            "round_number": session_info.get("round_number"),
        }

        # Handle None values
        for key, value in base_record.items():
            if value is None:
                base_record[key] = f"Unknown {key.title()}"
                context.add_warning(f"Missing value for {key}, using default")

        return base_record

    def _clean_session_data(
        self, session_record: Dict[str, Any], context: ProcessingContext
    ) -> Dict[str, Any]:
        """Clean and standardize session data fields"""

        self.logger.debug("Cleaning session data")

        # Clean event name
        event_name = str(session_record["event_name"]).strip()
        for pattern, replacement in self.event_name_cleaning.items():
            event_name = event_name.replace(pattern, replacement)

        session_record["event_name_clean"] = event_name.strip()

        # Clean location and country. Strip leading and trailing whitespace
        # and convert to Title case
        session_record["location_clean"] = (
            str(session_record["location"]).strip().title()
        )
        session_record["country_clean"] = str(session_record["country"]).strip().title()

        # Standardize session name
        session_name = str(session_record["session_name"]).strip()
        session_record["session_name_clean"] = session_name

        # Map to standard session type
        session_type = self.session_type_mapping.get(session_name, session_name)
        session_record["session_type"] = session_type

        # Parse session date
        session_date = session_record["session_date"]
        if session_date:
            try:
                if isinstance(session_date, str):
                    parsed_date = pd.to_datetime(session_date)
                else:
                    parsed_date = pd.to_datetime(str(session_date))

                session_record["session_date_clean"] = parsed_date
                session_record["year"] = parsed_date.year

            except Exception as e:
                self.logger.warning(
                    "Could not parse session date %s: %s", session_date, str(e)
                )
                session_record["session_date_clean"] = None
                session_record["year"] = None
                context.add_warning(f"Failed to parse session date: {str(e)}")
        else:
            session_record["session_date_clean"] = None
            session_record["year"] = None

        return session_record

    def _add_derived_fields(
        self, session_record: Dict[str, Any], context: ProcessingContext
    ) -> Dict[str, Any]:
        """Add derived fields to session record"""

        self.logger.debug("Adding derived fields")

        # Create unique session identifier
        year = session_record.get("year", "unknown")
        event = session_record.get("event_name_clean", "unknown").replace(" ", "_")
        session_type = session_record.get("session_type", "unknown")

        session_id = f"{year}_{event}_{session_type}"
        session_record["session_id"] = session_id

        # Determine weekend format
        event_format = session_record.get("event_format", "")
        if "sprint" in str(event_format).lower():
            weekend_format = "Sprint"
        else:
            weekend_format = "Conventional"

        session_record["weekend_format"] = weekend_format

        # Add processing metadata
        session_record["processed_at"] = datetime.now()
        session_record["processor_version"] = self.version

        return session_record

    def _final_cleanup(
        self, session_df: pd.DataFrame, context: ProcessingContext
    ) -> pd.DataFrame:
        """Final cleanup of session DataFrame"""

        self.logger.debug("Performing final cleanup")

        # Select only the columns we want in output
        output_columns = [
            "session_id",
            "event_name_clean",
            "session_name_clean",
            "session_type",
            "session_date_clean",
            "year",
            "location_clean",
            "country_clean",
            "weekend_format",
            "round_number",
            "processed_at",
        ]

        # Only keep columns that exist
        existing_columns = [col for col in output_columns if col in session_df.columns]
        session_df = session_df[existing_columns].copy()

        # Ensure data types
        if "year" in session_df.columns:
            session_df["year"] = pd.to_numeric(session_df["year"], errors="coerce")

        if "round_number" in session_df.columns:
            session_df["round_number"] = pd.to_numeric(
                session_df["round_number"], errors="coerce"
            )

        return session_df

    def get_required_config_keys(self) -> List[str]:
        """Required configuration keys for session processor"""
        return []  # No required config for now

    def get_default_config(self) -> Dict[str, Any]:
        """Default configuration for session processor"""
        return {
            "clean_sponsor_names": True,
            "standardize_session_types": True,
            "validate_dates": True,
        }
