"""
Lap processor for F1 lap timing data
Cleans, validates, and enriches lap timing information with calculated metrics
"""

from typing import Any, Dict, List

import pandas as pd

from ..base.base_processor import BaseProcessor, ProcessingContext
from ..base.data_contracts import (
    InputContract,
    OutputContract,
    ValidationLevel,
    ValidationResult,
)


class LapInputContract(InputContract):
    """Custom input contract for F1 laps data"""

    def __init__(self):
        super().__init__(
            name="f1_lap_input",
            required_columns=[
                "Driver",
                "DriverNumber",
                "LapTime",
                "LapNumber",
                "Stint",
                "Compound",
                "TyreLife",
            ],
            optional_columns=[
                "Time",
                "PitOutTime",
                "PitInTime",
                "Sector1Time",
                "Sector2Time",
                "Sector3Time",
                "Sector1SessionTime",
                "Sector2SessionTime",
                "Sector3SessionTime",
                "SpeedI1",
                "SpeedI2",
                "SpeedFL",
                "SpeedST",
                "IsPersonalBest",
                "FreshTyre",
                "Team",
                "LapStartTime",
                "LapStartDate",
                "TrackStatus",
                "Position",
                "Deleted",
                "DeletedReason",
                "FastF1Generated",
                "IsAccurate",
                "LapTimeSeconds",
                "EventName",
                "SessionName",
                "SessionDate",
            ],
            validation_level=ValidationLevel.WARNING,
        )

    def validate_schema(self, data: Any) -> ValidationResult:
        """Validate the schema of laps input data"""

        errors = []
        warnings = []
        metadata = {}

        if isinstance(data, dict):
            # Handle case where we get full session dict
            if "laps" in data:
                laps_data = data["laps"]

                if laps_data is None or (
                    isinstance(laps_data, pd.DataFrame) and laps_data.empty
                ):
                    errors.append(
                        self.create_validation_error(
                            field="laps",
                            error_type="no_lap_data",
                            message="No lap data available in session",
                            severity=ValidationLevel.STRICT,
                        )
                    )
                    return ValidationResult(
                        is_valid=False,
                        errors=errors,
                        warnings=warnings,
                        metadata=metadata,
                    )

                # Recursively validate the laps DataFrame
                return self.validate_schema(laps_data)

            else:
                errors.append(
                    self.create_validation_error(
                        field="data_structure",
                        error_type="invalid_structure",
                        message="Expected 'laps' key in session data dictionary",
                        severity=ValidationLevel.STRICT,
                    )
                )

        elif isinstance(data, pd.DataFrame):
            metadata["input_row_count"] = len(data)
            metadata["input_columns"] = list(data.columns)

            # Check required columns
            missing_required = [
                col for col in self.required_columns if col not in data.columns
            ]
            for col in missing_required:
                errors.append(
                    self.create_validation_error(
                        field=col,
                        error_type="missing_required_column",
                        message=f"Required column '{col}' is missing",
                        severity=ValidationLevel.STRICT,
                    )
                )

            # Check for optional but important columns
            important_optional = [
                "Sector1Time",
                "Sector2Time",
                "Sector3Time",
                "Position",
                "FreshTyre",
                "EventName",
                "SessionName",
                "SessionDate",
                "IsPersonalBest",
                "TrackStatus",
                "LapTimeSeconds",
            ]
            missing_optional = [
                col for col in important_optional if col not in data.columns
            ]

            if missing_optional:
                warnings.append(
                    self.create_validation_error(
                        field="optional_columns",
                        error_type="missing_optional_columns",
                        message=f"Important optional columns missing: {missing_optional}",
                        severity=ValidationLevel.WARNING,
                        actual_value=missing_optional,
                    )
                )

            # Check if DataFrame is empty
            if len(data) == 0:
                errors.append(
                    self.create_validation_error(
                        field="data",
                        error_type="empty_dataframe",
                        message="Lap DataFrame is empty",
                        severity=ValidationLevel.STRICT,
                    )
                )

        else:
            errors.append(
                self.create_validation_error(
                    field="data_type",
                    error_type="invalid_type",
                    message=f"Expected DataFrame or dict with 'laps', got {type(data)}",
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
        """Validates the content of the F1 laps data"""

        errors = []
        warnings = []
        metadata = {}

        # Extract DataFrame if wrapped in dict
        if isinstance(data, dict) and "laps" in data:
            data = data["laps"]

        # Validate lap times are reasonable
        if "LapTimeSeconds" in data.columns:
            lap_times = data["LapTimeSeconds"].dropna()

            if len(lap_times) > 0:
                min_lap_time = lap_times.min()
                max_lap_time = lap_times.max()

                # F1 laps are typically 70-120 seconds
                # Flag suspiciously fast/slow laps
                if min_lap_time < 60:  # Less than 1 minute is suspicious
                    warnings.append(
                        self.create_validation_error(
                            field="LapTimeSeconds",
                            error_type="suspiciously_fast_lap",
                            message=(
                                self.fast_lap_validation_error_message(min_lap_time)
                            ),
                            severity=ValidationLevel.WARNING,
                            actual_value=min_lap_time,
                        )
                    )

                if max_lap_time > 300:  # Over 5 minutes is very slow
                    warnings.append(
                        self.create_validation_error(
                            field="LapTimeSeconds",
                            error_type="suspiciously_slow_lap",
                            message=(
                                self.slow_lap_validation_error_message(max_lap_time)
                            ),
                            severity=ValidationLevel.WARNING,
                            actual_value=max_lap_time,
                        )
                    )

                metadata["lap_time_range"] = (float(min_lap_time), float(max_lap_time))
                metadata["median_lap_time"] = float(lap_times.median())

            # Validate lap numbers are sequential per driver
            if all(col in data.columns for col in ["Driver", "LapNumber"]):
                for driver in data["Driver"].unique():
                    driver_laps = data[data["Driver"] == driver][
                        "LapNumber"
                    ].sort_values()

                    if len(driver_laps) > 1:
                        # Check for gaps in lap numbers
                        expected_range = range(
                            int(driver_laps.min()), int(driver_laps.max()) + 1
                        )
                        if len(driver_laps) != len(expected_range):
                            warnings.append(
                                self.create_validation_error(
                                    field="LapNumber",
                                    error_type="non_sequential_laps",
                                    message=(
                                        self.non_sequential_laps_validation_error_message(
                                            driver
                                        )
                                    ),
                                    severity=ValidationLevel.WARNING,
                                    actual_value=driver,
                                )
                            )

            # Validate drivers have reasonable number of laps
            if "Driver" in data.columns:
                laps_per_driver = data["Driver"].value_counts()
                metadata["drivers_count"] = len(laps_per_driver)
                metadata["laps_per_driver_stats"] = {
                    "mean": float(laps_per_driver.mean()),
                    "min": int(laps_per_driver.min()),
                    "max": int(laps_per_driver.max()),
                }

                # Flag drivers with very few laps (possible data quality issue)
                low_lap_drivers = laps_per_driver[laps_per_driver < 3]
                if len(low_lap_drivers) > 0:
                    warnings.append(
                        self.create_validation_error(
                            field="Driver",
                            error_type="low_lap_count",
                            message=(
                                self.low_lap_count_validation_error_message(
                                    low_lap_drivers
                                )
                            ),
                            severity=ValidationLevel.WARNING,
                            actual_value=list(low_lap_drivers.index),
                        )
                    )

            # Check for null values in critical columns
            critical_columns = ["Driver", "LapNumber"]
            for col in critical_columns:
                if col in data.columns:
                    null_count = data[col].isnull().sum()
                    if null_count > 0:
                        null_percentage = (null_count / len(data)) * 100

                        if null_percentage > 10:  # More than 10% nulls is concerning
                            errors.append(
                                self.create_validation_error(
                                    field=col,
                                    error_type="high_null_percentage",
                                    message=(
                                        self.high_null_percentage_validation_error_message(
                                            col, null_percentage
                                        )
                                    ),
                                    severity=ValidationLevel.STRICT,
                                    actual_value=null_percentage,
                                )
                            )
                        else:
                            warnings.append(
                                self.create_validation_error(
                                    field=col,
                                    error_type="null_values",
                                    message=(
                                        self.null_values_validation_error_message(
                                            col, null_count, null_percentage
                                        )
                                    ),
                                    severity=ValidationLevel.WARNING,
                                    actual_value=null_count,
                                )
                            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )

    def fast_lap_validation_error_message(self, min_lap_time):
        """Generate message for suspeciously fast lap validation error"""
        return f"Minimum lap time {min_lap_time:.2f}s is suspiciously fast (< 60s)"

    def slow_lap_validation_error_message(self, max_lap_time):
        """Generate message for suspeciously slow lap validation error"""
        return f"Maximum lap time {max_lap_time:.2f}s is suspiciously slow (> 300s)"

    def non_sequential_laps_validation_error_message(self, driver):
        """Generate message for non-sequential lap numbers validation error"""
        return f"Driver {driver} has non-sequential lap numbers"

    def low_lap_count_validation_error_message(self, low_lap_drivers):
        """Generate message for low lap count validation error"""
        return f"{len(low_lap_drivers)} drivers have < 3 laps: {list(low_lap_drivers.index)}"

    def high_null_percentage_validation_error_message(self, col, null_percentage):
        """Generate message for high null percentage validation error"""
        return f"Column '{col}' has {null_percentage:.1f}% null values"

    def null_values_validation_error_message(self, col, null_count, null_percentage):
        """Generate message for null values validation error"""
        return f"Column '{col}' has {null_count} null values ({null_percentage:.1f}%)"


class LapOutputContract(OutputContract):
    """Validate the output data for processed F1 laps"""

    def __init__(self):
        super().__init__(
            name="f1_laps_output",
            expected_columns=[
                "lap_id",
                "session_id",
                "driver_clean",
                "driver_number_clean",
                "team_clean",
                "lap_number",
                "lap_time_seconds",
                "position",
                "compound_clean",
                "tyre_life",
                "fresh_tyre",
                "stint",
                "track_status",
                "is_valid_lap",
                "sector1_seconds",
                "sector2_seconds",
                "sector3_seconds",
                "lap_time_delta_to_fastest",
                "lap_time_delta_to_personal_best",
                "lap_time_improvement",
                "is_fastest_lap",
                "is_personal_best",
                "track_status",
                "deleted",
                "deleted_reason_clean",
                "driver_cumulative_laps",
                "processed_at",
                "processor_version",
            ],
            column_constraints={
                "lap_time_seconds": {"min": 0, "max": 300},  # 5 minutes max
                "lap_number": {"min": 1, "not_null": True},
                "lap_id": {"not_null": True},
            },
            validation_level=ValidationLevel.WARNING,
        )

    def validate_schema(self, data: Any) -> ValidationResult:
        """Validate the data structure of the processed F1 laps data"""

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
                # Some columns are optional based on available data
                optional_if_missing = [
                    "sector1_seconds",
                    "sector2_seconds",
                    "sector3_seconds",
                    "compound_clean",
                    "tyre_life",
                    "team_clean",
                ]

                severity = (
                    ValidationLevel.WARNING
                    if col in optional_if_missing
                    else ValidationLevel.STRICT
                )

                errors.append(
                    self.create_validation_error(
                        field=col,
                        error_type="missing_output_column",
                        message=f"Expected output column '{col}' is missing",
                        severity=severity,
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
            is_valid=len([e for e in errors if e.severity == ValidationLevel.STRICT])
            == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )

    def validate_content(self, data: Any) -> ValidationResult:
        """Validate the content of the processed F1 laps data"""

        errors = []
        warnings = []
        metadata = {}

        if isinstance(data, pd.DataFrame):
            # Check column constraints
            for col, constraints in self.column_constraints.items():
                if col in data.columns:
                    # not_null constraint
                    if constraints.get("not_null", False):
                        null_count = data[col].isnull().sum()
                        if null_count > 0:
                            errors.append(
                                self.create_validation_error(
                                    field=col,
                                    error_type="null_values",
                                    message=f"Column '{col}' has {null_count} null values",
                                    severity=ValidationLevel.STRICT,
                                )
                            )

                    # min/max constraints
                    if "min" in constraints and pd.api.types.is_numeric_dtype(
                        data[col]
                    ):
                        min_val = data[col].min()
                        if pd.notna(min_val) and min_val < constraints["min"]:
                            errors.append(
                                self.create_validation_error(
                                    field=col,
                                    error_type="value_below_minimum",
                                    message=(
                                        self.below_min_validation_error_message(
                                            col, constraints, min_val
                                        )
                                    ),
                                    severity=ValidationLevel.STRICT,
                                    actual_value=min_val,
                                    expected_value=constraints["min"],
                                )
                            )

                    if "max" in constraints and pd.api.types.is_numeric_dtype(
                        data[col]
                    ):
                        max_val = data[col].max()
                        if pd.notna(max_val) and max_val > constraints["max"]:
                            warnings.append(
                                self.create_validation_error(
                                    field=col,
                                    error_type="value_above_maximum",
                                    message=(
                                        self.over_max_validation_error_message(
                                            col, constraints, max_val
                                        )
                                    ),
                                    severity=ValidationLevel.WARNING,
                                    actual_value=max_val,
                                    expected_value=constraints["max"],
                                )
                            )

            # Check for duplicate lap IDs
            if "lap_id" in data.columns:
                dup_count = data["lap_id"].duplicated().sum()
                if dup_count > 0:
                    errors.append(
                        self.create_validation_error(
                            field="lap_id",
                            error_type="duplicate_ids",
                            message=f"Found {dup_count} duplicate lap IDs",
                            severity=ValidationLevel.STRICT,
                        )
                    )

            # Quality metrics
            metadata["total_laps"] = len(data)

            if "driver_clean" in data.columns:
                metadata["unique_drivers"] = data["driver_clean"].nunique()

            if "is_valid_lap" in data.columns:
                valid_laps = data["is_valid_lap"].sum()
                metadata["valid_laps"] = int(valid_laps)
                metadata["invalid_laps"] = len(data) - int(valid_laps)
                metadata["valid_lap_percentage"] = float((valid_laps / len(data)) * 100)

            if "lap_time_seconds" in data.columns:
                lap_times = data["lap_time_seconds"].dropna()
                if len(lap_times) > 0:
                    metadata["lap_time_stats"] = {
                        "min": float(lap_times.min()),
                        "max": float(lap_times.max()),
                        "mean": float(lap_times.mean()),
                        "median": float(lap_times.median()),
                    }

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )

    def below_min_validation_error_message(self, col, constraints, min_val):
        """Generate message for the below min validation error"""
        return f"Column '{col}' minimum {min_val} < expected {constraints['min']}"

    def over_max_validation_error_message(self, col, constraints, max_val):
        """Generate message for the over max validation error"""
        return f"Column '{col}' maximum {max_val} > expected {constraints['max']}"


class LapProcessor(BaseProcessor):
    """
    F1 Lap Processor

    Processes F1 lap timing data to create clean, enriched lap records

    Input: DataFrame with lap timing data from session
    Output: Clean DataFrame with standardized lap data and calculated metrics

    Processing steps:
    1. Clean and standardize driver/team names
    2. Parse and validate lap times
    3. Calculate sector times if not present
    4. Compute lap time deltas (to fastest, to personal best)
    5. Identify valid/invalid laps
    6. Add position tracking
    7. Enrich with tire and strategy information
    8. Generate unique lap identifiers
    """

    def __init__(self):
        super().__init__("lap_processor", "1.0.0")

        # Set up validation contracts
        self.set_input_contract(LapInputContract())
        self.set_output_contract(LapOutputContract())

        # Tire compound standardization
        self.compound_mapping = {
            "SOFT": "Soft",
            "MEDIUM": "Medium",
            "HARD": "Hard",
            "INTERMEDIATE": "Intermediate",
            "WET": "Wet",
        }

    def _process_data(self, data: Any, context: ProcessingContext) -> pd.DataFrame:
        """
        Core lap processing logic

        Args:
            data: Lap DataFrame or dict with 'laps' key
            context: Processing context

        Returns:
            Clean DataFrame with enriched lap data
        """

        self.logger.info("Processing F1 lap timing data")

        # Extract laps DataFrame
        if isinstance(data, dict):
            laps_df = data.get("laps")
            if laps_df is None:
                raise ValueError("No laps data in input dictionary")
        else:
            laps_df = data

        # Make a copy to avoid modifying original
        laps_df = laps_df.copy()

        if "Driver" in laps_df.columns:
            self.logger.info(
                "Processing %d laps from %s drivers",
                len(laps_df),
                laps_df["Driver"].nunique(),
            )
        else:
            self.logger.info(
                "Processing %d laps from unknown number of drivers",
                len(laps_df),
            )

        # Step 1: Clean and standardize basic fields
        laps_df = self._clean_basic_fields(laps_df, context)

        # Step 2: Process lap times
        laps_df = self._process_lap_times(laps_df, context)

        # Step 3: Process sector times
        laps_df = self._process_sector_times(laps_df, context)

        # Step 4: Calculate deltas and rankings
        laps_df = self._calculate_deltas_and_rankings(laps_df, context)

        # Step 5: Process tire information
        laps_df = self._process_tyre_information(laps_df, context)

        # Step 6: Identify valid/invalid laps
        laps_df = self._identify_valid_laps(laps_df, context)

        # Step 7: Add derived fields and identifiers
        laps_df = self._add_derived_fields(laps_df, context)

        # Step 8: Final cleanup and column selection
        laps_df = self._final_cleanup(laps_df, context)

        self.logger.info("Successfully processed %d laps", len(laps_df))

        # Add processing metrics
        context.metadata.add_custom_metric("laps_processed", len(laps_df))
        context.metadata.add_custom_metric(
            "unique_drivers", laps_df["driver_clean"].nunique()
        )
        context.metadata.add_custom_metric("valid_laps", laps_df["is_valid_lap"].sum())

        return laps_df

    def _clean_basic_fields(
        self, laps_df: pd.DataFrame, context: ProcessingContext
    ) -> pd.DataFrame:
        """Clean and standardize basic fields"""

        self.logger.debug("Cleaning basic fields")

        # Clean driver names
        if "Driver" in laps_df.columns:
            laps_df["driver_clean"] = laps_df["Driver"]
        else:
            laps_df["driver_clean"] = "UNK"
            context.add_warning("Driver column missing, using 'UNK': 'Unknown'")

        # Clean driver numbers
        if "DriverNumber" in laps_df.columns:
            laps_df["driver_number_clean"] = pd.to_numeric(
                laps_df["DriverNumber"], errors="coerce"
            )
        else:
            laps_df["driver_number_clean"] = 0

        # Clean team names
        if "Team" in laps_df.columns:
            laps_df["team_clean"] = str(laps_df["Team"]).strip().title()
        else:
            laps_df["team_clean"] = "Unknown"

        # Standardize lap number
        if "LapNumber" in laps_df.columns:
            laps_df["lap_number"] = pd.to_numeric(laps_df["LapNumber"], errors="coerce")
        elif laps_df["driver_clean"].isnull().sum() == 0:
            laps_df["lap_number"] = laps_df.groupby("driver_clean").cumcount() + 1
            context.add_warning("LapNumber column missing, using sequential numbers")
        else:
            context.add_warning(
                "LapNumber column missing, and missing values in the Driver column. "
                "As a result cannot add LapNumbers using sequential numbers."
            )

        # Clean position data
        if "Position" in laps_df.columns:
            laps_df["position"] = pd.to_numeric(laps_df["Position"], errors="coerce")
        else:
            laps_df["position"] = None

        # Add track status
        if "TrackStatus" in laps_df.columns:
            laps_df["track_status"] = pd.to_numeric(
                laps_df["TrackStatus"], errors="coerce"
            )
        else:
            laps_df["track_status"] = 0

        # Lap deleted flag
        if "Deleted" in laps_df.columns:
            laps_df["deleted"] = laps_df["Deleted"]

        if "DeletedReason" in laps_df.columns:
            laps_df["deleted_reason_clean"] = laps_df["DeletedReason"]

        return laps_df

    def _process_lap_times(
        self, laps_df: pd.DataFrame, context: ProcessingContext
    ) -> pd.DataFrame:
        """Process and validate lap times"""

        self.logger.debug("Processing lap times")

        # Convert LapTime to seconds if not already done
        if "LapTimeSeconds" in laps_df.columns:
            laps_df["lap_time_seconds"] = pd.to_numeric(
                laps_df["LapTimeSeconds"], errors="coerce"
            )
        elif "LapTime" in laps_df.columns:
            # Handle pandas Timedelta objects
            def convert_to_seconds(lap_time):
                if pd.isna(lap_time):
                    return None
                try:
                    if isinstance(lap_time, pd.Timedelta):
                        return lap_time.total_seconds()
                    elif isinstance(lap_time, (int, float)):
                        return float(lap_time)
                    else:
                        # Try to parse as timedelta
                        return pd.to_timedelta(lap_time).total_seconds()
                except Exception as e:
                    context.add_error(
                        f"Error while converting lap time to seconds: {str(e)}"
                    )
                    return None

            laps_df["lap_time_seconds"] = laps_df["LapTime"].apply(convert_to_seconds)
        else:
            laps_df["lap_time_seconds"] = None
            context.add_error("No lap time data available")

        # Flag suspiciously fast/slow laps
        if "lap_time_seconds" in laps_df.columns:
            valid_times = laps_df["lap_time_seconds"].dropna()

            if len(valid_times) > 0:
                # Calculate reasonable bounds (median ± 50%)
                median_time = valid_times.median()
                lower_bound = median_time * 0.5
                upper_bound = median_time * 1.5

                suspicious_count = (
                    (laps_df["lap_time_seconds"] < lower_bound)
                    | (laps_df["lap_time_seconds"] > upper_bound)
                ).sum()

                if suspicious_count > 0:
                    context.add_warning(
                        f"{suspicious_count} laps have suspicious lap times"
                    )

        return laps_df

    def _process_sector_times(
        self, laps_df: pd.DataFrame, context: ProcessingContext
    ) -> pd.DataFrame:
        """Process sector times"""

        self.logger.debug("Processing sector times")

        def convert_sector_to_seconds(sector_time):
            if pd.isna(sector_time):
                return None
            try:
                if isinstance(sector_time, pd.Timedelta):
                    return sector_time.total_seconds()

                if isinstance(sector_time, (int, float)):
                    return float(sector_time)

                return pd.to_timedelta(sector_time).total_seconds()

            except Exception as e:
                context.add_error(
                    f"Error while trying to convert SectorTime to seconds: {str(e)}"
                )
                return None

        # Convert sector times to seconds
        for sector_num in [1, 2, 3]:
            sector_col = f"Sector{sector_num}Time"
            output_col = f"sector{sector_num}_seconds"

            if sector_col in laps_df.columns:
                laps_df[output_col] = laps_df[sector_col].apply(
                    convert_sector_to_seconds
                )
            else:
                laps_df[output_col] = None

        # Validate sector times sum to lap time (approximately)
        if all(
            col in laps_df.columns
            for col in [
                "sector1_seconds",
                "sector2_seconds",
                "sector3_seconds",
                "lap_time_seconds",
            ]
        ):
            laps_df["sector_sum"] = (
                laps_df["sector1_seconds"].fillna(0)
                + laps_df["sector2_seconds"].fillna(0)
                + laps_df["sector3_seconds"].fillna(0)
            )

            # Check where sector sum significantly differs from lap time
            sector_mismatch = (laps_df["sector_sum"] > 0) & (
                abs(laps_df["sector_sum"] - laps_df["lap_time_seconds"]) > 1.0
            )

            mismatch_count = sector_mismatch.sum()
            if mismatch_count > 0:
                context.add_warning(
                    f"{mismatch_count} laps have sector times that don't sum to lap time"
                )

            # Drop temporary column
            laps_df = laps_df.drop(columns=["sector_sum"])

        return laps_df

    def _calculate_deltas_and_rankings(
        self, laps_df: pd.DataFrame, context: ProcessingContext
    ) -> pd.DataFrame:
        """Calculate lap time deltas and identify fastest laps"""

        self.logger.debug("Calculating deltas and rankings")

        if "lap_time_seconds" in laps_df.columns:
            valid_laps = laps_df[laps_df["lap_time_seconds"].notna()]

            if len(valid_laps) > 0:
                # Overall fastest lap
                fastest_lap_time = valid_laps["lap_time_seconds"].min()
                laps_df["lap_time_delta_to_fastest"] = (
                    laps_df["lap_time_seconds"] - fastest_lap_time
                )
                laps_df["is_fastest_lap"] = (
                    laps_df["lap_time_seconds"] == fastest_lap_time
                )

                # Personal best per driver
                if "driver_clean" in laps_df.columns:
                    driver_best_times = valid_laps.groupby("driver_clean")[
                        "lap_time_seconds"
                    ].min()

                    laps_df["personal_best_time"] = laps_df["driver_clean"].map(
                        driver_best_times
                    )
                    laps_df["lap_time_delta_to_personal_best"] = (
                        laps_df["lap_time_seconds"] - laps_df["personal_best_time"]
                    )
                    laps_df["is_personal_best"] = (
                        laps_df["lap_time_seconds"] == laps_df["personal_best_time"]
                    )

                    # Drop temporary column
                    laps_df = laps_df.drop(columns=["personal_best_time"])
                else:
                    laps_df["lap_time_delta_to_personal_best"] = None
                    laps_df["is_personal_best"] = False

                context.metadata.add_custom_metric(
                    "fastest_lap_time", float(fastest_lap_time)
                )
            else:
                laps_df["lap_time_delta_to_fastest"] = None
                laps_df["lap_time_delta_to_personal_best"] = None
                laps_df["is_fastest_lap"] = False
                laps_df["is_personal_best"] = False
                context.add_warning("No valid lap times to calculate deltas")

        else:
            laps_df["lap_time_delta_to_fastest"] = None
            laps_df["lap_time_delta_to_personal_best"] = None
            laps_df["is_fastest_lap"] = False
            laps_df["is_personal_best"] = False

        return laps_df

    def _process_tyre_information(
        self, laps_df: pd.DataFrame, context: ProcessingContext
    ) -> pd.DataFrame:
        """Process tire compound and life information"""

        self.logger.debug("Processing tire information")

        # Clean compound names
        if "Compound" in laps_df.columns:
            laps_df["compound_clean"] = laps_df["Compound"].apply(
                lambda x: self.compound_mapping.get(str(x).upper(), str(x))
                if pd.notna(x)
                else None
            )
        else:
            laps_df["compound_clean"] = None

        # Tire life
        if "TyreLife" in laps_df.columns:
            laps_df["tyre_life"] = pd.to_numeric(laps_df["TyreLife"], errors="coerce")
        else:
            laps_df["tyre_life"] = None

        # Stint information
        if "Stint" in laps_df.columns:
            laps_df["stint"] = pd.to_numeric(laps_df["Stint"], errors="coerce")
        else:
            laps_df["stint"] = None

        # Check if tyre is fresh
        if "FreshTyre" in laps_df.columns:
            laps_df["fresh_tyre"] = laps_df["FreshTyre"]
        else:
            laps_df["fresh_tyre"] = None

        return laps_df

    def _identify_valid_laps(
        self, laps_df: pd.DataFrame, context: ProcessingContext
    ) -> pd.DataFrame:
        """Identify valid racing laps vs pit laps, outliers, etc."""

        self.logger.debug(
            "Identifying valid racing laps i.e. no in/out laps and outliers"
        )

        # A lap is considered valid if:
        # 1. Has a lap time
        # 2. Lap time is within reasonable bounds
        # 3. Not a pit lap (if we have that info)
        # 4. Not under safety car/yellow flags (if we have that info)

        laps_df["is_valid_lap"] = True  # Start with all valid

        # Invalidate laps with no lap time
        if "lap_time_seconds" in laps_df.columns:
            laps_df.loc[laps_df["lap_time_seconds"].isna(), "is_valid_lap"] = False

        # Invalidate pit laps
        if "PitInTime" in laps_df.columns or "PitOutTime" in laps_df.columns:
            # If either pit time is present, it's likely a pit lap
            has_pit_in = laps_df.get(
                "PitInTime", pd.Series([None] * len(laps_df))
            ).notna()
            has_pit_out = laps_df.get(
                "PitOutTime", pd.Series([None] * len(laps_df))
            ).notna()
            laps_df.loc[has_pit_in | has_pit_out, "is_valid_lap"] = False

        # Invalidate laps under yellow/safety car
        if "TrackStatus" in laps_df.columns:
            # Track status codes: 1=Green, 2=Yellow, 4=Safety Car, 6=VSC, 7=Red Flag
            flagged_laps = laps_df["TrackStatus"].isin([2, 4, 5, 6, 7])
            laps_df.loc[flagged_laps, "is_valid_lap"] = False

        # Invalidate suspiciously fast/slow laps (outliers)
        if "lap_time_seconds" in laps_df.columns:
            valid_times = laps_df[laps_df["lap_time_seconds"].notna()][
                "lap_time_seconds"
            ]

            if (
                len(valid_times) > 10
            ):  # Need enough data for statistical outlier detection
                # Use IQR method for outlier detection
                q1 = valid_times.quantile(0.25)
                q3 = valid_times.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr  # 3*IQR for very lenient outlier detection
                upper_bound = q3 + 3 * iqr

                outliers = (laps_df["lap_time_seconds"] < lower_bound) | (
                    laps_df["lap_time_seconds"] > upper_bound
                )

                laps_df.loc[outliers, "is_valid_lap"] = False

                outlier_count = outliers.sum()
                if outlier_count > 0:
                    context.metadata.add_custom_metric(
                        "Outlier laps marked", int(outlier_count)
                    )

        valid_count = laps_df["is_valid_lap"].sum()
        invalid_count = len(laps_df) - valid_count

        self.logger.info("Valid laps: %d, Invalid laps: %d", valid_count, invalid_count)

        return laps_df

    def _add_derived_fields(
        self, laps_df: pd.DataFrame, context: ProcessingContext
    ) -> pd.DataFrame:
        """Add derived fields and unique identifiers"""

        self.logger.debug("Adding derived fields")

        # Create session_id from context
        session_id = context.get_session_identifier()
        laps_df["session_id"] = session_id

        # Create unique lap_id: session_id + driver + lap_number
        if all(col in laps_df.columns for col in ["driver_clean", "lap_number"]):
            laps_df["lap_id"] = (
                laps_df["session_id"]
                + "_"
                + laps_df["driver_clean"].str.replace(" ", "_")
                + "_"
                + laps_df["lap_number"].astype(str)
            )
        else:
            # Fallback to simple sequential IDs
            laps_df["lap_id"] = [f"{session_id}_lap_{i}" for i in range(len(laps_df))]
            context.add_warning("Could not create proper lap_ids, using sequential")

        # Add processing timestamp
        laps_df["processed_at"] = pd.Timestamp.now()

        # Add processor version
        laps_df["processor_version"] = self.version

        # Calculate lap time improvement from previous lap (per driver)
        if all(
            col in laps_df.columns
            for col in ["driver_clean", "lap_number", "lap_time_seconds"]
        ):
            laps_df = laps_df.sort_values(["driver_clean", "lap_number"])
            laps_df["lap_time_improvement"] = (
                laps_df.groupby("driver_clean")["lap_time_seconds"].diff() * -1
            )  # Negative diff means improvement
        else:
            laps_df["lap_time_improvement"] = None

        # Calculate cumulative laps per driver
        if "driver_clean" in laps_df.columns:
            laps_df = laps_df.sort_values(["driver_clean", "lap_number"])
            laps_df["driver_cumulative_laps"] = (
                laps_df.groupby("driver_clean").cumcount() + 1
            )
        else:
            laps_df["driver_cumulative_laps"] = None

        return laps_df

    def _final_cleanup(
        self, laps_df: pd.DataFrame, context: ProcessingContext
    ) -> pd.DataFrame:
        """Final cleanup and column selection"""

        self.logger.debug("Performing final cleanup")

        # Define output columns in desired order
        core_columns = [
            "lap_id",
            "session_id",
            "driver_clean",
            "driver_number_clean",
            "team_clean",
            "lap_number",
            "lap_time_seconds",
            "position",
        ]

        sector_columns = [
            "sector1_seconds",
            "sector2_seconds",
            "sector3_seconds",
        ]

        tire_columns = [
            "compound_clean",
            "tyre_life",
            "stint",
            "fresh_tyre",
        ]

        performance_columns = [
            "lap_time_delta_to_fastest",
            "lap_time_delta_to_personal_best",
            "lap_time_improvement",
            "is_fastest_lap",
            "is_personal_best",
            "is_valid_lap",
        ]

        meta_columns = [
            "track_status",
            "deleted",
            "deleted_reason_clean",
            "driver_cumulative_laps",
            "processed_at",
            "processor_version",
        ]

        all_output_columns = (
            core_columns
            + sector_columns
            + tire_columns
            + performance_columns
            + meta_columns
        )

        # Select only columns that exist
        existing_columns = [col for col in all_output_columns if col in laps_df.columns]
        laps_df = laps_df[existing_columns].copy()

        # Ensure proper data types
        numeric_columns = [
            "driver_number_clean",
            "lap_number",
            "lap_time_seconds",
            "position",
            "sector1_seconds",
            "sector2_seconds",
            "sector3_seconds",
            "tyre_life",
            "stint",
            "lap_time_delta_to_fastest",
            "lap_time_delta_to_personal_best",
            "lap_time_improvement",
            "driver_cumulative_laps",
            "track_status",
        ]

        for col in numeric_columns:
            if col in laps_df.columns:
                laps_df[col] = pd.to_numeric(laps_df[col], errors="coerce")

        # Ensure boolean columns
        boolean_columns = [
            "fresh_tyre",
            "is_fastest_lap",
            "is_personal_best",
            "is_valid_lap",
            "deleted",
        ]
        for col in boolean_columns:
            if col in laps_df.columns:
                laps_df[col] = laps_df[col].astype(bool)

        # Sort by driver and lap number for consistent output
        if all(col in laps_df.columns for col in ["driver_clean", "lap_number"]):
            laps_df = laps_df.sort_values(["driver_clean", "lap_number"]).reset_index(
                drop=True
            )

        return laps_df

    def get_required_config_keys(self) -> List[str]:
        """Required configuration keys"""
        return []

    def get_default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "remove_outliers": True,
            "invalidate_pit_laps": True,
            "invalidate_flagged_laps": True,
            "calculate_deltas": True,
            "iqr_multiplier": 3.0,  # For outlier detection
        }
