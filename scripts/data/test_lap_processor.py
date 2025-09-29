"""
Test script for LapProcessor
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.logging import setup_logging, get_logger
from src.data_processing.core.lap_processor import LapProcessor
from src.data_processing.base.processing_context import ProcessingContext
import pandas as pd
import json


def test_lap_processor_with_sample_data():
    """Test LapProcessor with synthetic sample data"""

    setup_logging()
    logger = get_logger("test_lap_processor")

    logger.info("=== Testing LapProcessor with Sample Data ===")

    # Create sample lap data (like what your ingestion produces)
    sample_laps = pd.DataFrame(
        {
            "Driver": ["VER", "VER", "VER", "HAM", "HAM", "HAM", "LEC", "LEC", "LEC"],
            "DriverNumber": [1, 1, 1, 44, 44, 44, 16, 16, 16],
            "Team": [
                "Red Bull Racing",
                "Red Bull Racing",
                "Red Bull Racing",
                "Mercedes",
                "Mercedes",
                "Mercedes",
                "Ferrari",
                "Ferrari",
                "Ferrari",
            ],
            "LapNumber": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "LapTime": [
                pd.Timedelta(seconds=78.5),
                pd.Timedelta(seconds=77.2),
                pd.Timedelta(seconds=76.8),
                pd.Timedelta(seconds=79.1),
                pd.Timedelta(seconds=78.3),
                pd.Timedelta(seconds=77.9),
                pd.Timedelta(seconds=78.8),
                pd.Timedelta(seconds=77.8),
                pd.Timedelta(seconds=77.5),
            ],
            "LapTimeSeconds": [78.5, 77.2, 76.8, 79.1, 78.3, 77.9, 78.8, 77.8, 77.5],
            "Position": [1, 1, 1, 3, 2, 2, 2, 3, 3],
            "Compound": [
                "SOFT",
                "SOFT",
                "SOFT",
                "MEDIUM",
                "MEDIUM",
                "MEDIUM",
                "SOFT",
                "SOFT",
                "SOFT",
            ],
            "TyreLife": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "Stint": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            "FreshTyre": [True, False, False, True, False, False, True, False, False],
            "Sector1Time": [
                pd.Timedelta(seconds=25.1),
                pd.Timedelta(seconds=24.8),
                pd.Timedelta(seconds=24.6),
                pd.Timedelta(seconds=25.4),
                pd.Timedelta(seconds=25.1),
                pd.Timedelta(seconds=25.0),
                pd.Timedelta(seconds=25.2),
                pd.Timedelta(seconds=24.9),
                pd.Timedelta(seconds=24.7),
            ],
            "Sector2Time": [
                pd.Timedelta(seconds=28.2),
                pd.Timedelta(seconds=27.8),
                pd.Timedelta(seconds=27.5),
                pd.Timedelta(seconds=28.5),
                pd.Timedelta(seconds=28.1),
                pd.Timedelta(seconds=27.9),
                pd.Timedelta(seconds=28.3),
                pd.Timedelta(seconds=27.9),
                pd.Timedelta(seconds=27.7),
            ],
            "Sector3Time": [
                pd.Timedelta(seconds=25.2),
                pd.Timedelta(seconds=24.6),
                pd.Timedelta(seconds=24.7),
                pd.Timedelta(seconds=25.2),
                pd.Timedelta(seconds=25.1),
                pd.Timedelta(seconds=25.0),
                pd.Timedelta(seconds=25.3),
                pd.Timedelta(seconds=25.0),
                pd.Timedelta(seconds=25.1),
            ],
            "EventName": ["Monaco"] * 9,
            "SessionName": ["Qualifying"] * 9,
            "SessionDate": ["2023-05-27"] * 9,
            "Deleted": [False] * 9,
            "DeletedReason": [""] * 9,
        }
    )

    # Wrap in session data structure
    session_data = {
        "session_info": {},
        "laps": sample_laps,
        "results": None,
        "weather": None,
        "telemetry": {},
    }

    try:
        # Create processor
        processor = LapProcessor()

        # Create context
        context = ProcessingContext(year=2023, event_name="Monaco", session_type="Q")

        # Process data
        logger.info("Processing sample lap data...")
        result_df, updated_context = processor.process(session_data, context)

        # Display results
        logger.info("✅ Processing completed successfully!")
        logger.info(f"Output shape: {result_df.shape}")
        logger.info(f"Output columns: {list(result_df.columns)}")

        logger.info("\nSample output (first 5 laps):")
        display_columns = [
            "lap_id",
            "driver_clean",
            "lap_number",
            "lap_time_seconds",
            "lap_time_delta_to_fastest",
            "is_fastest_lap",
            "is_valid_lap",
        ]
        print(result_df[display_columns].head())

        logger.info("\nLap statistics:")
        print(f"Total laps: {len(result_df)}")
        print(f"Unique drivers: {result_df['driver_clean'].nunique()}")
        print(f"Valid laps: {result_df['is_valid_lap'].sum()}")
        print(f"Fastest lap time: {result_df['lap_time_seconds'].min():.3f}s")
        print(f"Slowest lap time: {result_df['lap_time_seconds'].max():.3f}s")

        # Display processing stats
        stats = processor.get_processing_stats()
        logger.info(f"\nProcessing stats: {stats}")

        # Display validation results
        logger.info("\nValidation summary:")
        logger.info(f"Errors: {updated_context.metadata.errors_count}")
        logger.info(f"Warnings: {updated_context.metadata.warnings_count}")

        # if updated_context.has_warnings():
        #     logger.info("Warnings:")
        #     for warning in updated_context.warnings[:5]:  # Show first 5
        #         logger.info(f"  - {warning}")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_lap_processor_with_real_data():
    """Test LapProcessor with real ingested data"""

    setup_logging()
    logger = get_logger("test_lap_processor_real")

    logger.info("=== Testing LapProcessor with Real Data ===")

    try:
        # Try to load real lap data
        sample_file = Path(
            project_root / "data/raw/2023/Monaco Grand Prix/Q/laps.parquet"
        )

        if not sample_file.exists():
            logger.warning(f"Real data file not found: {sample_file}")
            logger.info("Skipping real data test")
            return True

        # Load real laps
        real_laps = pd.read_parquet(sample_file)
        logger.info(f"Loaded {len(real_laps)} real laps from {sample_file}")

        # Create data structure
        real_session_data = {
            "session_info": {},
            "laps": real_laps,
            "results": None,
            "weather": None,
            "telemetry": {},
        }

        # Process with LapProcessor
        processor = LapProcessor()
        context = ProcessingContext(year=2023, event_name="Monaco", session_type="Q")

        logger.info("Processing real lap data...")
        result_df, updated_context = processor.process(real_session_data, context)

        logger.info("✅ Real data processing completed!")
        logger.info(f"Output shape: {result_df.shape}")

        # Show some interesting statistics
        logger.info("\nReal data statistics:")
        print(f"Total laps processed: {len(result_df)}")
        print(f"Unique drivers: {result_df['driver_clean'].nunique()}")
        print(f"Valid laps: {result_df['is_valid_lap'].sum()}")
        print(f"Invalid laps: {(~result_df['is_valid_lap']).sum()}")

        if "lap_time_seconds" in result_df.columns:
            valid_laps = result_df[result_df["is_valid_lap"]]
            if len(valid_laps) > 0:
                print(f"\nLap time statistics (valid laps only):")
                print(f"Fastest: {valid_laps['lap_time_seconds'].min():.3f}s")
                print(f"Slowest: {valid_laps['lap_time_seconds'].max():.3f}s")
                print(f"Mean: {valid_laps['lap_time_seconds'].mean():.3f}s")
                print(f"Median: {valid_laps['lap_time_seconds'].median():.3f}s")

        # Show fastest lap holder
        if "is_fastest_lap" in result_df.columns:
            fastest = result_df[result_df["is_fastest_lap"]]
            if len(fastest) > 0:
                print(f"\nFastest lap:")
                print(f"Driver: {fastest.iloc[0]['driver_clean']}")
                print(f"Time: {fastest.iloc[0]['lap_time_seconds']:.3f}s")
                print(f"Lap Number: {fastest.iloc[0]['lap_number']}")

        # Sample of processed data
        logger.info("\nSample of processed laps:")
        display_columns = [
            "driver_clean",
            "lap_number",
            "lap_time_seconds",
            "compound_clean",
            "is_valid_lap",
            "is_fastest_lap",
        ]
        existing_display_cols = [
            col for col in display_columns if col in result_df.columns
        ]
        print(result_df[existing_display_cols].head(10).to_string())

        return True

    except Exception as e:
        logger.error(f"❌ Real data test failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests"""

    print("=" * 80)
    print("LAP PROCESSOR TESTS")
    print("=" * 80)

    test1_passed = test_lap_processor_with_sample_data()
    print("\n" + "=" * 80 + "\n")

    test2_passed = test_lap_processor_with_real_data()
    print("\n" + "=" * 80)

    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! LapProcessor is ready to use.")
        print("\nNext steps:")
        print("1. Process your full dataset with the LapProcessor")
        print("2. Save processed laps to data/processed/laps/")
        print("3. Build additional processors (ResultsProcessor, etc.)")
    else:
        print("\n❌ Some tests failed. Check the logs for details.")


if __name__ == "__main__":
    main()
