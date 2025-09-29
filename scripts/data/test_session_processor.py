"""
Test script for SessionProcessor
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.logging import setup_logging, get_logger
from src.data_processing.core.session_processor import SessionProcessor
from src.data_processing.base.processing_context import ProcessingContext
import json


def test_session_processor():
    """Test SessionProcessor with sample data"""

    setup_logging()
    logger = get_logger("test_session_processor")

    logger.info("=== Testing SessionProcessor ===")

    # Create sample session data (like what your ingestion produces)
    sample_session_data = {
        "event_name": "Monaco Grand Prix",
        "location": "Monte Carlo",
        "country": "Monaco",
        "session_name": "Qualifying",
        "session_date": "2023-05-27",
        "official_event_name": "FORMULA 1 GRAND PRIX DE MONACO 2023",
        "event_format": "conventional",
        "round_number": 6,
    }

    try:
        # Create processor
        processor = SessionProcessor()

        # Create context
        context = ProcessingContext(year=2023, event_name="Monaco", session_type="Q")

        # Process data
        logger.info("Processing sample session data...")
        result_df, updated_context = processor.process(sample_session_data, context)

        # Display results
        logger.info("✅ Processing completed successfully!")
        logger.info(f"Output shape: {result_df.shape}")
        logger.info(f"Output columns: {list(result_df.columns)}")
        logger.info("Sample output:")
        print(result_df.to_string())

        # Display processing stats
        stats = processor.get_processing_stats()
        logger.info(f"Processing stats: {stats}")

        # Display context metadata
        logger.info("Context metadata:")
        context_dict = updated_context.to_dict()
        print(json.dumps(context_dict, indent=2, default=str))

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_with_real_data():
    """Test with real ingested data"""

    setup_logging()
    logger = get_logger("test_session_processor_real")

    logger.info("=== Testing SessionProcessor with Real Data ===")

    # Try to load real session data
    try:
        sample_file = Path("data/raw/2023/Monaco/Q/session_info.json")

        if not sample_file.exists():
            logger.warning(f"Real data file not found: {sample_file}")
            logger.info("Skipping real data test")
            return True

        # Load real session info
        with open(sample_file, "r") as f:
            real_session_info = json.load(f)

        # Create data structure like your ingestion produces
        real_session_data = {
            "session_info": real_session_info,
            "laps": None,
            "results": None,
            "weather": None,
            "telemetry": {},
        }

        # Process with SessionProcessor
        processor = SessionProcessor()
        context = ProcessingContext(year=2023, event_name="Monaco", session_type="Q")

        result_df, updated_context = processor.process(real_session_data, context)

        logger.info("✅ Real data processing completed!")
        logger.info(f"Output shape: {result_df.shape}")
        print(result_df.to_string())

        return True

    except Exception as e:
        logger.error(f"❌ Real data test failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test1_passed = test_session_processor()
    print("Test 1 passed successfully.")
