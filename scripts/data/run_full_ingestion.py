"""
Production script to run complete F1 data ingestion
"""

import sys
from pathlib import Path
import argparse
from typing import List  # noqa: E402

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.logging import setup_logging, get_logger  # noqa: E402
from src.pipelines.data_pipeline import PipelineConfig, DataIngestionPipeline  # noqa: E402


def create_config_from_args(args) -> PipelineConfig:
    """Create pipeline configuration from command line arguments"""

    # Parse seasons
    if args.seasons:
        if "-" in args.seasons:
            # Range format: "2020-2024"
            start, end = map(int, args.seasons.split("-"))
            seasons = list(range(start, end + 1))
        else:
            # Comma-separated format: "2020,2021,2022"
            seasons = [int(s.strip()) for s in args.seasons.split(",")]
    else:
        seasons = list(range(2022, 2025))  # Default: 2022-2025

    # Parse session types
    if args.sessions:
        session_types = [s.strip() for s in args.sessions.split(",")]
    else:
        session_types = ["FP1", "FP2", "FP3", "Q", "R"]  # Default: all main sessions

    config = PipelineConfig(
        seasons=seasons,
        session_types=session_types,
        parallel_seasons=args.parallel_seasons,
        parallel_events=args.parallel_events,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        validate_data=args.validate,
        min_events_per_season=args.min_events,
        resume_on_failure=args.resume,
        skip_existing=args.skip_existing,
        progress_interval=args.progress_interval,
    )

    return config


def main():
    """Main ingestion script with configurable options"""

    parser = argparse.ArgumentParser(
        description="Run F1 data ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Ingest recent seasons with default settings
            python scripts/data/run_full_ingestion.py --seasons 2022-2024
            
            # Ingest specific seasons with only qualifying and race data
            python scripts/data/run_full_ingestion.py --seasons 2023,2024 --sessions Q,R
            
            # Conservative ingestion (sequential, more retries)
            python scripts/data/run_full_ingestion.py --preset conservative
            
            # High-performance ingestion (parallel, all sessions)
            python scripts/data/run_full_ingestion.py --preset performance
            
            # Resume failed ingestion with existing data
            python scripts/data/run_full_ingestion.py --seasons 2020-2024 --resume --skip-existing
        """,
    )

    # Preset configurations
    parser.add_argument(
        "--preset",
        choices=["default", "conservative", "performance"],
        help="Use predefined configuration preset",
    )

    # Season selection
    parser.add_argument(
        "--seasons",
        type=str,
        help='Seasons to ingest. Format: "2020-2024" or "2020,2021,2023"',
    )

    # Session types
    parser.add_argument(
        "--sessions",
        type=str,
        help='Session types to ingest. Format: "FP1,FP2,Q,R" (default: all main sessions)',
    )

    # Performance settings
    parser.add_argument(
        "--parallel-seasons",
        action="store_true",
        help="Process seasons in parallel (higher API load)",
    )

    parser.add_argument(
        "--parallel-events",
        action="store_true",
        default=True,
        help="Process events in parallel within each season (default: True)",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Maximum number of parallel workers (default: 3)",
    )

    # Reliability settings
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum retries for failed operations (default: 2)",
    )

    parser.add_argument(
        "--retry-delay",
        type=float,
        default=5.0,
        help="Delay between retries in seconds (default: 5.0)",
    )

    # Data quality
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate ingested data (default: True)",
    )

    parser.add_argument(
        "--min-events",
        type=int,
        default=15,
        help="Minimum expected events per season (default: 15)",
    )

    # Resume and skip options
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Continue processing on failures (default: True)",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip events that already have data (default: True)",
    )

    # Progress reporting
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=5,
        help="Report progress every N events (default: 5)",
    )

    # Execution control
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually ingesting data",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = get_logger("ingestion_script")

    logger.info("=== F1 Data Ingestion Pipeline ===")

    try:
        # Create configuration
        if args.preset:
            if args.preset == "conservative":
                config = PipelineConfig.conservative_config()
            elif args.preset == "performance":
                config = PipelineConfig.performance_config()
            else:  # default
                config = PipelineConfig.default_config()

            logger.info("Using preset configuration: %s", args.preset)
        else:
            config = create_config_from_args(args)
            logger.info("Using custom configuration from arguments")

        # Override specific settings from command line
        if args.seasons:
            if "-" in args.seasons:
                start, end = map(int, args.seasons.split("-"))
                config.seasons = list(range(start, end + 1))
            else:
                config.seasons = [int(s.strip()) for s in args.seasons.split(",")]

        if args.sessions:
            config.session_types = [s.strip() for s in args.sessions.split(",")]

        # Log configuration
        logger.info("Configuration:")
        logger.info("  Seasons: %s", config.seasons)
        logger.info("  Session Types: %s", config.session_types)
        logger.info(
            "  Parallel Processing: seasons=%d, events=%d",
            config.parallel_seasons,
            config.parallel_events,
        )
        logger.info("  Max Workers: %s", config.max_workers)
        logger.info("  Resume on Failure: %s", config.resume_on_failure)
        logger.info("  Skip Existing: %s", config.skip_existing)

        # Estimate scope
        total_seasons = len(config.seasons)
        estimated_events = total_seasons * 20  # ~20 events per season
        estimated_sessions = estimated_events * len(config.session_types)

        logger.info(
            "Estimated scope: %d seasons, ~%d events, ~%d sessions",
            total_seasons,
            estimated_events,
            estimated_sessions,
        )

        # Dry run option
        if args.dry_run:
            logger.info("DRY RUN - No data will be ingested")

            # Show what would be processed
            pipeline = DataIngestionPipeline(config)

            for year in config.seasons:
                try:
                    events = (
                        pipeline.data_fetcher.schedule_loader.get_events_for_ingestion(
                            year
                        )
                    )
                    logger.info(
                        "  %d: %d events - %s...", year, len(events), events[:3]
                    )
                except Exception as e:
                    logger.warning("  %d: Could not load event list - %s", year, str(e))

            logger.info("Dry run completed")
            return

        # Confirm with user for large ingestions
        if total_seasons > 3 or estimated_sessions > 300:
            logger.warning(
                "⚠️ Large ingestion scope: %d sessions estimated", estimated_sessions
            )
            logger.warning("This may take several hours and will make many API calls")

            response = input("Continue? [y/N]: ")
            if response.lower() != "y":
                logger.info("Ingestion cancelled by user")
                return

        # Run pipeline
        pipeline = DataIngestionPipeline(config)
        results = pipeline.run_full_ingestion()

        # Final summary
        summary = results.get("summary", {})
        logger.info("🎉 Pipeline completed!")
        logger.info(
            "Summary: %d/%d seasons successful",
            summary.get("successful_seasons", 0),
            summary.get("total_seasons", 0),
        )
        logger.info(
            "Sessions: %d successful, %d failed",
            summary.get("total_sessions_processed", 0),
            summary.get("total_sessions_failed", 0),
        )

        # Check for failures
        if (
            summary.get("failed_seasons", 0) > 0
            or summary.get("total_sessions_failed", 0) > 0
        ):
            logger.warning(
                "Some ingestion operations failed. Check logs and pipeline results for details."
            )
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Ingestion cancelled by user (Ctrl+C)")
        sys.exit(1)

    except Exception as e:
        logger.error("Pipeline failed: %s", str(e))
        logger.error("Check logs for detailed error information")
        sys.exit(1)


if __name__ == "__main__":
    main()
