"""
Script to resume failed or incomplete data ingestion
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.logging import get_logger, setup_logging  # noqa: E402
from config.settings import data_config  # noqa: E402
from src.pipelines.data_pipeline import DataIngestionPipeline, PipelineConfig  # noqa: E402


def find_failed_runs() -> List[Path]:
    """Find pipeline runs that failed or are incomplete"""

    pipeline_runs_dir = data_config.raw_data_path / "pipeline_runs"

    if not pipeline_runs_dir.exists():
        return []

    failed_runs = []

    for run_dir in pipeline_runs_dir.iterdir():
        if run_dir.is_dir() and run_dir.name.startswith("run_"):
            results_file = run_dir / "pipeline_results.json"

            if results_file.exists():
                try:
                    with open(results_file, "r", encoding="utf-8") as f:
                        results = json.load(f)

                    # Check if run failed or had significant failures
                    if (
                        results.get("pipeline_status") == "failed"
                        or results.get("summary", {}).get("failed_seasons", 0) > 0
                    ):
                        failed_runs.append(run_dir)

                except Exception:
                    # Assume failed if we can't read results
                    failed_runs.append(run_dir)

    return sorted(failed_runs, key=lambda x: x.name, reverse=True)


def analyze_failed_run(run_dir: Path) -> Dict[str, Any]:
    """Analyze a failed run to determine what needs to be resumed"""

    analysis = {
        "run_dir": run_dir,
        "config_found": False,
        "results_found": False,
        "failed_seasons": [],
        "incomplete_seasons": [],
        "total_failures": 0,
    }

    # Load configuration
    config_file = run_dir / "pipeline_config.json"
    if config_file.exists():
        analysis["config_found"] = True
        with open(config_file, "r", encoding="utf-8") as f:
            analysis["original_config"] = json.load(f)

    # Load results
    results_file = run_dir / "pipeline_results.json"
    if results_file.exists():
        analysis["results_found"] = True
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
            analysis["results"] = results

            # Analyze failures
            for year, season_data in results.get("seasons", {}).items():
                if season_data.get("status") == "failed":
                    analysis["failed_seasons"].append(int(year))
                    analysis["total_failures"] += season_data.get("sessions_failed", 0)
                elif season_data.get("sessions_failed", 0) > 0:
                    analysis["incomplete_seasons"].append(int(year))
                    analysis["total_failures"] += season_data.get("sessions_failed", 0)

    return analysis


def create_resume_config(
    analysis: Dict[str, Any], resume_all: bool = False
) -> PipelineConfig:
    """Create configuration to resume failed ingestion"""

    if not analysis["config_found"]:
        raise ValueError("Cannot resume: original configuration not found")

    original_config = analysis["original_config"]

    # Determine which seasons to retry
    seasons_to_retry = []

    if resume_all:
        # Resume all seasons from original config
        seasons_to_retry = original_config["seasons"]
    else:
        # Only retry failed and incomplete seasons
        seasons_to_retry = analysis["failed_seasons"] + analysis["incomplete_seasons"]

    if not seasons_to_retry:
        raise ValueError("No seasons to resume")

    # Create resume configuration based on original
    config = PipelineConfig(
        seasons=sorted(seasons_to_retry),
        session_types=original_config["session_types"],
        parallel_seasons=original_config.get("parallel_seasons", False),
        parallel_events=original_config.get("parallel_events", True),
        max_workers=original_config.get("max_workers", 3),
        max_retries=original_config.get("max_retries", 2)
        + 1,  # More retries for resume
        retry_delay=original_config.get("retry_delay", 5.0),
        validate_data=original_config.get("validate_data", True),
        min_events_per_season=original_config.get("min_events_per_season", 15),
        resume_on_failure=True,  # Always true for resume
        skip_existing=True,  # Always true for resume
        progress_interval=original_config.get("progress_interval", 5),
    )

    return config


def main():
    """Main resume script"""

    parser = argparse.ArgumentParser(description="Resume failed F1 data ingestion")
    parser.add_argument(
        "--run-id", help="Specific run ID to resume (e.g., run_20241201_143000)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Resume all seasons from failed run (not just failed ones)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available failed runs and exit"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = get_logger("resume_ingestion")

    logger.info("=== Resume Failed F1 Data Ingestion ===")

    try:
        # Find failed runs
        failed_runs = find_failed_runs()

        if not failed_runs:
            logger.info("No failed pipeline runs found")
            return

        # List mode
        if args.list:
            logger.info("Found %d failed runs:", len(failed_runs))
            for run_dir in failed_runs:
                analysis = analyze_failed_run(run_dir)
                logger.info(
                    "  %s: %d failed seasons, %d incomplete seasons, %d total session failures",
                    run_dir.name,
                    len(analysis["failed_seasons"]),
                    len(analysis["incomplete_seasons"]),
                    analysis["total_failures"],
                )
            return

        # Select run to resume
        if args.run_id:
            run_dir = data_config.raw_data_path / "pipeline_runs" / args.run_id
            if not run_dir.exists():
                logger.error("Run directory not found: %s", run_dir)
                sys.exit(1)
        else:
            # Use most recent failed run
            run_dir = failed_runs[0]
            logger.info("Using most recent failed run: %s", run_dir.name)

        # Analyze the failed run
        logger.info("Analyzing failed run...")
        analysis = analyze_failed_run(run_dir)

        if not analysis["config_found"]:
            logger.error("Cannot resume: original configuration not found")
            sys.exit(1)

        logger.info("Analysis:")
        logger.info("  Failed seasons: %s", analysis["failed_seasons"])
        logger.info("  Incomplete seasons: %s", analysis["incomplete_seasons"])
        logger.info("  Total session failures: %s", analysis["total_failures"])

        # Create resume configuration
        resume_config = create_resume_config(analysis, resume_all=args.all)

        logger.info("Resume configuration:")
        logger.info("  Seasons to retry: %s", resume_config.seasons)
        logger.info("  Session types: %s", resume_config.session_types)
        logger.info("  Skip existing: %s", resume_config.skip_existing)
        logger.info("  Max retries: %s", resume_config.max_retries)

        # Confirm with user
        response = input("Continue with resume? [y/N]: ")
        if response.lower() != "y":
            logger.info("Resume cancelled by user")
            return

        # Run resume pipeline
        logger.info("Starting resume ingestion...")
        pipeline = DataIngestionPipeline(resume_config)
        results = pipeline.run_full_ingestion()

        # Final summary
        summary = results.get("summary", {})
        logger.info("🎉 Resume completed!")
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

    except Exception as e:
        logger.error("Resume failed: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
