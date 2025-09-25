"""
Unit test for ScheduleLoader
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.logging import setup_logging  # noqa: E402
from src.data_ingestion.fastf1_client import FastF1Client  # noqa: E402
from src.data_ingestion.schedule_loader import ScheduleLoader  # noqa: E402

setup_logging()


def test_schedule_loader():
    """Function to test the ScheduleLoader"""

    client = FastF1Client()

    schedule_loader = ScheduleLoader(client=client)

    schedule_2022 = schedule_loader.load_season_schedule(2022, save_to_file=True)

    assert schedule_2022.shape == (22, 24)

    reloaded_schedule_2022 = schedule_loader.load_season_schedule(
        2022, save_to_file=False
    )

    assert reloaded_schedule_2022.shape == (22, 24)

    events_for_ingestion = schedule_loader.get_events_for_ingestion(2022)

    assert len(events_for_ingestion) == 22

    raw_data_status = schedule_loader.get_raw_data_status(2022)

    assert raw_data_status["file_exists"]
