"""
Unit test for session loader
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.logging import setup_logging
from src.data_ingestion.fastf1_client import FastF1Client
from src.data_ingestion.session_loader import SessionLoader

setup_logging()

client = FastF1Client()

session_loader = SessionLoader(client=client)


def test_session_load_light():
    """Only loading session summary for 2022 Monaco Qualifying"""

    session_summary = session_loader.get_session_summary(2022, "Monaco", "Q")

    assert len(session_summary.keys()) == 9
    assert session_summary["event_name"] == "Monaco Grand Prix"
    assert session_summary["lap_count"] == 404
    assert session_summary["driver_count"] == 20


def test_single_session_load():
    """Loading the entire data for a session"""

    session_data = session_loader.load_session_data(2022, "Monaco", "Q")

    assert len(session_data.keys()) == 5


def test_multiple_sessions_load():
    """Loading data for multiple session"""
    data = session_loader.load_multiple_sessions(2022, ["Monaco", "Monza"], ["R"])

    assert len(data.keys()) == 2
