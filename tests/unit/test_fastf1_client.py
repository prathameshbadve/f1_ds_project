import pandas as pd

from src.data_ingestion.fastf1_client import FastF1Client
from config.settings import fastf1_config


def test_ingest_single_session():
    """Test to check that F1 session is loaded correctly with the environment variables."""

    assert fastf1_config.cache_enabled
    assert not fastf1_config.force_renew
    assert fastf1_config.log_level == "INFO"
    assert fastf1_config.default_season == 2022
    assert fastf1_config.session_types == ["FP1", "FP2", "FP3", "Q", "R", "SQ", "S"]

    client = FastF1Client()

    year = fastf1_config.default_season
    session = client.get_session(year, "Monaco", "Q")

    assert session.event.EventName == "Monaco Grand Prix"
    assert session.name == "Qualifying"
    assert session.date == pd.Timestamp("2022-05-28 14:00:00")
    assert len(session.laps) == 404

    # Show what data is available based on configuration
    if fastf1_config.enable_telemetry and hasattr(session.laps, "pick_fastest"):
        fastest_lap = session.laps.pick_fastest()
        assert fastest_lap["Driver"] == "LEC"
        if fastest_lap is not None:
            telemetry = fastest_lap.get_telemetry()
            assert len(telemetry) == 535

    if fastf1_config.enable_weather and hasattr(session, "weather_data"):
        weather = session.weather_data
        if not weather.empty:
            assert len(weather) == 81
