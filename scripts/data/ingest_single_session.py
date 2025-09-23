"""Example of using FastF1 with environment configuration"""

from src.data_ingestion.fastf1_client import FastF1Client
from config.settings import fastf1_config


def main():
    """Function to test ingesting data from FastF1 API"""

    # Print current configuration
    print("=== FastF1 Configuration ===")
    print(f"Cache enabled: {fastf1_config.cache_enabled}")
    print(f"Cache directory: {fastf1_config.cache_dir}")
    print(f"Force renew: {fastf1_config.force_renew}")
    print(f"Log level: {fastf1_config.log_level}")
    print(f"Default season: {fastf1_config.default_season}")
    print(f"Session types to process: {fastf1_config.session_types}")

    # Create client
    client = FastF1Client()

    # Check cache status
    cache_path, cache_size = client.cache_info()
    if cache_path:
        print(f"Cache location: {cache_path}")
        print(f"Cache size: {cache_size / (1024 * 1024):.2f} MB")

    # Load a session based on environment configuration
    try:
        year = fastf1_config.default_season
        session = client.get_session(year, "Monaco", "Q")

        print("\n=== Loaded Session ===")
        print(f"Event: {session.event.EventName}")
        print(f"Session: {session.name}")
        print(f"Date: {session.date}")
        print(f"Number of laps: {len(session.laps)}")

        # Show what data is available based on configuration
        if fastf1_config.enable_telemetry and hasattr(session.laps, "pick_fastest"):
            fastest_lap = session.laps.pick_fastest()
            if fastest_lap is not None:
                telemetry = fastest_lap.get_telemetry()
                print(f"Telemetry data points: {len(telemetry)}")

        if fastf1_config.enable_weather and hasattr(session, "weather_data"):
            weather = session.weather_data
            if not weather.empty:
                print(f"Weather data points: {len(weather)}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
