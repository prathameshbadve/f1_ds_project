# FastF1 API

FastF1 provides data for all F1 races including testing, all sessions of the grands prix, laps, telemetry, track conditions, track status, race control messages, etc.

## Structure

The key object to interact with is the Session object. This object is associated with a particular session of a grand prix (free practice, qualifying, sprint, race). All data for a particular session is loaded from this Session object. More details can be found in this [link](https://docs.fastf1.dev).



## In context of this project

For this project, we will use the API to load the data on the event schedule, session-wise results, laps and telemetry. I am listing the methods/properties that are important to us along with the data columns that are returned by the API.

#### Session

- Call via `fastf1.get_session(year, race, session)`
- `session.load(laps=True, telemetry=True, weather=True, messages=True)` - Loads the data for that particular session from the API.
- `session.session_info` - (dict) Returns the details about the session including the name of the session ('Practice 1', 'Qualifying', 'Race'), name of grand prix, date, location, etc.

```
{'Meeting': {'Key': 1124,
'Name': 'Bahrain Grand Prix',
'OfficialName': 'FORMULA 1 GULF AIR BAHRAIN GRAND PRIX 2022',
'Location': 'Sakhir',
'Country': {'Key': 36, 'Code': 'BRN', 'Name': 'Bahrain'},
'Circuit': {'Key': 63, 'ShortName': 'Sakhir'}},
'ArchiveStatus': {'Status': 'Generating'},
'Key': 6979,
'Type': 'Practice',
'Number': 1,
'Name': 'Practice 1',
'StartDate': datetime.datetime(2022, 3, 18, 15, 0),
'EndDate': datetime.datetime(2022, 3, 18, 16, 0),
'GmtOffset': datetime.timedelta(seconds=10800),
'Path': '2022/2022-03-20_Bahrain_Grand_Prix/2022-03-18_Practice_1/'}
```

- `session.results` - (df) Returns a dataframe with the results for every driver.

|Column|Example Output|
|---|---|
|DriverNumber|1|
|BroadcastName|M VERSTAPPEN|
|Abbreviation|VER|
|DriverId|max_verstappen|
|TeamName|Red Bull Racing|
|TeamColor|1e5bc6|
|TeamId|red_bull|
|FirstName|Max|
|LastName|Verstappen|
|FullName|Max Verstappen|
|HeadshotUrl|<i>url</i>|
|CountryCode|NED|
|Position|1|
|ClassifiedPosition|1|
|GridPosition|1|
|Q1|NaN|
|Q2|NaN|
|Q3|NaN|
|Time|0 days 01:37:33.584000|
|Status|Finished|
|Points|25.0|
|Laps|57|

The structure of the dataframe is the same for all sessions and only the relevant timing and position columns are populated depending on the session. So we will store the results of all sessions in the same table and just add another column named 'sessionName' to differentiate between sessions.

- `session.laps` - (df) Returns data for every lap for every driver.

|Column|Example|
|---|---|
|Time|0 days 01:10:49.020000|
|Driver|VER|
|DriverNumber|1|
|LapTime|0 days 00:01:38.877000|
|LapNumber|5.0|
|Stint|1.0|
|PitOutTime|NaT|
|PitInTime|NaT|
|Sector1Time|0 days 00:00:31.498000|
|Sector2Time|0 days 00:00:42.854000|
|Sector3Time|0 days 00:00:24.525000|
|Sector1SessionTime|0 days 01:09:41.678000|
|Sector2SessionTime|0 days 01:10:24.532000|
|Sector3SessionTime|0 days 01:10:49.057000|
|SpeedI1|229.0|
|SpeedI2|256.0|
|SpeedFL|276.0|
|SpeedST|293.0|
|IsPersonalBest|False|
|Compound|SOFT|
|TyreLife|8.0|
|FreshTyre|False|
|Team|Red Bull Racing|
|LapStartTime|0 days 01:09:10.143000|
|LapStartDate|2022-03-20 15:10:10.160000|
|TrackStatus|1|
|Position|2.0|
|Deleted|False|
|DeletedReason| |
|FastF1Generated|False|
|IsAccurate|True|

- `session.total_laps` - (int) Total number of laps of the session
- `session.weather_data` - (df) Returns minute-wise weather data

|Column|Example|
|---|---|
|Time|0 days 00:05:03.218000|
|AirTemp|25.6|
|Humidity|17.0|
|Pressure|1010.0|
|Rainfall|False|
|TrackTemp|32.1|
|WindDirection|16|
|WindSpeed|0.5|

- `session.session_status` - (df) Timings of session level events i.e. start, finished, finalised, ends, etc.
- `session.track_status` - (df) Track status i.e. flag conditions, safety car conditions, etc.
- `session.race_control_messages` - (df) Returns all race control messages throughout the session

|Column|Example|
|---|---|
|Time|2022-03-20 15:06:56|
|Category|Drs|
|Message|DRS ENABLED|
|Status|ENABLED|
|Flag|None|
|Scope|None|
|Sector|NaN|
|RacingNumber|None|
|Lap|3|

------