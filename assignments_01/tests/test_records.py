# --- Task 5: The Test Suite --- # 
import json 
from pathlib import Path 
from weatherkit.records import HourlyReading, to_readings
from weatherkit.schemas import WeatherResponse

def load_weather():
    data_path = Path(__file__).parent.parent / "weather_raw.json"
    with open(data_path, "r") as f: 
        data = json.load(f)

    return WeatherResponse.model_validate(data)

def test_one_reading_per_hour():
    weather = load_weather()

    readings = to_readings(weather)
    assert len(readings) == len(weather.hourly.time)
    assert readings[0].timestamp == weather.hourly.time[0]
    assert readings[-1].timestamp == weather.hourly.time[-1]

def test_reading_values_match_index(): 
    weather = load_weather()

    readings = to_readings(weather) 
    i = 10

    assert readings[i].timestamp == weather.hourly.time[i]
    assert readings[i].temperature_c == weather.hourly.temperature_2m[i]
    assert readings[i].precipitation_mm == weather.hourly.precipitation[i]

def test_identical_hourly_readings():
    r1 = HourlyReading(
        timestamp="2026-04-08T00:00",
        temperature_c=10.0,
        precipitation_mm=0.0,
    )
    r2 = HourlyReading(
        timestamp="2026-04-08T00:00",
        temperature_c=10.0,
        precipitation_mm=0.0,
    )

    assert r1 == r2