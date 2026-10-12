# --- Task 5: The Test Suite --- # 
import json 
from pathlib import Path 
import pytest
from pydantic import ValidationError
from weatherkit.schemas import WeatherResponse

def test_valid_response_has_168_hours():
    data_path = Path(__file__).parent.parent / "weather_raw.json"
    with open(data_path, "r") as f: 
        data = json.load(f)
        # We use Path(__file__) because it will run no matter where pytest is ran from. 
        # We use this rather than the plain relative path because the relative path can break depending on what current directory is being used. 

    weather = WeatherResponse.model_validate(data)

    assert len(weather.hourly.time) == 168

def test_latitude_raises_validation_error():
    bad_data = {
        "latitude": 200.0,
        "longitude": -117.0,
        "timezone": "UTC",
        "elevation": 100.0,
        "hourly": {
            "time": ["2026-04-08T00:00"],
            "temperature_2m": [15.0],
            "precipitation": [0.0],
        },
    }

    with pytest.raises(ValidationError): WeatherResponse.model_validate(bad_data)

def test_mismatched_lengths_raise_validation_error():
    bad_data = {
        "latitude": 34.0,
        "longitude": -117.0,
        "timestamp": "UTC",
        "elevation": 100.0,
        "hourly": {
            "time": ["a", "b", "c"],
            "temperature_2m": [1.0, 2.0, 3.0],
            "precipitation": [0.0, 0.0],
        },
    }

    with pytest.raises(ValidationError): WeatherResponse.model_validate(bad_data)

def test_null_temp_raises_validation_error():
    bad_data = {
        "latitude": 34.0,
        "longitude": -117.0,
        "timestamp": "UTC",
        "elevation": 100.0,
        "hourly": {
            "time": ["2026-04-08T00:00"],
            "temperature_2m": [None],
            "precipitation": [0.0],
        },
    }
    
    with pytest.raises(ValidationError): WeatherResponse.model_validate(bad_data)
