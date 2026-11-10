import json 
import pytest

# --- Warmup Question 1 --- # 
class ModelInfo: 
    def __init__(self, path):
        with open (path, "r") as file: 
            self.data= json.load(file)

    def feature_count(self): 
        return len(self.data["features"])

sample_data = {
    "features": [
        "temperature", 
        "humidity",
        "wind_speed"
    ]
}

with open("sample_model.json", "w") as file:
    json.dump(sample_data, file)

info= ModelInfo("sample_model.json")

print(info.feature_count())

#comment: why is loading the file in __init__ better than loading it inside feature_count() if the method is called many times?
# It's better to load the file in __init__ because it will load the data once. While the feature_count() will load the file everytime it's called
# creating multiple files. 


# --- Warmup Question 2 --- # 
from dataclasses import dataclass

@dataclass
class DayForecast:
    date: str
    high_c: float
    will_rain: bool

    def summary(self) -> str: 
        if self.will_rain: rain_text = "rain expected"
        else: 
            rain_text = "no rain expected"

        return f"{self.date}: high {self.high_c}C, {rain_text}"

#create objects 
day1 = DayForecast(
    "2026-11-06",
    24.1,
    True
)

day2 = DayForecast(
    "2026-11-07",
    28.5,
    False
)

print(day1.summary())
print(day2.summary())

#comment: name one thing the @dataclass decorator wrote for you that you did not have to write by hand.
# I didn't have to write the __init__ method for the class. 

# --- Warmup Question 3 --- # 
def parse_temperature(text: str) -> float:
    try: 
        return float(text)
    except ValueError: 
        raise ValueError("Invalid temperature value")

#test 
def test_parse_temperature_valid():
    assert parse_temperature("24.1") == pytest.approx(24.1)
    
def test_parse_temperature_bad_input_raises():
    with pytest.raises(ValueError, match="temperature"):
        parse_temperature("hot")    

#comment: what would pytest.raises(ValueError) without match fail to catch?
# It will not catch incorrect error messages. 