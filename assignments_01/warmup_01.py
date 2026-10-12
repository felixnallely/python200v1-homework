# --- Classes --- Question 1
class Thermometer:
    def __init__ (self, location, readings=None):
        self.location = location 
        self.readings = readings if readings is not None else []

    def add(self, reading): 
        self.readings.append(reading)

    def average(self):
        if not self.readings: 
            return None 
        return sum(self.readings) / len(self.readings)

    def hottest(self):
        if not self.readings:
            return None
        return max(self.readings)

    def __repr__(self):
            avg = self.average()
            return (
                f"Thermometer(location='{self.location}', "
                f"n_readings={len(self.readings)}, "
                f"average={avg})"
            )

t = Thermometer("Hesperia, CA")
t.add(60.5)
t.add(58.2)
t.add(61.9)
t.add(57.4)

print("Average temperature:", t.average())
print("Hottest temperature:", t.hottest())

# Add comment
# why does average() need to handle the empty case? 
# So there isn't an error when the sum is divided by zero.
# what would happen without that check? 
# Without the check the program will crash if we ask for average() but there aren't any readings.

# --- Classes --- Question 2
#--- Testing __repr__ --- 

# The __repr__ method makes the object print in a useful, readable format.
# Without it, Python shows a default object representation like
# <__main__.Thermometer object at 0x...>
# which is not helpful for debugging.

t1 = Thermometer("Charlotte")
t1.add(18.2)
t1.add(19.1)
t1.add(17.5)
t1.add(19.7)

t2 = Thermometer("Hesperia")
t2.add(60.5)
t2.add(58.2)
t2.add(61.9)
t2.add(57.4)

print(t1)
print([t1, t2])

# Without __repr__ the output would be something like <__main__.Themometer object at ... This output would only show the class name and memory address which is technically correct
# but would not provide us with anything useful for debugging. 

# --- Classes --- Question 3
class TemperatureAlert:
    def __init__(self, threshold=30.0):
        self.threshold = threshold
    def breaches(self, thermometer):
        return [r for r in thermometer.readings if r > self.threshold]

t = Thermometer("Hesperia")
t.add(60.5)
t.add(58.2)
t.add(61.9)
t.add(57.4)

#alerts with 2 different thresholds
alert1 = TemperatureAlert(29.0)
alert2 = TemperatureAlert(30.5)

print("Alert1 breaches:", alert1.breaches(t))
print("Alert2 breaches:", alert2.breaches(t))

# --- Dataclass --- Question 1
from dataclasses import dataclass, FrozenInstanceError

@dataclass(frozen=True)
class Station:
    """
    Represents a weather or climate station monitoring weather using a unique ID, 
    readable name, geographic metadata that includes: latitude, longitude, and elevation. 
    """
    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float

station_a = Station(
    station_id="ABC123",
    name="South Eastern California Hesperia",
    latitude=34.426388,
    longitude=-117.300880,
    elevation=3190.0,
)

station_b = Station(
    station_id="ABC123",
    name="South Eastern California Hesperia",
    latitude=34.426388,
    longitude=-117.300880,
    elevation=3190.0,
)

print(station_a == station_b)

# The result is True because both station_a and station_b had identical fields and values. Therefore the comparison will return True.
# Dataclasses automatically generate an __eq__ that compares opbjects, since the handwritten class did not have an __eq__ then the result would have been false. 


# --- Dataclass --- Question 2
station = Station("ST01", "Hesperia Station", 34.4264, -117.3009, 3190)

try: 
    station.name = "New Name" 
except FrozenInstanceError as e:
    print("Caught error:", e)

a = Station("ST01", "Hesperia Station", 34.4264, -117.3009, 3190)
b = Station("ST01", "Hesperia Station", 34.4264, -117.3009, 3190)
c = Station("ST02", "Victorville Station", 34.5362, -117.2928, 2950)

stations = {a, b, c}
print(len(stations))

# It also makes the object hashable so identical fields and values that are compared are placed into one entry. 
# This is useful because stations are identified by data not memory location. 

# --- Dataclass --- Question 3
#from typing import List 

#@dataclass 
#class StationBatch: 
   # region: str
   # stations: List["Station"] = [] #ValueError: mutable default <class 'list'> for field stations is not allowed: use default_factory 
#batch = StationBatch("High Desert")

# - Correct -- # 
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass 
class StationBatch: 
    """
    Station obbjects that belong to specific regions. This provides 
    methods to add stations and see which one has the highest elevation.
    """
    region: str
    stations: List["Station"] = field(default_factory=list)

    def add(self, station: "Station") -> None: 
        """
        Add a station to the batch as an object.
        """
        self.stations.append(station)

    def highest(self) -> Optional["Station"]:
        """
        Return Station with greatest elevation, or None if empty.
        """
        if not self.stations:
            return None
        return max(self.stations, key=lambda s: s.elevation)

# Python refuses to allow station: likst[Station] = [] because every list given will keep appending to the same list which would 
# cause the data to cross contaminate. 


# --- Pydantic --- Question 1
from pydantic import BaseModel, Field, ValidationError 

class Reading(BaseModel):
    """
    Shows a single weather reading from a station, this includes temperature and 
    humidity constraints and a minimum length for station_id.
    """
    station_id: str = Field(..., min_length=3)
    timestamp: str 
    temperature_c: float = Field(..., ge=-90, le=60)
    humidity: float = Field(..., ge=0, le=100)

reading = Reading(
    station_id="ST01",
    timestamp="2026-10-06T18:54:00Z",
    temperature_c=22.5,
    humidity=45.0
)

print(reading)

# --- Pydantic --- Question 2
class Reading(BaseModel):
    """
    Shows reading for a single weather with validation rules for station_ID length, temp range and humidity range.
    """
    station_id: str = Field(..., min_length=3)
    timestamp: str
    temperature_c:float = Field(..., ge=-50, le=100) 
    humidity: int = Field(..., ge=0, le=100)

#-- A Missing Required Field (timestamp) --#
try: 
    Reading(
        station_id="ST01", 
        #timestamp missing
        temperature_c=20.0,
        humidity=50.0
    )
except ValidationError as e:
    print("Missing field error:")
    print(e)


#- Temperature_c of 150.0 --#
try: 
    Reading(
        station_id="ST01", 
        timestamp="2026-10-06T18:54:00Z",
        temperature_c=150.0, #invalid 
        humidity=50.0
    )
except ValidationError as e:
    print("\nTemperature error:")
    print(e)


#- Humidity of "very humid" --#
try: 
    Reading(
        station_id="ST01", 
        timestamp="2026-10-06T18:54:00Z",
        temperature_c=20.0, 
        humidity="very humid" # invalid 
    )
except ValidationError as e:
    print("\nHumidity error:")
    print(e)

#-- Successful example --#
reading = Reading(
    station_id="ST01", 
    timestamp="2026-10-06T18:54:00Z",
    temperature_c="21.5", #string, but we can convert
    humidity=40        #int
)
    
print("\nValid coerced reading:")
print(reading)

print("temperature_c type:", type(reading.temperature_c))
print("humidity type:", type(reading.humidity))

# Pydantic asscepts "21.5" because it can convert it into a float, but it 
# can not convert "very humid" since it isn't a string and can not be converted to an integer.
#
# Rule: 
# Pydantic will accept values that are converstions that actually make sense and if the value 
# entered cannot be converted to a required type then it will raise a ValidationError.

# --- Pydantic --- Question 3
from datetime import datetime 

   # station_id: str = Field(..., min_length=3)
    #timestamp: datetime 
    #temperature_c: float 

try: 
    reading = Reading(
        station_id_="AB",  #too short 
        temperature_c="not a number",  #non-numeric
        humidity=50
    ) #no timestamp 

except ValidationError as e: 
    print("Validation errors:")

    for error in e.errors():
        print(f"loc: {error['loc']}")
        print(f"msg: {error['msg']}")
        print()

# There are 3 errors reported: 
# 1. station_id is shorter than the minimum length. (Field required)
# 2. timestamp is missing. (Field required)
# 3. temperature_c cannot be a float. (Input should be a valid number...)
# Reporting all the errors as once is more useful because it can fix every problem in a single go, but 
# if the validation stops at the first error it will keep stopping at each rerun of the program until each problem is fixed one by one.
# Catching the errors in one go makes debugging more efficient 


# --- Pydantic --- Question 4
from pydantic import model_validator

class Reading(BaseModel): 
    station_id: str = Field(..., min_length=3)
    timestamp: str
    temperature_c: float = Field(..., ge=-50, le=100)
    humidity: float = Field(..., ge=0, le=100)

    @model_validator(mode="after")
    def check_failed_sensor(self):
        if self.humidity == 0.0 and self.temperature_c < -40:
            raise ValueError("Humidity of 0.0 combined with a temperature below -40" "indicates a failed sensor."
            )
        return self
    
#Valid reading 
valid_reading = Reading(
    station_id="ST01", 
    timestamp="2026-10-06T18:54:00Z",
    temperature_c=20.0,
    humidity=25.0 
)

print("Valid reading:")
print(valid_reading)

# invalid reading
try:
    Reading(
        station_id="ST01",
        timestamp="2026-10-06T18:54:00Z",
        temperature_c=-45.0,
        humidity=0.0,
    )
except ValidationError as e:
    print("\nInvalid reading rejected:")
    print(e)


# This rule cannot be expressed with Field constraints alone because they only check and validate 
# one field at a time. So it is better to use model_validator since this helps validate multiple required fields. 


# --- Pytest --- Question 1
import pytest 

def celsius_to_fanhrenheit(celsius:float) -> float:
    """
    To convert a temperature in Celsius to Fahrenheit.
    Args: celsius: Temperature in degrees Celsius 
    Returns: equivalent temperature in degrees Fahrenheit.
    """
    return (celsius * 9 / 5) + 32 

def test_celsius_to_fahrenheit():
    assert celsius_to_fanhrenheit(0) == 32
    assert celsius_to_fanhrenheit(100) == 212 
    assert celsius_to_fanhrenheit(37) == pytest.approx(98.6)   
    #since floats can cause rounding calculations, pytest.approx is used instead of == when it comes 
    #to comparing decimal values.


# --- Pytest --- Question 2
def mean(values: list[float]) -> float:
    """
    To return the sumed up mean of a list of numbers.
    Raises: ValueError: if the list is empty.
    """
    if not values: 
        raise ValueError("Can't calculate mean of empty list.")

    return sum(values) / len(values)

def test_mean_of_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        mean([])

# pytest.raises(ValueError) alone will fail to catch the match= catches because it only checks that a ValueError
# occurred, but will still pass wrong ValueError messages. But the matche="" makes sure that the exception message has the expected word to help 
# confirm the error does happen for a valid reason. 


# --- Pytest --- Question 3
def mean(values: list[float]) -> float: 
    """
    To return the sumed up mean of a list of numbers.
    """
    if not values: 
        raise ValueError("Can't calculate mean of empty list.")
    
    return sum(values) / len(values)
@pytest.mark.parametrize("values, expected", [([1, 2, 3], 2.0), ([5], 5.0), #single element list 
        ([10, 20, 30, 40], 25.0),]
    )

def test_mean_values(values, expected):
    assert mean(values) == expected

# One parametrized test with four cases are better than four nearly identical test functions because it reduces 
# duplicated code and helps keep any cases that are related to the test together. It also allows for adding more 
# entries onto the list of parameters. It also makes it easy to identify any failures. 


# --- Pytest --- Question 4 
#Break the function 
# def celsius_to_fanhrenheit(celsius:float) -> float:
    #return (celsius * 9 / 4) + 32 

# Failure output after breaking function: 
#  E assert 257.0 == 212
#  E        +  where 257.0 = celsius_to_fanhrenheit(100)
#
# This is more helpful than an "assertion failed" message because it shows what when wrong and by how much it went wrong. 
# Which makes it easier to debug and fix the problem. 



import json 

with open ("weather_raw.json") as f: 
    data = json.load(f)

    print(len(data["hourly"]["time"]))
    print(len(data["hourly"]["temperature_2m"]))
    print(len(data["hourly"]["precipitation"]))

