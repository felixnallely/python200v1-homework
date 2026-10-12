# --- Task 3: Inside the Boundary -- weatherkit/records.py --- # 
from dataclasses import dataclass
from .schemas import WeatherResponse

@dataclass
class HourlyReading:
    """
    This represent a single hourly reading of weather observation. 
    Units: temperature_c: degrees Celsius 
           precipitation_mm: millimeters (mm)
    """

    timestamp: str
    temperature_c: float
    precipitation_mm: float 

def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """
    Converts validated WeatherResponse into list of HourlyReading.
    Args: response: Validated WeatherResponse that contains hourly weather data.
    Returns: A list of HourlyReading occurances, one per hour, in the same order. 
    """

    readings = []

    for i in range(len(response.hourly.time)):
        readings.append(
            HourlyReading(
                timestamp=response.hourly.time[i],
                temperature_c=response.hourly.temperature_2m[i],
                precipitation_mm=response.hourly.precipitation[i],
                )
        )
    
    return readings 

# HourlyReading is a class instead of a Pydantic model because WeatherResponse and HourlyBlock are at the boundray between the code 
# and the external API. (The boundary is where the external API Json enters the application). These two help validate data that comes from a source that can be considered untrusted (the Json from the Open-Metro)
# so it's useful to have the Pydantic model there. But HourlyReading only exists after the data has already been validated and then converted into objects. 
# After this point the validation step is no longer needed so working with a dataclass is simpler with the cleaned data. 