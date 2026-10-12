from pydantic import BaseModel, Field, model_validator

# --- Task 2: The Boundary -- weatherkit/schemas.py --- #
class HourlyBlock(BaseModel):
    """
    This represents hourly weather data that is returned by the Open-Meteo API.
    The data is stored in columnar format where each index corresponds to the same observation hourly.
    """

    time: list[str]
    temperature_2m: list[float]
    precipitation: list[float]

    @model_validator(mode="after")
    def calidate_lengths(self):
        if not (
            len(self.time) == len(self.temperature_2m) == len(self.precipitation)
        ):
            raise ValueError(
                "time, temperature_2m, and precipiation must have the same length"
            )
        return self 

class WeatherResponse(BaseModel):
    """
    Top-level Open-Metor weather response.
    """

    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    timezone: str
    elevation: float
    hourly: HourlyBlock