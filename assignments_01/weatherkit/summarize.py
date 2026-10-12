# --- Task 4: The Aggregation -- weatherkit/summarize.py -- #
from dataclasses import dataclass
from collections import defaultdict
from .records import HourlyReading 

@dataclass
class DailySummary: 
    """
    This shows the observation of weather for a single day.
    Attributes: 
        date: in calendar date formatted as, YYYY-MM-DD. 
        temp_max: Highest observed temperature.
        temp_min: Lowest observed temperature.
        precipitation_sum: total precipitation for the day. 
        hours_observed: number of readings used hourly.
    """

    date: str
    temp_max: float
    temp_min: float
    precipitation_sum: float
    hours_observed: int

    def temp_range(self) -> float:
        """
        Returns the differemce between the day's maximum and minimum temperatures.
        Returns: 
        Temp range in degree Celsuis.
        """
        return self.temp_max - self.temp_min 

class DailyAggregator: 
    """
    This aggregates houly readings into summaries of weather daily. 
    """

    def __init__(self, min_hours: int = 24) -> None:
        """
        Starts the aggregator.
        Args: 
            min_hours: Minimum number of observations houraly that are required, 
            so a day can be used in summaries.
        """
        self.min_hours = min_hours

    def summarize(self, readings:list[HourlyReading]) -> list[DailySummary]:
        """
        Will create daily summaries from hourly readings. 
        But days with few min_hours will not be included. 
        Args: 
            readings: A list of hourly weather readings. 
        Returns: A list of DailySummary sorted in date.
        """
        grouped: dict[str, list[HourlyReading]] = defaultdict(list)

        for reading in readings: 
            date = reading.timestamp[:10]
            grouped[date].append(reading)

        summaries: list[DailySummary] = [] 

        for date, day_readings in grouped.items():
            if len(day_readings) < self.min_hours:
                continue

            summaries.append(
                DailySummary(
                    date=date, temp_max=max(reading.temperature_c for reading in day_readings),
                    temp_min=min(reading.temperature_c for reading in day_readings),
                    precipitation_sum=sum(reading.precipitation_mm for reading in day_readings),
                    hours_observed=len(day_readings),
                )
            )
        return sorted(summaries, key=lambda summary: summary.date)

    def incomplete_days(self, readings: list[HourlyReading]) -> list[str]:
        """
        Returns the dates that do not have enough hourly observations to be included.
        Args:
            readings: A list of hourly weather readings.
        Returns:
            A list of dates with too few readings.
        """
        grouped: dict[str, list[HourlyReading]] = defaultdict(list)

        for reading in readings:
            date = reading.timestamp[:10]
            grouped[date].append(reading)

        incomplete: list[str] = []

        for date, day_readings in grouped.items():
            if len(day_readings) < self.min_hours:
                incomplete.append(date)

        return sorted(incomplete)