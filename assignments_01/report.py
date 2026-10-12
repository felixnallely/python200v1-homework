import json 

from weatherkit import WeatherResponse, to_readings, DailyAggregator

#Testing --
with open ("weather_raw.json", "r") as f: 
    raw_data = json.load(f)

weather = WeatherResponse.model_validate(raw_data)

print("Latitude:", weather.latitude)
print("Timezone:", weather.timezone)
print("Hourly observations:", len(weather.hourly.time))

readings = to_readings(weather)

print(len(readings))
print(readings[0])
print(readings[-1])

# Testing aggregator
aggregator = DailyAggregator()
summaries = aggregator.summarize(readings)

print(len(summaries))

for summary in summaries[:2]:
    print(
        summary.date, 
        summary.temp_max,
        summary.temp_min,
        summary.precipitation_sum,
        summary.hours_observed,
    )

# -- Task 6: the Script -- report.py -- # 
def main() -> None: 
    """
    Loads weather and prints daily weather report.
    """
    with open ("weather_raw.json", "r") as f: 
        raw_data = json.load(f)

    weather = WeatherResponse.model_validate(raw_data)

    readings = to_readings(weather) 
    aggregator = DailyAggregator()
    summaries = aggregator.summarize(readings)
    dropped_days = aggregator.incomplete_days(readings)

    print(
        f"{'Date': <12}"
        f"{'High': >8}"
        f"{'Low': >8}"
        f"{'Precipitation': >10}"
        f"{'Range': >8}"
    )

    print("-" * 46)

    for summary in summaries: 
        print(
            f"{summary.date:<12}"
            f"{summary.temp_max:>8.1f}"
            f"{summary.temp_min:>8.1f}"
            f"{summary.precipitation_sum:>10.1f}"
            f"{summary.temp_range():>8.1f}"
        )

    if dropped_days: 
        print()
        print(
            "WARNING: Incomplete days were dropped:", ",".join(dropped_days),
        )

if __name__ == "__main__": main()

# If the guard was omitted, the report would automatically run printing the report and output. 
# If this report was being imported from another file. 

# -- Task 7: Reflection -- #
# 1. The file getting rejected as a whole when a single temperature is null can be useful when every reading and observation is 
#    critical for example when doing a scientific analysis. But I would rather tolerate the gap when doing a weather dashboard that is giving readings in real time.
#    In that situation, it is better to have missing values than to lose some missing data. In order to allow these missing temperatures the schema could change
#    in the field aspect for example instead of: temperature_2m: list[float] there could be something such as temperature_2m: list[float | None]. This will help the 
#    pipeline determine how it will handle missing values.
# 2. If the data pipeline runs at noon this means there will only be 12 hours worth of readings therefore the report will be dropped. This helps prevent any misleading statistics from 
#    being reported. Having the incomplete_days() helps list the dates that are dropped allowing users to know what data isn't included rather than the data disappearing altogether. 
# 3. In week 10, having done this as a package will make it easier to reuse, because we can just import components that are needed instead of copying code to other files. 