import pytest 
from weatherkit.records import HourlyReading
from weatherkit.summarize import DailyAggregator

@pytest.fixture
def sample_readings():
    return [
        HourlyReading("2026-04-08T00:00", 10.0, 0.5),
        HourlyReading("2026-04-08T01:00", 15.0, 1.0),
        HourlyReading("2026-04-09T00:00", 5.0, 0.2),
        HourlyReading("2026-04-09T01:00", 20.0, 0.3),
    ]

# Test 1 --
def test_grouping_two_summaries(sample_readings):
    aggregator = DailyAggregator(min_hours=2)

    summaries = aggregator.summarize(sample_readings)

    assert len(summaries) == 2

# Test 2 -- 
def test_temp_max_min_correct(sample_readings):
    aggregator = DailyAggregator(min_hours=2)

    summaries = aggregator.summarize(sample_readings)

    assert summaries[0].temp_max == 15.0
    assert summaries[0].temp_min == 10.0

# Test 3 -- 
def test_precipitation_sum(sample_readings):
    aggregator = DailyAggregator(min_hours=2)
    
    summaries = aggregator.summarize(sample_readings)
    
    assert summaries[0].precipitation_sum == pytest.approx(1.5)

# Test 4 --
def test_incomplete_days_dropped():
    readings = [HourlyReading("2026-04-08T00:00", 10.0, 0.0)]
    
    aggregator = DailyAggregator(min_hours=2)
    
    summaries = aggregator.summarize(readings)
    
    assert summaries == [] 
    assert aggregator.incomplete_days(readings) == ["2026-04-08"]

# Test 5 -- 
def test_lower_min_hours_keeps_day():
    readings = [HourlyReading("2026-04-08T00:00", 10.0, 0.0)]
    
    aggregator = DailyAggregator(min_hours=1)
    
    summaries = aggregator.summarize(readings)
    
    assert len(summaries) == 1


# Parametrize Test --
@pytest.mark.parametrize("temp_max, temp_min, expected",
        [
            (20.0, 10.0, 10.0),
            (15.5, 5.5, 10.0),
            (8.0, 8.0, 0.0),
            (-2.0, -6.0, 4.0)
        ],
    )
def test_temp_range(temp_max, temp_min, expected):
    from weatherkit.summarize import DailySummary

    summary = DailySummary(
        date="2026-04-08",
        temp_max=temp_max,
        temp_min=temp_min, 
        precipitation_sum=0.0,
        hours_observed=24,
    )

    assert summary.temp_range() == expected

## I checked to see if this test would fail. To do this I changed DailySummary.temp_range() (from summarize.py file) 
# to return temp_max + temp_min instead of temp_max - temp_min. Therefore the Parametrize test caught the error.
# FAILED tests/test_summarize.py::test_temp_range[20.0-10.0-10.0] - AssertionError: assert 30.0 == 10.0