# --- Task 4: The Test Suite --- # 
import joblib 
import pytest 

from sklearn.pipeline import Pipeline
from weather_model import WeatherClassifier, Prediction 
from sklearn.ensemble import RandomForestClassifier

@pytest.fixture
def classifier(tmp_path):
    """
    Create and save a tiny model, then load WeatherClassifier.
    """

    X = [
        [85, 65, 0.0, 5],  #good
        [80, 60, 0.0, 8],  #good
        [35, 25, 2.0, 30],  #skip day 
        [40, 30, 1.5, 25],  #skip day
    ]

    y = [ 1, 1, 0, 0]

    model = Pipeline([
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    model.fit(X, y)

    model_path = tmp_path / "weather_model.pkl"
    joblib.dump(model, model_path)

    return WeatherClassifier(str(model_path))

@pytest.mark.parametrize(
    "day, expected_label",
    [
        (
            {
                "temperature_2m_max": 85, 
                "temperature_2m_min": 65,
                "precipitation_sum": 0.0,
                "wind_speed_10m_max": 5,
            }, 
            "good", 
        ), 
        (
            {
                "temperature_2m_max": 35, 
                "temperature_2m_min": 25,
                "precipitation_sum": 2.0,
                "wind_speed_10m_max": 30,
            }, 
            "skip",
        ),
    ],
)

def test_day_labels(classifier, day, expected_label):
    """
    Checks good and bad day predictions.
    """
    result = classifier.predict([day])
    assert result[0].label == expected_label

def test_probability_between_zero_and_one(classifier):
    """
    Probability should be between 0 and 1.
    """

    day = {
        "temperature_2m_max": 80, 
        "temperature_2m_min": 60,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 8,
    }

    prediction = classifier.predict([day])[0] 
    assert 0.0 <= prediction.probability <= 1.0

def test_three_days_return_three_predictions(classifier):
    """
    Three input days should return three predictions.
    """

    days = [
        {
        "temperature_2m_max": 85, 
        "temperature_2m_min": 65,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 5,
    },
    {
        "temperature_2m_max": 35, 
        "temperature_2m_min": 25,
        "precipitation_sum": 2.0,
        "wind_speed_10m_max": 30,
    },
    {
        "temperature_2m_max": 75, 
        "temperature_2m_min": 55,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 10,
    },
]
    predictions = classifier.predict(days)

    assert len(predictions) == 3 
    assert all(isinstance(p, Prediction) for p in predictions)
    assert predictions[0].label == "good"
    assert predictions[1].label == "skip"

def test_empty_list_returns_empty_list(classifier): 
    """
    Empty input should return empty output. 
    """
    assert classifier.predict([]) == []

def test_missing_feature_value_error(classifier):
    """
    Missing features that are required should raise ValueError.
    """
    bad_day = {
        "temperature_2m_max": 80, 
        "temperature_2m_min": 60,
        "precipitation_sum": 0.0,
    }

    with pytest.raises(ValueError, match="wind_speed_10m_max",):
        classifier.predict([bad_day])


def test_missing_model_file_not_found():
    """
    Missing model file should raise a FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError, match="Model file not found"): WeatherClassifier("does_not_exist.pkl")
    
# comment which test caught the change, then undo the break.
# When the classifier.py was modified to return "skip", the test_day_labels failed.