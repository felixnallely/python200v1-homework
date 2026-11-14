from dataclasses import dataclass 
from typing import Any
import joblib 
import os 


# --- Task 1: The Prediction Dataclass --- #

@dataclass
class Prediction:
    """Stores weather prediction and its probability."""

    label: str  #the "good" or "skip"
    probability: float  #models confidence of good running day


# --- Task 2: The WeatherClassifier Class --- #
class WeatherClassifier:
    """
    Loads trained weather model and makes predictions.
    """
    required_features = [
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "wind_speed_10m_max",
    ]
     

    def __init__(self, model_path: str):
        """
        Starts the classifier.

        Args:
        model_path: Path to saved model.
        
        Raises: 
        FileNotFoundError: If model file does not exist.
        """

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = joblib.load(model_path)

    #predicted method 
    def predict(self, days: list[dict[str, float]]) -> list[Prediction]:
        """ 
        Predict good weather for running each day.

        Args:
            days: List of weather dictionaries.

        Return: 
            List of prediction objects. 
        
        Raises
            ValueError: If required features are missing.
        """

        if not days: 
            return[]

        missing = set()

        for day in days: 
            for feature in self.required_features:
                if feature not in day:
                    missing.add(feature)

        if missing: raise ValueError(f"Missing required features: {','.join(sorted(missing))}")

        features = [
            [
                day["temperature_2m_max"],
                day["temperature_2m_min"],
                day["precipitation_sum"],
                day["wind_speed_10m_max"], 
            ]
            for day in days
        ]

        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)[:, 1]

        return [
            Prediction(
                label = "good" if pred == 1 else "skip", 
                probability=float(prob)
            )
            for pred, prob in zip(predictions, probabilities)
        ]
#note where an error can be raised.
# Error can be raised in __init__ (FileNotFoundError) and predict (ValueError).

