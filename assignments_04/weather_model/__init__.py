# --- Task 3: The Package Entry Point --- # 
"""
Weather model package.
"""

from .classifier import WeatherClassifier, Prediction

__all__ = ["WeatherClassifier", "Prediction"]