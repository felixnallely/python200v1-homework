from weather_model import WeatherClassifier

# ---Task 5: The Script --- #
def main(): 
    classifier= WeatherClassifier("models/weather_classifier.pkl")  #load real model

    days = [
        { #clearly good
            "temperature_2m_max": 85, 
            "temperature_2m_min": 65,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 5,
        },
        { #Clearly good
            "temperature_2m_max": 78, 
            "temperature_2m_min": 58,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 8,
        },
        { #clearly skip
            "temperature_2m_max": 40, 
            "temperature_2m_min": 28,
            "precipitation_sum": 2.5,
            "wind_speed_10m_max": 30,
        },
        { #borderline
            "temperature_2m_max": 68, 
            "temperature_2m_min": 50,
            "precipitation_sum": 0.2,
            "wind_speed_10m_max": 15,
        },
    ]

    predictions = classifier.predict(days)

    for i, (day, prediction) in enumerate(zip(days, predictions), start=1):
        print(
            f"Day {i}: " 
            f"High={day['temperature_2m_max']}°C, "
            f"Low={day['temperature_2m_min']}°C, "
            f"Rain={day['precipitation_sum']} in, "
            f"Wind={day['wind_speed_10m_max']} mph -> "
            f"Label={prediction.label}, "
            f"Probability={prediction.probability:.2f}"
        )

if __name__ == "__main__":
    main()


#comment explaining what would happen if you omitted the guard and another script imported predict_weather.py to reuse a helper.
# Without the guard, the code inside main() will run immediately when any script imports predict_weather.py. 
# Having the guard allows the code to run only when it is ran directly from this file, which allows other scripts to safely import and
# reuse helpers without actually triggering the predictions. 


# --- Task 6: Reflection --- #
# 1.Your predict() selects the four features in a fixed order rather than using whatever order the caller passed. Describe a concrete bug this prevents.
#   Since the model was trained using similar features in a certain order, and if the predict() used whatever order the caller asked for the values would all get
#   mismatched. Which will result incorrect predictions, although the inputs where correct.

# 2.The graded tests train their own tiny model in a fixture instead of loading models/weather_classifier.pkl. Why is that a better design for a test suite than depending on the real file?
#   Having tests train their own tiny model makes the test independent and reproducible. By having a small model in the fixture, the tests always run in a secure environment which can help verify
#   how WeatherClassifier consistently behaves. 

# 3.In Week 10 a pipeline will import WeatherClassifier and call predict() on batches of weather records from a database. Name one thing about your component that makes it ready for that, and
#   one thing you might add before it runs in production.
#   Something that makes this component ready for week 10 is the predict() since it accepts a list of weather records and returns predictions for each record, this works well for processing batches from a database.
#   But before production more error handling needs to be added so that invalid data can be recorded without stopping the prediction process. 

