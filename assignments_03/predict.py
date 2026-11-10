# --- Task 7: Load and Predict --- # 
import joblib 
import json 
import pandas as pd 

#load model 
model = joblib.load("models/weather_classifier.pkl")

#load metadata 
with open( 
    "models/weather_classifier_metadata.json",
    "r",
    encoding="utf-8"
) as f: 
    metadata = json.load(f)

#Print key metadata 
print("---- Model Metadata ----")
print("City:", metadata["city"])
print("Features:", metadata["feature_names"])
print("Test Accuracy:", metadata["test_accuracy"])

features = metadata["feature_names"]
#four hypothetical days
days = pd.DataFrame(
    [[20, 10, 0.0, 10], #clearly good 
     [35, 20, 8.0, 45], #clearly bad
     [26, 5, 2.5, 28], #borderline
     [27, 5, 2.9, 29], #borderline 
    ],
    columns=features
)

#predictions 
predictions = model.predict(days)
probabilities = model.predict_proba(days)

#probability "good" columns 
good_index = list(model.classes_).index(1)

print("\n --- Predictions ---")

for i in range(len(days)):
    print(f"\nDay {i + 1}")

    for feature in features: 
        print(f"{feature}: {days.iloc[i][feature]}")

    label = "good" if predictions[i] == 1 else "skip"
    confidence = probabilities[i][good_index]

    print("Predicted Label:", label)
    print(f"Confidence (good): {confidence:.3f}")

# --- Task 8: Reflect --- # 
# 1. Pick your borderline day. What probability did the model give it? Would you call the model confident or uncertain, and how would you handle a day where it predicts 0.52?
# - Day 4 was a borderline day and it was given a "good" with a probability of 0.584. I would call the model uncertain because it is very close to 0.50. If the model predicted 0.52, I would have the app send 
#   a caution to show that weather is uncertain instead of just saying "skip" running today. 
# 2. predict.py has no training code. What would break if someone ran predict.py before train_classifier.py had ever been run? How could you make the error message more helpful?
# - The program would fail to load and the metafiles would not exist. Which would lead to a file not found error. To make the error more helpful a "Please run train_classifier.py first and make sure model is saved" 
#   would be added to the error message. 
# 3. In Week 10 this model will run inside a cloud pipeline that classifies each day's weather automatically. Name one thing about your current predict.py that would need to change to support that.
# - The script will need to be modified. Because in a cloud pipeline the weather data from the API will need to be automated instead of creating one manually using DataFrame. 