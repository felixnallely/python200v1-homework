# --- Task 1: Fetch the Data --- # 
import requests
import pandas as pd

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 34.11,
    "longitude": -117.29,
    "start_date": "2025-01-01",
    "end_date": "2025-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/Los_Angeles",
}
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

print("--Dataset shape:", df.shape)
print(df.head())

# The city I chose: San Bernardino, California 

# --- Task 2: Engineer the Label --- # 
df["good_for_running"] = (
    (df["temperature_2m_max"] >= 7) &
    (df["temperature_2m_max"] <= 26) &
    (df["temperature_2m_min"] >= 0) & 
    (df["precipitation_sum"] < 3) &
    (df["wind_speed_10m_max"] < 30)
    ).astype(int)

#Reasonable thresholds so I did not modify.
#True --> 1
#False --> 0

print("Class Balance:")
print(df["good_for_running"].value_counts())

good_fraction = df["good_for_running"].mean()
print(f"Fraction of good running days: {good_fraction:.2%}")

#comment: what fraction of days are good, and does that seem reasonable for your city's climate? If your data is very imbalanced, note which metrics will matter most because of it.
# 36% days were classified as good for running. This seems reasonable because San Bernardino has many windy days and often rainy days. 
# The data is very imbalanced with more skip days (63.56%) than good running days (36.44%). To make sure there is no misleading data, precision, recall and an F1 score would be good metrics to have.


# -- Task 3: Build and Evaluate a KNN Classifier --- # 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

#Features/target
X = df[[
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max"
]]

y = df["good_for_running"]

#Train split test (80/20) with stratification 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

#several k values (at least 5)
k_values = [1, 3, 5, 7, 9, 11, 13]

best_k = None
best_score = 0 

for k in k_values: 
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])

    scores = cross_val_score(
        pipeline,
        X_train, 
        y_train,
        cv=5,
        scoring="accuracy"
    )

    mean_score = scores.mean()

    print(f"k={k}: Mean CV Accuracy = {mean_score:.4f}")

    if mean_score > best_score: 
        best_score = mean_score 
        best_k = k

print("\nBest k:", best_k)
print("Best CV Accuracy:", round(best_score, 3))

#Train Final Model 
final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=best_k))
])

final_model.fit(X_train, y_train)

#predictions 
y_pred = final_model.predict(X_test)

#Evaluate model 
accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", round(accuracy, 3))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --- Task 4: Build and Evaluate a Logistic Regression Classifier --- #
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import (precision_score, recall_score, f1_score,)

#Create pipeline 
log_model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000))
])

log_model.fit(X_train, y_train)

#predictions 
y_pred = log_model.predict(X_test)

#Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#Print future coefficients
feature_names = X.columns 

coefficients = log_model.named_steps["logreg"].coef_[0]

print("\nFeature Coefficients:")

for feature, coef in zip(feature_names, coefficients): 
    print(f"{feature}: {coef:.4f}")

#comment interpreting them: which conditions most strongly push a day toward "skip"?
# Logistic Regression Results-
# Accuracy: 0.8904
# Precision: 0.8519
# Recall: 0.8519
# F1 Score: 0.8519

# Interpretation: 
# Precision and recall bout resulted in about 0.85, which indicates the model performs well when it 
# identifies good running days and keeps errors at a low rate. 
# The model classified 89% of the test days correctly. 

#-------------------
# Feature Coefficients:
# temperature_2m_max: -2.4258
# temperature_2m_min: -1.1397
# precipitation_sum: -4.4915
# wind_speed_10m_max: 0.2225

# Coefficient Interpretation: 
# Having a negative coefficient pushes predictions towards 0 which is a skip day, while a positive 
# coefficient pushes predictions towards 1 which is a good running day.
# Therefore precipitation_sum (-4.4915), temperature_2m_max (-2.4258), temperature_2m_min (-1.1397), all push towards 
# "skip" because they have a large negativer coefficient. This makes sense because rainy days high temperatures and low temperatures 
# are days people tend to not go out for a run. 
# And wind_speed_10m_max (0.2225) having a small positive coefficient indicates that wind speed has almost no influence on whether a day is good for running. 


# --- Task 5: Compare and Choose --- # 
#-------------------
# KNN results: 
# Accuracy: 0.918   Precision: 0.89 
# Recall: 0.89      F1 Score: 0.89

# Logistic Regression Results: 
# Accuracy: 0.8904  Precision: 0.8519
# Recall: 0.8519    F1 Score: 0.8519

#Which model had higher accuracy on your data?
# The KNN model performed better than Logisitic Regression. The KNN model scored an 91.8% in test accuracy, while Logistic Regression scored 89.04%.
# The KNN model also scored higher in precision, recall and F1 score for the good for running class. 

#-------------------
#For a running app, is accuracy the metric you care about most, or would you weigh precision or recall more heavily? Explain in terms of false positives (sent out on a bad day) and false negatives (missed a good day).
# Yes, for a runningg class accuracy is important, although it's not the only metric that should matter. 
# A false negative happens when the app recommends a skipping day but it was a suitable day for running. Although it is inconvenient, we perfer this since it is less harmful than having someone running in bad conditions. 
# A  false positive happens when the app recommends a running day but it is actually bad running conditions. Because of these false results having precision is more important than accuracy, because higher precision can reduce 
# the number of false positives and app users will be less likely to get poor recommendations.

#-------------------
#Which model will you deploy, and why? Consider not only the scores but also model size, prediction speed, interpretability, and whether you want probability confidence scores. Either choice can be correct if you justify it.
# I would deploy the Logisitic Regression model, because it is easier to interpret. Its coefficients shows how each weather feature influences the prediction clearly. It requires less memory and can store the model coefficients, 
# but the KNN model needs to retain training data, and also compare new observations to neighbors during prediction. 
# Although the KNN model resulted in a higher accuracy it wasn't by much. So the added interpretability and efficiency of the Logisitic Regression model makes the Logistic model the best choice to deploy. 


# --- Task 6: Save the Model --- # 
import os 
import json 
import joblib
import platform 
import sklearn 
from sklearn.metrics import accuracy_score, f1_score

os.makedirs("models", exist_ok=True)

#save trained model 
joblib.dump(log_model, "models/weather_classifier.pkl")

#create metadata 
metadata = {
    "python_version": platform.python_version(),
    "scikit_learn_version": sklearn.__version__,
    "city": "San Bernardino",
    "feature_names": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max"
    ],
    "label_rule": (
        "good_for_running = 1 when temperature_2m_max is between "
        "7 and 26 degrees C, temperature_2m_min >= 0 degrees C,"
        "precipitation_sum < 3 mm, and "
        "wind_speed_10m_max < 30 km/h; "
        "otherwise 0"
    ),
    "test_accuracy": float(accuracy), 
    "test_f1": float(f1)
}

#write metadats file 
with open(
    "models/weather_classifier_metadata.json", 
    "w",
    encoding="utf-8"
) as f: 
    json.dump(metadata, f, indent=4)

#Confirmation Message  
print("Model saved to models/weather_classifier.pkl")
print("Metadata saved to models/weather_classifier_metadata.json")