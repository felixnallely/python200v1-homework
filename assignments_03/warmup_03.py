import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
import joblib

df = pd.read_csv("weather_classification.csv")
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Preprocessing Question 1 --- # 
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# convert to dataframe
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print(X_train_scaled.mean())

##Comment explaining, in one sentence, why you fit the scaler on X_train only.
# Scaler is fit only on X_train to prevent any information from a test set from leaking into the training process. 

# --- Preprocessing Question 2 --- # 
season_data = df[["season"]]

#encoder 
encoder = OneHotEncoder(sparse_output=False)

#fit & transform 
season_encoded = encoder.fit_transform(season_data)

print("Encoded shape:")
print(season_encoded.shape)

#Comment: how many columns did the single season column become, and why is that number what it is?
# The single season column became one column for each season. This is because one-hot encoding creates a separate binary column when it finds an orignial feature. 

print("Feature names:")
print(encoder.get_feature_names_out())

# --- Preprocessing Question 3 --- # 
features = [
    "temperature_2m_max", 
    "temperature_2m_min", 
    "precipitation_sum", 
    "wind_speed_10m_max",
    "season"
]

X = df[features]
y = df["good_for_running"]

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Build preprocessing 
preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), numeric), ("cat", OneHotEncoder(sparse_output=False), ["season"])])

#Build pipeline
pipeline = Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression())])

#Train pipeline
pipeline.fit(X_train, y_train)

#Predictions 
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Test Accuracy:")
print(accuracy)

#comment: what does the pipeline do automatically that you would otherwise have to do by hand?
# The pipeline automatically applies both scaling and one-hot encoding. It does this to correct the 
# columns before training and prediction happens. 

# --- KNN Question 1 --- # 
#KNN pipeline 
numeric = [
    "temperature_2m_max", 
    "temperature_2m_min", 
    "precipitation_sum", 
    "wind_speed_10m_max",
]

X = df[numeric]
y = df["good_for_running"]

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


knn_pipeline = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=5))])

#train model
knn_pipeline.fit(X_train, y_train)

y_pred = knn_pipeline.predict(X_test)

#accuracy
accuracy = accuracy_score(y_test, y_pred)

print("KNN Test Accuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --- KNN Question 2 --- # 
k_values = [1, 3, 5, 7, 9, 11, 15, 21]

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
        scoring="accuracy")
    
    print(f"k={k}: Mean CV Accuracy = {scores.mean():.4f}")

#comment naming the k you would choose and why.
# I would choose the k value with the highest cross validation because it performed the best with 
# unseen 4 folds of the training data. So I would choose k=11 because it has the highest mean. 

# --- KNN Question 3 --- # 
#knn with no scaling
knn_raw = KNeighborsClassifier(n_neighbors=5)

knn_raw.fit(X_train, y_train)

y_pred_raw = knn_raw.predict(X_test)

raw_accuracy = accuracy_score(y_test, y_pred_raw)

print("Accuracy without scaling:")
print(raw_accuracy)

#knn with scaling 
knn_scaled = Pipeline([
    ("scaler", StandardScaler()), 
    ("knn", KNeighborsClassifier(n_neighbors=5))
    ])

knn_scaled.fit(X_train, y_train)

y_pred_scaled = knn_scaled.predict(X_test)

scaled_accuracy = accuracy_score(y_test, y_pred_scaled)

print("Accuracy with scaling:")
print(scaled_accuracy)

#comment: did scaling help, and why would scaling matter for a distance-based model?
# Yes, scaling helped, this is because KNN uses distances between points. Scaling puts the features all on 
# a similar feature so every feature can contribute the same. And with no scaling the features with larger ranges
# can affect the calculation. 

# --- Logistic Regression Question 1 --- # 
log_pipeline = Pipeline([
    ("scaler", StandardScaler()), 
    ("logreg", LogisticRegression(max_iter=1000))
    ])

#train model
log_pipeline.fit(X_train, y_train)

y_pred = log_pipeline.predict(X_test)

#Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:")
print(accuracy)
print("Precision:")
print(precision)
print("Recall:")
print(recall)
print("F1 Score:")
print(f1)


# --- Logistic Regression Question 2 --- # 
coefficients = log_pipeline.named_steps["logreg"].coef_[0]

#print feature names / coefficients together 
for feature, coefficient in zip(X_train.columns, coefficients):
    print(f"{feature}: {coefficient:.4f}")

#comment: which features push most strongly toward "skip," and does that match your intuition about running weather?
# wind_speed_10m_max and precipitation_sum push most strongly toward "skip" mainly because they both have large negative 
# coefficients. 
# Which matches my intuition because windy and raining conditions aren't the best to run in and make running less comfortable. 


# --- Logistic Regression Question 3 --- # 
probabilities = log_pipeline.predict_proba(X_test)

# "Good" probability class (class 1)
for i in range(5):
    print(f"Test Day {i+1}: {probabilities[i][1]:.2f}")
    
#comment: what does a probability near 0.5 tell you about the model's confidence for that day?
# Having a probability near 0.5 means the model is unsure/ not confident wheather the day can be classified 
# as "good" or "skip" because the day has both characteristics. 


# --- Evaluation Question 1 --- # 
#predictions 
y_pred = log_pipeline.predict(X_test)

#Confusion matrix
cm = confusion_matrix(y_test, y_pred)

#display cm
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, 
    display_labels=["skip", "good"]
    )

disp.plot(cmap="Blues")

#Save figure
plt.savefig("outputs/logreg_confusion_matrix.png")
plt.close()

print(cm)

#Extract values 
tn, fp, fn, tp = cm.ravel()

print("False Positives:", fp)
print("False Negatives:", fn)

#comment: how many false positives and how many false negatives did the model make, and which kind of error would matter more for a running app?
# The model made 9 false positives and 7 false negatives. A false positive would matter more for a running app because the app would recommend running 
# when the day isn't a good day for running. 


# --- Joblib Question 1 --- # 
#save fitted pipeline
joblib.dump(log_pipeline, "models/warmup_model.pkl")

print("Model saved.")

#load pipeline back
loaded = joblib.load("models/warmup_model.pkl")

#verify predictions match
assert (loaded.predict(X_test) == log_pipeline.predict(X_test)).all()
print("Predictions match.")

#comment: what would go wrong if you had saved only the LogisticRegression step without the scaler, then called .predict() on raw data?
# If only LogisticRegression was saved without scaler, the predictions on the raw data would be wrong because the model would have been trained with no scale.
# Which would lead to unreliable predictions. 