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
print()
print(df.head())

# The city I chose: San Bernardino, California 


# --- Task 2: Look at Features --- #
print("--Describe:")
print(df.describe())

print("--Missing values:", df.isna().sum())

# No missing values were found so no data cleaning is necessary. 

# Correlations
correlations = (df.corr(numeric_only=True)["temperature_2m_max"].sort_values(ascending=False))
print("--Correlations:")
print(correlations)

# Which feature relates most strongly to the daily high?
# 
# temperature_2m_min has the strongest relationship with temperature_2m_max with an approximate correlation of 0.85, 
# making it the most useful predictor.


# Plot
import matplotlib.pyplot as plt 

plt.figure(figsize=(8, 6))
plt.scatter(df["temperature_2m_min"], df["temperature_2m_max"])

plt.title("Daily Low vs Daily High")
plt.xlabel("Daily Low Temp (°C)")
plt.ylabel("Daily High Temp (°C)")

plt.savefig("outputs/low_vs_high_real_weather.png")
plt.close()

# Add a comment describing what you see.
# I see that the scatter plot shows a strong positive relationship 
# between daily high temperatures and daily low temperatures.


# --- Task 3: Engineer a Feature --- # 
df["is_summer"] = df["date"].dt.month.isin([6, 7, 8]).astype(int)

print("--Summer vs Non-Summer Days:")
print(df["is_summer"].value_counts())
# Do you expect is_summer to help predict the daily high, and why?
#
# I do expect is_summer to help predict daily high temperatures since summer is 
# when the highest temperatures happen. Adding is_summer as a variable may help the model make 
# accurate predictions. 


# --- Task 4: Baseline Model --- # 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np 
X = df[["temperature_2m_min"]]
y = df[["temperature_2m_max"]]

# Train-test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model 
model = LinearRegression()
model.fit(X_train, y_train)

# Predict 
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = model.score(X_test, y_test)

print("--Task 4")
print("Slope:", model.coef_[0])
print("RMSE:", rmse)
print("R2:", r2)

# The RMSE is in degrees Celsius. In plain English, how far off is a typical prediction?
# 
# Since the RMSE result is 3.94°C, this means the model's predictions will be off about 4°C
# from the actual high temperatures reported. 



# --- Task 5: Full Model --- # 
feature_cols = ["temperature_2m_min", 
                "precipitation_sum",
                "wind_speed_10m_max", 
                "is_summer"]

X = df[feature_cols].values
y = df["temperature_2m_max"].values


# Train-test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model 
model = LinearRegression()
model.fit(X_train, y_train)

# Predict 
y_pred = model.predict(X_test)

train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("--Task 5")
print("Train R2:", train_r2)
print("Test R2:", test_r2)
print("RMSE:", rmse)

print("Coefficients:")
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:22s}:{coef:+.3f}")

# Compare the test R² to your baseline from Task 4. How much does adding features help?
# --Task 4: RMSE: 3.95 R2: 0.7118
# --Task 5: RMSE: 3.425 Train R2: 0.7978 Test R2: 0.7833
# 
# The features performed better than they did in the baseline model. Although RMSE decreased 
# from 3.95°C to 3.43°C, this means prediction errors became smaller. Therefore adding features
# helped reduce prediction errors. 


# Add a comment interpreting the coefficients. Which features raise the predicted high, 
# and which lower it? Are any signs surprising? Then compare train R² to test R². Are they 
# close, or is there a gap, and what does that tell you?
# 
# temperature_2m_min has a positive coefficient +1.064, which means 1°C increase in daily low temperature 
# predicts a high temperature increase by 1.06°C, and also keeps the other features constant. 
#
# precipitation_sum is negative -0.193°C which mean days with high precipitation are lower on high daily temperature.
# 
# wind_speed_10m_max is positive +0.057°C suggests the effects are small. 
# 
# is_summer is positive +1.643°C which means summer days are 1.643°C warmer than days that aren't summer. 
# 
# I found wind speed to to have a small coefficient suprising because there is little to no correlation to its effect on high temperature daily. 
# Train R2 and test R2 are very close and have a small gap this means the model isn't overfitting the training set. 


# --- Task 6: Evaluate  and Summarize --- # 
y_pred = model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, y_test, alpha=0.07)

# Diagonal line 
# Diagonal line 
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.min())

plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted Daily High Temperature (°C)")
plt.xlabel("Actual Daily High Temperature (°C)")

plt.savefig("outputs/predicted_vs_actual_high.png")
plt.close()


# Add a comment: does the model struggle more at the high end, the low end, or is the error 
# roughly even across the range?
# Looking at the scatter plot, most points are near the diagonal line, which shows that the model's predictions are 
# pretty accurate. The errors are distributed across the temperature ranges, and aren't really concentrated at high or
# low ends. There does seem to be more spread around the lower and middle temperatures, but the model performs consistently throughout. 

# -- Plain Language Summary -- # 
# The size of the dataset and the test set
# The RMSE and R² of your full model, and what a typical error of that size means on a temperature scale
# Which feature has the largest effect on the predicted high, and in which direction
# One result that surprised you.
#
# The dataset has a total of 365 daily weather observations after the 80/20 split.
# The train-test split was done and trained on 292 observations and evalyated on 73 observations.
# 
# The full model test RMSE achieved 3.43°C and a test R2 of 0.783. This model explains 78% of variation on daily temperature.
# And that prediction is off by about 3.43°C, which is fairly accurate for a daily weather model predictor. 
#
# temperature_2m_min has the largest effect on daily high because it had a coeficient of +1.064, holding other features constant
# with a 1°C increase on low daily temperature and about 1.06°C increase on daily high. 
# 
# The thing that suprised me was wind_speed_10m_max having a positive coefficient since its effects on daily high was close to zero.
# But still the effect of it was small compared to the results of other features. 
