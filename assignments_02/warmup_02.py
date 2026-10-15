# --- scikit-learn API --- Question 1 --- # 
import numpy as np
from sklearn.linear_model import LinearRegression

temp_min = np.array([1, 4, 8, 12, 16, 20]).reshape(-1, 1)
temp_max = np.array([8, 10, 15, 18, 23, 27])

#Create,fit model 
model = LinearRegression()
model.fit(temp_min, temp_max)

#Predictions using low 6° and ow 18°
predicted_6 = model.predict([[6]])[0]
predicted_18 = model.predict([[18]])[0]

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted high for low of 6°C:", predicted_6)
print("Predicted high for low of 18°C:", predicted_18)

# --- scikit-learn API --- Question 2 --- # 
x = np.array([10, 20, 30, 40, 50])

print("Original Shape:", x.shape)

x_2d = x.reshape(-1, 1)
print("Reshaped:", x_2d.shape)

# Scikit-learn needs x to be 2d because it needs to know how many features are 
# being used so each 2d represents a sample and each column represents a feature.

# --- Linear Regression --- Question 1 --- # 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

rng = np.random.default_rng(42)
num_days  = 120
temp_min  = rng.uniform(-5, 22, num_days)
is_rainy  = rng.integers(0, 2, num_days).astype(float)
temp_max  = 1.0 * temp_min + 6 - 4 * is_rainy + rng.normal(0, 2, num_days)

#Scatter plot before fixing anything
plt.figure(figsize=(8, 6))
plt.scatter(temp_min, temp_max, c=is_rainy, cmap="coolwarm")

plt.title("Daily High vs Low")
plt.xlabel("Daily Low Temp in °C")
plt.ylabel("Daily High Temp in °C")

plt.savefig("outputs/high_vs_low.png")
plt.close

# Add a comment describing what you see. Are there two visible groups? What does that suggest about the is_rainy variable?
# 
# The scatter plot shows a positive relationship between temp_min and temp_high. The points show a group that run parallel to each other. 
# This suggests that is_rainy has an effect on temp-max. Since rainy days tend to have lower but still high temperatures than on dry days. 


# --- Linear Regression --- Question 2 --- # 
X = temp_min.reshape(-1, 1)
y = temp_max

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# --- Linear Regression --- Question 3 --- # 
model = LinearRegression()
model.fit(X_train, y_train)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Test predictions on test 
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = model.score(X_test, y_test)

print("RMSE:", rmse)
print("R2 on test set:", r2)

# The slope represents how much the predicted high temperature 
# changes for every 1°C. The slope is associated with 
# every daily low increase as well with the daily increase in daily high temperature.


# --- Linear Regression --- Question 4 --- # 
X_full = np.column_stack([temp_min, is_rainy])

# Split data 
X_train, X_test, y_train, y_test = train_test_split(X_full, temp_max, test_size=0.2, random_state=42)

# Fit model 
model_full = LinearRegression()
model_full.fit(X_train, y_train)

r2_full = model_full.score(X_test, y_test)

print("R2 with temp_min only:", r2)
print("R2 with temp_min and is_rainy:", r2_full)

print("temp_min coefficient:", model_full.coef_[0])
print("is_rainy coefficient:", model_full.coef_[1])

# Adding the rain flag helps make more predictions in temperature when the weather is rainy instead of 
# just focusing on dry weather. It allows us to get more explainations on daily temperature that temp_min 
# alone doesn't explain.
#
# is_rainy coefficient represents the predicted change in temperature for the daily high temperature
# when it's rainy instead of dry. If the result is a negative coefficient this means on rainy days the high daily temperatures
# tend to be lower when it rains than on dry days. The is_rainy coefficient is -2.9, which means that high daily temperature is usually
# -2.9°C lower than it would be on dry days. 


# --- Linear Regression --- Question 5 --- # 
y_pred = model_full.predict(X_test)

# predicted vs actual plot 
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, y_test)

# Diagonal line 
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.min())

plt.plot([min_val, max_val], [min_val, max_val], "r--")

plt.title("Predicted vs Actual")
plt.xlabel("Predicted High Temperature")
plt.xlabel("Actual high Temperature")

plt.savefig("outputs/predicted_vs_actual_warmup.png")
plt.close()


# what does it mean when a point falls above the diagonal? What about below?
# If a point falls above the diagonal line then the actual value was higher than the predicted which means the model underpredicted. 
# But if the actual value is under the diagonal line then the model overpredicted because the actual value was actually higher than the predicted. 