import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === Load training data ===
df = pd.read_csv("/home/ahmed/House_Prices/house-prices-advanced-regression-techniques/train.csv")

# === Fill missing values ===
df = df.fillna(df.median(numeric_only=True))  # numeric columns
df = df.fillna(df.mode().iloc[0])             # categorical columns

# === Prepare features and target ===
X = df.drop(columns=["Id", "SalePrice"])
y = df["SalePrice"]

# === Convert text to numbers ===
X = pd.get_dummies(X)

# === Split into training and validation sets ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train the model ===
model = XGBRegressor()
model.fit(X_train, y_train)

# === Predict on validation set ===
y_pred = model.predict(X_val)

# === Evaluate performance ===
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.4f}")
