import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("/home/ahmed/House_Prices/house-prices-advanced-regression-techniques/train.csv")

df = df.fillna(df.median(numeric_only=True))  
df = df.fillna(df.mode().iloc[0])             


X = df.drop(columns=["Id", "SalePrice"])
y = df["SalePrice"]


X = pd.get_dummies(X)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = XGBRegressor()
model.fit(X_train, y_train)


y_pred = model.predict(X_val)


rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.4f}")
