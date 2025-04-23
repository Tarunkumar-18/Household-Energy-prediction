import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the data
df = pd.read_csv('cleaned_data.csv', parse_dates=['Datetime'], index_col='Datetime')

# Feature Engineering (as you've done before)
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
df['lag_1'] = df['Global_active_power'].shift(1)
df['lag_24'] = df['Global_active_power'].shift(24)
df['rolling_3h'] = df['Global_active_power'].rolling(window=3).mean()
df['rolling_24h'] = df['Global_active_power'].rolling(window=24).mean()

df.dropna(inplace=True)

# Feature and target selection
features = ['hour', 'day_of_week', 'month', 'is_weekend', 'lag_1', 'lag_24', 'rolling_3h', 'rolling_24h']
X = df[features]
y = df['Global_active_power']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model (e.g., Random Forest or XGBoost)
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the metrics on Streamlit
st.title("Energy Consumption Prediction")
st.write(f"Root Mean Squared Error (RMSE): {rmse}")
st.write(f"Mean Absolute Error (MAE): {mae}")
st.write(f"R-Squared (R²): {r2}")

# Show actual vs predicted plot
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('Actual vs Predicted Energy Consumption')
plt.legend()
st.pyplot(plt)
