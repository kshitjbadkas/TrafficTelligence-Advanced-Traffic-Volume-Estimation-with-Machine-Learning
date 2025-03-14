import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("traffic_data.csv")

# Preprocessing
df.dropna(inplace=True)

# Extract hour from "Time" column
df['hour'] = pd.to_datetime(df['Time']).dt.hour  # Extract hour (0-23)

# Encoding categorical variable (Weather Condition)
label_encoder = LabelEncoder()
df['weather'] = label_encoder.fit_transform(df['weather'])

# Selecting features and target
X = df[['hour', 'temp', 'weather']]  # Use "temp" instead of "temperature"
y = df['traffic_volume']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scalers
with open("traffic_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

with open("label_encoder.pkl", "wb") as le_file:
    pickle.dump(label_encoder, le_file)

print("Model training completed and saved.")
