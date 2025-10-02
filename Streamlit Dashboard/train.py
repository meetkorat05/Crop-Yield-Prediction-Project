# train.py
from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd
from xgboost import XGBRegressor

# Load your training dataset
df = pd.read_csv("crop_yield.csv")

# Initialize encoders
crop_encoder = LabelEncoder()
season_encoder = LabelEncoder()
state_encoder = LabelEncoder()

# Encode categorical features
df["Crop"] = crop_encoder.fit_transform(df["Crop"])
df["Season"] = season_encoder.fit_transform(df["Season"])
df["State"] = state_encoder.fit_transform(df["State"])

# Define features and target
X = df.drop("Yield", axis=1)
y = df["Yield"]

# Train model
model = XGBRegressor()
model.fit(X, y)

# Save model and encoders
joblib.dump(model, "xgb_model.pkl")
joblib.dump(crop_encoder, "crop_encoder.pkl")
joblib.dump(season_encoder, "season_encoder.pkl")
joblib.dump(state_encoder, "state_encoder.pkl")
