from src.data_loader import generate_data
from src.preprocessing import clean_data
from src.features import create_features
from src.analysis import trend_analysis
from src.anomaly import detect_anomalies
from src.model import train_model

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ---------------------------
# CREATE FOLDERS (IMPORTANT)
# ---------------------------
os.makedirs("images", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("START RUNNING 🚀")

# ---------------------------
# Step 1: Load or Generate Data
# ---------------------------
if os.path.exists("data/climate_data.csv"):
    df = pd.read_csv("data/climate_data.csv")
    print("Loaded existing CSV")
else:
    df = generate_data()
    print("Generated new data")

# ---------------------------
# Step 2: Clean Data
# ---------------------------
df = clean_data(df)

print("Columns:", df.columns)
print("Rainfall data:", df["rainfall"].head())

# ---------------------------
# Step 3: Feature Engineering
# ---------------------------
df = create_features(df)

# ---------------------------
# Step 4: Trend Analysis
# ---------------------------
df = trend_analysis(df)

# ---------------------------
# Step 5: Anomaly Detection
# ---------------------------
df = detect_anomalies(df)

# ---------------------------
# Step 6: Train Model & Forecast
# ---------------------------
pred = train_model(df)

# ---------------------------
# Step 7: Save Outputs
# ---------------------------
df.to_csv("outputs/trends.csv", index=False)

forecast_df = pd.DataFrame({
    "Actual": df["temperature"],
    "Predicted": pred
})
forecast_df.to_csv("outputs/forecast.csv", index=False)

df[df["anomaly"] == True].to_csv("outputs/anomalies.csv", index=False)

print("Outputs saved ✅")

# ---------------------------
# Step 8: Visualization
# ---------------------------

# 🔮 Forecast
plt.figure()
plt.plot(df["temperature"], label="Actual")
plt.plot(pred, label="Predicted")
plt.legend()
plt.title("Climate Forecast")
plt.savefig("images/forecast_plot.png")
plt.show()

# 📈 Temperature Trend
plt.figure()
plt.plot(df["temp_trend"])
plt.title("Temperature Trend")
plt.savefig("images/temperature_trend.png")
plt.show()


# 🌧️ Rainfall Trend (FINAL FIX)
plt.figure()
plt.plot(df["rainfall"], label="Rainfall")
plt.legend()
plt.title("Rainfall Trend")

# FORCE SAVE (same folder)
plt.savefig("rainfall_trend.png")

print("Rainfall graph saved in root folder ✅")

plt.show()
# 🚨 Anomaly
plt.figure()
plt.plot(df["temperature"], label="Temperature")
plt.scatter(df.index[df["anomaly"]],
            df["temperature"][df["anomaly"]],
            color="red", label="Anomaly")
plt.legend()
plt.title("Anomaly Detection")
plt.savefig("images/anomaly_plot.png")
plt.show()

print("ALL DONE ✅")