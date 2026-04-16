import pandas as pd
import numpy as np

def generate_data():
    np.random.seed(42)

    # Date range
    dates = pd.date_range(start="2010-01-01", periods=1000)

    # Generate data
    temperature = 25 + 0.01*np.arange(1000) + np.random.normal(0, 2, 1000)
    rainfall = 100 + np.random.normal(0, 20, 1000)   # 🌧️ RAINFALL
    co2 = 380 + 0.05*np.arange(1000) + np.random.normal(0, 5, 1000)

    # Create DataFrame
    df = pd.DataFrame({
        "date": dates,
        "temperature": temperature,
        "rainfall": rainfall,   # 🌧️ IMPORTANT
        "co2": co2
    })

    # Save to CSV
    df.to_csv("data/climate_data.csv", index=False)

    print("Data generated with rainfall ✅")
    return df