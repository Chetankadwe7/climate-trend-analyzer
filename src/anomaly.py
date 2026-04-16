def detect_anomalies(df):
    mean = df["temperature"].mean()
    std = df["temperature"].std()

    df["anomaly"] = df["temperature"] > (mean + 2*std)
    return df