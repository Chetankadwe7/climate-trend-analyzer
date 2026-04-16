def create_features(df):
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["temp_rolling"] = df["temperature"].rolling(7).mean()
    return df