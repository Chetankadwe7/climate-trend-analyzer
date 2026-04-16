def trend_analysis(df):
    df["temp_trend"] = df["temperature"].rolling(30).mean()
    return df