import pandas as pd

def clean_data(df):
    df = df.dropna()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date")
    return df