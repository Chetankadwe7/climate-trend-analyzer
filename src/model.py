from sklearn.linear_model import LinearRegression
import numpy as np
import joblib

def train_model(df):
    X = np.arange(len(df)).reshape(-1,1)
    y = df["temperature"].values

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, "models/model.pkl")

    pred = model.predict(X)
    return pred