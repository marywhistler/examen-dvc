import json
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

if __name__ == "__main__":
    X_test_scaled = pd.read_csv("data/processed/X_test_scaled.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")

    y_test = y_test.squeeze()

    model = joblib.load("models/trained_model.pkl")

    y_pred = model.predict(X_test_scaled)

    predictions = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred
    })

    predictions.to_csv("data/processed/test_predictions.csv", index=False)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    scores = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
    }

    with open("metrics/scores.json", "w") as f:
        json.dump(scores, f, indent=4)

    print("Predictions: saved in data/processed/test_predictions.csv")
    print("Metrics: saved in metrics/scores.json")