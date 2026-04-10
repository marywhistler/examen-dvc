import joblib
import pandas as pd
from sklearn.linear_model import Ridge

if __name__ == "__main__":
    X_train_scaled = pd.read_csv("data/processed/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")

    y_train = y_train.squeeze()

    best_params = joblib.load("models/best_params.pkl")

    model = Ridge(**best_params)
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, "models/trained_model.pkl")

    print("Trained model: saved in models/trained_model")