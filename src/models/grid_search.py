import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":
    X_train_scaled = pd.read_csv("data/processed/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")

    y_train = y_train.squeeze()

    param_grid = {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        "fit_intercept": [True, False]
    }

    model = Ridge()

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    grid.fit(X_train_scaled, y_train)

    best_params = grid.best_params_

    joblib.dump(best_params, "models/best_params.pkl")

    print("Best parameters: saved in models/best_params")
    print(best_params)