from typing import Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.config import load_config


def train_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = None,
) -> Dict:
    """
    Обучает модель и возвращает метрики.
    Если random_state не передан — берётся из config.
    """

    # --- берем seed из конфига, если не передан ---
    if random_state is None:
        config = load_config()
        random_state = config["seed"]

    # --- split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # --- обучение ---
    model.fit(X_train, y_train)

    # --- предсказание ---
    y_pred = model.predict(X_test)

    # --- метрики ---
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return {
        "model": model,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }