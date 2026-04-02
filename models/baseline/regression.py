from typing import Dict
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.dummy import DummyRegressor

from models.train import train_model


def get_baseline_models() -> Dict:
    """
    Набор простых моделей для регрессии
    """
    return {
        "Dummy": DummyRegressor(strategy="median"),
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1, max_iter=100000),
    }


def run_baseline_regression(X, y) -> pd.DataFrame:
    """
    Запускает baseline модели и возвращает таблицу метрик
    """

    models = get_baseline_models()
    results = []

    for name, model in models.items():
        res = train_model(model, X, y)

        results.append({
            "model": name,
            "MAE": res["MAE"],
            "MSE": res["MSE"],
            "RMSE": res["RMSE"],
            "R2": res["R2"],
        })

    results_df = pd.DataFrame(results)

    # сортируем по RMSE (меньше = лучше)
    results_df = results_df.sort_values(by="RMSE").reset_index(drop=True)

    return results_df