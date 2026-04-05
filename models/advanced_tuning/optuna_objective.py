from typing import Dict, Optional

import numpy as np
import optuna
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from utils.config import load_config
from models.advanced_tuning.optuna_params import build_catboost_params


def objective_catboost(
    trial: optuna.trial.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: Optional[int] = None,
) -> float:
    """
    Одна попытка Optuna для CatBoost.

    Возвращает средний финальный RMSE по KFold.
    Дополнительно сохраняет:
    - mean_best_rmse
    - mean_best_iteration
    - best_iteration_list
    """

    config = load_config()

    if n_splits is None:
        n_splits = config["validation"]["n_splits"]

    params: Dict = build_catboost_params(trial)

    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=config["seed"],
    )

    final_rmse_list = []
    best_rmse_list = []
    best_iteration_list = []

    for train_idx, valid_idx in cv.split(X):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]

        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        model = CatBoostRegressor(
            **params,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            verbose=False,
        )

        y_pred = model.predict(X_valid)
        final_rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
        final_rmse_list.append(float(final_rmse))

        evals_result = model.get_evals_result()
        valid_rmse_history = evals_result["validation"]["RMSE"]

        best_idx = int(np.argmin(valid_rmse_history))
        best_rmse = float(valid_rmse_history[best_idx])
        best_iteration = best_idx + 1

        best_rmse_list.append(best_rmse)
        best_iteration_list.append(best_iteration)

    mean_rmse = float(np.mean(final_rmse_list))
    mean_best_rmse = float(np.mean(best_rmse_list))
    mean_best_iteration = float(np.mean(best_iteration_list))

    trial.set_user_attr("mean_best_rmse", mean_best_rmse)
    trial.set_user_attr("mean_best_iteration", mean_best_iteration)
    trial.set_user_attr("best_iteration_list", best_iteration_list)

    return mean_rmse