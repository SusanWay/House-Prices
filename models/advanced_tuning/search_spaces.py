from typing import Any, Dict, List

import numpy as np


def get_catboost_search_space() -> Dict[str, List[Any]]:
    """
    Возвращает диапазоны параметров для подбора CatBoost через Optuna.

    Это не grid для полного перебора,
    а просто описание допустимых диапазонов.
    """

    return {
        "learning_rate": np.arange(0.01, 0.11, 0.01).round(2).tolist(),
        "depth": list(range(2, 10)),
        "l2_leaf_reg": list(range(1, 10)),
        "random_strength": list(range(1, 6)),
        "subsample": [0.8, 1.0],
        "iterations": list(range(50, 1001, 10)),
        "loss_function": "RMSE",
        "verbose": 0,
        "bootstrap_type": "Bernoulli",
        "early_stopping_rounds": 50,
        "use_best_model": True,
    }