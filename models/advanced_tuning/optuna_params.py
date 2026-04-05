from typing import Any, Dict

import optuna

from utils.config import load_config
from models.advanced_tuning.search_spaces import get_catboost_search_space


def build_catboost_params(trial: optuna.trial.Trial) -> Dict[str, Any]:
    """
    Собирает параметры CatBoost для одного trial Optuna.
    """

    config = load_config()
    space = get_catboost_search_space()

    return {
        "iterations": trial.suggest_int(
            "iterations",
            min(space["iterations"]),
            max(space["iterations"]),
            step=10,
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            min(space["learning_rate"]),
            max(space["learning_rate"]),
        ),
        "depth": trial.suggest_int(
            "depth",
            min(space["depth"]),
            max(space["depth"]),
        ),
        "l2_leaf_reg": trial.suggest_int(
            "l2_leaf_reg",
            min(space["l2_leaf_reg"]),
            max(space["l2_leaf_reg"]),
        ),
        "random_strength": trial.suggest_int(
            "random_strength",
            min(space["random_strength"]),
            max(space["random_strength"]),
        ),
        "subsample": trial.suggest_categorical(
            "subsample",
            space["subsample"],
        ),
        "loss_function": space["loss_function"],
        "verbose": space["verbose"],
        "random_state": config["seed"],
        "bootstrap_type": space["bootstrap_type"],
        "early_stopping_rounds": space["early_stopping_rounds"],
        "use_best_model": space["use_best_model"],
    }