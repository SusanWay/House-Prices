from typing import Dict, Optional

import pandas as pd
from tqdm.auto import tqdm

from sklearn.ensemble import (
    BaggingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from models.advanced.metrics import collect_regression_metrics
from models.advanced.training_history import fit_history_model
from utils.config import load_config


def get_advanced_models(config: Optional[dict] = None) -> Dict[str, object]:
    """
    Все advanced модели
    """

    if config is None:
        config = load_config()

    seed = config["seed"]
    learning_rate = config["learning_rate"]
    n_estimators = config["n_estimators"]

    return {
        "Bagging": BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=10, random_state=seed),
            n_estimators=100,
            random_state=seed,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=12,
            random_state=seed,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=12,
            random_state=seed,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=3,
            random_state=seed,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            eval_metric="rmse",
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=6,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=n_estimators,
            learning_rate=learning_rate,
            depth=6,
            random_state=seed,
            loss_function="RMSE",
            verbose=0,
        ),
    }


def get_history_models(config: Optional[dict] = None) -> Dict[str, object]:
    """
    Модели с history
    """

    models = get_advanced_models(config)

    return {
        "XGBoost": models["XGBoost"],
        "LightGBM": models["LightGBM"],
        "CatBoost": models["CatBoost"],
    }


def collect_training_history(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    config: Optional[dict] = None,
    show_progress: bool = True,
) -> Dict[str, Dict]:
    """
    Собирает history для бустингов
    """

    if config is None:
        config = load_config()

    seed = config["seed"]
    early_stopping_rounds = config["early_stopping_rounds"]

    models = get_history_models(config)
    history = {}

    iterator = models.items()

    if show_progress:
        iterator = tqdm(
            iterator,
            total=len(models),
            desc="Collecting training history",
        )

    for name, model in iterator:
        history[name] = fit_history_model(
            model=model,
            X=X,
            y=y,
            test_size=test_size,
            random_state=seed,
            early_stopping_rounds=early_stopping_rounds,
        )

    return history


def run_advanced_regression(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    with_history: bool = True,
    config: Optional[dict] = None,
) -> Dict[str, object]:
    """
    Считает метрики и history
    """

    if config is None:
        config = load_config()

    seed = config["seed"]
    models = get_advanced_models(config)

    metrics_df = collect_regression_metrics(
        models=models,
        X=X,
        y=y,
        test_size=test_size,
        random_state=seed,
        show_progress=True,
    )

    history = {}

    if with_history:
        history = collect_training_history(
            X=X,
            y=y,
            test_size=test_size,
            config=config,
            show_progress=True,
        )

    return {
        "metrics": metrics_df,
        "training_history": history,
    }