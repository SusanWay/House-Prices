from typing import Dict, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from utils.config import load_config


def evaluate_regression_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Dict:
    """
    Обучает модель и считает метрики
    """

    if random_state is None:
        random_state = load_config()["seed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_test, y_pred),
    }


def collect_regression_metrics(
    models: Dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Собирает таблицу метрик
    """

    if random_state is None:
        random_state = load_config()["seed"]

    rows = []
    iterator = models.items()

    if show_progress:
        iterator = tqdm(
            iterator,
            total=len(models),
            desc="Collecting metrics",
        )

    for name, model in iterator:
        metric = evaluate_regression_model(
            model=model,
            X=X,
            y=y,
            test_size=test_size,
            random_state=random_state,
        )

        rows.append({
            "model": name,
            "MAE": metric["MAE"],
            "MSE": metric["MSE"],
            "RMSE": metric["RMSE"],
            "R2": metric["R2"],
        })

    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)