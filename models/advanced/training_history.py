from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split

from utils.config import load_config


def get_xgb_history(model) -> tuple[list[float], list[float]]:
    """
    История XGBoost
    """

    result = model.evals_result()
    return result["validation_0"]["rmse"], result["validation_1"]["rmse"]


def get_lgbm_history(model) -> tuple[list[float], list[float]]:
    """
    История LightGBM
    """

    result = model.evals_result_
    return result["training"]["rmse"], result["valid_1"]["rmse"]


def get_catboost_history(model) -> tuple[list[float], list[float]]:
    """
    История CatBoost
    """

    result = model.get_evals_result()
    return result["learn"]["RMSE"], result["validation"]["RMSE"]


def get_model_history(model) -> tuple[list[float], list[float]]:
    """
    Возвращает историю модели
    """

    match model.__class__.__name__:
        case "XGBRegressor":
            return get_xgb_history(model)
        case "LGBMRegressor":
            return get_lgbm_history(model)
        case "CatBoostRegressor":
            return get_catboost_history(model)
        case _:
            raise ValueError(
                f"Модель {model.__class__.__name__} не поддерживает history"
            )


def fit_history_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    early_stopping_rounds: Optional[int] = None,
) -> Dict:
    """
    Обучает модель и собирает history
    """

    config = load_config()

    if random_state is None:
        random_state = config["seed"]

    if early_stopping_rounds is None:
        early_stopping_rounds = config["early_stopping_rounds"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    match model.__class__.__name__:
        case "XGBRegressor":
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                verbose=False,
            )

        case "LGBMRegressor":
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                eval_metric="rmse",
            )

        case "CatBoostRegressor":
            model.fit(
                X_train,
                y_train,
                eval_set=(X_valid, y_valid),
                use_best_model=True,
                early_stopping_rounds=early_stopping_rounds,
                verbose=False,
            )

        case _:
            raise ValueError(
                f"Модель {model.__class__.__name__} не поддерживает history"
            )

    train_scores, valid_scores = get_model_history(model)

    best_idx = min(range(len(valid_scores)), key=lambda i: valid_scores[i])

    return {
        "model": model,
        "train_scores": train_scores,
        "valid_scores": valid_scores,
        "best_iteration": best_idx + 1,
        "best_valid_score": valid_scores[best_idx],
    }


def show_training_history(
    history: Dict,
    model_name: str,
    figsize: tuple = (10, 6),
    start_iter: int = 1,
):
    """
    Показывает один график (читаемый)
    """

    import matplotlib.pyplot as plt

    # --- фикс темы ---
    plt.style.use("default")

    train = history["train_scores"]
    valid = history["valid_scores"]

    iterations = list(range(1, len(train) + 1))

    if start_iter > 1:
        train = train[start_iter - 1:]
        valid = valid[start_iter - 1:]
        iterations = iterations[start_iter - 1:]

    best_idx = min(range(len(valid)), key=lambda i: valid[i])
    best_iter = iterations[best_idx]
    best_score = valid[best_idx]

    fig, ax = plt.subplots(figsize=figsize)

    # --- белый фон ---
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # --- линии ---
    ax.plot(iterations, train, label="Train", linewidth=2)
    ax.plot(iterations, valid, label="Valid", linewidth=2)

    # --- точка ---
    ax.scatter(
        best_iter,
        best_score,
        s=80,
        zorder=5,
    )

    # --- popup ---
    ax.annotate(
        f"RMSE = {best_score:.4f}\niter = {best_iter}",
        xy=(best_iter, best_score),
        xytext=(30, -30),
        textcoords="offset points",
        bbox=dict(
            boxstyle="round,pad=0.3",
            fc="white",
            ec="black",
            lw=1,
        ),
        arrowprops=dict(
            arrowstyle="->",
            color="black",
        ),
        zorder=6,
    )

    ax.set_title(f"{model_name}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMSE")

    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()