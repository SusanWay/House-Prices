import pandas as pd
import shap
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor


def build_best_catboost_model(random_state: int = 42) -> CatBoostRegressor:
    """
    Возвращает CatBoostRegressor с лучшими найденными параметрами.
    Параметры взяты из Optuna.
    """
    model = CatBoostRegressor(
        iterations=950,
        learning_rate=0.05725960671224448,
        depth=4,
        l2_leaf_reg=9,
        random_strength=3,
        subsample=0.8,
        loss_function="RMSE",
        bootstrap_type="Bernoulli",
        verbose=0,
        random_state=random_state,
    )
    return model


def fit_best_catboost_for_shap(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> CatBoostRegressor:
    """
    Обучает лучшую модель CatBoost на уже подготовленном датасете.
    """
    model = build_best_catboost_model(random_state=random_state)
    model.fit(X, y, verbose=False)
    return model


def get_shap_values(
    model: CatBoostRegressor,
    X: pd.DataFrame,
):
    """
    Вычисляет SHAP values для модели и датасета.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values


def get_shap_importance_df(
    model: CatBoostRegressor,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Возвращает DataFrame с важностью признаков
    по среднему абсолютному SHAP.
    """
    shap_values = get_shap_values(model, X)

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": abs(shap_values).mean(axis=0),
    })

    importance_df = (
        importance_df
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    return importance_df


def plot_shap_summary(
    model: CatBoostRegressor,
    X: pd.DataFrame,
) -> None:
    """
    Рисует SHAP summary plot.
    """
    shap_values = get_shap_values(model, X)
    shap.summary_plot(shap_values, X)


def plot_shap_bar(
    model: CatBoostRegressor,
    X: pd.DataFrame,
) -> None:
    """
    Рисует bar plot важности признаков по SHAP.
    """
    shap_values = get_shap_values(model, X)
    shap.summary_plot(shap_values, X, plot_type="bar")


def plot_top_shap_features(
    model: CatBoostRegressor,
    X: pd.DataFrame,
    top_n: int = 20,
) -> None:
    """
    Рисует top-N признаков по среднему абсолютному SHAP.
    """
    importance_df = get_shap_importance_df(model, X).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(
        importance_df["feature"][::-1],
        importance_df["mean_abs_shap"][::-1],
    )
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} features by SHAP importance")
    plt.tight_layout()
    plt.show()