import pandas as pd
import numpy as np

from analysis.numeric.numeric_features import get_numeric_columns


def get_outlier_mask(series: pd.Series) -> pd.Series:
    """
    Возвращает маску выбросов (IQR метод)
    """
    clean = series.dropna()

    if clean.empty:
        return pd.Series([False] * len(series), index=series.index)

    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return (series < lower) | (series > upper)


def detect_outliers(series: pd.Series) -> bool:
    """
    Есть ли выбросы
    """
    return get_outlier_mask(series).any()


def get_outlier_percent(series: pd.Series) -> float:
    """
    Доля выбросов
    """
    mask = get_outlier_mask(series)
    return mask.mean()  # доля Trueg


def build_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Строит summary по численным колонкам
    """

    numeric_cols = get_numeric_columns(df)

    rows = []

    for col in numeric_cols:
        s = df[col]

        has_na = s.isnull().any()

        outlier_mask = get_outlier_mask(s)
        outlier_percent = outlier_mask.mean()
        has_outliers = outlier_percent > 0

        row = {
            "feature": col,
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(),
            "min": s.min(),
            "max": s.max(),
            "missing": has_na,
            "missing_%": s.isnull().mean(),
            "has_outliers": has_outliers,
            "outlier_%": outlier_percent,
            "skew": s.skew(),
        }

        rows.append(row)

    return pd.DataFrame(rows).sort_values("missing_%", ascending=False)