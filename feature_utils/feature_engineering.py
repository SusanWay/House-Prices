# feature_engineering.py

import pandas as pd
from typing import List


def add_has_large_features(
    df: pd.DataFrame,
    features: List[str],
    quantile: float = 0.9,
    prefix: str = "Has",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Добавляет бинарные признаки вида HasLarge_<feature>.
    """

    if not inplace:
        df = df.copy()

    for col in features:
        threshold = df[col].quantile(quantile)

        if threshold == 0:
            df[f"{prefix}_{col}"] = (df[col] > 0).astype(int)
        else:
            df[f"{prefix}_{col}"] = (df[col] > threshold).astype(int)

    return df


def replace_with_has_large(
    df: pd.DataFrame,
    features: List[str],
    quantile: float = 0.9,
    prefix: str = "HasLarge",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Заменяет признаки:
    - создаёт HasLarge_<feature>
    - удаляет оригинальные колонки
    """

    if not inplace:
        df = df.copy()

    for col in features:
        threshold = df[col].quantile(quantile)

        if threshold == 0:
            df[f"{prefix}_{col}"] = (df[col] > 0).astype(int)
        else:
            df[f"{prefix}_{col}"] = (df[col] > threshold).astype(int)

    # удаляем оригинальные колонки
    df.drop(columns=features, inplace=True)

    return df