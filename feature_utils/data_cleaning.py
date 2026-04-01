import pandas as pd
from typing import Iterable


# ======================
# Missing values
# ======================

def fill_column(
    df: pd.DataFrame,
    column: str,
    strategy: str = "median",
) -> pd.DataFrame:

    if column not in df.columns:
        return df

    if df[column].isna().sum() == 0:
        return df

    if strategy == "mean":
        value = df[column].mean()

    elif strategy == "median":
        value = df[column].median()

    else:
        raise ValueError("strategy должен быть 'mean' или 'median'")

    df[column] = df[column].fillna(value)

    return df


def fill_missing_values(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    strategy: str = "median",
    inplace: bool = True,
) -> pd.DataFrame:

    if not inplace:
        df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()

    for col in columns:
        df = fill_column(df, col, strategy)

    return df


# ======================
# Outliers
# ======================

def clip_column(
    df: pd.DataFrame,
    column: str,
    quantile: float = 0.99,
) -> pd.DataFrame:
    """
    Ограничивает выбросы в одной колонке
    """

    if column not in df.columns:
        return df

    if df[column].isna().all():
        return df

    upper = df[column].quantile(quantile)

    df[column] = df[column].clip(upper=upper)

    return df


def handle_outliers(
    df: pd.DataFrame,
    columns: list[str],
    quantile: float = 0.99,
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Обрабатывает выбросы в указанных колонках
    """

    if not inplace:
        df = df.copy()

    for col in columns:
        df = clip_column(df, col, quantile)

    return df

def drop_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    errors: str = "ignore",
) -> pd.DataFrame:
    """
    Удаляет указанные столбцы из DataFrame.
    """
    return df.drop(columns=list(columns), errors=errors).copy()

import pandas as pd
from typing import Iterable


def fill_none_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Заполняет пропуски значением 'None' в указанных колонках.
    """

    if not inplace:
        df = df.copy()

    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    return df

def one_hot_encode_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    drop_original: bool = True,
    inplace: bool = True,
) -> pd.DataFrame:

    if not inplace:
        df = df.copy()

    new_columns = {}

    for col in columns:
        if col not in df.columns:
            continue

        unique_values = df[col].dropna().unique()

        for val in unique_values:
            new_col_name = f"{col}_{val}"
            new_columns[new_col_name] = (df[col] == val).astype(int)

    # создаём DataFrame сразу
    new_df = pd.DataFrame(new_columns, index=df.index)

    # объединяем одним разом
    df = pd.concat([df, new_df], axis=1)

    # удаляем оригинальные
    if drop_original:
        df = df.drop(columns=[col for col in columns if col in df.columns])

    return df