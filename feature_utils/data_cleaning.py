import pandas as pd


def fill_column(
    df: pd.DataFrame,
    column: str,
    strategy: str = "median",
) -> pd.DataFrame:
    """
    Заполняет пропуски в одной колонке.
    """
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
    """
    Заполняет пропуски в указанных колонках.
    Возвращает pd.DataFrame
    """
    if not inplace:
        df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()

    for col in columns:
        df = fill_column(df, col, strategy)

    return df