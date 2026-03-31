import pandas as pd


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """
    Возвращает список численных колонок
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return numeric_cols