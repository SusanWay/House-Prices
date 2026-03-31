import pandas as pd

def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """
    Возвращает список НЕ числовых колонок
    """
    return df.select_dtypes(exclude=["number"]).columns.tolist()