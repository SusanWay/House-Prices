import pandas as pd

from analysis.categorical.categorical_features import get_categorical_columns


def build_categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Строит summary по категориальным колонкам
    """

    categorical_cols = get_categorical_columns(df)

    rows = []

    for col in categorical_cols:
        s = df[col]

        total_count = len(s)
        missing = s.isnull().any()
        missing_percent = s.isnull().mean()

        clean = s.dropna().astype(str)

        unique_count = clean.nunique()
        unique_percent = unique_count / total_count if total_count > 0 else 0

        value_counts = clean.value_counts()

        top_value = value_counts.index[0] if not value_counts.empty else None
        top_count = value_counts.iloc[0] if not value_counts.empty else 0
        top_percent = (
            top_count / len(clean) if len(clean) > 0 else 0
        )

        rare_count = (value_counts == 1).sum()
        rare_percent = (
            rare_count / unique_count if unique_count > 0 else 0
        )

        row = {
            "feature": col,
            "missing": missing,
            "missing_%": missing_percent,
            "unique_count": unique_count,
            "unique_%": unique_percent,
            "top_value": top_value,
            "top_value_%": top_percent,
            "rare_count": rare_count,
            "rare_%": rare_percent,
            "is_binary": unique_count == 2,
            "is_high_cardinality": unique_count > 20,
        }

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["missing_%", "unique_count"],
        ascending=[False, False],
    )