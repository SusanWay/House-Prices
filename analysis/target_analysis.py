from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def analyze_numeric_feature(
    series: pd.Series,
    feature_name: Optional[str] = None,
    bins: int = 30,
) -> pd.DataFrame:
    """
    Анализ числового признака.
    """
    plt.style.use("default")

    if not isinstance(series, pd.Series):
        raise TypeError("Нужно передать pandas.Series")

    clean_series = series.dropna()

    if feature_name is None:
        feature_name = series.name if series.name else "feature"

    q1 = clean_series.quantile(0.25)
    q2 = clean_series.quantile(0.50)
    q3 = clean_series.quantile(0.75)

    stats_df = pd.DataFrame(
        {
            "stat": [
                "count",
                "mean",
                "median",
                "std",
                "min",
                "q1_25%",
                "q2_50%",
                "q3_75%",
                "max",
            ],
            "value": [
                clean_series.count(),
                clean_series.mean(),
                clean_series.median(),
                clean_series.std(),
                clean_series.min(),
                q1,
                q2,
                q3,
                clean_series.max(),
            ],
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Гистограмма
    axes[0].hist(clean_series, bins=bins)
    axes[0].axvline(clean_series.mean(), linestyle="--", label="mean")
    axes[0].axvline(clean_series.median(), linestyle="-", label="median")
    axes[0].set_title(f"Распределение: {feature_name}")
    axes[0].legend()

    # Boxplot
    axes[1].boxplot(clean_series)
    axes[1].axhline(q1, linestyle="--", label="Q1 (25%)")
    axes[1].axhline(q2, linestyle="-", label="Median (50%)")
    axes[1].axhline(q3, linestyle="--", label="Q3 (75%)")
    axes[1].set_title(f"Boxplot: {feature_name}")
    axes[1].set_ylabel(feature_name)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return stats_df


def analyze_categorical_feature(
    series: pd.Series,
    feature_name: Optional[str] = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Анализ категориального / строкового признака.
    """
    plt.style.use("default")

    if not isinstance(series, pd.Series):
        raise TypeError("Нужно передать pandas.Series")

    if feature_name is None:
        feature_name = series.name if series.name else "feature"

    total_count = len(series)
    missing_count = series.isna().sum()
    missing_percent = series.isna().mean()

    clean_series = series.dropna().astype(str)

    unique_count = clean_series.nunique()

    value_counts = clean_series.value_counts()
    top_categories = value_counts.head(top_n)

    most_frequent_value = value_counts.index[0] if not value_counts.empty else None
    most_frequent_count = value_counts.iloc[0] if not value_counts.empty else 0
    most_frequent_percent = (
        most_frequent_count / len(clean_series) if len(clean_series) > 0 else 0
    )

    rare_categories_count = (value_counts == 1).sum()
    rare_categories_percent = (
        rare_categories_count / unique_count if unique_count > 0 else 0
    )

    is_binary = unique_count == 2
    is_high_cardinality = unique_count > 20

    stats_df = pd.DataFrame(
        {
            "stat": [
                "count",
                "missing_count",
                "missing_%",
                "unique_count",
                "unique_%",
                "most_frequent_value",
                "most_frequent_count",
                "most_frequent_%",
                "rare_categories_count",
                "rare_categories_%",
                "is_binary",
                "is_high_cardinality",
            ],
            "value": [
                total_count,
                missing_count,
                missing_percent,
                unique_count,
                unique_count / total_count if total_count > 0 else 0,
                most_frequent_value,
                most_frequent_count,
                most_frequent_percent,
                rare_categories_count,
                rare_categories_percent,
                is_binary,
                is_high_cardinality,
            ],
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Топ категорий
    axes[0].bar(top_categories.index, top_categories.values)
    axes[0].set_title(f"Топ-{top_n} категорий: {feature_name}")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=45)

    # Доли
    if top_categories.sum() > 0:
        top_shares = top_categories / top_categories.sum()
    else:
        top_shares = top_categories

    axes[1].bar(top_shares.index, top_shares.values)
    axes[1].set_title(f"Доли топ-{top_n} категорий: {feature_name}")
    axes[1].set_ylabel("Share")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()

    return stats_df