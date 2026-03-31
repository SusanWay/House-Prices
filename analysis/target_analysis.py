from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def analyze_numeric_feature(
    series: pd.Series,
    feature_name: Optional[str] = None,
    bins: int = 30,
) -> pd.DataFrame:

    plt.style.use("default")  # белый фон

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

    # 1 Гистограмма
    axes[0].hist(clean_series, bins=bins)
    axes[0].axvline(clean_series.mean(), linestyle="--", label="mean")
    axes[0].axvline(clean_series.median(), linestyle="-", label="median")
    axes[0].set_title(f"Распределение: {feature_name}")
    axes[0].legend()

    # 2 boxplot
    axes[1].boxplot(clean_series)  # убрали vert=False
    axes[1].axhline(q1, linestyle="--", label="Q1 (25%)")
    axes[1].axhline(q2, linestyle="-", label="Median (50%)")
    axes[1].axhline(q3, linestyle="--", label="Q3 (75%)")
    axes[1].set_title(f"Boxplot: {feature_name}")
    axes[1].set_ylabel(feature_name)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return stats_df