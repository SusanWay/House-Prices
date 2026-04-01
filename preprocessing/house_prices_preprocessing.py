import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from feature_utils.feature_engineering import replace_with_has_large
from feature_utils.data_cleaning import (
    fill_missing_values,
    handle_outliers,
    drop_columns,
    fill_none_columns,
    one_hot_encode_columns,
)


class HousePricesPreprocessor:

    FEATURES_SPARSE = [
        "MiscVal",
        "PoolArea",
        "3SsnPorch",
        "LowQualFinSF",
    ]

    MISSING_NUMERIC_COLUMNS = [
        "LotFrontage",
        "GarageYrBlt",
        "MasVnrArea",
    ]

    OUTLIER_COLUMNS = [
        "LotFrontage",
        "LotArea",
        "MasVnrArea",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "1stFlrSF",
        "2ndFlrSF",
        "GrLivArea",
        "GarageArea",
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "ScreenPorch",
    ]

    LOW_INFORMATION_COLUMNS = [
        "MiscFeature",
        "GarageQual",
        "GarageCond",
        "BsmtFinType2",
        "BsmtCond",
        "Electrical",
        "Condition1",
        "SaleType",
        "Condition2",
        "RoofMatl",
        "Functional",
        "Heating",
        "ExterCond",
        "LandContour",
        "LandSlope",
        "PavedDrive",
        "Street",
        "Utilities",
        "CentralAir",
    ]

    HIGH_MISSING_COLUMNS = [
        "PoolQC",
        "Alley",
        "Fence",
        "MasVnrType",
        "FireplaceQu",
    ]

    NONE_FILL_COLUMNS = [
        "GarageType",
        "GarageFinish",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtQual",
    ]

    CATEGORICAL_COLUMNS = [
        "Neighborhood",
        "Exterior1st",
        "Exterior2nd",
        "HouseStyle",
        "GarageType",
        "RoofStyle",
        "Foundation",
        "SaleCondition",
        "MSZoning",
        "LotConfig",
        "BldgType",
        "BsmtQual",
        "BsmtExposure",
        "HeatingQC",
        "LotShape",
        "ExterQual",
        "KitchenQual",
        "GarageFinish",
        "BsmtFinType1",
    ]

    def __init__(self, outlier_quantile: float = 0.95):
        self.outlier_quantile = outlier_quantile
        self.feature_columns_ = None
        self.scaler = StandardScaler()
        self.numeric_columns_ = None

    def _apply_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 0. Удаляем Id если есть
        if "Id" in df.columns:
            df = df.drop(columns=["Id"])

        # 1. Sparse → бинарные
        df = replace_with_has_large(df, self.FEATURES_SPARSE)

        # 2. Missing
        df = fill_missing_values(
            df,
            columns=self.MISSING_NUMERIC_COLUMNS,
            strategy="median",
        )

        # 3. Outliers
        df = handle_outliers(
            df,
            columns=self.OUTLIER_COLUMNS,
            quantile=self.outlier_quantile,
        )

        # 4–5. Drop columns
        df = drop_columns(df, self.LOW_INFORMATION_COLUMNS)
        df = drop_columns(df, self.HIGH_MISSING_COLUMNS)

        # 6. Fill None
        df = fill_none_columns(df, self.NONE_FILL_COLUMNS)

        # 7. One-hot
        df = one_hot_encode_columns(df, self.CATEGORICAL_COLUMNS)

        return df

    def _get_numeric_columns(self, df: pd.DataFrame):
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # исключаем бинарные (0/1)
        non_binary_cols = [
            col for col in numeric_cols
            if not set(df[col].dropna().unique()).issubset({0, 1})
        ]

        return non_binary_cols

    def fit(self, df: pd.DataFrame):
        processed = self._apply_pipeline(df)

        # находим численные колонки
        self.numeric_columns_ = self._get_numeric_columns(processed)

        # обучаем scaler
        self.scaler.fit(processed[self.numeric_columns_])

        self.feature_columns_ = processed.columns.tolist()

        return self

    def transform(self, df: pd.DataFrame):
        if self.feature_columns_ is None:
            raise ValueError("Сначала вызови fit()")

        processed = self._apply_pipeline(df)

        # добавляем отсутствующие
        for col in self.feature_columns_:
            if col not in processed.columns:
                processed[col] = 0

        # удаляем лишние
        extra_cols = [c for c in processed.columns if c not in self.feature_columns_]
        if extra_cols:
            processed = processed.drop(columns=extra_cols)

        processed = processed[self.feature_columns_]

        # нормализация
        processed[self.numeric_columns_] = self.scaler.transform(
            processed[self.numeric_columns_]
        )

        return processed

    def fit_transform(self, df: pd.DataFrame):
        processed = self._apply_pipeline(df)

        self.numeric_columns_ = self._get_numeric_columns(processed)

        processed[self.numeric_columns_] = self.scaler.fit_transform(
            processed[self.numeric_columns_]
        )

        self.feature_columns_ = processed.columns.tolist()

        return processed