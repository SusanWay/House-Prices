import pandas as pd

from feature_utils.feature_engineering import replace_with_has_large
from feature_utils.data_cleaning import (
    fill_missing_values,
    handle_outliers,
    drop_columns,
    fill_none_columns,
    one_hot_encode_columns,
)


class HousePricesPreprocessor:
    """
    Единый препроцессор для датасета House Prices.

    Логика:
    1. Заменяет разреженные численные признаки на бинарные.
    2. Заполняет пропуски в численных колонках.
    3. Обрабатывает выбросы.
    4. Удаляет малоинформативные колонки.
    5. Удаляет колонки с большим количеством пропусков.
    6. Заполняет категориальные пропуски значением 'None'.
    7. Делает one-hot encoding.
    8. При transform выравнивает колонки под train.
    """

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
    ]

    def __init__(self, outlier_quantile: float = 0.95):
        self.outlier_quantile = outlier_quantile
        self.feature_columns_ = None

    def _apply_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. Разреженные признаки -> бинарные
        df = replace_with_has_large(df, self.FEATURES_SPARSE)

        # 2. Заполнение пропусков в численных колонках
        df = fill_missing_values(
            df,
            columns=self.MISSING_NUMERIC_COLUMNS,
            strategy="median",
        )

        # 3. Обработка выбросов
        df = handle_outliers(
            df,
            columns=self.OUTLIER_COLUMNS,
            quantile=self.outlier_quantile,
        )

        # 4. Удаление малоинформативных колонок
        df = drop_columns(df, self.LOW_INFORMATION_COLUMNS)

        # 5. Удаление колонок с большим числом пропусков
        df = drop_columns(df, self.HIGH_MISSING_COLUMNS)

        # 6. Заполнение категориальных пропусков
        df = fill_none_columns(df, self.NONE_FILL_COLUMNS)

        # 7. One-hot encoding
        df = one_hot_encode_columns(df, self.CATEGORICAL_COLUMNS)

        return df

    def fit(self, df: pd.DataFrame) -> "HousePricesPreprocessor":
        """
        Запоминает итоговый набор признаков после preprocessing на train.
        """
        processed = self._apply_pipeline(df)

        self.feature_columns_ = processed.columns.tolist()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет preprocessing и выравнивает колонки под train.
        """
        if self.feature_columns_ is None:
            raise ValueError("Сначала вызови fit() или fit_transform() на train данных.")

        processed = self._apply_pipeline(df)

        # Добавляем отсутствующие колонки
        for col in self.feature_columns_:
            if col not in processed.columns:
                processed[col] = 0

        # Удаляем лишние колонки, которых не было в train
        extra_cols = [col for col in processed.columns if col not in self.feature_columns_]
        if extra_cols:
            processed = processed.drop(columns=extra_cols)

        # Восстанавливаем порядок колонок
        processed = processed[self.feature_columns_]

        return processed

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Обучает препроцессор на train и сразу возвращает обработанный train.
        """
        processed = self._apply_pipeline(df)
        self.feature_columns_ = processed.columns.tolist()
        return processed