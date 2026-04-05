# House Prices — End-to-End ML Pipeline

## Описание

Проект посвящён решению задачи регрессии из соревнования Kaggle:

**House Prices: Advanced Regression Techniques**

Цель:

> Предсказать стоимость домов (`SalePrice`) на основе табличных данных.

---

## Основная идея

Модель обучается на логарифме таргета:

```python
y = log1p(SalePrice)
```

Это связано с метрикой соревнования:
`RMSLE`


---

## 📂 Структура проекта

```text
notebooks/
├── EDA/
│   ├── 01.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── eda_checklist.md
│
├── modeling/
│   ├── baseline/
│   │   └── 01_model.ipynb
│   │
│   ├── advanced/
│   │   └── 01_advanced.ipynb
│   │
│   ├── advanced_tuning/
│   │   └── 01_advanced.ipynb
│   │
│   └── dl/
│       └── 01_dl.ipynb
│
└── submission/
    └── 01_submission.ipynb
```

---

## Этап 1 — EDA

`notebooks/EDA/01.ipynb`

Что сделано:

* анализ распределения `SalePrice`
* выявлена сильная асимметрия (skewness)
* анализ пропусков
* базовый анализ признаков

Вывод:

* таргет требует логарифмирования

---

## ⚙Этап 2 — Feature Engineering

📁 `02_feature_engineering.ipynb`

Что пробовалось:

* бинаризация редких значений
* преобразование числовых признаков
* обработка выбросов

Итог:

* большинство новых признаков ухудшили качество
* оставлен минималистичный набор признаков

---

## Этап 3 — Базовые модели

📁 `modeling/baseline/01_model.ipynb`

| Модель           | RMSE   | R²     |
| ---------------- | ------ | ------ |
| Ridge            | 0.1299 | 0.9095 |
| LinearRegression | 0.1312 | 0.9076 |
| Lasso            | 0.2323 | 0.7108 |
| Dummy            | 0.4323 | ~0     |

Вывод:

* линейные модели дают неплохой baseline
* Ridge — лучший среди них

---

## Этап 4 — Продвинутые модели

📁 `modeling/advanced/01_advanced.ipynb`

| Модель          | RMSE   |
| --------------- | ------ |
| CatBoost      | 0.1275 |
| GradientBoosting | 0.1285 |
| XGBoost         | 0.1326 |
| LightGBM        | 0.1442 |

Вывод:

* CatBoost показывает лучший результат
* бустинг значительно лучше линейных моделей

---

## Этап 5 — Оптимизация (Optuna)

📁 `modeling/advanced_tuning/01_advanced.ipynb`

* 500 trials
* 5-fold CV

Лучшие параметры:

```python
{
    'iterations': 920,
    'learning_rate': 0.0615,
    'depth': 4,
    'l2_leaf_reg': 7,
    'random_strength': 3,
    'subsample': 0.8
}
```

Результат:

```text
CV RMSE ≈ 0.121
```

Особенность:

* топ-5 моделей имеют почти одинаковый результат
* пространство параметров хорошо исследовано

---

## Этап 6 — Deep Learning

📁 `modeling/dl/01_dl.ipynb`

* реализована модель на PyTorch
* использовалась нормализация признаков

Результат:

* DL значительно уступает бустингу
* высокая чувствительность к настройкам

Вывод:

> для табличных данных бустинг предпочтительнее DL

---

## Этап 7 — Финальный submission

📁 `submission/01_submission.ipynb`

Pipeline:

```python
# обучение
model.fit(X_prepared, y)

# предсказание
y_pred_log = model.predict(X_test)

# обратное преобразование
y_pred = np.expm1(y_pred_log)
```

---

## Итоговые результаты

| Метрика       | Значение    |
| ------------- | ----------- |
| CV RMSE       | ~0.121      |
| Kaggle Public | **0.12787** |

---

## Возможные улучшения

### 1. Bagging

```python
несколько CatBoost моделей с разными seed
```

### 2. Ансамбль

* CatBoost + Ridge
* CatBoost + LightGBM
