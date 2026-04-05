from typing import Optional, Tuple

import optuna
import pandas as pd
from tqdm.auto import tqdm

from models.advanced_tuning.optuna_objective import objective_catboost
from models.advanced_tuning.optuna_results import build_optuna_results_df


def run_optuna_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 100,
    n_splits: Optional[int] = None,
    study_name: Optional[str] = None,
) -> Tuple[optuna.Study, pd.DataFrame]:
    """
    Запускает Optuna-поиск по CatBoost и возвращает:
    - study
    - results_df
    """

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
    )

    pbar = tqdm(total=n_trials, desc="Optuna tuning")

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if len(study.trials) > 0:
            pbar.set_postfix(best_rmse=f"{study.best_value:.5f}")
        pbar.update(1)

    study.optimize(
        lambda trial: objective_catboost(
            trial=trial,
            X=X,
            y=y,
            n_splits=n_splits,
        ),
        n_trials=n_trials,
        callbacks=[callback],
    )

    pbar.close()

    results_df = build_optuna_results_df(study)

    return study, results_df