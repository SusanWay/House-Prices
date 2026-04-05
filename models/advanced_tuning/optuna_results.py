import numpy as np
import optuna
import pandas as pd


def build_optuna_results_df(study: optuna.Study) -> pd.DataFrame:
    """
    Превращает все trial из Optuna study в pandas DataFrame.
    """

    rows = []

    for trial in study.trials:
        row = {
            "number": trial.number,
            "mean_rmse": trial.value,
            "state": str(trial.state),
            "mean_best_rmse": trial.user_attrs.get("mean_best_rmse"),
            "mean_best_iteration": trial.user_attrs.get("mean_best_iteration"),
            "best_iteration_list": trial.user_attrs.get("best_iteration_list"),
        }

        row.update(trial.params)
        rows.append(row)

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values("mean_rmse", ascending=True).reset_index(drop=True)
    results_df["rank"] = np.arange(1, len(results_df) + 1)

    front_columns = [
        "rank",
        "number",
        "mean_rmse",
        "mean_best_rmse",
        "mean_best_iteration",
        "best_iteration_list",
        "state",
    ]
    other_columns = [col for col in results_df.columns if col not in front_columns]

    return results_df[front_columns + other_columns]