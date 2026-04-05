# models/dl/training_history.py
from typing import Dict

import matplotlib.pyplot as plt


def show_dl_training_history(
    history: Dict,
    model_name: str = "HousePricesDL",
    figsize: tuple = (10, 6),
    start_iter: int = 1,
):
    """
    Показывает график обучения DL модели
    """

    plt.style.use("default")

    train = history["train_scores"]
    valid = history["valid_scores"]

    iterations = list(range(1, len(train) + 1))

    if start_iter > 1:
        train = train[start_iter - 1:]
        valid = valid[start_iter - 1:]
        iterations = iterations[start_iter - 1:]

    best_idx = min(range(len(valid)), key=lambda i: valid[i])
    best_iter = iterations[best_idx]
    best_score = valid[best_idx]

    fig, ax = plt.subplots(figsize=figsize)

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(iterations, train, label="Train", linewidth=2)
    ax.plot(iterations, valid, label="Valid", linewidth=2)

    ax.scatter(
        best_iter,
        best_score,
        s=80,
        zorder=5,
    )

    ax.annotate(
        f"RMSE = {best_score:.4f}\niter = {best_iter}",
        xy=(best_iter, best_score),
        xytext=(30, -30),
        textcoords="offset points",
        bbox=dict(
            boxstyle="round,pad=0.3",
            fc="white",
            ec="black",
            lw=1,
        ),
        arrowprops=dict(
            arrowstyle="->",
            color="black",
        ),
        zorder=6,
    )

    ax.set_title(model_name)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMSE")

    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()