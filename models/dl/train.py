# models/dl/train.py
import math
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.config import load_config
from models.dl.dataset import HousePricesDataset
from models.dl.model import HousePricesModel


def train_model(X, y):
    config = load_config()

    # --- параметры ---
    seed = config["seed"]
    lr = config["learning_rate"]

    torch.manual_seed(seed)

    # --- split ---
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # --- dataset ---
    train_dataset = HousePricesDataset(X_train, y_train)
    val_dataset = HousePricesDataset(X_val, y_val)

    # --- dataloader ---
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # --- model ---
    model = HousePricesModel(input_dim=X.shape[1])

    # --- loss ---
    criterion = nn.MSELoss()

    # --- optimizer ---
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    # --- scheduler ---
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=25,
        gamma=0.5,
    )

    epochs = 250

    # --- history ---
    history = {
        "train_scores": [],
        "valid_scores": [],
        "learning_rates": [],
    }

    # --- progress bar ---
    pbar = tqdm(range(epochs), desc="Training")

    for epoch in pbar:
        # ===== TRAIN =====
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()

        # --- RMSE ---
        train_rmse = math.sqrt(train_loss / len(train_loader))
        valid_rmse = math.sqrt(val_loss / len(val_loader))

        # --- history ---
        history["train_scores"].append(train_rmse)
        history["valid_scores"].append(valid_rmse)
        history["learning_rates"].append(optimizer.param_groups[0]["lr"])

        # --- tqdm ---
        pbar.set_postfix({
            "train_rmse": f"{train_rmse:.4f}",
            "valid_rmse": f"{valid_rmse:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
        })

        # --- lr step ---
        scheduler.step()

    return model, history