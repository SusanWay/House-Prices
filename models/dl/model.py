# models/dl/model.py
import torch.nn as nn


class HousePricesModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)