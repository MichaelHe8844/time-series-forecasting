"""
src/models/baseline_lstm.py
Bidirectional LSTM — Huber + Ranking Loss optimization.
"""

import torch.nn as nn

from configs.config import cfg
from src.common import set_seed, load_data, run_training


MODEL_NAME = "lstm"


class BiLSTM(nn.Module):
    def __init__(self, input_dim, cfg_lstm):
        super().__init__()
        lstm_dropout = cfg_lstm["dropout"] if cfg_lstm["num_layers"] > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=cfg_lstm["hidden_dim"],
            num_layers=cfg_lstm["num_layers"],
            dropout=lstm_dropout,
            bidirectional=cfg_lstm["bidirectional"],
            batch_first=True
        )
        direction = 2 if cfg_lstm["bidirectional"] else 1
        self.fc = nn.Linear(cfg_lstm["hidden_dim"] * direction, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(-1)


def train():
    set_seed(cfg["seed"])
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    input_dim = X_train.shape[2]
    model = BiLSTM(input_dim, cfg["lstm"])
    run_training(model, MODEL_NAME, X_train, y_train, X_val, y_val, X_test, y_test)


if __name__ == "__main__":
    train()