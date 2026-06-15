"""
src/models/baseline_transformer.py
Transformer Baseline with positional encoding and LayerNorm.
"""

import torch.nn as nn

from configs.config import cfg
from src.common import set_seed, PositionalEncoding, load_data, run_training


MODEL_NAME = "transformer"


class Transformer(nn.Module):
    def __init__(self, input_dim, cfg_trans):
        super().__init__()
        self.d_model = cfg_trans["d_model"]

        self.input_proj = nn.Linear(input_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=cfg_trans.get("max_len", 512))
        self.input_norm = nn.LayerNorm(self.d_model)
        self.input_dropout = nn.Dropout(cfg_trans["dropout"])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=cfg_trans["nhead"],
            dim_feedforward=cfg_trans["dim_feedforward"],
            dropout=cfg_trans["dropout"],
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg_trans["num_layers"])
        self.fc = nn.Linear(self.d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)
        out = self.transformer(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(-1)


def train():
    set_seed(cfg["seed"])
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    input_dim = X_train.shape[2]
    model = Transformer(input_dim, cfg["transformer"])
    run_training(model, MODEL_NAME, X_train, y_train, X_val, y_val, X_test, y_test)


if __name__ == "__main__":
    train()