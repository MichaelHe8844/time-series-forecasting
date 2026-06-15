"""
src/models/lstm_transformer.py
LSTM-Transformer — LSTM extracts temporal features, Transformer models global dependencies.
"""

import torch.nn as nn

from configs.config import cfg
from src.common import set_seed, PositionalEncoding, load_data, run_training


MODEL_NAME = "lstm_transformer"


class LSTMTransformer(nn.Module):
    def __init__(self, input_dim, cfg_model):
        super().__init__()
        hidden_dim = cfg_model["hidden_dim"]
        num_lstm_layers = cfg_model.get("num_lstm_layers", 2)
        lstm_dropout = cfg_model.get("lstm_dropout", 0.1) if num_lstm_layers > 1 else 0.0
        bidirectional = cfg_model.get("bidirectional", False)
        d_model = cfg_model.get("d_model", hidden_dim)
        nhead = cfg_model.get("nhead", 4)
        num_tf_layers = cfg_model.get("num_layers", 2)
        dim_feedforward = cfg_model.get("dim_feedforward", 512)
        dropout = cfg_model.get("dropout", 0.12)
        max_len = cfg_model.get("max_len", 512)

        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_lstm_layers, dropout=lstm_dropout,
            bidirectional=bidirectional, batch_first=True
        )

        direction = 2 if bidirectional else 1
        lstm_output_dim = hidden_dim * direction
        self.proj = nn.Linear(lstm_output_dim, d_model) if lstm_output_dim != d_model else nn.Identity()

        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_tf_layers)

        hidden_out = d_model // 2
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_out, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.proj(lstm_out)
        out = self.pos_encoder(out)
        out = self.input_norm(out)
        out = self.input_dropout(out)
        out = self.transformer(out)
        out = out[:, -1, :]
        return self.head(out).squeeze(-1)


def train():
    set_seed(cfg["seed"])
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    input_dim = X_train.shape[2]
    model = LSTMTransformer(input_dim, cfg["lstm_transformer"])
    run_training(model, MODEL_NAME, X_train, y_train, X_val, y_val, X_test, y_test)


if __name__ == "__main__":
    train()