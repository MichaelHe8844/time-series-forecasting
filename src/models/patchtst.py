"""
src/models/patchtst.py
PatchTST — Patch + Transformer architecture for multivariate time series (ICLR 2023).
"""

import torch
import torch.nn as nn

from configs.config import cfg
from src.common import set_seed, PositionalEncoding, load_data, run_training


MODEL_NAME = "patchtst"


class PatchEmbedding(nn.Module):
    def __init__(self, patch_len: int, stride: int, input_dim: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.patch_linear = nn.Linear(patch_len * input_dim, d_model)

    def forward(self, x):
        B, L, F = x.shape
        patches = []
        for i in range(0, L - self.patch_len + 1, self.stride):
            patch = x[:, i:i + self.patch_len, :]
            patch = patch.reshape(B, -1)
            patches.append(patch)
        patches = torch.stack(patches, dim=1)
        return self.patch_linear(patches)


class PatchTST(nn.Module):
    def __init__(self, input_dim, cfg_model):
        super().__init__()

        self.patch_len = cfg_model.get("patch_len", 16)
        self.stride = cfg_model.get("stride", 8)
        d_model = cfg_model.get("d_model", 192)
        nhead = cfg_model.get("nhead", 8)
        num_layers = cfg_model.get("num_layers", 3)
        dim_feedforward = cfg_model.get("dim_feedforward", 512)
        dropout = cfg_model.get("dropout", 0.1)
        max_len = cfg_model.get("max_len", 512)

        self.patch_embedding = PatchEmbedding(
            patch_len=self.patch_len, stride=self.stride,
            input_dim=input_dim, d_model=d_model
        )

        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        hidden_dim = d_model // 2
        self.shared_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.reg_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.pos_encoder(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)
        out = self.transformer(x)
        out = out[:, -1, :]
        feat = self.shared_head(out)
        return self.reg_head(feat).squeeze(-1)


def train():
    set_seed(cfg["seed"])
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    input_dim = X_train.shape[2]
    model = PatchTST(input_dim, cfg[MODEL_NAME])
    run_training(model, MODEL_NAME, X_train, y_train, X_val, y_val, X_test, y_test)


if __name__ == "__main__":
    train()