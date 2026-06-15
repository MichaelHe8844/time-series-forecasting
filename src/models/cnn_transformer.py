"""
src/models/cnn_transformer.py
CNN-Transformer with causal convolutions (Chomp) and attention pooling.
"""

import torch
import torch.nn as nn

from configs.config import cfg
from src.common import set_seed, Chomp1d, PositionalEncoding, load_data, run_training


MODEL_NAME = "cnn_transformer"


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.1, use_residual=True):
        super().__init__()
        self.chomp_size = kernel_size - 1
        self.use_residual = use_residual

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.chomp_size)
        self.chomp1 = Chomp1d(self.chomp_size)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=self.chomp_size)
        self.chomp2 = Chomp1d(self.chomp_size)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.res_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.res_proj(x)
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.drop1(out)
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.drop2(out)
        if self.use_residual:
            out = out + residual
        return out


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(weights * x, dim=1)


class CNNTransformer(nn.Module):
    def __init__(self, input_dim, cfg_model):
        super().__init__()

        conv_channels = cfg_model.get("conv_channels", 64)
        kernel_size = cfg_model.get("kernel_size", 3)
        num_conv_layers = cfg_model.get("num_conv_layers", 2)
        d_model = cfg_model.get("d_model", 128)
        nhead = cfg_model.get("nhead", 4)
        dim_feedforward = cfg_model.get("dim_feedforward", 256)
        num_layers = cfg_model.get("num_layers", 2)
        dropout = cfg_model.get("dropout", 0.1)
        max_len = cfg_model.get("max_len", 512)
        use_residual = cfg_model.get("use_residual", True)
        pool = cfg_model.get("pool", "last")

        self.pool = pool

        conv_layers = []
        in_channels = input_dim
        for _ in range(num_conv_layers):
            conv_layers.append(ConvBlock(
                in_channels=in_channels, out_channels=conv_channels,
                kernel_size=kernel_size, dropout=dropout, use_residual=use_residual
            ))
            in_channels = conv_channels
        self.conv_extractor = nn.Sequential(*conv_layers)

        self.proj = nn.Linear(conv_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if self.pool == "attn":
            self.pooler = AttentionPooling(d_model)
        else:
            self.pooler = None

        hidden_dim = d_model // 2
        self.shared_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.reg_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_extractor(x)
        x = x.transpose(1, 2)

        x = self.proj(x)
        x = self.pos_encoder(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        out = self.transformer(x)

        if self.pool == "mean":
            out = out.mean(dim=1)
        elif self.pool == "attn":
            out = self.pooler(out)
        else:
            out = out[:, -1, :]

        feat = self.shared_head(out)
        return self.reg_head(feat).squeeze(-1)


def train():
    set_seed(cfg["seed"])
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    input_dim = X_train.shape[2]
    model = CNNTransformer(input_dim, cfg[MODEL_NAME])
    run_training(model, MODEL_NAME, X_train, y_train, X_val, y_val, X_test, y_test)


if __name__ == "__main__":
    train()