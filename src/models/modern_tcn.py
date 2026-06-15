"""
src/models/modern_tcn.py
ModernTCN — large-kernel causal convolutions with modern residual blocks.
"""

import torch.nn as nn

from configs.config import cfg
from src.common import set_seed, Chomp1d, load_data, run_training


MODEL_NAME = "modern_tcn"


class ModernTemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               padding=self.padding, dilation=dilation)
        self.chomp1 = Chomp1d(self.padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.gelu1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               padding=self.padding, dilation=dilation)
        self.chomp2 = Chomp1d(self.padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.gelu2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.residual = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.gelu1(out)
        out = self.drop1(out)
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.gelu2(out)
        out = self.drop2(out)
        return out + residual


class ModernTCN(nn.Module):
    def __init__(self, input_dim, cfg_model):
        super().__init__()
        hidden_dim = cfg_model.get("hidden_dim", 192)
        num_layers = cfg_model.get("num_layers", 4)
        kernel_size = cfg_model.get("kernel_size", 31)
        dropout = cfg_model.get("dropout", 0.1)

        self.input_bn = nn.BatchNorm1d(input_dim)

        layers = []
        in_channels = input_dim
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(ModernTemporalBlock(
                n_inputs=in_channels, n_outputs=hidden_dim,
                kernel_size=kernel_size, dilation=dilation, dropout=dropout
            ))
            in_channels = hidden_dim

        self.tcn = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_bn(x)
        out = self.tcn(x)
        out = out[:, :, -1]
        return self.head(out).squeeze(-1)


def train():
    set_seed(cfg["seed"])
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    input_dim = X_train.shape[2]
    model = ModernTCN(input_dim, cfg[MODEL_NAME])
    run_training(model, MODEL_NAME, X_train, y_train, X_val, y_val, X_test, y_test)


if __name__ == "__main__":
    train()