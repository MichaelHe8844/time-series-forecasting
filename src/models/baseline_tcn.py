"""
src/models/baseline_tcn.py
TCN Baseline — causal convolutions with residual connections.
"""

import torch.nn as nn

from configs.config import cfg
from src.common import set_seed, Chomp1d, load_data, run_training


MODEL_NAME = "tcn"


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, input_dim, cfg_tcn):
        super().__init__()
        self.hidden_dim = cfg_tcn["hidden_dim"]
        self.num_layers = cfg_tcn["num_layers"]
        self.kernel_size = cfg_tcn["kernel_size"]
        self.dropout = cfg_tcn["dropout"]

        self.input_bn = nn.BatchNorm1d(input_dim)

        layers = []
        num_channels = [self.hidden_dim] * self.num_layers
        for i in range(self.num_layers):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (self.kernel_size - 1) * dilation_size
            layers.append(
                TemporalBlock(in_channels, out_channels, self.kernel_size,
                              stride=1, dilation=dilation_size, padding=padding, dropout=self.dropout)
            )

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(self.hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_bn(x)
        out = self.tcn(x)
        out = out[:, :, -1]
        out = self.fc(out)
        return out.squeeze(-1)


def train():
    set_seed(cfg["seed"])
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    input_dim = X_train.shape[2]
    model = TCN(input_dim, cfg["tcn"])
    run_training(model, MODEL_NAME, X_train, y_train, X_val, y_val, X_test, y_test)


if __name__ == "__main__":
    train()