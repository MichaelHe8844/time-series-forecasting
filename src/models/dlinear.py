"""
src/models/dlinear.py
DLinear — Are Transformers Effective for Time Series Forecasting? (AAAI 2023).
Decomposition-based linear model: moving-average trend + seasonal residual → separate linear layers.
Channel-independent by design, matching the original paper's architecture.
"""

import torch
import torch.nn as nn

from configs.config import cfg
from src.common import set_seed, load_data, run_training


MODEL_NAME = "dlinear"


class MovingAvg(nn.Module):
    """Moving average with symmetric padding for trend extraction."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        # AvgPool1d with padding=kernel_size//2 preserves sequence length
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1,
                                padding=kernel_size // 2,
                                count_include_pad=False)

    def forward(self, x):
        # x: (B, L, F) → transpose for Conv1d → (B, F, L)
        x = x.transpose(1, 2)
        x_avg = self.avg(x)
        return x_avg.transpose(1, 2)  # (B, L, F)


class DLinear(nn.Module):
    """
    DLinear: Decomposition Linear model for time series.

    Decomposes each input channel into:
      - Trend:     moving-average smoothed component
      - Seasonal:  residual (raw - trend)

    Then applies separate linear transformations (L → 1) per channel
    and aggregates via a learnable weighted sum.

    Reference:
      Zeng et al., "Are Transformers Effective for Time Series Forecasting?",
      AAAI 2023.
    """

    def __init__(self, input_dim, cfg_model):
        super().__init__()

        kernel_size = cfg_model.get("moving_avg_kernel", 25)
        self.use_individual = cfg_model.get("individual", True)
        self.moving_avg = MovingAvg(kernel_size)

        if self.use_individual:
            # Channel-independent: one Linear(L,1) per feature per component
            self.trend_linears = nn.ModuleList([
                nn.Linear(cfg["lookback"], 1) for _ in range(input_dim)
            ])
            self.seasonal_linears = nn.ModuleList([
                nn.Linear(cfg["lookback"], 1) for _ in range(input_dim)
            ])
        else:
            # Shared linear across all channels (less common)
            self.trend_linear = nn.Linear(cfg["lookback"] * input_dim, 1)
            self.seasonal_linear = nn.Linear(cfg["lookback"] * input_dim, 1)

        self.feature_weight = nn.Parameter(torch.ones(input_dim) / input_dim)

    def forward(self, x):
        # x: (B, L, F)
        trend = self.moving_avg(x)       # (B, L, F) — smoothed
        seasonal = x - trend             # (B, L, F) — residual

        if self.use_individual:
            # Per-channel prediction: each feature independently
            trends = []
            seasonals = []
            for i in range(x.shape[-1]):
                # (B, L) → Linear(L, 1) → (B, 1)
                t = self.trend_linears[i](trend[:, :, i])
                s = self.seasonal_linears[i](seasonal[:, :, i])
                trends.append(t)
                seasonals.append(s)
            trend_out = torch.cat(trends, dim=-1)         # (B, F)
            seasonal_out = torch.cat(seasonals, dim=-1)   # (B, F)
        else:
            # Flatten all channels together
            B, L, F = x.shape
            trend_out = self.trend_linear(trend.reshape(B, L * F))
            seasonal_out = self.seasonal_linear(seasonal.reshape(B, L * F))

        # Weighted aggregation across features
        weights = torch.softmax(self.feature_weight, dim=0)  # (F,)
        out = (trend_out + seasonal_out) @ weights            # (B,)
        return out


def train():
    set_seed(cfg["seed"])
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    input_dim = X_train.shape[2]
    model = DLinear(input_dim, cfg[MODEL_NAME])
    run_training(model, MODEL_NAME, X_train, y_train, X_val, y_val, X_test, y_test)


if __name__ == "__main__":
    train()
