"""
src/models/timemixer.py
TimeMixer — Multi-Scale Mixing for Time Series Forecasting (ICLR 2024).
Past-Decomposable-Mixing (PDM) + Future-Multipredictor-Mixing (FMM).
Adapted for single-step (H=1) financial return prediction.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import cfg
from src.common import set_seed, load_data, run_training


MODEL_NAME = "timemixer"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _create_downsample_series(length: int, num_scales: int):
    """Generate downsample factors that evenly cover the lookback window."""
    factors = []
    for i in range(num_scales):
        f = max(1, int(length / (2 ** (num_scales - 1 - i))))
        if f not in factors:
            factors.append(f)
    # Ensure smallest scale = 1 (original resolution)
    if 1 not in factors:
        factors = [1] + factors
    return sorted(factors)[:num_scales]


# ---------------------------------------------------------------------------
# Seasonal / Trend decomposition
# ---------------------------------------------------------------------------
class SeriesDecomp(nn.Module):
    """Moving-average based seasonal-trend decomposition."""

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1,
                                padding=kernel_size // 2,
                                count_include_pad=False)

    def forward(self, x):
        # x: (B, F, L_seq)
        trend = self.avg(x)             # (B, F, L) — same length due to padding
        seasonal = x - trend
        return seasonal, trend


# ---------------------------------------------------------------------------
# Past Decomposable Mixing (PDM) block
# ---------------------------------------------------------------------------
class PDMBlock(nn.Module):
    """
    Single PDM block operating on multi-scale representations.

    For each scale:
      1) Decompose into seasonal / trend
      2) Seasonal mixing: bottom-up (coarse → fine) via inter-scale attention
      3) Trend mixing:     top-down  (fine → coarse) via inter-scale attention
      4) Fuse and feed-forward
    """

    def __init__(self, d_model: int, num_scales: int, decomp_kernel: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        self.num_scales = num_scales
        self.decomp = SeriesDecomp(kernel_size=decomp_kernel)

        # Cross-scale mixing for seasonal & trend
        self.seasonal_mix = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=1),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=1),
            ) for _ in range(num_scales)
        ])
        self.trend_mix = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=1),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=1),
            ) for _ in range(num_scales)
        ])

        # Post-mixing LayerNorm + FFN
        self.seasonal_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_scales)])
        self.trend_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_scales)])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.Dropout(dropout),
            ) for _ in range(num_scales)
        ])
        self.out_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_scales)])

    def forward(self, scale_reprs):
        """
        Args:
            scale_reprs: list of (B, F, L_s) tensors — one per scale

        Returns:
            list of (B, F, L_s) — updated scale representations
        """
        B = scale_reprs[0].shape[0]
        d_model = scale_reprs[0].shape[1]
        num_scales = len(scale_reprs)

        # Decompose each scale
        seasonals, trends = [], []
        for s in range(num_scales):
            sea, tr = self.decomp(scale_reprs[s])
            seasonals.append(sea)  # (B, F, L_s)
            trends.append(tr)

        # --- Seasonal mixing: bottom-up then top-down ---
        # Coarse-to-fine (top-down): propagate coarse seasonal info to finer scales
        mixed_seasonals = []
        coarse_sea = seasonals[-1]  # coarsest scale
        mixed_seasonals.append(coarse_sea)
        for s in range(num_scales - 2, -1, -1):
            # Upsample coarse to match current scale
            target_len = seasonals[s].shape[-1]
            coarse_up = F.interpolate(mixed_seasonals[0], size=target_len,
                                       mode='linear', align_corners=False)
            # Mix: current fine + upsampled coarse info
            combined = seasonals[s] + coarse_up
            # Refine
            refined = self.seasonal_mix[s](combined)
            mixed_seasonals.insert(0, refined)

        # --- Trend mixing: fine-to-coarse ---
        mixed_trends = []
        fine_tr = trends[0]  # finest scale
        mixed_trends.append(fine_tr)
        for s in range(1, num_scales):
            target_len = trends[s].shape[-1]
            fine_down = F.interpolate(mixed_trends[-1], size=target_len,
                                       mode='linear', align_corners=False)
            combined = trends[s] + fine_down
            refined = self.trend_mix[s](combined)
            mixed_trends.append(refined)

        # --- Fuse seasonal + trend per scale ---
        outputs = []
        for s in range(num_scales):
            fused = mixed_seasonals[s] + mixed_trends[s]  # (B, F, L_s)
            fused_t = fused.transpose(1, 2)                # (B, L_s, F)
            fused_t = self.seasonal_norm[s](fused_t)
            fused_t = self.ffn[s](fused_t)
            fused_t = self.out_norm[s](fused_t)
            outputs.append(fused_t.transpose(1, 2))        # (B, F, L_s)

        return outputs


# ---------------------------------------------------------------------------
# Future Multipredictor Mixing (FMM)
# ---------------------------------------------------------------------------
class FMM(nn.Module):
    """
    Predict at each scale, then ensemble with learnable weights.
    """

    def __init__(self, d_model: int, num_scales: int, dropout: float = 0.1):
        super().__init__()
        self.num_scales = num_scales
        # Per-scale predictor: takes last time-step + pooled
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            ) for _ in range(num_scales)
        ])
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)

    def forward(self, scale_reprs):
        """
        Args:
            scale_reprs: list of (B, F, L_s)
        Returns:
            (B,) — ensemble prediction
        """
        preds = []
        for s in range(self.num_scales):
            rep = scale_reprs[s]  # (B, F, L_s)
            # Last step + global mean pooling
            last = rep[:, :, -1]            # (B, F)
            mean_pool = rep.mean(dim=-1)     # (B, F)
            feat = torch.cat([last, mean_pool], dim=-1)  # (B, 2F)
            pred = self.predictors[s](feat).squeeze(-1)   # (B,)
            preds.append(pred)

        preds = torch.stack(preds, dim=-1)               # (B, S)
        weights = torch.softmax(self.scale_weights, dim=0)  # (S,)
        return (preds * weights).sum(dim=-1)              # (B,)


# ---------------------------------------------------------------------------
#  TimeMixer main model
# ---------------------------------------------------------------------------
class TimeMixer(nn.Module):
    """
    TimeMixer: Multi-Scale Mixing for Time Series.

    Reference:
      Wang et al., "TimeMixer: Decomposable Multiscale Mixing for Time Series
      Forecasting", ICLR 2024.
    """

    def __init__(self, input_dim, cfg_model):
        super().__init__()

        d_model = cfg_model.get("d_model", 128)
        num_scales = cfg_model.get("num_scales", 4)
        num_blocks = cfg_model.get("num_blocks", 2)
        decomp_kernel = cfg_model.get("decomp_kernel", 5)
        dropout = cfg_model.get("dropout", 0.1)
        lookback = cfg["lookback"]

        self.num_scales = num_scales
        self.downsample_factors = _create_downsample_series(lookback, num_scales)
        self.num_scales = len(self.downsample_factors)  # may differ from requested
        print(f"[TimeMixer] Lookback={lookback}, scales={self.downsample_factors}")

        # Input projection per scale
        self.scale_projections = nn.ModuleList([
            nn.Linear(input_dim, d_model) for _ in range(self.num_scales)
        ])
        self.scale_dropouts = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(self.num_scales)
        ])

        # PDM blocks
        self.pdm_blocks = nn.ModuleList([
            PDMBlock(d_model=d_model, num_scales=self.num_scales,
                     decomp_kernel=decomp_kernel, dropout=dropout)
            for _ in range(num_blocks)
        ])

        # FMM predictor
        self.fmm = FMM(d_model=d_model, num_scales=self.num_scales, dropout=dropout)

    def _downsample(self, x):
        """Create multi-scale representations via avg-pool downsampling."""
        # x: (B, L, F)
        x_t = x.transpose(1, 2)  # (B, F, L) — Conv1d expects channels in dim=1
        scales = []
        for factor in self.downsample_factors:
            if factor == 1:
                scales.append(x_t)
            else:
                pooled = F.avg_pool1d(x_t, kernel_size=factor, stride=factor)
                scales.append(pooled)
        return scales  # list of (B, F, L_s)

    def forward(self, x):
        # x: (B, L, F)
        B = x.shape[0]

        # 1) Multi-scale downsampling
        raw_scales = self._downsample(x)  # list of (B, F, L_s)

        # 2) Per-scale projection to d_model
        scale_reprs = []
        for s in range(self.num_scales):
            rep = raw_scales[s].transpose(1, 2)           # (B, L_s, F)
            rep = self.scale_projections[s](rep)          # (B, L_s, d_model)
            rep = self.scale_dropouts[s](rep)
            rep = rep.transpose(1, 2)                     # (B, d_model, L_s)
            scale_reprs.append(rep)

        # 3) PDM blocks
        for block in self.pdm_blocks:
            scale_reprs = block(scale_reprs)

        # 4) FMM ensemble prediction
        out = self.fmm(scale_reprs)  # (B,)
        return out


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------
def train():
    set_seed(cfg["seed"])
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    input_dim = X_train.shape[2]
    model = TimeMixer(input_dim, cfg[MODEL_NAME])
    run_training(model, MODEL_NAME, X_train, y_train, X_val, y_val, X_test, y_test)


if __name__ == "__main__":
    train()
