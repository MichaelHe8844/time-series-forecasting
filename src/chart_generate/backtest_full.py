#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Backtest chart generator (Figure 5 in the paper).
Generates cumulative return curves for all models on the full feature set
(seed=1) with strategy metrics. Includes DLinear and TimeMixer.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import cfg
from src.common import set_seed, load_data
from src.metrics import calc_strategy_returns

CHART_DIR = PROJECT_ROOT / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

# All models for backtest
ALL_MODELS = [
    "cnn_transformer",
    "lstm_transformer",
    "transformer",
    "lstm",
    "tcn",
    "patchtst",
    "modern_tcn",
    "xgboost",
    "dlinear",
    "timemixer",
]

MODEL_LABELS = {
    "cnn_transformer":   "CNN-Transformer",
    "lstm_transformer":  "LSTM-Transformer",
    "transformer":       "Transformer",
    "lstm":              "LSTM",
    "tcn":               "TCN",
    "patchtst":          "PatchTST",
    "modern_tcn":        "ModernTCN",
    "xgboost":           "XGBoost",
    "dlinear":           "DLinear",
    "timemixer":         "TimeMixer",
}

# Plot colours — one per model
MODEL_COLORS = {
    "cnn_transformer":   "#2196F3",  # Blue
    "lstm_transformer":  "#4CAF50",  # Green
    "transformer":       "#FF9800",  # Orange
    "lstm":              "#9C27B0",  # Purple
    "tcn":               "#F44336",  # Red
    "patchtst":          "#00BCD4",  # Cyan
    "modern_tcn":        "#795548",  # Brown
    "xgboost":           "#607D8B",  # Blue Grey
    "dlinear":           "#E91E63",  # Pink
    "timemixer":         "#3F51B5",  # Indigo
}

sns.set_style("whitegrid")
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11


def load_model_predictions(model_name, feature_type, seed):
    """Load trained model and generate predictions. Uses cost_sensitivity's function."""
    from src.cost_sensitivity import load_model_predictions as lmp
    return lmp(model_name, feature_type, seed)


def compute_strategy_metrics(preds, trues, fee=0.0005):
    """Compute strategy metrics from predictions."""
    mask = np.isfinite(preds) & np.isfinite(trues)
    preds = preds[mask]
    trues = trues[mask]
    preds = np.clip(preds, cfg["pred_clip_min"], cfg["pred_clip_max"])

    returns = calc_strategy_returns(preds, trues, fee=fee)
    equity = np.cumprod(1.0 + returns)

    # Drawdown: take the most negative (min), then abs for reporting
    dd_curve = equity / np.maximum.accumulate(equity) - 1.0
    max_dd = float(abs(dd_curve.min()))

    sharpe = float(np.sqrt(2190) * returns.mean() / (returns.std(ddof=1) + 1e-12))
    ann_ret = float(returns.mean() * 2190)

    return {
        "equity": equity,
        "returns": returns,
        "sharpe": sharpe,
        "ann_ret": ann_ret,
        "max_dd": max_dd,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Backtest chart — all models on full feature set")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed (default: 1)")
    parser.add_argument("--feature-type", type=str, default="full",
                        help="Feature set (default: full)")
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        help="Models to include (default: all 10)")
    parser.add_argument("--top-n", type=int, default=None,
                        help="Only plot top-N models by Sharpe (default: all)")
    args = parser.parse_args()

    seed = args.seed
    feature_type = args.feature_type
    model_list = args.models if args.models else ALL_MODELS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Feature: {feature_type}, Seed: {seed}")

    # ------------------------------------------------------------------
    # Load predictions for all models
    # ------------------------------------------------------------------
    results = {}
    failed = []

    for model_name in model_list:
        ckpt_file = "best.json" if model_name == "xgboost" else "best.pth"
        ckpt_path = PROJECT_ROOT / "checkpoint" / f"{model_name}_{feature_type}_seed{seed}" / ckpt_file
        if not ckpt_path.exists():
            print(f"  [SKIP] {model_name} — no checkpoint at {ckpt_path}")
            failed.append(model_name)
            continue

        try:
            print(f"  Loading {model_name}...")
            preds, trues = load_model_predictions(model_name, feature_type, seed)
            metrics = compute_strategy_metrics(preds, trues)
            results[model_name] = metrics
            print(f"    Sharpe={metrics['sharpe']:.2f}  "
                  f"AnnRet={metrics['ann_ret']*100:.1f}%  "
                  f"MaxDD={metrics['max_dd']*100:.1f}%  "
                  f"Periods={len(metrics['returns'])}")
        except Exception as e:
            print(f"  [FAIL] {model_name}: {e}")
            failed.append(model_name)

    if failed:
        print(f"\n[WARN] Failed/Skipped ({len(failed)}): {failed}")

    if len(results) < 2:
        raise RuntimeError(f"Only {len(results)} model(s) loaded successfully, need ≥2")

    # ------------------------------------------------------------------
    # Sort by Sharpe (descending) and optionally filter top-N
    # ------------------------------------------------------------------
    sorted_models = sorted(results.keys(),
                           key=lambda m: results[m]["sharpe"],
                           reverse=True)
    if args.top_n:
        sorted_models = sorted_models[:args.top_n]

    print(f"\n[INFO] Plotting {len(sorted_models)} models (sorted by Sharpe):")
    for m in sorted_models:
        r = results[m]
        print(f"  {MODEL_LABELS.get(m, m):20s}  Sharpe={r['sharpe']:.2f}  "
              f"MaxDD={r['max_dd']*100:.1f}%")

    # ------------------------------------------------------------------
    # Plot — (A) Cumulative Equity  (B) Drawdown
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 10),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True,
    )

    n_periods = max(len(results[m]["equity"]) for m in sorted_models)
    for model_name in sorted_models:
        color = MODEL_COLORS.get(model_name, "#999999")
        label = MODEL_LABELS.get(model_name, model_name)
        equity = results[model_name]["equity"]

        # Pad shorter equity curves with NaN
        if len(equity) < n_periods:
            equity_padded = np.full(n_periods, np.nan)
            equity_padded[:len(equity)] = equity
        else:
            equity_padded = equity

        ax1.plot(equity_padded, color=color, linewidth=1.3, label=label, alpha=0.9)

        # Drawdown
        eq = equity  # unpadded for correct running max
        dd = eq / np.maximum.accumulate(eq) - 1.0
        if len(dd) < n_periods:
            dd_padded = np.full(n_periods, np.nan)
            dd_padded[:len(dd)] = dd
        else:
            dd_padded = dd
        ax2.plot(dd_padded, color=color, linewidth=0.8, alpha=0.7)

    ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.axhline(y=0.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    ax1.set_ylabel('Cumulative Return', fontsize=13)
    ax2.set_ylabel('Drawdown', fontsize=13)
    ax2.set_xlabel('Test Period (4-hour steps)', fontsize=13)
    ax1.set_title(
        f'Backtest — Full Feature Set (seed={seed}, {feature_type})',
        fontsize=15, pad=12,
    )

    # Legend in two columns
    ax1.legend(fontsize=10, loc='upper left', ncol=2, framealpha=0.9)
    ax1.grid(True, alpha=0.25)
    ax2.grid(True, alpha=0.25)
    ax2.set_ylim(-1.0, 0.05)

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_path = CHART_DIR / f"backtest_{feature_type}_all_models"
    for fmt in ('png', 'pdf'):
        plt.savefig(save_path.with_suffix(f'.{fmt}'), bbox_inches='tight')

    # Save CSV with equity curves and metrics
    csv_rows = []
    max_len = 0
    for model_name in sorted_models:
        r = results[model_name]
        equity = r["equity"]
        dd = equity / np.maximum.accumulate(equity) - 1.0
        csv_rows.append({
            "model": model_name,
            "equity": equity,
            "drawdown": dd,
            "sharpe": r["sharpe"],
            "ann_ret": r["ann_ret"],
            "max_dd": r["max_dd"],
        })
        max_len = max(max_len, len(equity))

    # Build aligned CSV
    csv_arrays = {}
    for row in csv_rows:
        label = MODEL_LABELS.get(row["model"], row["model"])
        eq = np.pad(row["equity"], (0, max_len - len(row["equity"])),
                    constant_values=np.nan)
        dd = np.pad(row["drawdown"], (0, max_len - len(row["drawdown"])),
                    constant_values=np.nan)
        csv_arrays[f"{label}_equity"] = eq
        csv_arrays[f"{label}_drawdown"] = dd

    csv_df = {k: v for k, v in csv_arrays.items()}
    # Also save metrics summary
    metrics_rows = []
    for model_name in sorted_models:
        r = results[model_name]
        label = MODEL_LABELS.get(model_name, model_name)
        metrics_rows.append({
            "Model": label,
            "Sharpe": r["sharpe"],
            "AnnualReturn": r["ann_ret"],
            "MaxDrawdown": r["max_dd"],
        })
    import pandas as pd
    metrics_df = pd.DataFrame(metrics_rows)

    # Equity CSV
    max_cols = max(len(arr) for arr in csv_df.values())
    aligned = {}
    for k, arr in csv_df.items():
        if len(arr) < max_cols:
            aligned[k] = np.pad(arr, (0, max_cols - len(arr)), constant_values=np.nan)
        else:
            aligned[k] = arr
    equity_csv = np.column_stack([aligned[k] for k in csv_df])
    np.savetxt(
        save_path.with_suffix('.csv'),
        equity_csv,
        delimiter=',',
        header=','.join(csv_df.keys()),
        comments='',
        fmt='%.8f',
    )
    metrics_df.to_csv(
        CHART_DIR / f"backtest_{feature_type}_metrics.csv",
        index=False, encoding='utf-8-sig', float_format='%.4f',
    )

    print(f"\nSaved:")
    for ext in ('png', 'pdf', 'csv'):
        print(f"   {save_path}.{ext}")
    print(f"   {CHART_DIR / f'backtest_{feature_type}_metrics.csv'}")

    plt.show()


if __name__ == "__main__":
    main()
