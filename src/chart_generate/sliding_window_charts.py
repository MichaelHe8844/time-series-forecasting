#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sliding window validation chart generator.
Reads sliding_window_summary.csv, produces:
  1. Per-window metric line chart (2x2: Sharpe, IC, AnnRet, MaxDD)
  2. Model stability bar chart (mean +/- std Sharpe across windows)
  3. Rank consistency plot (Sharpe-based rank vs window index)
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULT_DIR = PROJECT_ROOT / "results"
CHART_DIR = PROJECT_ROOT / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

MODEL_LABELS = {
    "cnn_transformer":   "CNN-Transformer",
    "lstm_transformer":  "LSTM-Transformer",
    "transformer":       "Transformer",
    "lstm":              "LSTM",
    "tcn":               "TCN",
    "modern_tcn":        "ModernTCN",
    "patchtst":          "PatchTST",
    "xgboost":           "XGBoost",
    "dlinear":           "DLinear",
    "timemixer":         "TimeMixer",
}

MODEL_ORDER = [
    "cnn_transformer", "lstm_transformer", "transformer", "lstm",
    "patchtst", "xgboost", "tcn", "modern_tcn", "dlinear", "timemixer",
]

MODEL_PALETTE = {
    "cnn_transformer":   "#2196F3",
    "lstm_transformer":  "#4CAF50",
    "transformer":       "#FF9800",
    "lstm":              "#9C27B0",
    "tcn":               "#F44336",
    "modern_tcn":        "#795548",
    "patchtst":          "#00BCD4",
    "xgboost":           "#607D8B",
    "dlinear":           "#E91E63",
    "timemixer":         "#3F51B5",
}

MODEL_MARKERS = {
    "cnn_transformer":   "o",
    "lstm_transformer":  "s",
    "transformer":       "D",
    "lstm":              "^",
    "tcn":               "v",
    "modern_tcn":        "p",
    "patchtst":          "h",
    "xgboost":           "*",
    "dlinear":           "P",
    "timemixer":         "X",
}

sns.set_style("whitegrid")
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11


# ---------------------------------------------------------------------------
# Chart 1: Per-window metric line chart (2x2)
# ---------------------------------------------------------------------------
def plot_per_window_metrics(df):
    models_present = [m for m in MODEL_ORDER if m in df["Model"].values]
    windows = sorted(df["Window"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    metrics_config = [
        ("Test_Sharpe", "Test Sharpe Ratio", axes[0, 0], True),
        ("Test_IC", "Test Rank IC", axes[0, 1], True),
        ("Test_AnnualReturn", "Annualized Return", axes[1, 0], True),
        ("Test_MaxDrawdown", "Max Drawdown", axes[1, 1], False),
    ]

    for metric, ylabel, ax, draw_zero in metrics_config:
        for model in models_present:
            mdata = df[df["Model"] == model].sort_values("Window")
            label = MODEL_LABELS.get(model, model)
            color = MODEL_PALETTE.get(model, "gray")
            marker = MODEL_MARKERS.get(model, "o")
            lw = 2.5 if model == "cnn_transformer" else 1.8
            ms = 10 if model == "cnn_transformer" else 7
            zorder = 10 if model == "cnn_transformer" else 1
            ax.plot(mdata["Window"], mdata[metric], marker=marker,
                    color=color, label=label, linewidth=lw, markersize=ms,
                    zorder=zorder, markeredgewidth=0.5, markeredgecolor='white')
        if draw_zero:
            ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("Window Index", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(ylabel, fontsize=14, pad=10)
        ax.set_xticks(windows)
        ax.grid(True, alpha=0.3)

    # Single shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Sliding Window Validation — Per-Window Test Metrics (Full Feature Set, Seed=1)",
                 fontsize=16, y=1.01)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])

    for fmt in ("png", "pdf"):
        path = CHART_DIR / f"sliding_window_metrics.{fmt}"
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Chart 2: Model stability bar chart
# ---------------------------------------------------------------------------
def plot_stability_bars(df):
    models_present = [m for m in MODEL_ORDER if m in df["Model"].values]

    # Compute mean and std per model
    stats = []
    for model in models_present:
        sub = df[df["Model"] == model]["Test_Sharpe"]
        stats.append({
            "Model": model,
            "Label": MODEL_LABELS.get(model, model),
            "Sharpe_Mean": sub.mean(),
            "Sharpe_Std": sub.std(),
        })
    stats_df = pd.DataFrame(stats)
    stats_df = stats_df.sort_values("Sharpe_Mean", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = [MODEL_PALETTE.get(m, "gray") for m in stats_df["Model"]]

    bars = ax.barh(range(len(stats_df)), stats_df["Sharpe_Mean"], xerr=stats_df["Sharpe_Std"],
                   color=colors, edgecolor="black", capsize=4, height=0.6, alpha=0.85)
    ax.set_yticks(range(len(stats_df)))
    ax.set_yticklabels(stats_df["Label"], fontsize=11)
    ax.set_xlabel("Test Sharpe Ratio (mean +/- std across windows)", fontsize=12)
    ax.set_title("Model Stability Across Sliding Windows\n(Full Feature Set, Seed=1)",
                 fontsize=14, pad=12)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)

    # Annotate bar ends
    for i, (bar, row) in enumerate(zip(bars, stats_df.itertuples())):
        ax.text(bar.get_width() + 0.1, i,
                f"{row.Sharpe_Mean:.2f} +/- {row.Sharpe_Std:.2f}",
                va="center", fontsize=9)

    fig.tight_layout()
    for fmt in ("png", "pdf"):
        path = CHART_DIR / f"sliding_window_stability.{fmt}"
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Chart 3: Rank consistency plot
# ---------------------------------------------------------------------------
def plot_rank_consistency(df):
    models_present = [m for m in MODEL_ORDER if m in df["Model"].values]
    windows = sorted(df["Window"].unique())

    # Compute Sharpe rank per window (1 = best)
    rank_data = []
    for w in windows:
        sub = df[df["Window"] == w].copy()
        sub["Rank"] = sub["Test_Sharpe"].rank(ascending=False)
        rank_data.append(sub[["Model", "Window", "Rank"]])
    rank_df = pd.concat(rank_data)

    fig, ax = plt.subplots(figsize=(12, 7))

    for model in models_present:
        mdata = rank_df[rank_df["Model"] == model].sort_values("Window")
        label = MODEL_LABELS.get(model, model)
        color = MODEL_PALETTE.get(model, "gray")
        marker = MODEL_MARKERS.get(model, "o")
        ax.plot(mdata["Window"], mdata["Rank"], marker=marker,
                color=color, label=label, linewidth=2, markersize=10)

    ax.set_yticks(range(1, len(models_present) + 1))
    ax.set_yticklabels([str(i) for i in range(1, len(models_present) + 1)])
    ax.invert_yaxis()
    ax.set_xlabel("Window Index", fontsize=12)
    ax.set_ylabel("Sharpe Ratio Rank (1 = best)", fontsize=12)
    ax.set_title("Model Rank Consistency Across Sliding Windows\n(Full Feature Set, Seed=1)",
                 fontsize=14, pad=12)
    ax.set_xticks(windows)
    ax.legend(fontsize=10, ncol=2, framealpha=0.9)

    # Annotate stability: models consistently at the top
    for model in models_present[:3]:
        sub = rank_df[rank_df["Model"] == model]["Rank"]
        mean_rank = sub.mean()
        ax.annotate(f"{MODEL_LABELS.get(model, model)}\nmean rank={mean_rank:.1f}",
                    xy=(windows[-1], sub.iloc[-1]),
                    fontsize=9, color=MODEL_PALETTE.get(model, "gray"),
                    fontweight="bold")

    fig.tight_layout()
    for fmt in ("png", "pdf"):
        path = CHART_DIR / f"sliding_window_ranks.{fmt}"
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Sliding window chart generator")
    parser.add_argument("--results-csv", type=str,
                        default=str(RESULT_DIR / "sliding_window_summary.csv"))
    args = parser.parse_args()

    csv_path = Path(args.results_csv)
    if not csv_path.exists():
        print(f"[WARN] Results CSV not found: {csv_path}")
        print("Run sliding_window.py first to generate results.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path.name}")
    print(f"Models: {sorted(df['Model'].unique())}")
    print(f"Windows: {sorted(df['Window'].unique())}")

    # Print quick summary
    print("\nCross-window test Sharpe summary:")
    for model in MODEL_ORDER:
        sub = df[df["Model"] == model]
        if sub.empty:
            continue
        s = sub["Test_Sharpe"]
        print(f"  {MODEL_LABELS.get(model, model):20s}  "
              f"mean={s.mean():+.2f}  std={s.std():.2f}  "
              f"min={s.min():+.2f}  max={s.max():+.2f}  ({len(sub)} windows)")

    # Generate charts
    print("\nGenerating charts...")
    plot_per_window_metrics(df)
    plot_stability_bars(df)
    plot_rank_consistency(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
