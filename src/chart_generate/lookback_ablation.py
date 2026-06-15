#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lookback window (L) ablation chart generator.
Reads results_summary_lookback.csv, aggregates across seeds,
produces Sharpe-vs-L and IC-vs-L line charts + LaTeX table.
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


# ---------------------------------------------------------------------------
# LaTeX table generator
# ---------------------------------------------------------------------------
def generate_latex_table(agg: pd.DataFrame, output_path: Path):
    """Generate a LaTeX tabular fragment (no float wrapper) for \input in paper."""
    models_present = [m for m in MODEL_ORDER if m in agg['Model'].values]
    lookbacks = sorted(agg['Lookback'].unique())

    lines = []
    n_l = len(lookbacks)
    col_spec = r"l" + r"c" * n_l + r"|" + r"c" * n_l
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")
    # Header
    lb_headers = " & ".join([str(l) for l in lookbacks])
    lines.append(r"\multirow{2}{*}{Model} & \multicolumn{" + str(n_l) +
                 r"}{c}{Test Sharpe (L=)} & \multicolumn{" + str(n_l) +
                 r"}{c}{Test IC (L=)} \\")
    lines.append(r"\cmidrule(lr){2-" + str(n_l + 1) + r"} \cmidrule(lr){" +
                 str(n_l + 2) + r"-" + str(2 * n_l + 1) + r"}")
    lines.append(r" & " + lb_headers + r" & " + lb_headers + r" \\")
    lines.append(r"\midrule")

    for model in models_present:
        mdata = agg[agg['Model'] == model]
        label = MODEL_LABELS.get(model, model)
        row = [label]
        # Sharpe values
        for lb in lookbacks:
            val = mdata.loc[mdata['Lookback'] == lb, 'Sharpe_Mean']
            row.append(f"{val.values[0]:.2f}" if len(val) > 0 else "--")
        # IC values
        for lb in lookbacks:
            val = mdata.loc[mdata['Lookback'] == lb, 'IC_Mean']
            row.append(f"{val.values[0]:.3f}" if len(val) > 0 else "--")
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"LaTeX table saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Lookback ablation chart generator")
    parser.add_argument("--results-csv", type=str,
                        default=str(RESULT_DIR / "results_summary_lookback.csv"))
    args = parser.parse_args()

    csv_path = Path(args.results_csv)
    if not csv_path.exists():
        print(f"[WARN] Results CSV not found: {csv_path}")
        print("Waiting for lookback ablation to complete...")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path.name}")
    print(f"Lookbacks: {sorted(df['Lookback'].unique())}")
    print(f"Seeds: {sorted(df['Seed'].unique())}")
    print(f"Models: {sorted(df['Model'].unique())}")

    # Filter to full feature set only
    df = df[df['Feature_Type'].str.strip() == "full"].copy()

    # Aggregate across seeds
    agg = df.groupby(['Model', 'Lookback']).agg(
        Sharpe_Mean=('Test_Sharpe', 'mean'),
        Sharpe_Std=('Test_Sharpe', 'std'),
        IC_Mean=('Test_IC', 'mean'),
        IC_Std=('Test_IC', 'std'),
        DA_Mean=('Test_DA', 'mean'),
        DA_Std=('Test_DA', 'std'),
        MaxDD_Mean=('Test_MaxDrawdown', 'mean'),
        AnnRet_Mean=('Test_AnnualReturn', 'mean'),
    ).reset_index()

    # Fill NaN std (single seed) with 0
    for col in ['Sharpe_Std', 'IC_Std', 'DA_Std']:
        agg[col] = agg[col].fillna(0)

    lookbacks = sorted(agg['Lookback'].unique())
    models_present = [m for m in MODEL_ORDER if m in agg['Model'].values]

    print("\nAggregated results (5-seed mean):")
    for _, row in agg.iterrows():
        print(f"  {row['Model']:20s} L={int(row['Lookback']):2d}  "
              f"Sharpe={row['Sharpe_Mean']:7.2f}+/-{row['Sharpe_Std']:.2f}  "
              f"IC={row['IC_Mean']:.4f}+/-{row['IC_Std']:.4f}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Subplot 1: Sharpe vs L
    ax = axes[0]
    for model in models_present:
        mdata = agg[agg['Model'] == model]
        label = MODEL_LABELS.get(model, model)
        color = MODEL_PALETTE.get(model, 'gray')
        marker = MODEL_MARKERS.get(model, 'o')
        ax.errorbar(mdata['Lookback'], mdata['Sharpe_Mean'],
                    yerr=mdata['Sharpe_Std'], label=label,
                    color=color, marker=marker, capsize=4, linewidth=2,
                    markersize=8, capthick=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Lookback Window L (4h periods)', fontsize=12)
    ax.set_ylabel('Test Sharpe Ratio (5-seed mean +/- std)', fontsize=12)
    ax.set_title('Test Sharpe vs Lookback Window', fontsize=14, pad=12)
    ax.set_xticks(lookbacks)
    ax.legend(fontsize=9, ncol=2, framealpha=0.9)

    # Subplot 2: IC vs L
    ax = axes[1]
    for model in models_present:
        mdata = agg[agg['Model'] == model]
        label = MODEL_LABELS.get(model, model)
        color = MODEL_PALETTE.get(model, 'gray')
        marker = MODEL_MARKERS.get(model, 'o')
        ax.errorbar(mdata['Lookback'], mdata['IC_Mean'],
                    yerr=mdata['IC_Std'], label=label,
                    color=color, marker=marker, capsize=4, linewidth=2,
                    markersize=8, capthick=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Lookback Window L (4h periods)', fontsize=12)
    ax.set_ylabel('Test Rank IC (5-seed mean +/- std)', fontsize=12)
    ax.set_title('Test IC vs Lookback Window', fontsize=14, pad=12)
    ax.set_xticks(lookbacks)
    ax.legend(fontsize=9, ncol=2, framealpha=0.9)

    fig.suptitle('Lookback Window (L) Ablation Study -- Full Feature Set',
                 fontsize=16, y=1.02)
    fig.tight_layout()

    # ---- Save charts ----
    for fmt in ('png', 'pdf'):
        path = CHART_DIR / f"lookback_ablation.{fmt}"
        plt.savefig(path, bbox_inches='tight')
        print(f"Saved: {path}")

    # ---- Save aggregated CSV ----
    csv_out = CHART_DIR / "lookback_ablation_summary.csv"
    agg.to_csv(csv_out, index=False, encoding='utf-8-sig')
    print(f"Saved: {csv_out}")

    # ---- Generate LaTeX table ----
    tex_out = RESULT_DIR / "lookback_ablation.tex"
    generate_latex_table(agg, tex_out)

    plt.show()


if __name__ == "__main__":
    main()
