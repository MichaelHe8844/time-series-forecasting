#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Feature ablation study chart generator.
Reads results_summary_seed*.md from results/ and produces bar charts with error bars.
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

# Human-readable labels for feature types
FEATURE_LABELS = {
    "price_only":          "Price Only",
    "price_funding":       "Price + Funding",
    "price_funding_fng":   "Price + Funding + F&G",
    "price_long_onchain":  "Price + Long On-chain",
    "price_onchain":       "Price + On-chain",
    "full":                "Full (All Features)",
}

sns.set_style("whitegrid")
plt.rcParams['savefig.dpi'] = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_markdown_table(md_text: str) -> pd.DataFrame:
    """Parse a markdown table string into a DataFrame."""
    lines = [line.strip() for line in md_text.split('\n')
             if line.strip().startswith('|')]
    if len(lines) < 3:
        return pd.DataFrame()

    # Skip the separator line (contains ---)
    body_lines = [ln for ln in lines if '---' not in ln]
    if not body_lines:
        return pd.DataFrame()

    # Manual split by | — more robust than pd.read_csv(sep='|')
    header = [c.strip() for c in body_lines[0].split('|') if c.strip()]
    rows = []
    for ln in body_lines[1:]:
        cols = [c.strip() for c in ln.split('|') if c != '']
        cols = cols[1:-1] if len(cols) > len(header) + 2 else cols
        if len(cols) == len(header):
            rows.append(cols)

    df = pd.DataFrame(rows, columns=header)
    for col in df.columns:
        if col not in ("Model", "Feature_Type"):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def load_all_results() -> pd.DataFrame:
    """Load all seed result files into a single DataFrame."""
    md_files = sorted(RESULT_DIR.glob("results_summary_seed*.md"))
    print(f"Found {len(md_files)} result file(s): {[f.name for f in md_files]}")

    all_dfs = []
    for f in md_files:
        df = parse_markdown_table(f.read_text(encoding='utf-8'))
        if not df.empty:
            df['seed'] = f.stem.split('_')[-1]
            all_dfs.append(df)

    if not all_dfs:
        raise FileNotFoundError(
            f"No results_summary_seed*.md files found in {RESULT_DIR}")
    return pd.concat(all_dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Feature ablation chart generator")
    parser.add_argument('--model', type=str, default='cnn_transformer',
                        help='Model name to plot (default: cnn_transformer)')
    parser.add_argument('--metrics', type=str, nargs='+',
                        default=['Test_Sharpe', 'Test_IC'],
                        help='Metrics to plot (default: Test_Sharpe Test_IC)')
    args = parser.parse_args()

    model_name = args.model
    metric_list = args.metrics

    # ---- Load & filter ----
    df = load_all_results()
    df_model = df[df['Model'].str.strip() == model_name].copy()
    if df_model.empty:
        raise ValueError(f"No data found for model '{model_name}'")

    n_seeds = df_model['seed'].nunique()
    print(f"Extracted {model_name}: {len(df_model)} rows, {n_seeds} seed(s)")

    # ---- Aggregate across seeds ----
    records = []
    feature_types_seen = []
    for ft in df_model['Feature_Type'].str.strip().unique():
        subset = df_model[df_model['Feature_Type'].str.strip() == ft]
        if len(subset) < 2:
            print(f"  Skipping {ft}: only {len(subset)} seed(s)")
            continue
        feature_types_seen.append(ft)
        for metric in metric_list:
            vals = pd.to_numeric(subset[metric], errors='coerce')
            records.append({
                'Feature_Type': ft,
                'Metric': metric,
                'Mean': vals.mean(),
                'Std':  vals.std(ddof=1),
            })

    summary = pd.DataFrame(records)
    if summary.empty:
        raise RuntimeError("Not enough data to plot")

    # Sort feature types by the first metric's mean (descending)
    primary_metric = metric_list[0]
    order_data = summary[summary['Metric'] == primary_metric].set_index('Feature_Type')
    ordered_fts = order_data['Mean'].sort_values(ascending=False).index.tolist()

    print(f"\n{n_seeds}-seed mean results ({model_name}):")
    print(summary.round(4))

    # ---- Plot ----
    n_metrics = len(metric_list)
    fig, axes = plt.subplots(1, n_metrics, figsize=(7 * n_metrics, 6), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    palette_dict = dict(zip(ordered_fts,
        sns.color_palette("Blues_d", n_colors=len(ordered_fts))))

    for ax, metric in zip(axes, metric_list):
        metric_data = summary[summary['Metric'] == metric].copy()
        # Order bars by descending mean
        metric_data['Feature_Type'] = pd.Categorical(
            metric_data['Feature_Type'], categories=ordered_fts, ordered=True)
        metric_data = metric_data.sort_values('Feature_Type')

        bars = sns.barplot(data=metric_data, x='Feature_Type', y='Mean',
                           hue='Feature_Type', palette=palette_dict,
                           legend=False, ax=ax,
                           edgecolor='black', linewidth=1)

        # Error bars — use enumerate for correct x alignment
        for pos, (_, row) in enumerate(metric_data.iterrows()):
            ax.errorbar(pos, row['Mean'], yerr=row['Std'], fmt='none',
                        ecolor='black', capsize=6, elinewidth=2.0)

        # Tighten y-axis for small-range metrics (e.g., IC) so error bars are visible
        data_range = metric_data['Mean'].max() - metric_data['Mean'].min()
        if data_range < 0.5:
            y_min = metric_data['Mean'].min() - metric_data['Std'].max() * 2.5
            y_max = metric_data['Mean'].max() + metric_data['Std'].max() * 2.5
            ax.set_ylim(y_min, y_max)

        # Human-readable x labels
        readable = [FEATURE_LABELS.get(ft, ft) for ft in metric_data['Feature_Type']]
        ax.set_xticks(range(len(readable)))
        ax.set_xticklabels(readable, rotation=30, ha='right', fontsize=10)

        metric_label = metric.replace('Test_', '')
        ax.set_title(f'Test {metric_label} ({n_seeds} seeds)', fontsize=14, pad=15)
        ax.set_xlabel('Feature Set', fontsize=12)
        ax.set_ylabel(f'Mean Test {metric_label} ± Std', fontsize=12)

    fig.suptitle(f'Feature Ablation Study — {model_name.upper()}', fontsize=16, y=1.03)
    fig.tight_layout()

    # ---- Save ----
    save_path = CHART_DIR / f"feature_ablation_{model_name}"
    for fmt in ('png', 'pdf'):
        plt.savefig(save_path.with_suffix(f'.{fmt}'), bbox_inches='tight')
    summary.to_csv(CHART_DIR / f"feature_ablation_{model_name}.csv",
                   index=False, encoding='utf-8-sig')

    print(f"\nSaved:")
    for ext in ('png', 'pdf', 'csv'):
        print(f"   {save_path}.{ext}")

    plt.show()


if __name__ == "__main__":
    main()
