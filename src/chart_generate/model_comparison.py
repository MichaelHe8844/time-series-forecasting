#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model comparison chart generator (Figure 2 in the paper).
Reads per-seed results.txt from checkpoint/ directories, computes cross-seed
mean ± std for all models on the full feature set, and produces a grouped bar
chart of Test Sharpe Ratio and Test IC.
"""

import argparse
import re
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
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoint"
CHART_DIR = PROJECT_ROOT / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

# Human-readable model labels (matching paper convention)
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

# All models expected in the comparison
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

sns.set_style("whitegrid")
plt.rcParams['savefig.dpi'] = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_results_txt(file_path: str) -> dict:
    """Parse a single results.txt into a flat dict of metric values."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    result = {}
    # Extract feature_type and lookback
    ft_match = re.search(r"feature_type:\s*(\S+)", content)
    result["feature_type"] = ft_match.group(1) if ft_match else "unknown"
    lb_match = re.search(r"lookback:\s*(\d+)", content)
    result["lookback"] = int(lb_match.group(1)) if lb_match else 48

    # Extract per-split metrics
    for split in ["Train", "Val", "Test"]:
        # Find the line for this split
        line_re = re.compile(rf"^{split}\s+(.+)$", re.MULTILINE)
        m = line_re.search(content)
        if not m:
            continue
        line = m.group(1)
        # Parse out each metric:  IC: 0.1234  PIC: 0.1234  ...
        for token in re.finditer(
            r"(IC|PIC|DA|MSE|Sharpe|IR|MaxDrawdown|AnnualReturn):\s*([-\d\.eE]+)",
            line,
        ):
            metric_name = token.group(1)
            metric_val = float(token.group(2))
            result[f"{split}_{metric_name}"] = metric_val
    return result


def extract_model_seed(folder_name: str):
    """Parse model name and seed from checkpoint folder name.

    Folder names look like: cnn_transformer_full_seed0
    or: cnn_transformer_full_L96_seed0
    """
    # Split from end: find _seed\d+ at the end
    seed_match = re.search(r"_seed(\d+)$", folder_name)
    seed = int(seed_match.group(1)) if seed_match else None

    # Remove the seed suffix to get model_ft
    prefix = folder_name[:seed_match.start()] if seed_match else folder_name
    # Also strip lookback suffix if present: _L\d+
    prefix = re.sub(r"_L\d+$", "", prefix)

    # The model name is the part before _<feature_type>
    # We know the feature types; find the longest match
    known_fts = [
        "full", "price_only", "price_funding", "price_funding_fng",
        "price_onchain", "price_long_onchain",
    ]
    model = None
    ft = None
    for known_ft in known_fts:
        if prefix.endswith("_" + known_ft):
            model = prefix[: -(len(known_ft) + 1)]  # +1 for the underscore
            ft = known_ft
            break

    if model is None:
        # Fallback: take everything before the last underscore as model guess
        parts = prefix.rsplit("_", 1)
        model = parts[0] if len(parts) > 1 else prefix
        ft = parts[1] if len(parts) > 1 else "unknown"

    return model, ft, seed


def load_checkpoint_results(
    feature_type: str = "full",
    lookback: int = None,
    models: list = None,
) -> pd.DataFrame:
    """Scan checkpoint/ folders and collect results for the given feature set."""
    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {CHECKPOINT_DIR}")

    records = []
    for folder in CHECKPOINT_DIR.iterdir():
        if not folder.is_dir():
            continue
        results_path = folder / "results.txt"
        if not results_path.exists():
            continue

        model, ft, seed = extract_model_seed(folder.name)
        if seed is None:
            continue
        if ft != feature_type:
            continue
        if models is not None and model not in models:
            continue

        metrics = parse_results_txt(str(results_path))

        # Detect if this folder is from a lookback ablation (has _L\d+ in name)
        # Default folders (no _L suffix) are the main L=48 experiments
        folder_lookback = metrics.get("lookback")
        is_ablation = bool(re.search(r"_L\d+", folder.name))

        # Apply lookback filter if specified
        if lookback is not None:
            if folder_lookback != lookback:
                continue
        # If no lookback specified, prefer non-ablation folders (L=48 default)
        # but if a model only has ablation results, include those

        records.append({
            "Model": model,
            "Feature_Type": ft,
            "Lookback": folder_lookback,
            "Seed": seed,
            "Is_Ablation": is_ablation,
            "Test_IC": metrics.get("Test_IC"),
            "Test_PIC": metrics.get("Test_PIC"),
            "Test_DA": metrics.get("Test_DA"),
            "Test_MSE": metrics.get("Test_MSE"),
            "Test_Sharpe": metrics.get("Test_Sharpe"),
            "Test_IR": metrics.get("Test_IR"),
            "Test_MaxDrawdown": metrics.get("Test_MaxDrawdown"),
            "Test_AnnualReturn": metrics.get("Test_AnnualReturn"),
        })

    if not records:
        raise RuntimeError(
            f"No checkpoint results found for feature_type='{feature_type}'"
            + (f" lookback={lookback}" if lookback else "")
        )

    df = pd.DataFrame(records)
    df = df.dropna(subset=["Test_Sharpe", "Test_IC"], how="any")

    # When lookback is not explicitly specified, prefer default (L=48) results
    # over ablation results. For each (model, seed), keep non-ablation if
    # available, otherwise keep the ablation result.
    if lookback is None:
        df = df.sort_values("Is_Ablation")  # False (default) before True (ablation)
        df = df.groupby(["Model", "Seed"], as_index=False).first()
        df = df.drop(columns=["Is_Ablation"])

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Model comparison chart (all models, full feature set)")
    parser.add_argument(
        "--metrics", type=str, nargs="+",
        default=["Test_Sharpe", "Test_IC"],
        help="Metrics to plot (default: Test_Sharpe Test_IC)",
    )
    parser.add_argument(
        "--feature-type", type=str, default="full",
        help="Feature set to compare (default: full)",
    )
    parser.add_argument(
        "--lookback", type=int, default=None,
        help="Lookback window filter (default: no filter, latest per model)",
    )
    parser.add_argument(
        "--models", type=str, nargs="*", default=None,
        help="Models to include (default: all 10)",
    )
    args = parser.parse_args()

    metric_list = args.metrics
    feature_type = args.feature_type
    model_list = args.models if args.models else ALL_MODELS

    # ---- Load ----
    df = load_checkpoint_results(
        feature_type=feature_type,
        lookback=args.lookback,
        models=model_list,
    )

    n_models = df["Model"].nunique()
    n_seeds = df["Seed"].nunique()
    print(f"Loaded {len(df)} records: {n_models} models × up to {n_seeds} seeds")
    print(f"Models: {sorted(df['Model'].unique())}")
    print(f"Seeds:  {sorted(df['Seed'].unique())}")
    if "Lookback" in df.columns:
        lb_counts = df.groupby("Model")["Lookback"].first()
        print(f"Lookback per model:\n{lb_counts.to_string()}")

    # ---- Aggregate across seeds ----
    records = []
    models_found = []
    for model in df["Model"].unique():
        subset = df[df["Model"] == model]
        if len(subset) < 2:
            print(f"  Skipping {model}: only {len(subset)} seed(s)")
            continue
        models_found.append(model)
        for metric in metric_list:
            vals = pd.to_numeric(subset[metric], errors="coerce").dropna()
            mean_val = vals.mean()
            std_val = vals.std(ddof=1) if len(vals) > 1 else 0.0
            records.append({
                "Model": model,
                "Metric": metric,
                "Mean": mean_val,
                "Std": std_val,
                "N": len(vals),
            })

    summary = pd.DataFrame(records)
    if summary.empty:
        raise RuntimeError("Not enough data to plot (need ≥2 seeds per model)")

    # Sort models by the primary metric's mean (descending)
    primary_metric = metric_list[0]
    order_data = summary[summary["Metric"] == primary_metric].set_index("Model")
    ordered_models = order_data["Mean"].sort_values(ascending=False).index.tolist()

    print(f"\n=== Cross-seed results ({feature_type}, {n_seeds} seeds) ===")
    print(summary.round(4).to_string(index=False))

    # ---- Plot ----
    n_metrics = len(metric_list)
    fig, axes = plt.subplots(1, n_metrics, figsize=(8 * n_metrics, 6), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    n_bars = len(ordered_models)
    colors = sns.color_palette("viridis", n_colors=n_bars)

    for ax, metric in zip(axes, metric_list):
        metric_data = summary[summary["Metric"] == metric].copy()
        # Order by descending mean
        metric_data["Model"] = pd.Categorical(
            metric_data["Model"], categories=ordered_models, ordered=True)
        metric_data = metric_data.sort_values("Model")

        # Human-readable labels
        labels = [MODEL_LABELS.get(m, m) for m in metric_data["Model"]]

        x_pos = np.arange(len(labels))
        bars = ax.bar(
            x_pos, metric_data["Mean"].values,
            yerr=metric_data["Std"].values,
            capsize=6, error_kw={"elinewidth": 2, "ecolor": "black"},
            color=colors, edgecolor="black", linewidth=1.2,
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=11)

        metric_label = metric.replace("Test_", "Test ")
        ax.set_title(
            f"{metric_label} — Full Feature Set ({n_seeds} seeds)",
            fontsize=14, pad=15,
        )
        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel(f"Mean {metric_label} ± Std", fontsize=12)
        ax.grid(axis="y", alpha=0.3)

        # Annotate bar tops with mean value
        for bar, mean_val in zip(bars, metric_data["Mean"].values):
            offset = bar.get_height() * 0.01  # slight offset above bar
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (offset if bar.get_height() >= 0 else offset),
                f"{mean_val:.2f}" if "Sharpe" in metric else f"{mean_val:.3f}",
                ha="center",
                va="bottom" if bar.get_height() >= 0 else "top",
                fontsize=9,
                fontweight="bold",
            )

    fig.suptitle(
        "Model Comparison — Full Feature Set (Cross-Seed)",
        fontsize=16, y=1.03,
    )
    fig.tight_layout()

    # ---- Save ----
    save_path = CHART_DIR / "model_comparison_full"
    for fmt in ("png", "pdf"):
        plt.savefig(save_path.with_suffix(f".{fmt}"), bbox_inches="tight")

    # Save CSV with per-model cross-seed stats
    pivot = summary.pivot_table(
        index="Model", columns="Metric", values=["Mean", "Std", "N"])
    pivot.columns = [f"{agg}_{met}" for agg, met in pivot.columns]
    pivot = pivot.reindex(ordered_models)
    pivot.to_csv(save_path.with_suffix(".csv"), encoding="utf-8-sig", float_format="%.6f")

    print(f"\nSaved:")
    for ext in ("png", "pdf", "csv"):
        print(f"   {save_path}.{ext}")

    plt.show()


if __name__ == "__main__":
    main()
