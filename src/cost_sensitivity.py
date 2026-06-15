"""
Transaction cost sensitivity analysis.
Tests how Sharpe / AnnualReturn / MaxDD degrade as trading fees increase
from 0 bps to 50 bps, across the top-performing models on the full feature set.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import cfg
from src.common import set_seed, SeqDataset, load_data
from src.metrics import calc_strategy_returns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHART_DIR = PROJECT_ROOT / "charts"
RESULT_DIR = PROJECT_ROOT / "results"
CHART_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

FEE_LEVELS_BPS = [0, 5, 10, 15, 20, 25, 30, 40, 50]
MODELS = ["cnn_transformer", "transformer", "lstm_transformer", "xgboost", "dlinear", "timemixer"]
MODEL_LABELS = {
    "cnn_transformer": "CNN-Transformer",
    "transformer": "Transformer",
    "lstm_transformer": "LSTM-Transformer",
    "xgboost": "XGBoost",
    "dlinear": "DLinear",
    "timemixer": "TimeMixer",
}
FEATURE_TYPE = "full"
SEEDS = (0, 1, 2, 42, 123)
PERIODS_PER_YEAR = 2190

sns.set_style("whitegrid")
plt.rcParams["savefig.dpi"] = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def calc_strategy_returns_at_fee(pred, target, fee_bps):
    fee = fee_bps / 10000.0
    positions = np.sign(pred)
    raw_returns = positions * target
    if len(positions) > 1:
        position_changes = np.abs(positions[1:] - positions[:-1])
        costs = fee * position_changes
        costs = np.concatenate(([0.0], costs))
    else:
        costs = np.zeros_like(raw_returns)
    net_returns = raw_returns - costs
    return np.clip(net_returns, -0.3, 0.3)


def calc_sharpe(returns):
    returns = returns[np.isfinite(returns)]
    if returns.size < 2:
        return 0.0
    std = returns.std(ddof=1)
    if std <= 1e-12 or np.isnan(std):
        return 0.0
    return float(np.sqrt(PERIODS_PER_YEAR) * returns.mean() / std)


def calc_annual_return(returns):
    returns = returns[np.isfinite(returns)]
    if returns.size == 0:
        return 0.0
    return float(np.mean(returns) * PERIODS_PER_YEAR)


def calc_max_drawdown(returns):
    returns = returns[np.isfinite(returns)]
    if returns.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / np.where(running_max == 0, 1.0, running_max) - 1.0
    return float(abs(drawdown.min()))


def load_model_predictions(model_name, feature_type, seed):
    """Load a trained model and generate predictions. Avoids importing statistical_tests."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = PROJECT_ROOT / "checkpoint" / f"{model_name}_{feature_type}_seed{seed}"

    # Temporarily override config
    original_seed = cfg["seed"]
    original_ft = cfg["data"]["feature_type"]
    cfg["seed"] = seed
    cfg["data"]["feature_type"] = feature_type

    set_seed(seed)
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    if model_name == "xgboost":
        try:
            import xgboost as xgb
        except ImportError:
            cfg["seed"] = original_seed
            cfg["data"]["feature_type"] = original_ft
            raise RuntimeError("xgboost not installed")
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        json_path = ckpt_dir / "best.json"
        if not json_path.exists():
            cfg["seed"] = original_seed
            cfg["data"]["feature_type"] = original_ft
            raise FileNotFoundError(f"XGBoost checkpoint not found: {json_path}")
        model = xgb.Booster()
        model.load_model(str(json_path))
        dtest = xgb.DMatrix(X_test_flat)
        preds = model.predict(dtest)
        cfg["seed"] = original_seed
        cfg["data"]["feature_type"] = original_ft
        return preds, y_test

    # Deep learning models
    input_dim = X_train.shape[2]

    if model_name == "cnn_transformer":
        from src.models.cnn_transformer import CNNTransformer
        model = CNNTransformer(input_dim, cfg[model_name])
    elif model_name == "lstm_transformer":
        from src.models.lstm_transformer import LSTMTransformer
        model = LSTMTransformer(input_dim, cfg[model_name])
    elif model_name == "transformer":
        from src.models.baseline_transformer import Transformer
        model = Transformer(input_dim, cfg[model_name])
    elif model_name == "lstm":
        from src.models.baseline_lstm import BiLSTM
        model = BiLSTM(input_dim, cfg[model_name])
    elif model_name == "tcn":
        from src.models.baseline_tcn import TCN
        model = TCN(input_dim, cfg[model_name])
    elif model_name == "patchtst":
        from src.models.patchtst import PatchTST
        model = PatchTST(input_dim, cfg[model_name])
    elif model_name == "modern_tcn":
        from src.models.modern_tcn import ModernTCN
        model = ModernTCN(input_dim, cfg[model_name])
    elif model_name == "dlinear":
        from src.models.dlinear import DLinear
        model = DLinear(input_dim, cfg[model_name])
    elif model_name == "timemixer":
        from src.models.timemixer import TimeMixer
        model = TimeMixer(input_dim, cfg[model_name])
    else:
        cfg["seed"] = original_seed
        cfg["data"]["feature_type"] = original_ft
        raise ValueError(f"Unknown model: {model_name}")

    ckpt_path = ckpt_dir / "best.pth"
    if not ckpt_path.exists():
        cfg["seed"] = original_seed
        cfg["data"]["feature_type"] = original_ft
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
    model = model.to(device)
    model.eval()

    test_dataset = SeqDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=cfg["train"]["batch_size"],
                             shuffle=False, num_workers=0, pin_memory=False)

    preds_list = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            pred = model(x)
            preds_list.append(pred.cpu().numpy())
    preds = np.concatenate(preds_list)
    preds = np.clip(preds, cfg["pred_clip_min"], cfg["pred_clip_max"])

    cfg["seed"] = original_seed
    cfg["data"]["feature_type"] = original_ft

    return preds, y_test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # 1. Collect predictions across seeds
    print("Loading predictions across seeds...")
    all_preds = defaultdict(list)
    all_trues = defaultdict(list)

    for model_name in MODELS:
        for seed in SEEDS:
            ckpt_file = "best.json" if model_name == "xgboost" else "best.pth"
            ckpt_path = PROJECT_ROOT / "checkpoint" / f"{model_name}_{FEATURE_TYPE}_seed{seed}" / ckpt_file
            if not ckpt_path.exists():
                print(f"  [SKIP] {model_name} seed={seed} — no checkpoint")
                continue
            try:
                preds, trues = load_model_predictions(model_name, FEATURE_TYPE, seed)
                mask = np.isfinite(preds) & np.isfinite(trues)
                all_preds[model_name].append(preds[mask])
                all_trues[model_name].append(trues[mask])
                print(f"  [OK] {model_name} seed={seed}")
            except Exception as e:
                print(f"  [ERR] {model_name} seed={seed}: {e}")

    # 2. For each fee level, compute cross-seed averaged metrics
    print(f"\nComputing cost sensitivity across {len(FEE_LEVELS_BPS)} fee levels...")
    records = []

    for model_name in MODELS:
        if model_name not in all_preds or len(all_preds[model_name]) == 0:
            continue

        n_seeds_avail = len(all_preds[model_name])
        print(f"  {model_name}: {n_seeds_avail}/{len(SEEDS)} seeds loaded")

        preds_stack = np.stack(all_preds[model_name], axis=0)
        preds_avg = preds_stack.mean(axis=0)
        trues_ref = all_trues[model_name][0]

        for fee_bps in FEE_LEVELS_BPS:
            returns = calc_strategy_returns_at_fee(preds_avg, trues_ref, fee_bps)
            sharpe = calc_sharpe(returns)
            ann_ret = calc_annual_return(returns)
            max_dd = calc_max_drawdown(returns)

            records.append({
                "Model": MODEL_LABELS[model_name],
                "Fee_bps": fee_bps,
                "Sharpe": sharpe,
                "AnnualReturn": ann_ret,
                "MaxDrawdown": max_dd,
            })

    df = pd.DataFrame(records)

    # 3. Save CSV
    csv_path = RESULT_DIR / "cost_sensitivity.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\nSaved: {csv_path}")

    # 4. Print summary
    print("\n" + "=" * 80)
    print("Cost Sensitivity Summary (Cross-Seed Avg)")
    print("=" * 80)
    pivot_sharpe = df.pivot(index="Fee_bps", columns="Model", values="Sharpe")
    print("\n--- Sharpe Ratio ---")
    print(pivot_sharpe.round(2).to_string())

    # 5. Breakeven fees
    print("\n--- Approximate Breakeven Fee (Sharpe -> 0) ---")
    for model_name in MODELS:
        label = MODEL_LABELS[model_name]
        sub = df[df["Model"] == label].sort_values("Fee_bps")
        if sub.empty:
            continue
        sharpes = sub["Sharpe"].values
        fees = sub["Fee_bps"].values
        if sharpes[0] <= 0:
            print(f"  {label}: N/A (Sharpe <= 0 even at 0 bps)")
            continue
        found = False
        for i in range(len(sharpes) - 1):
            if sharpes[i] >= 0 and sharpes[i + 1] <= 0:
                t = sharpes[i] / (sharpes[i] - sharpes[i + 1])
                be_fee = fees[i] + t * (fees[i + 1] - fees[i])
                print(f"  {label}: ~{be_fee:.1f} bps")
                found = True
                break
        if not found:
            print(f"  {label}: >{fees[-1]} bps (still profitable at max fee)")

    # 6. Sharpe decay relative to 5 bps baseline
    print("\n--- Sharpe Decay vs 5 bps Baseline ---")
    for model_name in MODELS:
        label = MODEL_LABELS[model_name]
        sub = df[df["Model"] == label].sort_values("Fee_bps")
        if sub.empty:
            continue
        base_sharpe = sub[sub["Fee_bps"] == 5]["Sharpe"].values[0]
        for _, row in sub.iterrows():
            decay = (row["Sharpe"] - base_sharpe) / base_sharpe * 100 if base_sharpe != 0 else 0
            if row["Fee_bps"] in [10, 20, 30, 50]:
                print(f"  {label} @ {int(row['Fee_bps'])} bps: Sharpe={row['Sharpe']:.2f} ({decay:+.1f}%)")

    # 7. Generate chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    palette = sns.color_palette("Set2", n_colors=len(MODELS))

    for ax_idx, (metric, ylabel, title) in enumerate([
        ("Sharpe", "Annualized Sharpe Ratio", "Sharpe Ratio vs Transaction Cost"),
        ("AnnualReturn", "Annualized Return", "Annual Return vs Transaction Cost"),
        ("MaxDrawdown", "Max Drawdown", "Max Drawdown vs Transaction Cost"),
    ]):
        ax = axes[ax_idx]
        for idx, model_name in enumerate(MODELS):
            label = MODEL_LABELS[model_name]
            sub = df[df["Model"] == label]
            if sub.empty:
                continue
            ax.plot(sub["Fee_bps"], sub[metric], "o-", color=palette[idx],
                    label=label, linewidth=2, markersize=6)
        if metric != "MaxDrawdown":
            ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(x=5, color="red", linestyle=":", linewidth=0.8, alpha=0.5, label="Baseline (5 bps)")
        ax.set_xlabel("Transaction Cost (bps)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.set_xlim(0, 52)

    fig.suptitle("Transaction Cost Sensitivity Analysis (Full Feature Set, Cross-Seed Avg)",
                 fontsize=16, y=1.02)
    fig.tight_layout()

    for fmt in ("png", "pdf"):
        save_path = CHART_DIR / f"cost_sensitivity.{fmt}"
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close()

    # 8. Generate LaTeX table (compact format to fit page width)
    latex_path = RESULT_DIR / "cost_sensitivity.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated cost sensitivity LaTeX table\n")
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\caption{Transaction cost sensitivity analysis --- strategy performance across fee levels (cross-seed average)}\n")
        f.write("\\label{tab:cost_sensitivity}\n")
        f.write("\\footnotesize\n")
        f.write("\\setlength{\\tabcolsep}{4pt}\n")
        available_models = [m for m in MODELS if m in all_preds and len(all_preds[m]) > 0]
        n_models = len(available_models)
        f.write("\\begin{tabular}{l" + "c" * n_models + "}\n")
        f.write("\\toprule\n")
        f.write("Fee & " + " & ".join(MODEL_LABELS[m] for m in available_models) + " \\\\\n")
        f.write("\\midrule\n")

        for fee_bps in FEE_LEVELS_BPS:
            row = [f"{fee_bps} bps"]
            for model_name in available_models:
                label = MODEL_LABELS[model_name]
                sub_ref = df[(df["Model"] == label) & (df["Fee_bps"] == fee_bps)]
                if sub_ref.empty:
                    row.append("--")
                else:
                    s = sub_ref.iloc[0]
                    sr = f"{s['Sharpe']:.2f}"
                    ar = f"{s['AnnualReturn']*100:.0f}\\%"
                    dd = f"{s['MaxDrawdown']*100:.0f}\\%"
                    # Use $-$ for negative numbers
                    sr_str = f"${'-' if s['Sharpe'] < 0 else ''}{abs(s['Sharpe']):.2f}$" if s['Sharpe'] < 0 else f"{s['Sharpe']:.2f}"
                    ar_str = f"${'-' if s['AnnualReturn'] < 0 else ''}{abs(s['AnnualReturn']*100):.0f}\\%$" if s['AnnualReturn'] < 0 else f"{s['AnnualReturn']*100:.0f}\\%"
                    row.append(f"{sr_str} / {ar_str} / {dd}")
            f.write(" & ".join(row) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\multicolumn{" + str(n_models + 1) + "}{l}{\\footnotesize Format per cell: Sharpe ratio / Annual return / Max drawdown. "
                "Baseline at 5 bps. Computed from cross-seed averaged predictions.} \\\\\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Saved: {latex_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
