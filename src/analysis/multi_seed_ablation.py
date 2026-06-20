"""
多种子特征消融分析 —— 从已有 checkpoint 中提取跨 5 种子的特征消融统计。

输出:
  1. 每个模型 × 特征集的跨种子 Sharpe/IC (均值±标准差)
  2. ΔSharpe(fng→full) 跨种子统计，验证三组分类的符号一致性
  3. LaTeX 表格可直接替换论文中的 seed=1 单种子表
"""
import re
import os
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoint"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

KNOWN_MODELS = [
    "cnn_transformer", "lstm_transformer", "transformer", "lstm",
    "tcn", "modern_tcn", "patchtst", "xgboost", "dlinear", "timemixer",
]

FEATURE_TYPES = [
    "price_only", "price_funding", "price_funding_fng",
    "price_onchain", "price_long_onchain", "full",
]

FEATURE_SHORT = {
    "price_only": "only", "price_funding": "fund", "price_funding_fng": "fng",
    "price_onchain": "onch", "price_long_onchain": "lon", "full": "full",
}

MODEL_LABELS = {
    "cnn_transformer": "CNN-Transformer", "lstm_transformer": "LSTM-Transformer",
    "transformer": "Transformer", "lstm": "LSTM", "tcn": "TCN",
    "modern_tcn": "ModernTCN", "patchtst": "PatchTST", "xgboost": "XGBoost",
    "dlinear": "DLinear", "timemixer": "TimeMixer",
}

SEEDS = [0, 1, 2, 42, 123]


def parse_results(file_path):
    """Parse a results.txt file, return dict of all metrics."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    ft_match = re.search(r"feature_type:\s*(\S+)", content)
    feature_type = ft_match.group(1) if ft_match else "unknown"

    metrics = {}
    metric_patterns = {
        "IC": r"IC:\s*([-0-9.eE]+)",
        "PIC": r"PIC:\s*([-0-9.eE]+)",
        "DA": r"DA:\s*([-0-9.eE]+)",
        "MSE": r"MSE:\s*([-0-9.eE]+)",
        "Sharpe": r"Sharpe:\s*([-0-9.eE]+)",
        "IR": r"IR:\s*([-0-9.eE]+)",
        "MaxDrawdown": r"MaxDrawdown:\s*([-0-9.eE]+)",
        "AnnualReturn": r"AnnualReturn:\s*([-0-9.eE]+)",
    }

    for split in ["Train", "Val", "Test"]:
        line_match = re.search(rf"^{split}.*$", content, flags=re.MULTILINE)
        line = line_match.group(0) if line_match else ""
        for mn, mp in metric_patterns.items():
            m = re.search(mp, line)
            metrics[f"{split}_{mn}"] = float(m.group(1)) if m else None

    return feature_type, metrics


def infer_model(folder_name):
    for m in KNOWN_MODELS:
        if folder_name == m or folder_name.startswith(m + "_"):
            return m
    return folder_name


def extract_seed(folder_name):
    m = re.search(r'_seed(\d+)$', folder_name)
    return int(m.group(1)) if m else None


def collect_all_results():
    """Scan all checkpoints, return dict: (model, ft) -> list of per-seed metrics."""
    data = defaultdict(list)
    for folder in CHECKPOINT_DIR.iterdir():
        if not folder.is_dir():
            continue
        rp = folder / "results.txt"
        if not rp.exists():
            continue

        model = infer_model(folder.name)
        seed = extract_seed(folder.name)
        ft, metrics = parse_results(str(rp))

        # Skip non-standard lookback checkpoints (L12, L24, L96)
        if re.search(r'_L\d+_seed', folder.name):
            continue

        if seed is None:
            continue

        data[(model, ft)].append({
            "seed": seed,
            **{k: v for k, v in metrics.items() if k.startswith("Test_")}
        })

    return data


def build_ablation_table(data):
    """Build cross-seed Sharpe table (mean ± std) for all models × features."""
    rows = []
    for model in KNOWN_MODELS:
        row = {"Model": MODEL_LABELS.get(model, model)}
        row_sharpes = {}
        for ft in FEATURE_TYPES:
            entries = data.get((model, ft), [])
            sharpes = [e["Test_Sharpe"] for e in entries if e.get("Test_Sharpe") is not None]
            if len(sharpes) >= 3:
                row_sharpes[ft] = {
                    "mean": np.mean(sharpes),
                    "std": np.std(sharpes, ddof=1),
                    "n": len(sharpes),
                }
                row[f"{FEATURE_SHORT[ft]}"] = f"{row_sharpes[ft]['mean']:.2f}±{row_sharpes[ft]['std']:.2f}"
            elif len(sharpes) > 0:
                row_sharpes[ft] = {"mean": np.mean(sharpes), "std": 0.0, "n": len(sharpes)}
                row[f"{FEATURE_SHORT[ft]}"] = f"{row_sharpes[ft]['mean']:.2f}"
            else:
                row[f"{FEATURE_SHORT[ft]}"] = "--"

        # Add cross-feature stats
        valid_means = [v["mean"] for v in row_sharpes.values() if v["n"] >= 3]
        if valid_means:
            row["Mean"] = f"{np.mean(valid_means):.2f}"
            row["Std"] = f"{np.std(valid_means, ddof=1):.2f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def build_delta_sharpe_table(data):
    """Build ΔSharpe (fng→full) cross-seed statistics for all models."""
    rows = []
    for model in KNOWN_MODELS:
        fng_entries = data.get((model, "price_funding_fng"), [])
        full_entries = data.get((model, "full"), [])

        # Match by seed
        fng_by_seed = {e["seed"]: e.get("Test_Sharpe") for e in fng_entries}
        full_by_seed = {e["seed"]: e.get("Test_Sharpe") for e in full_entries}

        deltas = []
        for seed in SEEDS:
            if seed in fng_by_seed and seed in full_by_seed:
                sf = fng_by_seed[seed]
                sfull = full_by_seed[seed]
                if sf is not None and sfull is not None:
                    deltas.append(sfull - sf)

        if len(deltas) >= 3:
            mean_d = np.mean(deltas)
            std_d = np.std(deltas, ddof=1)
            pos_seeds = sum(1 for d in deltas if d > 0)
            if mean_d > 2:
                synergy = "强正向协同"
            elif mean_d > 0.5:
                synergy = "弱正向协同"
            elif mean_d > -0.5:
                synergy = "中性"
            elif mean_d > -1.5:
                synergy = "弱负向干扰"
            else:
                synergy = "负向干扰"

            rows.append({
                "Model": MODEL_LABELS.get(model, model),
                "Sharpe_fng": f"{np.mean([fng_by_seed[s] for s in SEEDS if s in fng_by_seed]):.2f}",
                "Sharpe_full": f"{np.mean([full_by_seed[s] for s in SEEDS if s in full_by_seed]):.2f}",
                "ΔSharpe": f"{mean_d:+.2f}±{std_d:.2f}",
                "正种子数": f"{pos_seeds}/{len(deltas)}",
                "协同类型": synergy,
            })

    df = pd.DataFrame(rows)
    # Sort by ΔSharpe descending
    df["_sort"] = df["ΔSharpe"].apply(lambda x: float(x.split("±")[0]))
    df = df.sort_values("_sort", ascending=False).drop(columns=["_sort"])
    return df


def build_seed_consistency_table(data):
    """
    Per-seed ΔSharpe (fng→full) for key models, to verify sign consistency.
    Output: LaTeX-ready table showing per-seed delta.
    """
    key_models = ["cnn_transformer", "timemixer", "lstm_transformer", "transformer", "lstm"]
    rows = []
    for model in key_models:
        fng_by_seed = {
            e["seed"]: e.get("Test_Sharpe")
            for e in data.get((model, "price_funding_fng"), [])
        }
        full_by_seed = {
            e["seed"]: e.get("Test_Sharpe")
            for e in data.get((model, "full"), [])
        }
        row = {"Model": MODEL_LABELS.get(model, model)}
        all_pos = True
        for seed in SEEDS:
            sf = fng_by_seed.get(seed)
            sfull = full_by_seed.get(seed)
            if sf is not None and sfull is not None:
                delta = sfull - sf
                row[f"seed={seed}"] = f"{delta:+.2f}"
                if delta <= 0:
                    all_pos = False
            else:
                row[f"seed={seed}"] = "N/A"
        row["全正"] = "✓" if all_pos and len([v for v in row.values() if v != "N/A"]) >= 4 else "✗"
        rows.append(row)

    return pd.DataFrame(rows)


def to_latex_ablation(df):
    """Generate LaTeX table for multi-seed feature ablation Sharpe."""
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{各模型在六种特征组合下跨5种子的测试集夏普比率（均值$\pm$标准差）}")
    lines.append(r"\label{tab:feature_ablation_multi_seed}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{1.5pt}")
    cols = "l" + "c" * (len(FEATURE_TYPES) + 2)
    lines.append(r"\begin{tabular}{" + cols + "}")
    lines.append(r"\toprule")
    header = "模型 & " + " & ".join(FEATURE_SHORT[ft] for ft in FEATURE_TYPES) + r" & 均值 & 标准差 \\"
    lines.append(header)
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        cells = [row["Model"]]
        for ft in FEATURE_TYPES:
            cells.append(str(row.get(FEATURE_SHORT[ft], "--")))
        cells.append(str(row.get("Mean", "--")))
        cells.append(str(row.get("Std", "--")))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\multicolumn{" + str(len(FEATURE_TYPES) + 3) + r"}{p{\textwidth}}{\footnotesize ")
    lines.append(r"注：only=price\_only; fund=price\_funding; fng=price\_funding\_fng; "
                 r"onch=price\_onchain; lon=price\_long\_onchain; full=全特征。"
                 r"跨5种子（0,1,2,42,123）。XGBoost仅price\_funding\_fng和full特征集有可用模型。}")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def to_latex_delta(df):
    """Generate LaTeX table for ΔSharpe cross-seed statistics."""
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{链上数据协同效应跨种子统计：$\Delta$Sharpe (fng$\to$full) 的均值、标准差与符号一致性}")
    lines.append(r"\label{tab:delta_multi_seed}")
    lines.append(r"\small")
    cols = "lcccc"
    lines.append(r"\begin{tabular}{" + cols + "}")
    lines.append(r"\toprule")
    lines.append(r"模型 & Sharpe$_{\text{fng}}$ & Sharpe$_{\text{full}}$ & $\Delta$Sharpe & 协同类型 \\")
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        cells = [str(row[c]) for c in ["Model", "Sharpe_fng", "Sharpe_full", "ΔSharpe", "协同类型"]]
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\multicolumn{5}{p{0.85\textwidth}}{\footnotesize "
                 r"注：跨5种子（0,1,2,42,123）均值$\pm$标准差。$\Delta$Sharpe = Sharpe$_{\text{full}}$ $-$ Sharpe$_{\text{fng}}$。"
                 r"协同分类阈值：$>+2$强正向, $(0, +2]$弱正向, $\approx 0$中性, $[-1.5,0)$弱负向, $<-1.5$负向干扰。}")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Multi-seed feature ablation analysis")
    parser.add_argument("--output", type=str, default="all",
                        help="Output type: table, delta, consistency, latex, all")
    args = parser.parse_args()

    print("[INFO] Scanning checkpoints...")
    data = collect_all_results()
    total = sum(len(v) for v in data.values())
    print(f"[INFO] Found {total} trained models across {len(data)} (model,ft) combinations")

    if args.output in ("table", "all"):
        df_ablation = build_ablation_table(data)
        print("\n" + "=" * 100)
        print("特征消融跨5种子夏普比率（均值±标准差）")
        print("=" * 100)
        print(df_ablation.to_string(index=False))
        df_ablation.to_csv(RESULTS_DIR / "feature_ablation_multi_seed.csv", index=False)

    if args.output in ("delta", "all"):
        df_delta = build_delta_sharpe_table(data)
        print("\n" + "=" * 80)
        print("ΔSharpe (fng→full) 跨种子统计")
        print("=" * 80)
        print(df_delta.to_string(index=False))
        df_delta.to_csv(RESULTS_DIR / "delta_sharpe_multi_seed.csv", index=False)

    if args.output in ("consistency", "all"):
        df_cons = build_seed_consistency_table(data)
        print("\n" + "=" * 80)
        print("逐种子ΔSharpe符号一致性")
        print("=" * 80)
        print(df_cons.to_string(index=False))

    if args.output in ("latex", "all"):
        df_ablation = build_ablation_table(data)
        df_delta = build_delta_sharpe_table(data)
        latex_abl = to_latex_ablation(df_ablation)
        latex_del = to_latex_delta(df_delta)

        abl_path = RESULTS_DIR / "feature_ablation_multi_seed.tex"
        delta_path = RESULTS_DIR / "delta_sharpe_multi_seed.tex"

        with open(abl_path, "w", encoding="utf-8") as f:
            f.write(latex_abl)
        with open(delta_path, "w", encoding="utf-8") as f:
            f.write(latex_del)

        print(f"\n[OK] LaTeX tables saved:")
        print(f"  {abl_path}")
        print(f"  {delta_path}")


if __name__ == "__main__":
    main()
