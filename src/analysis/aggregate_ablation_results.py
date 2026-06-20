"""
汇总缺口 3-5 的消融实验结果，生成 LaTeX 表格。

用法:
  python src/analysis/aggregate_ablation_results.py --exp loss
  python src/analysis/aggregate_ablation_results.py --exp arch
  python src/analysis/aggregate_ablation_results.py --exp hparam
  python src/analysis/aggregate_ablation_results.py --exp all
"""

import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoint"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def parse_results(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    metrics = {}
    patterns = {
        "IC": r"IC:\s*([-0-9.eE]+)", "Sharpe": r"Sharpe:\s*([-0-9.eE]+)",
        "MSE": r"MSE:\s*([-0-9.eE]+)", "MaxDrawdown": r"MaxDrawdown:\s*([-0-9.eE]+)",
        "AnnualReturn": r"AnnualReturn:\s*([-0-9.eE]+)",
    }
    for split in ["Test"]:
        line_match = re.search(rf"^{split}.*$", content, flags=re.MULTILINE)
        line = line_match.group(0) if line_match else ""
        for mn, mp in patterns.items():
            m = re.search(mp, line)
            metrics[f"{split}_{mn}"] = float(m.group(1)) if m else None
    return metrics


def aggregate_loss_ablation():
    """缺口 3: 损失函数消融结果"""
    variants = ["huber", "mse", "huber_rank"]
    models = ["cnn_transformer", "timemixer"]
    seeds = [0, 1, 2]

    rows = []
    for model in models:
        for variant in variants:
            sharpes, ics = [], []
            for seed in seeds:
                # 确定 checkpoint 路径
                ckpt_pattern = f"{model}_ablation_{variant}"
                found = False
                for ckpt_dir in CHECKPOINT_DIR.iterdir():
                    if not ckpt_dir.is_dir():
                        continue
                    if ckpt_dir.name.startswith(ckpt_pattern) and f"_seed{seed}" in ckpt_dir.name:
                        rp = ckpt_dir / "results.txt"
                        if rp.exists():
                            m = parse_results(str(rp))
                            if m.get("Test_Sharpe") is not None:
                                sharpes.append(m["Test_Sharpe"])
                            if m.get("Test_IC") is not None:
                                ics.append(m["Test_IC"])
                            found = True
                            break

                if not found:
                    # 检查标准 checkpoint (huber_rank 变体 = 标准训练)
                    if variant == "huber_rank":
                        std_ckpt = CHECKPOINT_DIR / f"{model}_full_seed{seed}" / "results.txt"
                        if std_ckpt.exists():
                            m = parse_results(str(std_ckpt))
                            if m.get("Test_Sharpe") is not None:
                                sharpes.append(m["Test_Sharpe"])
                            if m.get("Test_IC") is not None:
                                ics.append(m["Test_IC"])

            if sharpes:
                rows.append({
                    "模型": {"cnn_transformer": "CNN-Transformer",
                             "timemixer": "TimeMixer"}[model],
                    "损失函数": {"huber": "纯Huber", "mse": "纯MSE",
                                "huber_rank": "Huber+Ranking(当前)"}[variant],
                    "Sharpe": f"{np.mean(sharpes):.2f}±{np.std(sharpes, ddof=1):.2f}" if len(sharpes) > 1 else f"{sharpes[0]:.2f}",
                    "IC": f"{np.mean(ics):.4f}" if ics else "N/A",
                    "种子数": len(sharpes),
                })

    df = pd.DataFrame(rows)
    return df


def aggregate_arch_ablation():
    """缺口 4: 架构消融结果"""
    variants = ["standard", "cnn_only", "tr_only", "tr_matched"]
    seeds = [0, 1, 2]

    rows = []
    for variant in variants:
        sharpes, ics = [], []
        for seed in seeds:
            if variant == "standard":
                ckpt_path = CHECKPOINT_DIR / f"cnn_transformer_full_seed{seed}" / "results.txt"
            else:
                ckpt_path = None
                ckpt_pattern = f"cnn_transformer_ablation_{variant}_full_seed{seed}"
                for d in CHECKPOINT_DIR.iterdir():
                    if d.is_dir() and d.name == ckpt_pattern:
                        ckpt_path = d / "results.txt"
                        break

            if ckpt_path and ckpt_path.exists():
                m = parse_results(str(ckpt_path))
                if m.get("Test_Sharpe") is not None:
                    sharpes.append(m["Test_Sharpe"])
                if m.get("Test_IC") is not None:
                    ics.append(m["Test_IC"])

        if sharpes:
            rows.append({
                "架构变体": {
                    "standard": "CNN-Transformer (完整)",
                    "cnn_only": "CNN-only (去Transformer)",
                    "tr_only": "Transformer-only (去CNN)",
                    "tr_matched": "Transformer-only (参数量匹配)",
                }[variant],
                "Sharpe": f"{np.mean(sharpes):.2f}±{np.std(sharpes, ddof=1):.2f}" if len(sharpes) > 1 else f"{sharpes[0]:.2f}",
                "IC": f"{np.mean(ics):.4f}" if ics else "N/A",
                "种子数": len(sharpes),
            })

    df = pd.DataFrame(rows)
    return df


def aggregate_hparam_ablation():
    """缺口 5: 超参数敏感性"""
    models = ["cnn_transformer", "timemixer", "transformer"]
    params = {"lr": [5e-5, 1e-4, 2e-4], "dropout": [0.10, 0.13, 0.20]}

    rows = []
    for model in models:
        for param, values in params.items():
            for val in values:
                sharpes = []
                for seed in [0, 1]:
                    if val == (1e-4 if param == "lr" else 0.13):
                        # 基线
                        ckpt_path = CHECKPOINT_DIR / f"{model}_full_seed{seed}" / "results.txt"
                    else:
                        # 查找消融 checkpoint
                        val_str = f"{val:.0e}" if param == "lr" else f"{val:.2f}"
                        ckpt_pattern = f"{model}_ablation_standard_full_seed{seed}"
                        ckpt_path = None
                        # 简化：从 checkpoint 目录中搜索
                        for d in CHECKPOINT_DIR.iterdir():
                            if d.is_dir() and f"{model}_ablation" in d.name and f"_seed{seed}" in d.name:
                                rp = d / "results.txt"
                                if rp.exists():
                                    # 读取 params
                                    with open(rp, "r") as f:
                                        content = f.read()
                                    if param == "lr" and f"lr: {val}" in content:
                                        ckpt_path = rp
                                        break
                                    elif param == "dropout" and f"dropout: {val}" in content:
                                        ckpt_path = rp
                                        break

                    if ckpt_path and ckpt_path.exists():
                        m = parse_results(str(ckpt_path))
                        if m.get("Test_Sharpe") is not None:
                            sharpes.append(m["Test_Sharpe"])

                if sharpes:
                    model_label = {"cnn_transformer": "CNN-Transformer",
                                   "timemixer": "TimeMixer",
                                   "transformer": "Transformer"}[model]
                    param_label = "学习率" if param == "lr" else "Dropout"
                    rows.append({
                        "模型": model_label,
                        "超参数": param_label,
                        "值": f"{val:.0e}" if param == "lr" else f"{val:.2f}",
                        "Sharpe": f"{np.mean(sharpes):.2f}",
                        "种子数": len(sharpes),
                    })

    df = pd.DataFrame(rows)
    return df


def to_latex_loss(df):
    lines = [r"\begin{table}[ht]", r"\centering",
             r"\caption{损失函数消融：不同损失函数下的测试集夏普比率（full特征集）}",
             r"\label{tab:loss_ablation}", r"\small",
             r"\begin{tabular}{lccc}", r"\toprule",
             r"模型 & 损失函数 & Sharpe & IC \\", r"\midrule"]
    for _, row in df.iterrows():
        lines.append(f"{row['模型']} & {row['损失函数']} & {row['Sharpe']} & {row['IC']} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def to_latex_arch(df):
    lines = [r"\begin{table}[ht]", r"\centering",
             r"\caption{架构组件消融：CNN-Transformer各组件对full集预测性能的贡献}",
             r"\label{tab:arch_ablation}", r"\small",
             r"\begin{tabular}{lccc}", r"\toprule",
             r"架构变体 & Sharpe & IC & 说明 \\", r"\midrule"]
    for _, row in df.iterrows():
        lines.append(f"{row['架构变体']} & {row['Sharpe']} & {row['IC']} & -- \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True,
                        choices=["loss", "arch", "hparam", "all"])
    args = parser.parse_args()

    if args.exp in ("loss", "all"):
        df = aggregate_loss_ablation()
        print("\n=== 缺口 3: 损失函数消融 ===")
        print(df.to_string(index=False))
        latex = to_latex_loss(df)
        with open(RESULTS_DIR / "loss_ablation.tex", "w", encoding="utf-8") as f:
            f.write(latex)
        print(f"\n[OK] Saved: {RESULTS_DIR / 'loss_ablation.tex'}")

    if args.exp in ("arch", "all"):
        df = aggregate_arch_ablation()
        print("\n=== 缺口 4: 架构消融 ===")
        print(df.to_string(index=False))
        latex = to_latex_arch(df)
        with open(RESULTS_DIR / "arch_ablation.tex", "w", encoding="utf-8") as f:
            f.write(latex)
        print(f"\n[OK] Saved: {RESULTS_DIR / 'arch_ablation.tex'}")

    if args.exp in ("hparam", "all"):
        df = aggregate_hparam_ablation()
        print("\n=== 缺口 5: 超参数敏感性 ===")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
