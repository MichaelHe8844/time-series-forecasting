import os
import re
import pandas as pd
from pathlib import Path
import argparse
from collections import defaultdict

KNOWN_MODELS = [
    "lstm_transformer",
    "cnn_transformer",
    "transformer",
    "lstm",
    "tcn",
    "xgboost",
    "patchtst",
    "modern_tcn",
    "dlinear",
    "timemixer",
]


def parse_results_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    ft_match = re.search(r"feature_type:\s*(\S+)", content)
    feature_type = ft_match.group(1) if ft_match else "unknown"

    lb_match = re.search(r"lookback:\s*(\d+)", content)
    lookback = int(lb_match.group(1)) if lb_match else 48

    metrics = {}
    metric_patterns = {
        "IC": r"IC:\s*([-\d\.eE]+)",
        "PIC": r"PIC:\s*([-\d\.eE]+)",
        "DA": r"DA:\s*([-\d\.eE]+)",
        "MSE": r"MSE:\s*([-\d\.eE]+)",
        "Sharpe": r"Sharpe:\s*([-\d\.eE]+)",
        "IR": r"IR:\s*([-\d\.eE]+)",
        "MaxDrawdown": r"MaxDrawdown:\s*([-\d\.eE]+)",
        "AnnualReturn": r"AnnualReturn:\s*([-\d\.eE]+)",
    }

    for split in ["Train", "Val", "Test"]:
        line_match = re.search(rf"^{split}.*$", content, flags=re.MULTILINE)
        line = line_match.group(0) if line_match else ""

        for metric_name, metric_pattern in metric_patterns.items():
            metric_match = re.search(metric_pattern, line)
            metrics[f"{split}_{metric_name}"] = float(metric_match.group(1)) if metric_match else None

    return feature_type, lookback, metrics


def infer_model_name(folder_name: str):
    for model in KNOWN_MODELS:
        if folder_name == model or folder_name.startswith(model + "_"):
            return model
    return folder_name


def extract_seed(folder_name: str):
    m = re.search(r'_seed(\d+)$', folder_name)
    return int(m.group(1)) if m else None


def extract_lookback_from_folder(folder_name: str):
    m = re.search(r'_L(\d+)_seed', folder_name)
    return int(m.group(1)) if m else 48


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--include-seed", action="store_true",
                        help="Include all seeds (no dedup by model/ft)")
    parser.add_argument("--lookback-filter", type=int, default=None,
                        help="Only include results with this lookback value")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "configs" / "config.py"

    # 读取 seed
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()
        seed_match = re.search(r'"seed":\s*(\d+)', content)
        seed_to_use = int(seed_match.group(1)) if seed_match else 42
    else:
        seed_to_use = 42

    if args.seed is not None:
        seed_to_use = args.seed

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    checkpoint_dir = project_root / "checkpoint"

    if not checkpoint_dir.exists():
        print(f"[ERROR] 未找到 checkpoint 目录: {checkpoint_dir}")
        return

    latest_results = defaultdict(dict)  # (model, feature_type) -> (results_path, mtime)

    for folder in checkpoint_dir.iterdir():
        if not folder.is_dir():
            continue

        results_path = folder / "results.txt"
        if not results_path.exists():
            continue

        folder_name = folder.name
        model_name = infer_model_name(folder_name)

        ft, lookback, _ = parse_results_file(str(results_path))
        seed = extract_seed(folder_name)

        # When a specific seed is requested, skip folders that do not match
        if args.seed is not None and seed is not None and seed != seed_to_use:
            continue

        if args.lookback_filter is not None and lookback != args.lookback_filter:
            continue

        if args.include_seed:
            key = (model_name, ft, lookback, seed)
        else:
            key = (model_name, ft)

        mtime = results_path.stat().st_mtime

        if key not in latest_results or mtime > latest_results[key][1]:
            latest_results[key] = (results_path, mtime, lookback)

    records = []
    for key, (results_path, _, lookback) in latest_results.items():
        if args.include_seed:
            model_name, feature_type, lookback_val, seed = key
        else:
            model_name, feature_type = key
            lookback_val = lookback
            seed = None

        _, _, metrics = parse_results_file(str(results_path))
        record = {
            "Model": model_name,
            "Feature_Type": feature_type,
            "Lookback": lookback_val,
            **metrics
        }
        if seed is not None:
            record["Seed"] = seed
        records.append(record)

    if not records:
        print("[ERROR] 未找到任何 results.txt 文件")
        return

    df = pd.DataFrame(records)
    df = df.sort_values(by=["Feature_Type", "Model"]).reset_index(drop=True)

    preferred_metric_order = [
        "Train_IC", "Train_PIC", "Train_DA", "Train_MSE", "Train_Sharpe", "Train_IR", "Train_MaxDrawdown", "Train_AnnualReturn",
        "Val_IC", "Val_PIC", "Val_DA", "Val_MSE", "Val_Sharpe", "Val_IR", "Val_MaxDrawdown", "Val_AnnualReturn",
        "Test_IC", "Test_PIC", "Test_DA", "Test_MSE", "Test_Sharpe", "Test_IR", "Test_MaxDrawdown", "Test_AnnualReturn",
    ]
    metric_cols = [col for col in preferred_metric_order if col in df.columns]
    extra_cols = ["Model", "Feature_Type", "Lookback"]
    if args.include_seed:
        extra_cols.append("Seed")
    df = df[extra_cols + metric_cols]

    print("\n" + "=" * 80)
    print("所有实验结果汇总表（已去重）")
    print("=" * 80)
    print(df.to_string(index=False, float_format="%.4f"))
    print("=" * 80)

    if args.include_seed:
        output_csv = results_dir / "results_summary_lookback.csv"
        output_md = results_dir / "results_summary_lookback.md"
    else:
        output_csv = results_dir / f"results_summary_seed{seed_to_use}.csv"
        output_md = results_dir / f"results_summary_seed{seed_to_use}.md"

    df.to_csv(output_csv, index=False, float_format="%.6f")
    df.to_markdown(output_md, index=False, floatfmt=".4f")

    print(f"\n[INFO] 汇总结果已保存（已自动去重）:")
    print(f"   CSV  -> {output_csv}")
    print(f"   Markdown -> {output_md}")


if __name__ == "__main__":
    main()
