"""
缺口 2：滑动窗口验证在 price_funding_fng 集上的补充实验。

问题：当前只在 full 集上做了8窗口滑动验证。核心论点"条件有效性"
（fng→full 的 ΔSharpe）在滑动窗口下没有被验证。

方案：对 CNN-Transformer 和 TimeMixer，在 price_funding_fng 上跑同样的
8窗口滑动验证，与已有的 full 集结果对比，计算每个窗口的 ΔSharpe。

使用滑动窗口数据构建方法：
- 从原始 merged CSV 中按时间窗口切分
- 每个窗口: train 2年, test 6个月, step 6个月
- 覆盖 2022 H1 ~ 2025 H2

"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.py"


# Sliding window definitions: (train_start, train_end, test_start, test_end)
# 8 windows, 2-year train, 6-month test, 6-month step
WINDOWS = [
    # (train_start, train_end, test_start, test_end)
    ("2020-01-01", "2021-12-31", "2022-01-01", "2022-06-30"),   # W0
    ("2020-07-01", "2022-06-30", "2022-07-01", "2022-12-31"),   # W1
    ("2021-01-01", "2022-12-31", "2023-01-01", "2023-06-30"),   # W2
    ("2021-07-01", "2023-06-30", "2023-07-01", "2023-12-31"),   # W3
    ("2022-01-01", "2023-12-31", "2024-01-01", "2024-06-30"),   # W4
    ("2022-07-01", "2024-06-30", "2024-07-01", "2024-12-31"),   # W5
    ("2023-01-01", "2024-12-31", "2025-01-01", "2025-06-30"),   # W6
    ("2023-07-01", "2025-06-30", "2025-07-01", "2025-12-31"),   # W7
]


def build_window_features(feature_type, window_idx, window_def):
    """
    为指定窗口构建特征。
    复用 build_features.py 的逻辑，但用时间窗口而非固定比例切分。
    """
    train_start, train_end, test_start, test_end = window_def

    # Read merged CSV
    merged_dir = PROJECT_ROOT / "data" / "processed" / "merged"
    dataset_map = {
        "price_funding_fng": "merged_price_funding_fng_4h.csv",
        "full": "merged_full_4h.csv",
    }
    csv_name = dataset_map.get(feature_type)
    if not csv_name:
        raise ValueError(f"Unknown feature type: {feature_type}")

    csv_path = merged_dir / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Merged data not found: {csv_path}")

    print(f"[INFO] Loading {csv_path}...")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # Filter time ranges
    train_df = df.loc[train_start:train_end].copy()
    test_df = df.loc[test_start:test_end].copy()

    if len(train_df) < 100 or len(test_df) < 50:
        print(f"[WARN] Window {window_idx}: insufficient data "
              f"(train={len(train_df)}, test={len(test_df)}), skipping")
        return None

    # Build features using the same logic as build_features.py
    # We import the feature construction functions
    sys.path.insert(0, str(SRC_DIR / "features_construct"))
    from build_features import (
        build_price_features, build_funding_features, build_fng_features,
        build_onchain_features, build_long_onchain_features, build_interaction_features,
        Config
    )

    print(f"[INFO] Building features for window {window_idx}...")

    # Build base features
    price_feat = build_price_features(df) if hasattr(sys.modules[__name__], 'build_price_features') else None

    # For simplicity, we re-run the full feature building pipeline on the subset
    # This is done by running build_features.py with modified time ranges
    # Alternative: use numpy feature files but slice by index

    # TODO: The simplest approach is to modify config and run the model training,
    # with custom data loaders that slice by time index.
    # For now, we provide the training orchestration approach.

    return train_df, test_df


def run_sliding_window_training(feature_type, model_name, seeds):
    """
    Run sliding window training by modifying config seed and using time-sliced data.

    Strategy: We re-use the EXISTING feature .npy files, but slice them
    according to each window's time range. The existing 70/15/15 split corresponds
    approximately to:
      Train: 2020-01 ~ 2024-06 (index 0 ~ 9701)
      Val:   2024-06 ~ 2025-06 (index 9702 ~ 11807)
      Test:  2025-06 ~ 2026-05 (index 11808 ~ 13894)

    For sliding windows, we need to dynamically load the full feature array
    and slice by window.
    """
    results = {}
    for seed in seeds:
        for wi, wdef in enumerate(WINDOWS):
            print(f"\n{'='*60}")
            print(f"[SW] {model_name} | {feature_type} | Window {wi} | Seed {seed}")
            print(f"[SW] Train: {wdef[0]}~{wdef[1]} | Test: {wdef[2]}~{wdef[3]}")
            print(f"{'='*60}")

            # Approach: Monkey-patch config and load_data to use time slices
            # This avoids duplicating the entire training infrastructure

            cmd = [
                sys.executable, "-c", f"""
import sys
sys.path.insert(0, r"{SRC_DIR}")

# Monkey-patch before imports
import numpy as np

# Override load_data to use time-sliced data
import src.common as common

_original_load = common.load_data

def sliced_load_data():
    X_train, y_train, X_val, y_val, X_test, y_test = _original_load()
    # We need the full sequence, not pre-split
    # Actually, the existing data is already pre-split 70/15/15
    # For sliding window, we need a different approach:
    # We concatenate all splits, then re-split by window

    X_all = np.concatenate([X_train, X_val, X_test], axis=0)
    y_all = np.concatenate([y_train, y_val, y_test], axis=0)

    # Sliding window {wi}: train 2 years (~4380 4h periods), test 6 months (~1095)
    samples_per_window = 2 * 365 * 6  # ~4380
    test_samples = 182 * 6  # ~1092
    step = 182 * 6  # ~1092

    train_start = {wi} * step
    train_end = train_start + samples_per_window
    test_start = train_end
    test_end = test_start + test_samples

    if test_end > len(X_all):
        print(f"[WARN] Window {{wi}} exceeds data length, truncating")
        test_end = len(X_all)
        train_end = test_start

    return (
        X_all[train_start:train_end], y_all[train_start:train_end],
        X_all[test_start-200:test_start], y_all[test_start-200:test_start],  # validation: 200 samples before test
        X_all[test_start:test_end], y_all[test_start:test_end]
    )

common.load_data = sliced_load_data

# Now run the model training
from configs.config import cfg
cfg["seed"] = {seed}
cfg["data"]["feature_type"] = "{feature_type}"

from src.models.{model_name} import train
train()
"""
            ]
            try:
                subprocess.run(cmd, check=True, cwd=str(SRC_DIR), timeout=7200)
            except subprocess.TimeoutExpired:
                print(f"[WARN] Window {wi} seed {seed} timed out")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Window {wi} seed {seed} failed: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Sliding window validation on price_funding_fng feature set"
    )
    parser.add_argument("--feature", type=str, default="price_funding_fng",
                        choices=["price_funding_fng", "full"])
    parser.add_argument("--model", type=str, default="cnn_transformer",
                        choices=["cnn_transformer", "timemixer", "all"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    models = ["cnn_transformer", "timemixer"] if args.model == "all" else [args.model]

    print("=" * 60)
    print("滑动窗口补充实验方案")
    print(f"  特征集: {args.feature}")
    print(f"  模型:   {models}")
    print(f"  种子:   {args.seeds}")
    print(f"  窗口数: {len(WINDOWS)}")
    print(f"  总训练: {len(models) * len(args.seeds) * len(WINDOWS)} 次")
    print("=" * 60)

    for model in models:
        run_sliding_window_training(args.feature, model, args.seeds)

    print("\n[DONE] Sliding window experiments complete.")


if __name__ == "__main__":
    main()
