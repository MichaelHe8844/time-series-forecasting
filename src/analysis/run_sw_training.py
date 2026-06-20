"""
缺口 2：滑动窗口训练脚本（干净的实现）。

对指定模型和特征集，在8个滑动窗口上完成训练。
每个窗口: 训练2年 ~4380条序列, 测试随后6个月 ~1095条序列, 步长6个月。

用法:
  # CNN-Transformer on price_funding_fng, sliding windows
  python src/analysis/run_sw_training.py --model cnn_transformer --feature price_funding_fng --seed 1

  # TimeMixer on price_funding_fng
  python src/analysis/run_sw_training.py --model timemixer --feature price_funding_fng --seed 1

工作原理:
  1. 加载原始 merged CSV (带时间戳)
  2. 对每个窗口找到时间范围对应的行索引
  3. 重建该窗口的特征 (需要历史上下文做 rolling)
  4. 训练模型并将 checkpoint 保存到 checkpoint/{model}_{feature}_swW{idx}_seed{seed}/
"""

import os
import sys
import argparse
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_FILE = PROJECT_ROOT / "configs" / "config.py"

# 8 sliding windows
# train 2 years, test 6 months, stride 6 months
# Window 0: train 2020-01 ~ 2021-12, test 2022-01 ~ 2022-06
# Window 7: train 2023-07 ~ 2025-06, test 2025-07 ~ 2025-12
WINDOWS = [
    ("2020-01-01", "2021-12-31", "2022-01-01", "2022-06-30"),
    ("2020-07-01", "2022-06-30", "2022-07-01", "2022-12-31"),
    ("2021-01-01", "2022-12-31", "2023-01-01", "2023-06-30"),
    ("2021-07-01", "2023-06-30", "2023-07-01", "2023-12-31"),
    ("2022-01-01", "2023-12-31", "2024-01-01", "2024-06-30"),
    ("2022-07-01", "2024-06-30", "2024-07-01", "2024-12-31"),
    ("2023-01-01", "2024-12-31", "2025-01-01", "2025-06-30"),
    ("2023-07-01", "2025-06-30", "2025-07-01", "2025-12-31"),
]

DATASET_FILES = {
    "price_funding_fng": "merged_price_funding_fng_4h.csv",
    "full": "merged_full_4h.csv",
}


def get_window_indices(csv_path, window_def):
    """给定时间范围，返回在 CSV 中的行索引区间。"""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    train_start, train_end, test_start, test_end = window_def

    # 需要额外的历史数据来做 rolling/lookback
    # lookback=48 条 (=8 days of 4h)，加上 rolling window=24 的额外历史
    extra_history = pd.Timedelta(days=60)  # 2 months extra for rolling features

    data_start = pd.Timestamp(train_start) - extra_history

    train_mask = (df.index >= data_start) & (df.index <= pd.Timestamp(train_end))
    test_mask = (df.index >= pd.Timestamp(test_start)) & (df.index <= pd.Timestamp(test_end))

    train_df = df[train_mask].copy()
    test_df_full = df[(df.index >= data_start) & (df.index <= pd.Timestamp(test_end))].copy()

    print(f"  Train rows: {len(train_df)}, Test rows: {len(test_df_full) - len(train_df)}")
    return train_df, test_df_full


def main():
    parser = argparse.ArgumentParser(description="Sliding window training")
    parser.add_argument("--model", type=str, required=True,
                        choices=["cnn_transformer", "timemixer", "lstm_transformer",
                                 "transformer", "lstm"])
    parser.add_argument("--feature", type=str, required=True,
                        choices=["price_funding_fng", "full"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--windows", type=int, nargs="+", default=list(range(8)),
                        help="Which windows to run (0-7)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned runs without executing")
    args = parser.parse_args()

    csv_name = DATASET_FILES[args.feature]
    csv_path = PROJECT_ROOT / "data" / "processed" / "merged" / csv_name

    if not csv_path.exists():
        print(f"[ERROR] Data file not found: {csv_path}")
        print("  Run src/data_preprocess/merge_ablation_datasets.py first.")
        sys.exit(1)

    total_runs = len(args.windows)
    print("=" * 70)
    print(f"滑动窗口训练计划")
    print(f"  模型:     {args.model}")
    print(f"  特征集:   {args.feature}")
    print(f"  种子:     {args.seed}")
    print(f"  窗口:     {args.windows}")
    print(f"  总训练数: {total_runs}")
    print("=" * 70)

    for wi in args.windows:
        wdef = WINDOWS[wi]
        print(f"\n{'─'*50}")
        print(f"Window {wi}: Train {wdef[0]}~{wdef[1]} → Test {wdef[2]}~{wdef[3]}")

        # 构建特征并训练
        # 策略：复用一个包装脚本，该脚本:
        #   1) 重写 cfg["seed"] 和 cfg["data"]["feature_type"]
        #   2) 调用模型训练
        # 为了处理滑动窗口的时间切片，我们需要自定义 load_data

        # 最简单的方法：用环境变量传递窗口参数
        env = os.environ.copy()
        env["SW_WINDOW_IDX"] = str(wi)
        env["SW_FEATURE"] = args.feature
        env["SW_SEED"] = str(args.seed)

        cmd = [
            sys.executable,
            str(SRC_DIR / "analysis" / "_sw_train_one.py"),
            "--model", args.model,
            "--feature", args.feature,
            "--seed", str(args.seed),
            "--window", str(wi),
        ]

        if args.dry_run:
            print(f"  [DRY RUN] {' '.join(cmd)}")
        else:
            print(f"  Running...")
            result = subprocess.run(cmd, cwd=str(SRC_DIR), env=env)
            if result.returncode != 0:
                print(f"  [FAILED] Window {wi} returned code {result.returncode}")
            else:
                print(f"  [OK] Window {wi} completed")

    print("\n[DONE] All sliding window runs complete.")


if __name__ == "__main__":
    main()
