"""
单窗口训练 worker —— 由 run_sw_training.py 调用。
不要直接运行此脚本。

对指定的 (model, feature, seed, window_idx)，完成一次完整的训练+评估。
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# 确定项目根目录
SRC_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

# 滑动窗口定义
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

# 特征构建函数（从 build_features.py 复制核心逻辑）
# 避免循环导入
LOOKBACK = 48
HORIZON = 1


def multi_scale_features(series, windows=None, prefix=""):
    if windows is None:
        windows = [4, 8, 12, 24]
    df = pd.DataFrame(index=series.index)
    for w in windows:
        mean = series.rolling(w, min_periods=1).mean()
        std = series.rolling(w, min_periods=1).std()
        z = (series - mean) / (std + 1e-8)
        df[f"{prefix}z_{w}"] = np.clip(z, -20, 20)
        shifted = series.shift(w)
        mom = np.where(np.abs(shifted) < 1e-8, 0.0, series / shifted - 1.0)
        df[f"{prefix}mom_{w}"] = np.clip(mom, -10, 10)
    return df


def build_price_features(df):
    features = pd.DataFrame(index=df.index)
    for lag in [1, 2, 3, 4, 6, 8, 12]:
        features[f"ret_{lag}"] = np.log(df["close"] / df["close"].shift(lag))
    features["ret_acc_1"] = features["ret_1"] - features["ret_1"].shift(1)
    features["ret_acc_2"] = features["ret_2"] - features["ret_2"].shift(2)
    features["vol_log"] = np.log1p(df["volume"])
    features["vol_chg"] = np.log((df["volume"] + 1e-8) / (df["volume"].shift(1) + 1e-8))
    for w in [4, 8, 12, 24]:
        features[f"vol_{w}"] = features["ret_1"].rolling(w, min_periods=1).std()
    high, low, close = df["high"], df["low"], df["close"]
    features["hh_4"] = (high.rolling(4).max() - close) / (close + 1e-8)
    features["ll_4"] = (close - low.rolling(4).min()) / (close + 1e-8)
    features["position_4"] = (close - low.rolling(4).min()) / (
        high.rolling(4).max() - low.rolling(4).min() + 1e-8
    )
    return features


def build_funding_features(df):
    if "fundingRate" not in df.columns:
        return pd.DataFrame(index=df.index)
    fr = df["fundingRate"]
    features = pd.DataFrame(index=df.index)
    features["funding_raw"] = fr
    features["funding_diff"] = fr.diff()
    features = pd.concat(
        [features, multi_scale_features(fr, prefix="fr_")], axis=1
    )
    return features


def build_fng_features(df):
    if "fng_value" not in df.columns:
        return pd.DataFrame(index=df.index)
    fng = df["fng_value"]
    features = pd.DataFrame(index=df.index)
    features["fng_raw"] = fng
    features["fng_diff"] = fng.diff()
    features = pd.concat(
        [features, multi_scale_features(fng, prefix="fng_")], axis=1
    )
    return features


def build_onchain_features(df):
    features = pd.DataFrame(index=df.index)
    for col in ["sopr", "cdd"]:
        if col not in df.columns:
            continue
        s = df[col]
        log_s = np.log(s.clip(lower=1e-8))
        features[f"{col}_raw"] = s
        features[f"{col}_log"] = log_s
        features[f"{col}_diff1"] = s.diff()
        features[f"{col}_diff2"] = s.diff().diff()
        features[f"{col}_log_diff1"] = log_s.diff()
        features = pd.concat(
            [features, multi_scale_features(s, prefix=f"{col}_")], axis=1
        )
    return features


def build_interaction_features(df, feature_type):
    features = pd.DataFrame(index=df.index)
    has_fng = "fng_value" in df.columns
    has_sopr = "sopr" in df.columns
    has_cdd = "cdd" in df.columns
    has_fr = "fundingRate" in df.columns

    if has_fng and has_sopr:
        fng = df["fng_value"]
        sopr = df["sopr"]
        features["fng_sopr_int"] = (fng - 50) * (sopr - 1.0)
    if has_cdd:
        cdd_log = np.log(df["cdd"].clip(lower=1e-8))
        vol_log = np.log1p(df["volume"])
        features["cdd_vol_int"] = cdd_log * vol_log
    if has_fr:
        fr = df["fundingRate"]
        vol_std_12 = np.log(df["close"] / df["close"].shift(1)).rolling(12).std()
        features["fr_vol_int"] = fr * vol_std_12
    if has_fng:
        fng = df["fng_value"]
        ret_1 = np.log(df["close"] / df["close"].shift(1))
        features["fng_mom_int"] = (fng - 50) * ret_1

    return features


def build_label(df):
    return np.log(df["close"].shift(-1) / df["close"])


def build_sequences(X_df, y_series, lookback=48):
    """从特征 DataFrame 构建 (N, L, F) 序列。"""
    X_arr = X_df.values.astype(np.float32)
    y_arr = y_series.values.astype(np.float32)

    n = len(X_arr) - lookback
    X_seq = np.zeros((n, lookback, X_arr.shape[1]), dtype=np.float32)
    y_seq = np.zeros(n, dtype=np.float32)

    for i in range(n):
        X_seq[i] = X_arr[i : i + lookback]
        y_seq[i] = y_arr[i + lookback]

    # 删除含 NaN/Inf 的序列
    mask = np.isfinite(X_seq).all(axis=(1, 2)) & np.isfinite(y_seq)
    return X_seq[mask], y_seq[mask]


def train_one_window(model_name, feature_type, seed, window_idx):
    """核心：为单个滑动窗口完成训练。"""
    import torch
    from configs.config import cfg
    from src.common import set_seed, run_training

    wdef = WINDOWS[window_idx]
    train_start, train_end, test_start, test_end = wdef

    # 加载原始 merged CSV
    dataset_map = {
        "price_funding_fng": "merged_price_funding_fng_4h.csv",
        "full": "merged_full_4h.csv",
    }
    csv_name = dataset_map[feature_type]
    csv_path = PROJECT_ROOT / "data" / "processed" / "merged" / csv_name

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # 需要额外历史数据用于 rolling/lookback
    extra = pd.Timedelta(days=90)
    data_start = pd.Timestamp(train_start) - extra
    test_end_ts = pd.Timestamp(test_end)

    subset = df.loc[data_start:test_end_ts].copy()
    print(f"[SW W{window_idx}] Data subset: {len(subset)} rows, "
          f"{subset.index[0]} to {subset.index[-1]}")

    # 构建特征
    feat_parts = [
        build_price_features(subset),
        build_funding_features(subset),
    ]
    if "fng" in feature_type:
        feat_parts.append(build_fng_features(subset))
    if "onchain" in feature_type or feature_type == "full":
        feat_parts.append(build_onchain_features(subset))
    feat_parts.append(build_interaction_features(subset, feature_type))

    X_df = pd.concat(feat_parts, axis=1)
    y_series = build_label(subset)

    # 删除所有 NaN 行（来自 rolling/diff）
    X_df = X_df.dropna()
    common_idx = X_df.index.intersection(y_series.dropna().index)
    X_df = X_df.loc[common_idx]
    y_series = y_series.loc[common_idx]

    print(f"[SW W{window_idx}] Features: {X_df.shape[1]} dims, {len(X_df)} rows after dropna")

    # 切分 train/val/test
    train_mask = X_df.index <= pd.Timestamp(train_end)
    test_mask = X_df.index >= pd.Timestamp(test_start)

    X_all_train = X_df[train_mask]
    y_all_train = y_series[train_mask]
    X_test_df = X_df[test_mask]
    y_test_s = y_series[test_mask]

    # 从训练集末尾取 200 条作为验证集
    val_size = min(200, len(X_all_train) // 5)
    X_train_df = X_all_train.iloc[:-val_size]
    y_train_s = y_all_train.iloc[:-val_size]
    X_val_df = X_all_train.iloc[-val_size:]
    y_val_s = y_all_train.iloc[-val_size:]

    print(f"[SW W{window_idx}] Split: train={len(X_train_df)}, val={len(X_val_df)}, test={len(X_test_df)}")

    if len(X_train_df) < LOOKBACK + 100 or len(X_test_df) < 50:
        print(f"[SW W{window_idx}] [SKIP] Insufficient data")
        return None

    # RobustScaler (fit on train only)
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler(quantile_range=(5, 95))
    X_train_arr = scaler.fit_transform(X_train_df.values)
    X_val_arr = scaler.transform(X_val_df.values) if len(X_val_df) > 0 else np.zeros((0, X_train_arr.shape[1]))
    X_test_arr = scaler.transform(X_test_df.values)

    y_train_arr = y_train_s.values
    y_val_arr = y_val_s.values if len(y_val_s) > 0 else np.zeros(0)
    y_test_arr = y_test_s.values

    # 构建序列
    X_train_seq, y_train_seq = build_sequences(
        pd.DataFrame(X_train_arr, index=X_train_df.index), pd.Series(y_train_arr), LOOKBACK
    )
    X_val_seq, y_val_seq = build_sequences(
        pd.DataFrame(X_val_arr), pd.Series(y_val_arr), LOOKBACK
    ) if len(X_val_arr) > LOOKBACK else (np.zeros((0, LOOKBACK, X_train_arr.shape[1]), dtype=np.float32), np.zeros(0, dtype=np.float32))
    X_test_seq, y_test_seq = build_sequences(
        pd.DataFrame(X_test_arr), pd.Series(y_test_arr), LOOKBACK
    )

    print(f"[SW W{window_idx}] Sequences: train={len(X_train_seq)}, val={len(X_val_seq)}, test={len(X_test_seq)}")

    if len(X_train_seq) < 100:
        print(f"[SW W{window_idx}] [SKIP] Not enough sequences")
        return None

    # 重写 cfg 并训练
    set_seed(seed)
    original_seed = cfg.get("seed", 0)
    original_ft = cfg["data"]["feature_type"]

    cfg["seed"] = seed
    cfg["data"]["feature_type"] = feature_type

    # 直接导入模型类（避免动态检测到 AttentionPooling 等辅助类）
    MODEL_CLASS_MAP = {
        "cnn_transformer": "CNNTransformer",
        "lstm_transformer": "LSTMTransformer",
        "transformer": "TransformerModel",
        "lstm": "BiLSTM",
        "tcn": "TCN",
        "modern_tcn": "ModernTCN",
        "patchtst": "PatchTST",
        "dlinear": "DLinear",
        "timemixer": "TimeMixer",
    }
    class_name = MODEL_CLASS_MAP.get(model_name)
    model_module = __import__(f"src.models.{model_name}", fromlist=[class_name])
    ModelClass = getattr(model_module, class_name)

    input_dim = X_train_seq.shape[2]
    model = ModelClass(input_dim, cfg[model_name])

    # 修改 checkpoint 路径
    import src.common as common
    original_ckpt_logic = common.run_training.__code__  # can't modify directly

    # Monkey-patch checkpoint saving
    original_run_training = common.run_training

    def sw_run_training(model, mname, X_tr, y_tr, X_v, y_v, X_te, y_te):
        """与 run_training 相同，但 checkpoint 目录加上 _swW{idx} 后缀"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model_cfg = cfg.get(mname, {})

        from torch.utils.data import Dataset, DataLoader
        from src.metrics import (
            calc_ic, calc_pearson_ic, calc_da, calc_mse,
            calc_strategy_returns, calc_sharpe, calc_ic_ir,
            calc_max_drawdown, calc_annual_return,
        )
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        # 简化版 DataLoader
        class SeqDataset(Dataset):
            def __init__(self, X, y):
                self.X = torch.from_numpy(X).float()
                self.y = torch.from_numpy(y).float()
            def __len__(self):
                return len(self.y)
            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]

        train_dataset = SeqDataset(X_tr, y_tr)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        train_eval_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

        val_dataset = SeqDataset(X_v, y_v) if len(y_v) > 0 else train_dataset
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        test_dataset = SeqDataset(X_te, y_te)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        lr = model_cfg.get("lr", 1e-4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        huber_criterion = nn.HuberLoss(delta=0.3)
        ranking_weight = model_cfg.get("ranking_loss_weight", 0.13)
        ranking_margin = model_cfg.get("ranking_margin", 0.0005)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        grad_clip = model_cfg.get("grad_clip", 1.0)

        # Checkpoint dir
        ckpt_dir = PROJECT_ROOT / "checkpoint" / f"{mname}_{feature_type}_swW{window_idx}_seed{seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = ckpt_dir / "best.pth"

        def sw_evaluate(model, loader, criterion=None):
            model.eval()
            preds, trues = [], []
            total_loss, count = 0.0, 0
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(y.detach().cpu().numpy())
                    if criterion is not None:
                        total_loss += criterion(pred, y).item() * len(y)
                        count += len(y)

            preds = np.concatenate(preds)
            trues = np.concatenate(trues)
            mask = np.isfinite(preds) & np.isfinite(trues)
            preds, trues = preds[mask], trues[mask]

            if len(preds) == 0:
                return {
                    "IC": 0, "PIC": 0, "DA": 0.5, "MSE": 0,
                    "Sharpe": 0, "IR": 0, "MaxDrawdown": 0, "AnnualReturn": 0,
                }

            preds = np.clip(preds, -0.005, 0.005)
            sr = calc_strategy_returns(preds, trues, fee=0.0005)
            m = {
                "IC": calc_ic(preds, trues),
                "PIC": calc_pearson_ic(preds, trues),
                "DA": calc_da(preds, trues),
                "MSE": calc_mse(preds, trues),
                "Sharpe": calc_sharpe(sr),
                "IR": calc_ic_ir(preds, trues),
                "MaxDrawdown": calc_max_drawdown(sr),
                "AnnualReturn": calc_annual_return(sr),
            }
            if criterion is not None and count > 0:
                m["Loss"] = total_loss / count
            return m

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(200):
            model.train()
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss_reg = huber_criterion(pred, y)

                bs = pred.size(0)
                if bs > 1:
                    perm = torch.randperm(bs, device=device)
                    half = bs // 2
                    idx_i, idx_j = perm[:half], perm[half:2*half]
                    target = torch.sign(y[idx_i] - y[idx_j])
                    mask = target != 0
                    if mask.any():
                        loss_rank = F.margin_ranking_loss(
                            pred[idx_i][mask], pred[idx_j][mask],
                            target[mask].float(), margin=ranking_margin
                        )
                    else:
                        loss_rank = torch.tensor(0.0, device=device)
                else:
                    loss_rank = torch.tensor(0.0, device=device)

                loss = (1.0 - ranking_weight) * loss_reg + ranking_weight * loss_rank
                optimizer.zero_grad()
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                train_loss += loss.item() * len(y)

            train_loss /= len(train_loader.dataset)
            val_m = sw_evaluate(model, val_loader, huber_criterion)
            val_loss = val_m.get("Loss", float('inf'))
            scheduler.step(val_loss)

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | "
                      f"Val IC: {val_m['IC']:.4f} | Val Sharpe: {val_m['Sharpe']:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        model.load_state_dict(torch.load(best_model_path, map_location=device))
        train_m = sw_evaluate(model, train_eval_loader)
        val_m = sw_evaluate(model, val_loader)
        test_m = sw_evaluate(model, test_loader)

        # Save results
        result_path = ckpt_dir / "results.txt"
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(f"feature_type: {feature_type}\n")
            f.write(f"lookback: {LOOKBACK}\n")
            f.write(f"window_idx: {window_idx}\n")
            f.write(f"window_range: {train_start}_{train_end}__{test_start}_{test_end}\n\n")
            for split, m in [("Train", train_m), ("Val", val_m), ("Test", test_m)]:
                f.write(f"{split}  IC: {m['IC']:.4f}  PIC: {m['PIC']:.4f}  "
                        f"DA: {m['DA']:.4f}  MSE: {m['MSE']:.6f}  "
                        f"Sharpe: {m['Sharpe']:.4f}  IR: {m['IR']:.4f}  "
                        f"MaxDrawdown: {m['MaxDrawdown']:.4f}  "
                        f"AnnualReturn: {m['AnnualReturn']:.4f}\n")

        print(f"  Test: IC={test_m['IC']:.4f} Sharpe={test_m['Sharpe']:.4f} "
              f"MaxDD={test_m['MaxDrawdown']:.4f} AnnRet={test_m['AnnualReturn']:.4f}")
        return train_m, val_m, test_m

    # 运行训练
    sw_run_training(model, model_name, X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq)

    # 恢复配置
    cfg["seed"] = original_seed
    cfg["data"]["feature_type"] = original_ft

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--feature", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--window", type=int, required=True)
    args = parser.parse_args()

    result = train_one_window(args.model, args.feature, args.seed, args.window)
    if result:
        print(f"[OK] Window {args.window} complete")
    else:
        print(f"[SKIP] Window {args.window} skipped (insufficient data)")
        sys.exit(2)


if __name__ == "__main__":
    main()
