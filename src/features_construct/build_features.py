"""
build_features.py (4h 短期预测专版) —— 已回滚到原始预测方式（带 timing lag）
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


class Config:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "processed", "merged")
    FEATURES_ROOT = os.path.join(PROJECT_ROOT, "features")
    LOOKBACK = 48
    HORIZON = 1
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    SEED = 42

    DATASETS = {
        "price_only":          {"file": "merged_price_only_4h.csv"},
        "price_funding":       {"file": "merged_price_funding_4h.csv"},
        "price_funding_fng":   {"file": "merged_price_funding_fng_4h.csv"},
        "price_onchain":       {"file": "merged_price_onchain_4h.csv"},
        "price_long_onchain":  {"file": "merged_price_long_onchain_4h.csv"},
        "full":                {"file": "merged_full_4h.csv"}
    }


def multi_scale_features(series, windows=[4, 8, 12, 24], prefix=""):
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

    high = df["high"]
    low = df["low"]
    close = df["close"]
    features["hh_4"] = (high.rolling(4).max() - close) / (close + 1e-8)
    features["ll_4"] = (close - low.rolling(4).min()) / (close + 1e-8)
    features["position_4"] = (close - low.rolling(4).min()) / (high.rolling(4).max() - low.rolling(4).min() + 1e-8)

    return features


def build_funding_features(df):
    if "fundingRate" not in df.columns:
        return pd.DataFrame(index=df.index)
    fr = df["fundingRate"]
    features = pd.DataFrame(index=df.index)
    features["funding_raw"] = fr
    features["funding_diff"] = fr.diff()
    features = pd.concat([features, multi_scale_features(fr, windows=[4,8,12,24], prefix="fr_")], axis=1)
    return features


def build_fng_features(df):
    if "fng_value" not in df.columns:
        return pd.DataFrame(index=df.index)
    fng = df["fng_value"].ffill()
    features = pd.DataFrame(index=df.index)
    features["fng_diff"] = fng.diff()
    features["fng_extreme"] = ((fng < 25) | (fng > 75)).astype(int)
    features = pd.concat([features, multi_scale_features(fng, windows=[4,8,12,24], prefix="fng_")], axis=1)
    return features


def build_onchain_features(df):
    features = pd.DataFrame(index=df.index)
    if "sopr" in df.columns:
        sopr = df["sopr"].ffill()
        features["sopr_raw"] = sopr
        features["sopr_diff"] = sopr.diff()
        features["sopr_acc"] = sopr.diff().diff()
        features = pd.concat([features, multi_scale_features(sopr, windows=[4,8,12,24], prefix="sopr_")], axis=1)

    if "cdd" in df.columns:
        cdd = df["cdd"].ffill().replace(0, np.nan).ffill().fillna(0)
        cdd_log = np.log1p(cdd)
        features["cdd_log"] = cdd_log
        features["cdd_log_diff"] = cdd_log.diff()
        features["cdd_log_acc"] = cdd_log.diff().diff()
        features = pd.concat([features, multi_scale_features(cdd_log, windows=[4,8,12,24], prefix="cdd_")], axis=1)
    return features


def build_cross_modal_features(price_feat, funding_feat, fng_feat, onchain_feat):
    features = pd.DataFrame(index=price_feat.index)

    if "fng_z_12" in fng_feat.columns and "sopr_z_12" in onchain_feat.columns:
        features["fng_sopr_int"] = fng_feat["fng_z_12"] * onchain_feat["sopr_z_12"]

    if "fr_z_8" in funding_feat.columns and "vol_8" in price_feat.columns:
        features["fr_vol_int"] = funding_feat["fr_z_8"] * price_feat["vol_8"]

    if "cdd_z_8" in onchain_feat.columns and "vol_chg" in price_feat.columns:
        features["cdd_vol_int"] = onchain_feat["cdd_z_8"] * price_feat["vol_chg"]

    if "fng_extreme" in fng_feat.columns and "ret_2" in price_feat.columns:
        features["fng_mom_int"] = fng_feat["fng_extreme"] * price_feat["ret_2"]

    return features


def build_all_features(df, feature_type: str):
    print(f"\n[{feature_type}] Building features...")
    price_feat = build_price_features(df)
    funding_feat = build_funding_features(df)
    fng_feat = build_fng_features(df)
    onchain_feat = build_onchain_features(df)

    features = price_feat.copy()
    if feature_type in ["price_funding", "price_funding_fng", "full"]:
        features = pd.concat([features, funding_feat], axis=1)
    if feature_type in ["price_funding_fng", "full"]:
        features = pd.concat([features, fng_feat], axis=1)
    if feature_type == "price_onchain":
        features = pd.concat([features, onchain_feat], axis=1)
    if feature_type == "full":
        features = pd.concat([features, onchain_feat], axis=1)

    if feature_type == "price_long_onchain":
        for col in ["total_btc_on_exchange", "active_addresses", "netflow_btc"]:
            if col in df.columns:
                s = df[col].ffill()
                features[col] = s
                features[f"{col}_diff"] = s.diff()
                features = pd.concat([features, multi_scale_features(s, windows=[4,8,12,24], prefix=f"{col}_")], axis=1)

    if feature_type in ["full", "price_funding_fng"]:
        cross_feat = build_cross_modal_features(price_feat, funding_feat, fng_feat, onchain_feat)
        features = pd.concat([features, cross_feat], axis=1)

    features["target"] = np.log(df["close"].shift(-Config.HORIZON) / df["close"])

    features = features.replace([np.inf, -np.inf], np.nan)

    # 先 drop 掉 target 为 NaN 的行（最后 HORIZON 行），再对特征做 ffill
    features = features.sort_index()
    features = features.dropna(subset=["target"])

    features = features.ffill().fillna(0)

    print(f"[{feature_type}] Final shape: {features.shape}")
    return features


def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, :-1])
        y.append(data[i, -1])
    return np.array(X), np.array(y)


def train_val_test_split(X, y):
    n = len(X)
    train_end = int(n * Config.TRAIN_RATIO)
    val_end = int(n * (Config.TRAIN_RATIO + Config.VAL_RATIO))
    return (X[:train_end], X[train_end:val_end], X[val_end:],
            y[:train_end], y[train_end:val_end], y[val_end:])


def normalize_data(X_train, X_val, X_test):
    n_train, lookback, n_features = X_train.shape
    X_train_flat = X_train.reshape(-1, n_features)
    X_val_flat = X_val.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)

    scaler = RobustScaler(quantile_range=(5, 95))
    X_train_norm = scaler.fit_transform(X_train_flat).reshape(n_train, lookback, n_features)
    X_val_norm = scaler.transform(X_val_flat).reshape(X_val.shape[0], lookback, n_features)
    X_test_norm = scaler.transform(X_test_flat).reshape(X_test.shape[0], lookback, n_features)

    return X_train_norm, X_val_norm, X_test_norm, scaler


def process_dataset(name, info):
    print(f"\n{'='*70}")
    print(f"Processing: {name}")
    print(f"{'='*70}")
    path = os.path.join(Config.DATA_ROOT, info["file"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing data file: {path}")

    df = pd.read_csv(path)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()

    features = build_all_features(df, name)
    data = features.values.astype(np.float32)

    X, y = create_sequences(data, Config.LOOKBACK)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    X_train, X_val, X_test, _ = normalize_data(X_train, X_val, X_test)

    if Config.LOOKBACK == 48:
        out = os.path.join(Config.FEATURES_ROOT, name)
    else:
        out = os.path.join(Config.FEATURES_ROOT, f"{name}_L{Config.LOOKBACK}")
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "X_train.npy"), X_train)
    np.save(os.path.join(out, "X_val.npy"), X_val)
    np.save(os.path.join(out, "X_test.npy"), X_test)
    np.save(os.path.join(out, "y_train.npy"), y_train)
    np.save(os.path.join(out, "y_val.npy"), y_val)
    np.save(os.path.join(out, "y_test.npy"), y_test)

    print(f"[OK] Saved: {name} | X_train: {X_train.shape}, Features: {X_train.shape[-1]}")


def main():
    parser = argparse.ArgumentParser(description="Build time-series features with sliding window")
    parser.add_argument("--lookback", type=int, default=Config.LOOKBACK,
                        help=f"Lookback window size (default: {Config.LOOKBACK})")
    parser.add_argument("--feature_type", type=str, default=None,
                        help="Single feature type to build (default: all)")
    args = parser.parse_args()

    Config.LOOKBACK = args.lookback
    np.random.seed(Config.SEED)
    os.makedirs(Config.FEATURES_ROOT, exist_ok=True)

    if args.feature_type:
        info = Config.DATASETS.get(args.feature_type)
        if info is None:
            raise ValueError(f"Unknown feature type: {args.feature_type}. "
                             f"Available: {list(Config.DATASETS.keys())}")
        process_dataset(args.feature_type, info)
    else:
        for name, info in Config.DATASETS.items():
            try:
                process_dataset(name, info)
            except Exception as e:
                print(f"[FAIL] {name} failed: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()