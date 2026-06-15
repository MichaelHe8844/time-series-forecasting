#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Systematic data audit: cross-check paper claims against source data."""
import os, sys, json, re
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT = Path("D:/MyCodes/Time_series_forecasting")
RESULTS = PROJECT / "results"
FEATURES = PROJECT / "features"
CHECKPOINT = PROJECT / "checkpoint"

print("=" * 70)
print("PAPER DATA AUDIT")
print("=" * 70)

# -------------------------------------------------------------------
# 1. FEATURE DIMENSIONS
# -------------------------------------------------------------------
print("\n--- 1. FEATURE DIMENSIONS ---")
expected_dims = {
    "price_only": 18, "price_funding": 28, "price_funding_fng": 40,
    "price_onchain": 40, "price_long_onchain": 48, "full": 64,
}
all_ok = True
for ft, expected in expected_dims.items():
    path = FEATURES / ft / "X_train.npy"
    if path.exists():
        actual = np.load(path).shape[2]
        status = "OK" if actual == expected else "MISMATCH"
        if actual != expected:
            all_ok = False
        print(f"  {ft:25s}: expected {expected:3d}, actual {actual:3d}  {status}")
    else:
        print(f"  {ft:25s}: FILE NOT FOUND  MISMATCH")
        all_ok = False
print(f"  => {'ALL OK' if all_ok else 'ERRORS FOUND'}")

# -------------------------------------------------------------------
# 2. DATASET SIZES
# -------------------------------------------------------------------
print("\n--- 2. DATASET SIZES (paper claims: train=9736, val=2086, test=2087) ---")
for ft in ["full", "price_funding_fng"]:
    X_train = np.load(FEATURES / ft / "X_train.npy")
    X_val = np.load(FEATURES / ft / "X_val.npy")
    X_test = np.load(FEATURES / ft / "X_test.npy")
    total = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    print(f"  {ft}: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}, total={total}")

# -------------------------------------------------------------------
# 3. LOAD ALL SEED=1 RESULTS
# -------------------------------------------------------------------
print("\n--- 3. SEED=1: FULL FEATURE SET (Table 2 in paper) ---")
df1 = pd.read_csv(RESULTS / "results_summary_seed1.csv")
df1_full = df1[df1["Feature_Type"] == "full"].set_index("Model")
# Paper claims (Table 2):
table2_claims = {
    "cnn_transformer":  {"IC": 0.149, "Sharpe": 5.652, "MaxDrawdown": 0.186, "AnnualReturn": 2.306},
    "lstm_transformer": {"IC": 0.108, "Sharpe": 3.259, "MaxDrawdown": 0.255, "AnnualReturn": 1.334},
    "transformer":      {"IC": 0.135, "Sharpe": 4.333, "MaxDrawdown": 0.207, "AnnualReturn": 1.773},
    "lstm":             {"IC": 0.119, "Sharpe": 2.257, "MaxDrawdown": 0.291, "AnnualReturn": 0.925},
    "tcn":              {"IC": 0.062, "Sharpe": 0.050, "MaxDrawdown": 0.366, "AnnualReturn": 0.021},
    "modern_tcn":       {"IC": 0.063, "Sharpe": 0.168, "MaxDrawdown": 0.415, "AnnualReturn": 0.069},
    "patchtst":         {"IC": 0.128, "Sharpe": 3.981, "MaxDrawdown": 0.261, "AnnualReturn": 1.630},
    "xgboost":          {"IC": 0.128, "Sharpe": 4.496, "MaxDrawdown": 0.183, "AnnualReturn": 1.838},
    "dlinear":          None,  # 待训练
    "timemixer":        None,  # 待训练
}
for model, claims in table2_claims.items():
    row = df1_full.loc[model] if model in df1_full.index else None
    if row is None:
        print(f"  {model}: NOT FOUND")
        continue
    issues = []
    for metric, expected in claims.items():
        actual = row[f"Test_{metric}"]
        if abs(actual - expected) > 0.001:
            issues.append(f"{metric}: paper={expected:.4f} actual={actual:.4f}")
    if issues:
        print(f"  {model}: MISMATCH {'; '.join(issues)}")
    else:
        print(f"  {model}: OK")

# -------------------------------------------------------------------
# 4. SEED=1: FEATURE ABLATION (Table 3 in paper)
# -------------------------------------------------------------------
print("\n--- 4. SEED=1: FEATURE ABLATION SHARPE (Table 3 in paper) ---")
# Paper claims (Test_Sharpe for each model x feature set):
table3_claims = {
    "cnn_transformer":  {"only": -2.27, "funding": -2.52, "fng": 2.20, "onchain": -0.85, "long_on": -2.90, "full": 5.65},
    "transformer":      {"only": -1.94, "funding": -2.00, "fng": 4.37, "onchain": -1.42, "long_on": -2.70, "full": 4.33},
    "lstm":             {"only": -1.10, "funding": -0.19, "fng": 4.28, "onchain": -2.01, "long_on": -1.87, "full": 2.26},
    "lstm_transformer": {"only": -0.07, "funding": -0.81, "fng": 0.46, "onchain": -0.94, "long_on": -1.94, "full": 3.26},
    "tcn":              {"only": -2.79, "funding": -3.31, "fng": 1.75, "onchain": -3.11, "long_on": -2.76, "full": 0.05},
    "modern_tcn":       {"only": -2.35, "funding": -1.20, "fng": 2.39, "onchain": -3.69, "long_on": -0.61, "full": 0.17},
    "patchtst":         {"only": -1.44, "funding": -3.29, "fng": 3.52, "onchain": -0.73, "long_on": -2.68, "full": 3.98},
    "xgboost":          {"fng": 4.50, "full": 4.50},  # others are --
}
ft_map = {"only": "price_only", "funding": "price_funding", "fng": "price_funding_fng",
          "onchain": "price_onchain", "long_on": "price_long_onchain", "full": "full"}
for model, claims in table3_claims.items():
    for ft_short, expected in claims.items():
        ft = ft_map[ft_short]
        sub = df1[(df1["Model"] == model) & (df1["Feature_Type"] == ft)]
        if len(sub) == 0:
            if expected != "--":
                print(f"  {model}/{ft_short}: expected {expected}, but DATA NOT FOUND")
            continue
        actual = sub["Test_Sharpe"].values[0]
        if abs(actual - expected) > 0.01:
            print(f"  {model}/{ft_short}: paper={expected:.2f} actual={actual:.4f} MISMATCH")
print("  (no output = all OK)")

# -------------------------------------------------------------------
# 5. SYNERGY TABLE (Table 4 in paper) - Verify Delta Sharpe
# -------------------------------------------------------------------
print("\n--- 5. SYNERGY Delta SHARPE (Table 4 in paper, seed=1) ---")
synergy_claims = {
    "cnn_transformer":  {"fng": 2.20, "full": 5.65, "delta": 3.45},
    "lstm_transformer": {"fng": 0.46, "full": 3.26, "delta": 2.80},
    "patchtst":         {"fng": 3.52, "full": 3.98, "delta": 0.46},
    "transformer":      {"fng": 4.37, "full": 4.33, "delta": -0.04},
    "xgboost":          {"fng": 4.50, "full": 4.50, "delta": 0.00},
    "lstm":             {"fng": 4.28, "full": 2.26, "delta": -2.02},
    "tcn":              {"fng": 1.75, "full": 0.05, "delta": -1.70},
    "modern_tcn":       {"fng": 2.39, "full": 0.17, "delta": -2.22},
}
for model, claims in synergy_claims.items():
    fng_row = df1[(df1["Model"] == model) & (df1["Feature_Type"] == "price_funding_fng")]
    full_row = df1[(df1["Model"] == model) & (df1["Feature_Type"] == "full")]
    if len(fng_row) == 0 or len(full_row) == 0:
        print(f"  {model}: DATA MISSING")
        continue
    actual_fng = fng_row["Test_Sharpe"].values[0]
    actual_full = full_row["Test_Sharpe"].values[0]
    actual_delta = actual_full - actual_fng
    # Check
    issues = []
    if abs(actual_fng - claims["fng"]) > 0.01:
        issues.append(f"fng: {claims['fng']} vs {actual_fng:.4f}")
    if abs(actual_full - claims["full"]) > 0.01:
        issues.append(f"full: {claims['full']} vs {actual_full:.4f}")
    if abs(actual_delta - claims["delta"]) > 0.01:
        issues.append(f"delta: {claims['delta']} vs {actual_delta:.4f}")
    if issues:
        print(f"  {model}: MISMATCH {'; '.join(issues)}")
    else:
        print(f"  {model}: OK (fng={actual_fng:.4f}, full={actual_full:.4f}, delta={actual_delta:.4f})")

# -------------------------------------------------------------------
# 6. XGBOOST TRAIN SHARPE: full == fng (paper claims both = 5.3363)
# -------------------------------------------------------------------
print("\n--- 6. XGBOOST TRAIN SHARPE: full vs fng (paper claims identical = 5.3363) ---")
xg_full = df1[(df1["Model"] == "xgboost") & (df1["Feature_Type"] == "full")]
xg_fng = df1[(df1["Model"] == "xgboost") & (df1["Feature_Type"] == "price_funding_fng")]
if len(xg_full) > 0 and len(xg_fng) > 0:
    ts_full = xg_full["Train_Sharpe"].values[0]
    ts_fng = xg_fng["Train_Sharpe"].values[0]
    print(f"  Train Sharpe: full={ts_full:.4f}, fng={ts_fng:.4f}")
    if abs(ts_full - ts_fng) < 0.0001:
        print(f"  OK Identical (both ~ {ts_full:.4f})")
    else:
        print(f"  MISMATCH DIFFER by {abs(ts_full-ts_fng):.4f}")
    if abs(ts_full - 5.3363) > 0.001:
        print(f"  MISMATCH Paper claims 5.3363 but actual is {ts_full:.4f}")

# -------------------------------------------------------------------
# 7. MODERNTCN TRAIN SHARPE (paper: cross-seed mean 16.31, max 19.30 seed=1)
# -------------------------------------------------------------------
print("\n--- 7. MODERNTCN TRAIN SHARPE ---")
# Check seed=1 first
mtcn1 = df1[(df1["Model"] == "modern_tcn") & (df1["Feature_Type"] == "full")]
if len(mtcn1) > 0:
    actual = mtcn1["Train_Sharpe"].values[0]
    print(f"  seed=1 full Train Sharpe: {actual:.4f} (paper claims 19.30)")

# Check all seeds
seeds_data = []
for seed in [0, 1, 2, 42, 123]:
    df_s = pd.read_csv(RESULTS / f"results_summary_seed{seed}.csv")
    mtcn = df_s[(df_s["Model"] == "modern_tcn") & (df_s["Feature_Type"] == "full")]
    if len(mtcn) > 0:
        seeds_data.append(mtcn["Train_Sharpe"].values[0])
if seeds_data:
    print(f"  Cross-seed Train Sharpe: mean={np.mean(seeds_data):.2f}, values={[f'{x:.2f}' for x in seeds_data]}")
    print(f"  (paper claims mean=16.31, max=19.30 at seed=1)")

# -------------------------------------------------------------------
# 8. CROSS-SEED: FULL FEATURE SET (Table 5 in paper)
# -------------------------------------------------------------------
print("\n--- 8. CROSS-SEED FULL SET (Table 5 in paper) ---")
all_seeds = []
for seed in [0, 1, 2, 42, 123]:
    df_s = pd.read_csv(RESULTS / f"results_summary_seed{seed}.csv")
    df_s["seed"] = seed
    all_seeds.append(df_s[df_s["Feature_Type"] == "full"])
df_all = pd.concat(all_seeds)

# Paper claims for cross-seed stats
table5_claims = {
    "cnn_transformer":  {"IC_mean": 0.150, "IC_std": 0.014, "Sharpe_mean": 5.187, "Sharpe_std": 0.441},
    "transformer":      {"IC_mean": 0.129, "IC_std": 0.008, "Sharpe_mean": 3.531, "Sharpe_std": 0.903},
    "lstm_transformer": {"IC_mean": 0.125, "IC_std": 0.015, "Sharpe_mean": 3.016, "Sharpe_std": 1.855},
    "lstm":             {"IC_mean": 0.111, "IC_std": 0.020, "Sharpe_mean": 3.000, "Sharpe_std": 0.844},
    "patchtst":         {"IC_mean": 0.101, "IC_std": 0.026, "Sharpe_mean": 2.807, "Sharpe_std": 0.758},
    "xgboost":          {"IC_mean": 0.127, "IC_std": 0.005, "Sharpe_mean": 4.189, "Sharpe_std": 0.720},
    "tcn":              {"IC_mean": 0.069, "IC_std": 0.039, "Sharpe_mean": 0.616, "Sharpe_std": 1.966},
    "modern_tcn":       {"IC_mean": 0.100, "IC_std": 0.027, "Sharpe_mean": 1.888, "Sharpe_std": 1.162},
}
for model, claims in table5_claims.items():
    sub = df_all[df_all["Model"] == model]
    if len(sub) == 0:
        print(f"  {model}: NO DATA")
        continue
    actual_ic_mean = sub["Test_IC"].mean()
    actual_ic_std = sub["Test_IC"].std(ddof=1)
    actual_sh_mean = sub["Test_Sharpe"].mean()
    actual_sh_std = sub["Test_Sharpe"].std(ddof=1)
    issues = []
    for label, actual, expected in [
        ("IC_mean", actual_ic_mean, claims["IC_mean"]),
        ("IC_std", actual_ic_std, claims["IC_std"]),
        ("Sharpe_mean", actual_sh_mean, claims["Sharpe_mean"]),
        ("Sharpe_std", actual_sh_std, claims["Sharpe_std"]),
    ]:
        if abs(actual - expected) > 0.0015:  # tighter for std
            issues.append(f"{label}: paper={expected:.4f} actual={actual:.4f}")
    if issues:
        print(f"  {model}: MISMATCH {'; '.join(issues)}")
    else:
        print(f"  {model}: OK")

# -------------------------------------------------------------------
# 9. CROSS-SEED SYNERGY DELTA (Section 5.4 point 4)
# -------------------------------------------------------------------
print("\n--- 9. CROSS-SEED SYNERGY DELTA for CNN-Transformer ---")
print("  Paper claims: seed=0 +3.47, seed=1 +3.45, seed=2 +4.22, seed=42 +2.34, seed=123 +3.01")
print("  Paper claims: mean=+3.30, std=0.72, min=+2.34")
deltas = []
for seed in [0, 1, 2, 42, 123]:
    df_s = pd.read_csv(RESULTS / f"results_summary_seed{seed}.csv")
    full = df_s[(df_s["Model"] == "cnn_transformer") & (df_s["Feature_Type"] == "full")]
    fng = df_s[(df_s["Model"] == "cnn_transformer") & (df_s["Feature_Type"] == "price_funding_fng")]
    if len(full) > 0 and len(fng) > 0:
        d = full["Test_Sharpe"].values[0] - fng["Test_Sharpe"].values[0]
        deltas.append(d)
        print(f"  seed={seed:3d}: full={full['Test_Sharpe'].values[0]:.4f}, fng={fng['Test_Sharpe'].values[0]:.4f}, delta={d:+.4f}")
if deltas:
    print(f"  Mean={np.mean(deltas):.4f}, Std={np.std(deltas, ddof=1):.4f}, Min={np.min(deltas):.4f}")

# -------------------------------------------------------------------
# 10. XGBOOST CONVERGENCE FAILURE (IC=0 on non-full sets)
# -------------------------------------------------------------------
print("\n--- 10. XGBOOST CONVERGENCE FAILURES (IC=0 claims) ---")
for ft in ["price_only", "price_funding", "price_onchain", "price_long_onchain"]:
    sub = df1[(df1["Model"] == "xgboost") & (df1["Feature_Type"] == ft)]
    if len(sub) > 0:
        test_ic = sub["Test_IC"].values[0]
        test_sh = sub["Test_Sharpe"].values[0]
        status = "OK IC=0" if abs(test_ic) < 0.0001 else f"MISMATCH IC={test_ic:.4f} (not zero!)"
        print(f"  xgboost/{ft}: Test_IC={test_ic:.6f}, Test_Sharpe={test_sh:.4f}  {status}")

# -------------------------------------------------------------------
# 11. BACKTEST FINAL EQUITY (Synergy chart)
# -------------------------------------------------------------------
print("\n--- 11. BACKTEST SYNERGY FINAL EQUITY ---")
bt_csv = PROJECT / "charts" / "backtest_synergy_full_vs_fng.csv"
if bt_csv.exists():
    data = np.loadtxt(bt_csv, delimiter=',', skiprows=1)
    full_eq = data[:, 0]; full_eq = full_eq[~np.isnan(full_eq)]
    fng_eq = data[:, 2]; fng_eq = fng_eq[~np.isnan(fng_eq)]
    full_final = full_eq[-1]
    fng_final = fng_eq[-1]
    ratio = full_final / fng_final
    print(f"  Full final equity: {full_final:.4f}x  (paper: 8.25x)")
    print(f"  FNG final equity:  {fng_final:.4f}x  (paper: 2.21x)")
    print(f"  Ratio: {ratio:.2f}x  (paper: 3.7x)")
    # Compute Sharpe from equity
    full_ret = np.diff(full_eq) / full_eq[:-1]
    fng_ret = np.diff(fng_eq) / fng_eq[:-1]
    full_sh = np.sqrt(2190) * full_ret.mean() / full_ret.std()
    fng_sh = np.sqrt(2190) * fng_ret.mean() / fng_ret.std()
    print(f"  Full approx Sharpe from equity: {full_sh:.2f}  (paper: 5.65)")
    print(f"  FNG approx Sharpe from equity:  {fng_sh:.2f}  (paper: 2.20)")
    # MaxDD
    full_dd = np.abs((full_eq / np.maximum.accumulate(full_eq) - 1).min())
    fng_dd = np.abs((fng_eq / np.maximum.accumulate(fng_eq) - 1).min())
    print(f"  Full MaxDD: {full_dd*100:.1f}%  (paper: 18.6%)")
    print(f"  FNG MaxDD:  {fng_dd*100:.1f}%  (paper: 23.4%)")

# -------------------------------------------------------------------
# 12. TABLE 1 LITERATURE COMPARISON - verify our numbers
# -------------------------------------------------------------------
print("\n--- 12. LITERATURE TABLE: OUR NUMBERS ---")
print("  Paper claims: DA=54.9%, Sharpe=5.19, IC=0.150")
sub = df_all[df_all["Model"] == "cnn_transformer"]
actual_da = sub["Test_DA"].mean()
actual_sh = sub["Test_Sharpe"].mean()
actual_ic = sub["Test_IC"].mean()
print(f"  Actual: DA={actual_da*100:.1f}%, Sharpe={actual_sh:.2f}, IC={actual_ic:.3f}")

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
