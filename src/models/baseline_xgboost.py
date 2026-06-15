"""
src/models/baseline_xgboost.py
XGBoost Baseline — tree model baseline for ablation experiments.
"""

import os
import numpy as np
import xgboost as xgb

from configs.config import cfg
from src.common import set_seed, load_data
from src.metrics import (calc_ic, calc_pearson_ic, calc_da, calc_mse, calc_strategy_returns,
                         calc_sharpe, calc_ic_ir, calc_max_drawdown, calc_annual_return)


MODEL_NAME = "xgboost"


def evaluate(model, X, y):
    """XGBoost 评估函数（无需 DataLoader）"""
    dmatrix = xgb.DMatrix(X)
    preds = model.predict(dmatrix)
    preds = np.clip(preds, cfg["pred_clip_min"], cfg["pred_clip_max"])

    strategy_returns = calc_strategy_returns(preds, y, fee=0.0005)

    return {
        "IC": calc_ic(preds, y),
        "PIC": calc_pearson_ic(preds, y),
        "DA": calc_da(preds, y),
        "MSE": calc_mse(preds, y),
        "Sharpe": calc_sharpe(strategy_returns),
        "IR": calc_ic_ir(preds, y),
        "MaxDrawdown": calc_max_drawdown(strategy_returns),
        "AnnualReturn": calc_annual_return(strategy_returns),
    }


def train():
    set_seed(cfg["seed"])

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # === XGBoost 必须 flatten 时序特征 ===
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    print(f"[INFO] Flattened X_train shape: {X_train_flat.shape}")

    dtrain = xgb.DMatrix(X_train_flat, label=y_train)
    dval = xgb.DMatrix(X_val_flat, label=y_val)

    # 从 config 读取参数
    xgb_cfg = cfg["xgboost"].copy()
    num_boost_round = xgb_cfg.pop("num_boost_round", 1000)
    early_stopping_rounds = xgb_cfg.pop("early_stopping_rounds", 60)
    verbosity = xgb_cfg.pop("verbosity", 1)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": cfg["seed"],
        **xgb_cfg,
    }

    print(f"[INFO] XGBoost params: {params}")

    # 训练（内置 early stopping）
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbosity > 0,
    )

    # 保存最佳模型（XGBoost 格式）
    project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    lookback = cfg.get("lookback", 48)
    ft = cfg['data']['feature_type']
    l_suffix = f"_L{lookback}" if lookback != 48 else ""
    ckpt_dir = os.path.join(project_root, "checkpoint", f"{MODEL_NAME}_{ft}{l_suffix}_seed{cfg['seed']}")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_model_path = os.path.join(ckpt_dir, "best.json")
    model.save_model(best_model_path)

    # 最终评估（使用 flatten 数据）
    train_metrics = evaluate(model, X_train_flat, y_train)
    val_metrics = evaluate(model, X_val_flat, y_val)
    test_metrics = evaluate(model, X_test_flat, y_test)

    print("\n" + "=" * 60)
    print(f"最终评估结果 ({MODEL_NAME.upper()} | {cfg['data']['feature_type']} | L={lookback})")
    print("=" * 60)
    print(f"Train  | IC: {train_metrics['IC']:.4f} | PIC: {train_metrics['PIC']:.4f} | DA: {train_metrics['DA']:.4f} | MSE: {train_metrics['MSE']:.6f} | "
          f"Sharpe: {train_metrics['Sharpe']:.4f} | IR: {train_metrics['IR']:.4f} | "
          f"MaxDD: {train_metrics['MaxDrawdown']:.4f} | AnnRet: {train_metrics['AnnualReturn']:.4f}")
    print(f"Val    | IC: {val_metrics['IC']:.4f} | PIC: {val_metrics['PIC']:.4f} | DA: {val_metrics['DA']:.4f} | MSE: {val_metrics['MSE']:.6f} | "
          f"Sharpe: {val_metrics['Sharpe']:.4f} | IR: {val_metrics['IR']:.4f} | "
          f"MaxDD: {val_metrics['MaxDrawdown']:.4f} | AnnRet: {val_metrics['AnnualReturn']:.4f}")
    print(f"Test   | IC: {test_metrics['IC']:.4f} | PIC: {test_metrics['PIC']:.4f} | DA: {test_metrics['DA']:.4f} | MSE: {test_metrics['MSE']:.6f} | "
          f"Sharpe: {test_metrics['Sharpe']:.4f} | IR: {test_metrics['IR']:.4f} | "
          f"MaxDD: {test_metrics['MaxDrawdown']:.4f} | AnnRet: {test_metrics['AnnualReturn']:.4f}")
    print("=" * 60)

    # 保存 results.txt（与主模型格式完全一致）
    result_path = os.path.join(ckpt_dir, "results.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"feature_type: {cfg['data']['feature_type']}\n")
        f.write(f"lookback: {lookback}\n\n")
        f.write(f"Train  IC: {train_metrics['IC']:.4f}  PIC: {train_metrics['PIC']:.4f}  DA: {train_metrics['DA']:.4f}  MSE: {train_metrics['MSE']:.6f}  "
                f"Sharpe: {train_metrics['Sharpe']:.4f}  IR: {train_metrics['IR']:.4f}  "
                f"MaxDrawdown: {train_metrics['MaxDrawdown']:.4f}  AnnualReturn: {train_metrics['AnnualReturn']:.4f}\n")
        f.write(f"Val    IC: {val_metrics['IC']:.4f}  PIC: {val_metrics['PIC']:.4f}  DA: {val_metrics['DA']:.4f}  MSE: {val_metrics['MSE']:.6f}  "
                f"Sharpe: {val_metrics['Sharpe']:.4f}  IR: {val_metrics['IR']:.4f}  "
                f"MaxDrawdown: {val_metrics['MaxDrawdown']:.4f}  AnnualReturn: {val_metrics['AnnualReturn']:.4f}\n")
        f.write(f"Test   IC: {test_metrics['IC']:.4f}  PIC: {test_metrics['PIC']:.4f}  DA: {test_metrics['DA']:.4f}  MSE: {test_metrics['MSE']:.6f}  "
                f"Sharpe: {test_metrics['Sharpe']:.4f}  IR: {test_metrics['IR']:.4f}  "
                f"MaxDrawdown: {test_metrics['MaxDrawdown']:.4f}  AnnualReturn: {test_metrics['AnnualReturn']:.4f}\n")

    print(f"\n[INFO] 最佳模型已保存: {best_model_path}")
    print(f"[INFO] 完整指标已保存: {result_path}")


if __name__ == "__main__":
    train()