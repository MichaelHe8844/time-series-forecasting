"""
缺口 3: 损失函数消融 —— 直接修改 config + 调用标准训练流程。

用法:
  python src/analysis/run_loss_ablation.py --model cnn_transformer --feature full --loss huber --seed 0
  python src/analysis/run_loss_ablation.py --model all --feature full
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from configs.config import cfg
from src.common import set_seed, load_data, SeqDataset, PositionalEncoding, Chomp1d
from src.metrics import (
    calc_ic, calc_pearson_ic, calc_da, calc_mse,
    calc_strategy_returns, calc_sharpe, calc_ic_ir,
    calc_max_drawdown, calc_annual_return,
)
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

MODEL_CLASS_NAMES = {
    "cnn_transformer": "CNNTransformer",
    "timemixer": "TimeMixer",
    "transformer": "TransformerModel",
    "lstm": "BiLSTM",
    "lstm_transformer": "LSTMTransformer",
    "tcn": "TCN",
    "modern_tcn": "ModernTCN",
    "patchtst": "PatchTST",
    "dlinear": "DLinear",
}


def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            preds.append(pred.detach().cpu().numpy())
            trues.append(y.detach().cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    mask = np.isfinite(preds) & np.isfinite(trues)
    preds, trues = preds[mask], trues[mask]
    if len(preds) == 0:
        return {"IC": 0, "PIC": 0, "DA": 0.5, "MSE": 0,
                "Sharpe": 0, "IR": 0, "MaxDrawdown": 0, "AnnualReturn": 0}
    preds = np.clip(preds, -0.005, 0.005)
    sr = calc_strategy_returns(preds, trues, fee=0.0005)
    return {
        "IC": calc_ic(preds, trues), "PIC": calc_pearson_ic(preds, trues),
        "DA": calc_da(preds, trues), "MSE": calc_mse(preds, trues),
        "Sharpe": calc_sharpe(sr), "IR": calc_ic_ir(preds, trues),
        "MaxDrawdown": calc_max_drawdown(sr), "AnnualReturn": calc_annual_return(sr),
    }


def train_with_loss(model_name, feature_type, seed, ranking_weight, loss_type="huber"):
    """核心：使用指定损失函数训练模型。"""
    set_seed(seed)
    cfg["seed"] = seed
    cfg["data"]["feature_type"] = feature_type

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    input_dim = X_train.shape[2]

    # 导入模型类
    class_name = MODEL_CLASS_NAMES[model_name]
    model_module = __import__(f"src.models.{model_name}", fromlist=[class_name])
    ModelClass = getattr(model_module, class_name)
    model = ModelClass(input_dim, cfg[model_name])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model_cfg = cfg[model_name]

    # DataLoaders
    train_ds = SeqDataset(X_train, y_train)
    val_ds = SeqDataset(X_val, y_val)
    test_ds = SeqDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    train_eval_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # Optimizer
    lr = model_cfg.get("lr", 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    ranking_margin = model_cfg.get("ranking_margin", 0.0005)
    grad_clip = model_cfg.get("grad_clip", 1.0)

    # 损失函数
    if loss_type == "mse":
        reg_criterion = nn.MSELoss()
    else:
        reg_criterion = nn.HuberLoss(delta=0.3)

    # Checkpoint dir
    loss_label = f"{loss_type}" + (f"_rank{ranking_weight}" if ranking_weight > 0 else "")
    ckpt_dir = PROJECT_ROOT / "checkpoint" / f"{model_name}_loss_{loss_label}_{feature_type}_seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = ckpt_dir / "best.pth"

    print(f"\n[Loss Ablation] {model_name} | {feature_type} | {loss_type} | "
          f"rank_w={ranking_weight} | seed={seed}")

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(200):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss_reg = reg_criterion(pred, y)

            # Ranking loss (if enabled)
            bs = pred.size(0)
            if bs > 1 and ranking_weight > 0:
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
        val_m = evaluate(model, val_loader, device)
        val_loss = val_m["MSE"]
        scheduler.step(val_loss)

        if (epoch + 1) % 30 == 0:
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
    train_m = evaluate(model, train_eval_loader, device)
    val_m = evaluate(model, val_loader, device)
    test_m = evaluate(model, test_loader, device)

    print(f"\n{'='*50}")
    print(f"Result: loss={loss_type} rank_w={ranking_weight} seed={seed}")
    print(f"Test | IC: {test_m['IC']:.4f} | Sharpe: {test_m['Sharpe']:.4f} | "
          f"MaxDD: {test_m['MaxDrawdown']:.4f} | AnnRet: {test_m['AnnualReturn']:.4f}")
    print(f"{'='*50}\n")

    # Save results
    result_path = ckpt_dir / "results.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"feature_type: {feature_type}\n")
        f.write(f"loss_type: {loss_type}\n")
        f.write(f"ranking_weight: {ranking_weight}\n")
        f.write(f"lookback: 48\n\n")
        for split, m in [("Train", train_m), ("Val", val_m), ("Test", test_m)]:
            f.write(f"{split}  IC: {m['IC']:.4f}  PIC: {m['PIC']:.4f}  "
                    f"DA: {m['DA']:.4f}  MSE: {m['MSE']:.6f}  "
                    f"Sharpe: {m['Sharpe']:.4f}  IR: {m['IR']:.4f}  "
                    f"MaxDrawdown: {m['MaxDrawdown']:.4f}  "
                    f"AnnualReturn: {m['AnnualReturn']:.4f}\n")

    return test_m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cnn_transformer")
    parser.add_argument("--feature", type=str, default="full")
    parser.add_argument("--loss", type=str, default="all",
                        choices=["huber", "mse", "huber_rank", "all"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds_all", action="store_true",
                        help="Run seeds 0, 1, 2 for the specified loss")
    args = parser.parse_args()

    models = [args.model]
    if args.model == "all":
        models = ["cnn_transformer", "timemixer"]

    # 损失函数配置: (loss_type, ranking_weight)
    loss_configs = {
        "huber":      ("huber", 0.0),
        "mse":        ("mse",   0.0),
        "huber_rank": ("huber", 0.13),
    }

    if args.loss == "all":
        variants = list(loss_configs.items())
    else:
        variants = [(args.loss, loss_configs[args.loss])]

    seeds = [0, 1, 2] if args.seeds_all else [args.seed]

    print("=" * 60)
    print(f"损失函数消融实验")
    print(f"  模型: {models}  特征: {args.feature}")
    print(f"  变体: {[v[0] for v in variants]}  种子: {seeds}")
    print(f"  总训练: {len(models) * len(variants) * len(seeds)}")
    print("=" * 60)

    results = {}
    for model in models:
        for variant_name, (loss_type, rank_w) in variants:
            for seed in seeds:
                key = f"{model}/{variant_name}/seed{seed}"
                try:
                    m = train_with_loss(model, args.feature, seed, rank_w, loss_type)
                    results[key] = m
                except Exception as e:
                    print(f"[FAILED] {key}: {e}")
                    import traceback
                    traceback.print_exc()

    # 汇总
    print("\n" + "=" * 70)
    print("损失函数消融结果汇总")
    print(f"{'配置':<40} {'IC':>8} {'Sharpe':>8} {'AnnRet':>8}")
    print("-" * 70)
    for key, m in sorted(results.items()):
        print(f"{key:<40} {m['IC']:8.4f} {m['Sharpe']:8.4f} {m['AnnualReturn']:8.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
