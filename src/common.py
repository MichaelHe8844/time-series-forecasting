"""
Shared PyTorch utilities for all deep learning models.
"""
import math
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from configs.config import cfg
from src.metrics import (calc_ic, calc_pearson_ic, calc_da, calc_mse, calc_strategy_returns,
                         calc_sharpe, calc_ic_ir, calc_max_drawdown, calc_annual_return)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


def load_data():
    common_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(common_dir, ".."))
    root = os.path.join(project_root, "features")

    ft = cfg["data"]["feature_type"]
    lookback = cfg.get("lookback", 48)
    if lookback == 48:
        path = os.path.join(root, ft)
    else:
        path = os.path.join(root, f"{ft}_L{lookback}")

    print(f"[INFO] Feature set: {ft}")
    print(f"[INFO] Loading from: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature path not found: {path}\n请先运行 build_features.py")

    def load(name):
        file_path = os.path.join(path, name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing file: {file_path}")
        return np.load(file_path)

    X_train = load("X_train.npy")
    y_train = load("y_train.npy")
    X_val = load("X_val.npy")
    y_val = load("y_val.npy")
    X_test = load("X_test.npy")
    y_test = load("y_test.npy")

    print(f"[INFO] X_train shape: {X_train.shape}")
    print(f"[INFO] X_val shape:   {X_val.shape}")
    print(f"[INFO] X_test shape:  {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def run_training(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Unified training loop for all deep learning models.

    Args:
        model: nn.Module instance (not yet on device)
        model_name: e.g. "lstm", "cnn_transformer"
        X_train ... y_test: numpy arrays from load_data()
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = model.to(device)
    model_cfg = cfg.get(model_name, {})

    # DataLoaders
    train_dataset = SeqDataset(X_train, y_train)
    val_dataset = SeqDataset(X_val, y_val)
    test_dataset = SeqDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["train"]["num_workers"], pin_memory=cfg["train"]["pin_memory"]
    )
    train_eval_loader = DataLoader(
        train_dataset, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["train"]["num_workers"], pin_memory=cfg["train"]["pin_memory"]
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["train"]["num_workers"], pin_memory=cfg["train"]["pin_memory"]
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["train"]["num_workers"], pin_memory=cfg["train"]["pin_memory"]
    )

    # Optimizer
    lr = model_cfg.get("lr", 1e-4)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=cfg["train"]["weight_decay"]
    )

    # Huber + Ranking Loss
    huber_criterion = nn.HuberLoss(delta=0.3)
    ranking_weight = model_cfg.get("ranking_loss_weight", 0.13)
    ranking_margin = model_cfg.get("ranking_margin", 0.0005)
    print(f"[INFO] Ranking loss weight: {ranking_weight}, margin: {ranking_margin}")

    # Scheduler & early stopping
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    best_val_loss = float('inf')
    patience_counter = 0
    grad_clip = model_cfg.get("grad_clip", 1.0)

    # Checkpoint
    project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    lookback = cfg.get("lookback", 48)
    ft = cfg['data']['feature_type']
    l_suffix = f"_L{lookback}" if lookback != 48 else ""
    ckpt_dir = os.path.join(project_root, "checkpoint", f"{model_name}_{ft}{l_suffix}_seed{cfg['seed']}")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_model_path = os.path.join(ckpt_dir, "best.pth")

    # Training loop
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x)

            loss_reg = huber_criterion(pred, y)

            batch_size = pred.size(0)
            if batch_size > 1:
                perm = torch.randperm(batch_size, device=device)
                half = batch_size // 2
                idx_i = perm[:half]
                idx_j = perm[half:2 * half]

                pred_i = pred[idx_i]
                pred_j = pred[idx_j]
                y_i = y[idx_i]
                y_j = y[idx_j]

                target = torch.sign(y_i - y_j)
                mask = (target != 0)
                if mask.any():
                    loss_rank = F.margin_ranking_loss(
                        pred_i[mask], pred_j[mask],
                        target[mask].float(),
                        margin=ranking_margin
                    )
                else:
                    loss_rank = torch.tensor(0.0, device=device)
            else:
                loss_rank = torch.tensor(0.0, device=device)

            loss = (1.0 - ranking_weight) * loss_reg + ranking_weight * loss_rank

            optimizer.zero_grad()
            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()
            train_loss += loss.item() * len(y)

        train_loss /= len(train_loader.dataset)

        val_metrics = evaluate(model, val_loader, device, huber_criterion)
        val_loss = val_metrics["Loss"]
        scheduler.step(val_loss)

        print(
            f"\nEpoch {epoch + 1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"Val IC: {val_metrics['IC']:.4f} PIC: {val_metrics['PIC']:.4f} DA: {val_metrics['DA']:.4f} MSE: {val_metrics['MSE']:.6f} "
            f"Sharpe: {val_metrics['Sharpe']:.4f} IR: {val_metrics['IR']:.4f} "
            f"MDD: {val_metrics['MaxDrawdown']:.4f} AnnRet: {val_metrics['AnnualReturn']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg["train"]["patience"]:
                print("[INFO] Early stopping triggered")
                break

    # Final evaluation
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    train_metrics = evaluate(model, train_eval_loader, device)
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    print("\n" + "=" * 60)
    print(f"Final evaluation ({model_name.upper()} | {cfg['data']['feature_type']} | L={lookback})")
    print("=" * 60)
    for split, m in [("Train", train_metrics), ("Val", val_metrics), ("Test", test_metrics)]:
        print(f"{split}  | IC: {m['IC']:.4f} | PIC: {m['PIC']:.4f} | DA: {m['DA']:.4f} | MSE: {m['MSE']:.6f} | "
              f"Sharpe: {m['Sharpe']:.4f} | IR: {m['IR']:.4f} | "
              f"MaxDD: {m['MaxDrawdown']:.4f} | AnnRet: {m['AnnualReturn']:.4f}")
    print("=" * 60)

    # Save results.txt
    result_path = os.path.join(ckpt_dir, "results.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"feature_type: {cfg['data']['feature_type']}\n")
        f.write(f"lookback: {lookback}\n\n")
        for split, m in [("Train", train_metrics), ("Val", val_metrics), ("Test", test_metrics)]:
            f.write(f"{split}  IC: {m['IC']:.4f}  PIC: {m['PIC']:.4f}  DA: {m['DA']:.4f}  MSE: {m['MSE']:.6f}  "
                    f"Sharpe: {m['Sharpe']:.4f}  IR: {m['IR']:.4f}  "
                    f"MaxDrawdown: {m['MaxDrawdown']:.4f}  AnnualReturn: {m['AnnualReturn']:.4f}\n")

    print(f"\n[INFO] Best model saved: {best_model_path}")
    print(f"[INFO] Full metrics saved: {result_path}")

    return train_metrics, val_metrics, test_metrics


def evaluate(model, loader, device, criterion=None):
    model.eval()
    preds, trues = [], []
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            preds.append(pred.detach().cpu().numpy())
            trues.append(y.detach().cpu().numpy())
            if criterion is not None:
                loss = criterion(pred, y)
                total_loss += loss.item() * len(y)
                count += len(y)

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    mask = np.isfinite(preds) & np.isfinite(trues)
    preds = preds[mask]
    trues = trues[mask]

    if len(preds) == 0:
        return {
            "IC": 0.0, "PIC": 0.0, "DA": 0.5, "MSE": 0.0,
            "Sharpe": 0.0, "IR": 0.0,
            "MaxDrawdown": 0.0, "AnnualReturn": 0.0
        }

    preds = np.clip(preds, cfg["pred_clip_min"], cfg["pred_clip_max"])

    strategy_returns = calc_strategy_returns(preds, trues, fee=0.0005)

    metrics = {
        "IC": calc_ic(preds, trues),
        "PIC": calc_pearson_ic(preds, trues),
        "DA": calc_da(preds, trues),
        "MSE": calc_mse(preds, trues),
        "Sharpe": calc_sharpe(strategy_returns),
        "IR": calc_ic_ir(preds, trues),
        "MaxDrawdown": calc_max_drawdown(strategy_returns),
        "AnnualReturn": calc_annual_return(strategy_returns)
    }
    if criterion is not None and count > 0:
        metrics["Loss"] = total_loss / count
    return metrics
