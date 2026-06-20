"""
缺口 4: CNN-Transformer 架构组件消融。

消融变体:
  - full:      完整 CNN-Transformer (3层因果卷积 → 2层Transformer, 12头)
  - cnn_only:  去Transformer, CNN后直接AdaptiveAvgPool → MLP输出
  - tr_only:   去CNN, 原始输入投影后直接进Transformer (4头, 2层)
  - tr_matched: 去CNN, Transformer加深加宽至~1.75M参数 (8头, 4层, FF=768)

用法:
  python src/analysis/run_arch_ablation.py --variant cnn_only --feature full --seed 0
  python src/analysis/run_arch_ablation.py --variant all --feature full --seeds_all
"""

import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from configs.config import cfg
from src.common import set_seed, load_data, SeqDataset, PositionalEncoding
from src.models.cnn_transformer import ConvBlock
from src.metrics import (
    calc_ic, calc_pearson_ic, calc_da, calc_mse,
    calc_strategy_returns, calc_sharpe, calc_ic_ir,
    calc_max_drawdown, calc_annual_return,
)
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F


class CNNOnlyModel(nn.Module):
    """仅保留 CNN 部分，去掉 Transformer。"""
    def __init__(self, input_dim, cfg_model):
        super().__init__()
        conv_channels = cfg_model.get("conv_channels", 192)
        kernel_size = cfg_model.get("kernel_size", 5)
        num_conv_layers = cfg_model.get("num_conv_layers", 3)
        dropout = cfg_model.get("dropout", 0.13)

        conv_layers = []
        in_ch = input_dim
        for _ in range(num_conv_layers):
            conv_layers.append(ConvBlock(in_ch, conv_channels, kernel_size, dropout, True))
            in_ch = conv_channels
        self.conv_extractor = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        hidden = conv_channels // 2
        self.head = nn.Sequential(
            nn.Linear(conv_channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)          # (B, F, L)
        x = self.conv_extractor(x)     # (B, C, L)
        x = self.pool(x).squeeze(-1)   # (B, C)
        return self.head(x).squeeze(-1)


class TROnlyModel(nn.Module):
    """仅保留 Transformer 部分，去掉 CNN。"""
    def __init__(self, input_dim, cfg_model):
        super().__init__()
        d_model = cfg_model.get("d_model", 192)
        nhead = 4
        num_layers = 2
        dim_feedforward = 512
        dropout = cfg_model.get("dropout", 0.13)

        self.proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=512)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        hidden = d_model // 2
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.pos_encoder(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)
        out = self.transformer(x)
        out = out[:, -1, :]
        return self.head(out).squeeze(-1)


class TRMatchedModel(nn.Module):
    """Transformer-only，参数量匹配完整 CNN-Transformer (~1.75M)。"""
    def __init__(self, input_dim, cfg_model):
        super().__init__()
        d_model = 192
        nhead = 8
        num_layers = 4
        dim_feedforward = 768
        dropout = cfg_model.get("dropout", 0.13)

        self.proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=512)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        hidden = d_model // 2
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.pos_encoder(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)
        out = self.transformer(x)
        out = out[:, -1, :]
        return self.head(out).squeeze(-1)


# 变体映射
ARCH_VARIANTS = {
    "cnn_only": CNNOnlyModel,
    "tr_only": TROnlyModel,
    "tr_matched": TRMatchedModel,
}


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


def train_arch(variant, feature_type, seed):
    """训练一个架构消融变体。"""
    set_seed(seed)
    cfg["seed"] = seed
    cfg["data"]["feature_type"] = feature_type

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    input_dim = X_train.shape[2]

    ModelClass = ARCH_VARIANTS[variant]
    model = ModelClass(input_dim, cfg["cnn_transformer"])
    n_params = count_params(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_ds = SeqDataset(X_train, y_train)
    val_ds = SeqDataset(X_val, y_val)
    test_ds = SeqDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    train_eval_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    lr = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    huber_criterion = nn.HuberLoss(delta=0.3)
    ranking_weight = 0.13
    ranking_margin = 0.0005
    grad_clip = 1.0

    ckpt_dir = PROJECT_ROOT / "checkpoint" / f"cnn_transformer_arch_{variant}_{feature_type}_seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = ckpt_dir / "best.pth"

    print(f"\n[Arch Ablation] variant={variant} params={n_params/1e6:.2f}M "
          f"feature={feature_type} seed={seed}")

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

    print(f"\n  Arch={variant} params={n_params/1e6:.2f}M | "
          f"Test IC={test_m['IC']:.4f} Sharpe={test_m['Sharpe']:.4f} "
          f"MaxDD={test_m['MaxDrawdown']:.4f} AnnRet={test_m['AnnualReturn']:.4f}")

    result_path = ckpt_dir / "results.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"feature_type: {feature_type}\n")
        f.write(f"arch_variant: {variant}\n")
        f.write(f"params: {n_params}\n")
        f.write(f"lookback: 48\n\n")
        for split, m in [("Train", train_m), ("Val", val_m), ("Test", test_m)]:
            f.write(f"{split}  IC: {m['IC']:.4f}  PIC: {m['PIC']:.4f}  "
                    f"DA: {m['DA']:.4f}  MSE: {m['MSE']:.6f}  "
                    f"Sharpe: {m['Sharpe']:.4f}  IR: {m['IR']:.4f}  "
                    f"MaxDrawdown: {m['MaxDrawdown']:.4f}  "
                    f"AnnualReturn: {m['AnnualReturn']:.4f}\n")

    return test_m, n_params


# 基线：从已有 checkpoint 读取
BASELINE_CHECKPOINTS = {
    0: PROJECT_ROOT / "checkpoint" / "cnn_transformer_full_seed0" / "results.txt",
    1: PROJECT_ROOT / "checkpoint" / "cnn_transformer_full_seed1" / "results.txt",
}

def get_baseline(seed):
    """读取完整 CNN-Transformer 的基线结果。"""
    import re
    ckpt = BASELINE_CHECKPOINTS.get(seed)
    if ckpt and ckpt.exists():
        with open(ckpt, "r") as f:
            content = f.read()
        m = re.search(r"Test.*Sharpe:\s*([-0-9.eE]+)", content)
        if m:
            return float(m.group(1))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="all",
                        choices=["cnn_only", "tr_only", "tr_matched", "all"])
    parser.add_argument("--feature", type=str, default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds_all", action="store_true")
    args = parser.parse_args()

    variants = list(ARCH_VARIANTS.keys()) if args.variant == "all" else [args.variant]
    seeds = [0, 1] if args.seeds_all else [args.seed]

    print("=" * 60)
    print(f"架构组件消融实验")
    print(f"  变体: {variants}  特征: {args.feature}  种子: {seeds}")
    print(f"  总训练: {len(variants) * len(seeds)}")
    print("=" * 60)

    results = {}
    for variant in variants:
        for seed in seeds:
            key = f"{variant}/seed{seed}"
            try:
                test_m, n_params = train_arch(variant, args.feature, seed)
                baseline_sharpe = get_baseline(seed)
                results[key] = {**test_m, "params": n_params, "baseline_sharpe": baseline_sharpe}
            except Exception as e:
                print(f"[FAILED] {key}: {e}")
                import traceback
                traceback.print_exc()

    # 汇总
    print("\n" + "=" * 80)
    print("架构消融结果汇总")
    print(f"{'变体':<20} {'参数量':>8} {'IC':>8} {'Sharpe':>8} {'基线Sharpe':>10} {'Diff':>8}")
    print("-" * 80)
    for key, m in sorted(results.items()):
        diff = m['Sharpe'] - m['baseline_sharpe'] if m.get('baseline_sharpe') else 0
        print(f"{key:<20} {m['params']/1e6:7.2f}M {m['IC']:8.4f} {m['Sharpe']:8.4f} "
              f"{m.get('baseline_sharpe', 0):10.2f} {diff:+8.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
