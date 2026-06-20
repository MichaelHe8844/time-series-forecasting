"""
缺口 3-5 统一实验脚本：损失函数消融 + 架构消融 + 超参数敏感性。

用法:
  # 损失函数消融
  python src/analysis/run_ablation_experiments.py --exp loss --model cnn_transformer --feature full

  # 架构消融
  python src/analysis/run_ablation_experiments.py --exp arch --feature full

  # 超参数敏感性
  python src/analysis/run_ablation_experiments.py --exp hparam --model cnn_transformer --feature full

  # 全部 (可能需要几小时)
  python src/analysis/run_ablation_experiments.py --exp all --feature full
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_FILE = PROJECT_ROOT / "configs" / "config.py"

SEEDS = [0, 1, 2]
FEATURE_TYPES = ["full", "price_funding_fng"]

MODEL_CLASS_NAMES = {
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


# ============================================================================
# 缺口 3: 损失函数消融
# ============================================================================
LOSS_VARIANTS = {
    "huber":        {"ranking_loss_weight": 0.0},           # 纯 Huber
    "mse":          {"ranking_loss_weight": 0.0, "loss_type": "mse"},  # 纯 MSE
    "huber_rank":   {"ranking_loss_weight": 0.13},          # 当前 (Huber + Ranking)
}

# ============================================================================
# 缺口 4: 架构组件消融
# ============================================================================
ARCH_VARIANTS = {
    "full":      "cnn_transformer",        # 完整 CNN-Transformer
    "cnn_only":  "cnn_transformer_cnn_only",  # 仅 CNN 部分
    "tr_only":   "cnn_transformer_tr_only",   # 仅 Transformer 部分
    "tr_matched": "cnn_transformer_tr_matched",  # Transformer 参数量匹配版
}

# ============================================================================
# 缺口 5: 超参数敏感性
# ============================================================================
HPARAM_GRID = {
    "lr":   [5e-5, 1e-4, 2e-4],
    "dropout": [0.10, 0.13, 0.20],
}


def modify_config_and_run(model_name, feature, seed, overrides):
    """
    临时修改 config，运行单次训练，保存结果。

    策略：写一个临时 Python 脚本，在其中修改 cfg 后调用模型训练。
    避免直接修改 config.py 造成并发问题。
    """
    # 构建 Python 代码来运行训练
    override_code = "\n".join([
        f'cfg["{k}"] = {repr(v)}' if "." not in k
        else f'cfg{generate_nested_setter(k, v)}'
        for k, v in overrides.items()
    ])

    # 检查是否需要特殊处理（如 loss_type=mse）
    use_mse = overrides.get("loss_type") == "mse"
    use_special_arch = overrides.get("_arch_variant") is not None

    temp_script = PROJECT_ROOT / "src" / "analysis" / f"_temp_train_{model_name}_{feature}_seed{seed}.py"

    code = f'''
import sys
sys.path.insert(0, r"{PROJECT_ROOT}")
sys.path.insert(0, r"{SRC_DIR}")

from configs.config import cfg
from src.common import set_seed, load_data

# --- 覆盖配置 ---
{override_code}

cfg["seed"] = {seed}
cfg["data"]["feature_type"] = "{feature}"

set_seed(cfg["seed"])

# --- 加载数据 ---
X_train, y_train, X_val, y_val, X_test, y_test = load_data()
input_dim = X_train.shape[2]

# --- 构建模型 ---
import torch
import torch.nn as nn
import torch.nn.functional as F
'''

    if use_special_arch:
        arch_variant = overrides["_arch_variant"]
        code += f'''
# 架构消融变体: {arch_variant}
from src.models.cnn_transformer import CNNTransformer, ConvBlock, AttentionPooling
from src.common import PositionalEncoding, Chomp1d

class AblationModel(nn.Module):
    def __init__(self, input_dim, cfg_model):
        super().__init__()
        d_model = cfg_model.get("d_model", 192)
        dropout = cfg_model.get("dropout", 0.13)
'''
        if arch_variant == "cnn_only":
            code += f'''
        conv_channels = cfg_model.get("conv_channels", 192)
        kernel_size = cfg_model.get("kernel_size", 5)
        num_conv_layers = cfg_model.get("num_conv_layers", 3)

        conv_layers = []
        in_ch = input_dim
        for _ in range(num_conv_layers):
            conv_layers.append(ConvBlock(in_ch, conv_channels, kernel_size, dropout, True))
            in_ch = conv_channels
        self.conv_extractor = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(conv_channels, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
'''
        elif arch_variant == "tr_only":
            code += f'''
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=512)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=512,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
'''
        elif arch_variant == "tr_matched":
            # 增大 Transformer 使其参数量接近 CNN-Transformer (~1.75M)
            code += f'''
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=512)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=768,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
'''

        if arch_variant == "cnn_only":
            code += f'''
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_extractor(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x).squeeze(-1)
'''
        else:
            code += f'''
    def forward(self, x):
        x = self.proj(x)
        x = self.pos_encoder(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)
        out = self.transformer(x)
        out = out[:, -1, :]
        return self.head(out).squeeze(-1)
'''

        code += f'''
model = AblationModel(input_dim, cfg.get("{model_name}", {{}}))
arch_name = "{arch_variant}"
'''
    else:
        code += f'''
from src.models.{model_name} import {MODEL_CLASS_NAMES[model_name]}

model = {MODEL_CLASS_NAMES[model_name]}(input_dim, cfg["{model_name}"])
arch_name = "standard"
'''

    if use_mse:
        code += f'''
# MSE loss 变体
ranking_weight = 0.0
'''
    else:
        code += f'''
ranking_weight = cfg["{model_name}"].get("ranking_loss_weight", 0.13)
'''

    code += f'''
# --- 简化版训练循环 (同 common.run_training 但允许 loss 变体) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from src.metrics import (
    calc_ic, calc_pearson_ic, calc_da, calc_mse,
    calc_strategy_returns, calc_sharpe, calc_ic_ir,
    calc_max_drawdown, calc_annual_return,
)

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = SeqDataset(X_train, y_train)
val_ds = SeqDataset(X_val, y_val)
test_ds = SeqDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
train_eval_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

lr = cfg["{model_name}"].get("lr", 1e-4)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

ranking_margin = cfg["{model_name}"].get("ranking_margin", 0.0005)
grad_clip = cfg["{model_name}"].get("grad_clip", 1.0)

huber_criterion = nn.HuberLoss(delta=0.3)
mse_criterion = nn.MSELoss()

def sw_evaluate(model, loader):
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
        return {{"IC": 0, "PIC": 0, "DA": 0.5, "MSE": 0, "Sharpe": 0, "IR": 0, "MaxDrawdown": 0, "AnnualReturn": 0}}
    preds = np.clip(preds, -0.005, 0.005)
    sr = calc_strategy_returns(preds, trues, fee=0.0005)
    return {{
        "IC": calc_ic(preds, trues), "PIC": calc_pearson_ic(preds, trues),
        "DA": calc_da(preds, trues), "MSE": calc_mse(preds, trues),
        "Sharpe": calc_sharpe(sr), "IR": calc_ic_ir(preds, trues),
        "MaxDrawdown": calc_max_drawdown(sr), "AnnualReturn": calc_annual_return(sr),
    }}

# Checkpoint dir
ckpt_dir = r"{PROJECT_ROOT / 'checkpoint' / f'{model_name}_ablation_{{{{arch_name}}}}_{feature}_seed{{seed}}'}"
import os as _os
_os.makedirs(ckpt_dir, exist_ok=True)
best_model_path = _os.path.join(ckpt_dir, "best.pth")

best_val_loss = float('inf')
patience_counter = 0

print(f"[Ablation] model={{arch_name}}, feature={{'{feature}'}}, seed={seed}")
print(f"[Ablation] ranking_weight={{ranking_weight}}")

for epoch in range(200):
    model.train()
    train_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)

        use_mse = {str(use_mse)}
        if use_mse:
            loss_reg = mse_criterion(pred, y)
        else:
            loss_reg = huber_criterion(pred, y)

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
    val_m = sw_evaluate(model, val_loader)
    val_loss = val_m.get("MSE", float('inf'))
    scheduler.step(val_loss)

    if (epoch + 1) % 30 == 0:
        print(f"  Epoch {{epoch+1:3d}} | Train Loss: {{train_loss:.6f}} | "
              f"Val IC: {{val_m['IC']:.4f}} | Val Sharpe: {{val_m['Sharpe']:.4f}}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        patience_counter += 1
        if patience_counter >= 20:
            print(f"  Early stopping at epoch {{epoch+1}}")
            break

model.load_state_dict(torch.load(best_model_path, map_location=device))
train_m = sw_evaluate(model, train_eval_loader)
val_m = sw_evaluate(model, val_loader)
test_m = sw_evaluate(model, test_loader)

print(f"\\n{{'='*50}}")
print(f"Result: arch={{arch_name}} feature={{'{feature}'}} seed={seed}")
print(f"Test | IC: {{test_m['IC']:.4f}} | Sharpe: {{test_m['Sharpe']:.4f}} | "
      f"MaxDD: {{test_m['MaxDrawdown']:.4f}} | AnnRet: {{test_m['AnnualReturn']:.4f}}")
print(f"{{'='*50}}\\n")

result_path = _os.path.join(ckpt_dir, "results.txt")
with open(result_path, "w", encoding="utf-8") as f:
    f.write(f"feature_type: {{'{feature}'}}\\n")
    f.write(f"arch_variant: {{arch_name}}\\n")
    f.write(f"ranking_weight: {{ranking_weight}}\\n")
    f.write(f"lookback: 48\\n\\n")
    for split, m in [("Train", train_m), ("Val", val_m), ("Test", test_m)]:
        f.write(f"{{split}}  IC: {{m['IC']:.4f}}  PIC: {{m['PIC']:.4f}}  "
                f"DA: {{m['DA']:.4f}}  MSE: {{m['MSE']:.6f}}  "
                f"Sharpe: {{m['Sharpe']:.4f}}  IR: {{m['IR']:.4f}}  "
                f"MaxDrawdown: {{m['MaxDrawdown']:.4f}}  "
                f"AnnualReturn: {{m['AnnualReturn']:.4f}}\\n")

print(f"[OK] Results saved: {{result_path}}")
'''

    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(code)

    try:
        result = subprocess.run(
            [sys.executable, str(temp_script)],
            cwd=str(SRC_DIR),
            timeout=7200,
            capture_output=True,
            text=True,
        )
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        if result.returncode != 0:
            print(f"[ERROR] stderr:\n{result.stderr[-1000:]}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("[ERROR] Training timed out (2 hours)")
        return False
    finally:
        # 清理临时脚本
        if temp_script.exists():
            temp_script.unlink()


def generate_nested_setter(key_path, value):
    """生成嵌套字典的设置代码，如 'data.feature_type' -> ['data']['feature_type'] = v"""
    parts = key_path.split(".")
    setter = ""
    for part in parts[:-1]:
        setter += f'["{part}"]'
    setter += f'["{parts[-1]}"] = {repr(value)}'
    return setter


def run_loss_ablation(args):
    """缺口 3: 损失函数消融"""
    models = [args.model] if args.model != "all" else ["cnn_transformer", "timemixer"]
    features = [args.feature] if args.feature != "all" else ["full"]

    print("=" * 60)
    print("缺口 3: 损失函数消融实验")
    print(f"  模型: {models}  特征集: {features}  种子: {SEEDS}")
    print("=" * 60)

    for model in models:
        for ft in features:
            for variant_name, overrides in LOSS_VARIANTS.items():
                for seed in SEEDS:
                    label = f"{model}/{ft}/{variant_name}/seed{seed}"
                    print(f"\n[Loss Ablation] {label}")

                    # 构建覆盖
                    cfg_overrides = {}
                    for k, v in overrides.items():
                        cfg_overrides[f"{model}.{k}"] = v

                    success = modify_config_and_run(model, ft, seed, cfg_overrides)
                    if not success:
                        print(f"  [FAILED] {label}")


def run_arch_ablation(args):
    """缺口 4: 架构组件消融"""
    features = [args.feature] if args.feature != "all" else ["full"]

    print("=" * 60)
    print("缺口 4: 架构组件消融实验")
    print(f"  特征集: {features}  种子: {SEEDS}")
    print("=" * 60)

    for ft in features:
        for arch_name in ["cnn_only", "tr_only", "tr_matched"]:
            for seed in SEEDS:
                label = f"{arch_name}/{ft}/seed{seed}"
                print(f"\n[Arch Ablation] {label}")

                overrides = {"cnn_transformer._arch_variant": arch_name}
                success = modify_config_and_run("cnn_transformer", ft, seed, overrides)
                if not success:
                    print(f"  [FAILED] {label}")

    # 基线（完整 CNN-Transformer）已在标准实验中有，但确保种子匹配
    for ft in features:
        for seed in SEEDS:
            # 检查是否已有 checkpoint
            ckpt = PROJECT_ROOT / "checkpoint" / f"cnn_transformer_{ft}_seed{seed}" / "results.txt"
            if ckpt.exists():
                print(f"[Arch Baseline] {ft}/seed{seed} already exists, skipping")
            else:
                print(f"[Arch Baseline] {ft}/seed{seed} — running standard training")
                # 直接调用标准训练
                cmd = [
                    sys.executable, "-c",
                    f"import sys; sys.path.insert(0, r'{SRC_DIR}'); "
                    f"from configs.config import cfg; "
                    f"cfg['seed']={seed}; cfg['data']['feature_type']='{ft}'; "
                    f"from src.models.cnn_transformer import train; train()"
                ]
                subprocess.run(cmd, cwd=str(SRC_DIR), timeout=7200)


def run_hparam_ablation(args):
    """缺口 5: 超参数敏感性"""
    models = [args.model] if args.model != "all" else ["cnn_transformer", "timemixer", "transformer"]
    features = [args.feature] if args.feature != "all" else ["full"]

    print("=" * 60)
    print("缺口 5: 超参数敏感性实验")
    print(f"  模型: {models}  特征集: {features}")
    print(f"  超参数网格: lr={HPARAM_GRID['lr']}, dropout={HPARAM_GRID['dropout']}")
    print("=" * 60)

    for model in models:
        for ft in features:
            # 只跑 seed=0,1 节省时间
            for seed in [0, 1]:
                # 学习率敏感性
                for lr in HPARAM_GRID["lr"]:
                    if lr == 1e-4:
                        continue  # 基线已有
                    label = f"{model}/{ft}/lr{lr}/seed{seed}"
                    print(f"\n[HP LR] {label}")
                    overrides = {f"{model}.lr": lr}
                    modify_config_and_run(model, ft, seed, overrides)

                # Dropout 敏感性
                for do in HPARAM_GRID["dropout"]:
                    if do == 0.13:
                        continue  # 基线已有
                    label = f"{model}/{ft}/do{do}/seed{seed}"
                    print(f"\n[HP DO] {label}")
                    overrides = {f"{model}.dropout": do}
                    modify_config_and_run(model, ft, seed, overrides)


def main():
    parser = argparse.ArgumentParser(description="Ablation experiments (Gaps 3-5)")
    parser.add_argument("--exp", type=str, required=True,
                        choices=["loss", "arch", "hparam", "all"])
    parser.add_argument("--model", type=str, default="cnn_transformer")
    parser.add_argument("--feature", type=str, default="full")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] No training will be executed.")
        if args.exp in ("loss", "all"):
            print(f"  Loss ablation: {len(LOSS_VARIANTS)} variants × 2 models × 3 seeds = 18 runs")
        if args.exp in ("arch", "all"):
            print(f"  Arch ablation: 3 variants × 3 seeds = 9 runs")
        if args.exp in ("hparam", "all"):
            print(f"  HP sensitivity: 4 params × 2 models × 2 seeds = 16 runs")
        return

    if args.exp in ("loss", "all"):
        run_loss_ablation(args)
    if args.exp in ("arch", "all"):
        run_arch_ablation(args)
    if args.exp in ("hparam", "all"):
        run_hparam_ablation(args)

    print("\n[DONE] All ablation experiments complete.")


if __name__ == "__main__":
    main()
