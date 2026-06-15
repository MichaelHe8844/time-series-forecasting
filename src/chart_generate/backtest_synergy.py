#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate backtest comparison chart: CNN-Transformer full vs price_funding_fng.
Demonstrates the onchain synergy effect visually.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import cfg
from src.common import set_seed, SeqDataset, load_data, evaluate
from src.metrics import calc_strategy_returns

sns.set_style("whitegrid")
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11


def load_model_and_predict(model_module, model_name, ckpt_name, X_test, y_test, device):
    """Load a trained model and compute predictions & strategy returns."""
    model_cfg = cfg[model_name]
    model = model_module.CNNTransformer(X_test.shape[-1], model_cfg)

    ckpt_path = PROJECT_ROOT / "checkpoint" / ckpt_name / "best.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()

    test_dataset = SeqDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg["train"]["batch_size"], shuffle=False
    )

    preds = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            pred = model(x)
            preds.append(pred.detach().cpu().numpy())
    preds = np.concatenate(preds)

    mask = np.isfinite(preds) & np.isfinite(y_test)
    preds = preds[mask]
    trues = y_test[mask]
    preds = np.clip(preds, cfg["pred_clip_min"], cfg["pred_clip_max"])

    returns = calc_strategy_returns(preds, trues, fee=0.0005)
    equity = np.cumprod(1.0 + returns)

    return equity, returns


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ------------------------------------------------------------------
    # Load data for full and fng
    # ------------------------------------------------------------------
    import src.models.cnn_transformer as cnn_transformer_module

    results = {}

    for ft, ckpt_name in [
        ("full", "cnn_transformer_full_seed1"),
        ("price_funding_fng", "cnn_transformer_price_funding_fng_seed1"),
    ]:
        # Temporarily modify config to load correct data
        cfg["data"]["feature_type"] = ft
        set_seed(1)

        X_train, y_train, X_val, y_val, X_test, y_test = load_data()

        equity, returns = load_model_and_predict(
            cnn_transformer_module, "cnn_transformer", ckpt_name,
            X_test, y_test, device
        )

        results[ft] = {
            "equity": equity,
            "returns": returns,
        }

        n_periods = len(returns)
        sharpe = np.sqrt(2190) * returns.mean() / (returns.std() + 1e-12)
        ann_ret = returns.mean() * 2190
        max_dd = np.abs((equity / np.maximum.accumulate(equity) - 1.0).min())

        print(f"\n[{ft}]")
        print(f"  Sharpe: {sharpe:.4f}")
        print(f"  Annual Return: {ann_ret:.4f} ({ann_ret*100:.1f}%)")
        print(f"  Max Drawdown: {max_dd:.4f} ({max_dd*100:.1f}%)")
        print(f"  Periods: {n_periods}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                     gridspec_kw={'height_ratios': [3, 1]},
                                     sharex=True)

    # Color scheme
    color_full = '#2196F3'      # Blue for full
    color_fng = '#FF9800'       # Orange for fng

    equity_full = results["full"]["equity"]
    equity_fng = results["price_funding_fng"]["equity"]

    # Equity curves
    ax1.plot(equity_full, color=color_full, linewidth=1.2, label='Full (Price + Funding + F&G + SOPR/CDD)')
    ax1.plot(equity_fng, color=color_fng, linewidth=1.2, label='Price + Funding + F&G (w/o on-chain)')
    ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax1.set_ylabel('Cumulative Return', fontsize=12)
    ax1.set_title('CNN-Transformer Backtest: Full vs Price+Funding+F&G (seed=1)', fontsize=14, pad=12)
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Annotate final values
    final_full = equity_full[-1]
    final_fng = equity_fng[-1]
    ax1.annotate(f'{final_full:.2f}x', xy=(len(equity_full)-1, final_full),
                 xytext=(10, 10), textcoords='offset points', fontsize=11,
                 color=color_full, fontweight='bold')
    ax1.annotate(f'{final_fng:.2f}x', xy=(len(equity_fng)-1, final_fng),
                 xytext=(10, -15), textcoords='offset points', fontsize=11,
                 color=color_fng, fontweight='bold')

    # Drawdown
    dd_full = equity_full / np.maximum.accumulate(equity_full) - 1.0
    dd_fng = equity_fng / np.maximum.accumulate(equity_fng) - 1.0

    ax2.fill_between(range(len(dd_full)), dd_full, 0, color=color_full, alpha=0.3)
    ax2.fill_between(range(len(dd_fng)), dd_fng, 0, color=color_fng, alpha=0.3)
    ax2.plot(dd_full, color=color_full, linewidth=0.8)
    ax2.plot(dd_fng, color=color_fng, linewidth=0.8)
    ax2.set_ylabel('Drawdown', fontsize=12)
    ax2.set_xlabel('Test Period (4-hour steps)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.0, 0.05)

    # Summary stats box
    stats_text = (
        f"Full:       Sharpe={np.sqrt(2190)*results['full']['returns'].mean()/results['full']['returns'].std():.2f}  "
        f"AnnRet={results['full']['returns'].mean()*2190*100:.1f}%  "
        f"MaxDD={np.abs(dd_full.min())*100:.1f}%\n"
        f"F&G only:   Sharpe={np.sqrt(2190)*results['price_funding_fng']['returns'].mean()/results['price_funding_fng']['returns'].std():.2f}  "
        f"AnnRet={results['price_funding_fng']['returns'].mean()*2190*100:.1f}%  "
        f"MaxDD={np.abs(dd_fng.min())*100:.1f}%"
    )
    fig.text(0.12, 0.01, stats_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    # Save
    save_path = CHART_DIR / "backtest_synergy_full_vs_fng"
    for fmt in ('png', 'pdf'):
        plt.savefig(save_path.with_suffix(f'.{fmt}'), bbox_inches='tight')

    # Also save CSV for reproducibility
    csv_data = {
        'full_equity': equity_full,
        'full_drawdown': dd_full,
        'fng_equity': equity_fng,
        'fng_drawdown': dd_fng,
    }
    max_len = max(len(v) for v in csv_data.values())
    for k, v in csv_data.items():
        if len(v) < max_len:
            csv_data[k] = np.pad(v, (0, max_len - len(v)), constant_values=np.nan)
    np_csv = np.column_stack([csv_data[k] for k in csv_data])
    np.savetxt(
        CHART_DIR / "backtest_synergy_full_vs_fng.csv",
        np_csv,
        delimiter=',',
        header=','.join(csv_data.keys()),
        comments='',
        fmt='%.8f'
    )

    print(f"\nSaved:")
    for ext in ('png', 'pdf', 'csv'):
        print(f"   {save_path}.{ext}")

    plt.show()


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    CHART_DIR = PROJECT_ROOT / "charts"
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    main()
