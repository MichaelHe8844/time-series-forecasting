# Multi-Source Heterogeneous Data Fusion for Cryptocurrency Return Forecasting

[![DOI](https://zenodo.org/badge/1269770764.svg)](https://doi.org/10.5281/zenodo.20696269)

Source code for the paper: **"Multi-Source Heterogeneous Data Fusion for Cryptocurrency Return Forecasting: On the Conditional Effectiveness of On-Chain Indicators"**

## Overview

This repository provides a complete experimental pipeline for systematic multi-source data fusion in cryptocurrency return forecasting. We evaluate whether on-chain indicators (SOPR, CDD, exchange balances, active addresses, netflow) provide incremental predictive value for Bitcoin 4-hour returns, and under what conditions.

**10 models × 6 feature configurations × 5 random seeds × 8 sliding windows = 2,400+ trained models.**

### Models
| Family | Models |
|--------|--------|
| Recurrent | LSTM |
| Transformer | Transformer, PatchTST |
| Convolutional | TCN, ModernTCN |
| Hybrid | CNN-Transformer, LSTM-Transformer |
| Linear | DLinear |
| Multi-scale Mixing | TimeMixer (ICLR 2024) |
| Tree-based | XGBoost |

### Feature Configurations
- `price_only` (18 dims) — Baseline
- `price_funding` (28 dims) — + Funding rate
- `price_funding_fng` (40 dims) — + Fear & Greed sentiment
- `price_onchain` (40 dims) — + Short-term on-chain (SOPR/CDD)
- `price_long_onchain` (48 dims) — + Long-term on-chain
- `full` (64 dims) — All sources fused

## Project Structure

```
.
├── src/
│   ├── models/              # 10 model implementations (PyTorch + XGBoost)
│   ├── data_download/       # Data fetching from Binance, Dune Analytics, Alternative.me
│   ├── data_preprocess/     # Data cleaning, forward-fill, merge
│   ├── features_construct/  # Feature engineering pipeline
│   ├── chart_generate/      # Backtest, ablation, sliding-window charts
│   ├── evaluate_all.py      # Unified evaluation runner
│   ├── metrics.py           # Trading strategy metrics
│   ├── cost_sensitivity.py  # Transaction cost analysis
│   ├── count_params.py      # Model parameter counter
│   └── common.py            # Shared utilities
├── configs/
│   └── config.py            # Global configuration & hyperparameters
├── paper_english/           # LaTeX source (Elsevier cas-dc template)
├── paper/                   # Chinese manuscript
├── charts/                  # Generated figures (PDF + PNG)
├── results/                 # Result tables (LaTeX + CSV)
└── data/                    # Raw and processed data (excluded from repo)
```

## Environment Setup

### Requirements
- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU training)

### Installation
```bash
pip install torch numpy pandas scikit-learn xgboost matplotlib seaborn
```

### Data
Raw data is excluded from this repository due to size. To reproduce:

1. **Price & Volume**: BTC/USDT perpetual 1h candlesticks from [Binance](https://www.binance.com)
2. **Funding Rate**: Binance perpetual contract funding rate
3. **Fear & Greed Index**: [Alternative.me](https://alternative.me/crypto/fear-and-greed-index/)
4. **On-chain data**: SOPR, CDD, exchange balances, active addresses, netflow from [Dune Analytics](https://dune.com) and [OKLink](https://www.oklink.com)

Data fetching scripts are in `src/data_download/`. After downloading, run:

```bash
cd src/data_preprocess
python data_ffill.py          # Forward-fill missing values
python merge_ablation_datasets.py  # Merge into 6 feature configurations
```

Then generate features:

```bash
cd src/features_construct
python build_features.py       # Build features for all configurations
```

## Usage

### Quick Start: Evaluate All Models

```bash
cd src

# Single feature set, single seed
python evaluate_all.py --feature full --seed 1

# All feature sets, all seeds (parallel execution recommended)
python evaluate_all.py --feature all --seed all
```

### Individual Model Training

```bash
cd src
# Train CNN-Transformer on full feature set
python models/cnn_transformer.py
```

Model configurations (feature set, seed, hyperparameters) are controlled via `configs/config.py`.

### Generate Charts

```bash
cd src/chart_generate
python model_comparison.py      # Cross-model comparison charts
python feature_ablation.py      # Feature ablation charts
python backtest_full.py         # Full backtest curves
python backtest_synergy.py      # On-chain synergy visualization
python sliding_window_charts.py # Sliding window metrics
python cost_sensitivity.py      # Cost sensitivity analysis
```

## Key Results

| Model | IC | Sharpe | MaxDD | Ann. Return |
|-------|-----|--------|-------|-------------|
| CNN-Transformer | 0.150 | **5.19** | 17.4% | 211.5% |
| XGBoost | 0.127 | 4.19 | 20.4% | 171.3% |
| Transformer | 0.129 | 3.53 | 19.8% | 144.6% |
| TimeMixer | 0.103 | 3.48 | 21.8% | 142.4% |
| LSTM-Transformer | 0.125 | 3.02 | 25.3% | 123.4% |
| DLinear | 0.074 | 0.37 | 37.8% | 15.1% |

*Cross-5-seed mean on full feature set. All metrics on test set (2025-06 to 2026-05).*

## Citation

```bibtex
@article{he2025multi,
  title={Multi-Source Heterogeneous Data Fusion for Cryptocurrency Return
         Forecasting: On the Conditional Effectiveness of On-Chain Indicators},
  author={He, Minghan},
  journal={Big Data Research},
  year={2025},
  note={Under review}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

Supported by the Humanities and Social Sciences Project of the Ministry of Education of China (Grant No. 19YJCZH031) and the Shanghai Municipal Education Science Research Project (Grant No. C2023068).
