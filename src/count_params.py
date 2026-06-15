"""
Count trainable parameters for all 8 models.
"""
import sys
sys.path.insert(0, ".")

from configs.config import cfg
from src.models.baseline_lstm import BiLSTM
from src.models.baseline_transformer import Transformer
from src.models.baseline_tcn import TCN
from src.models.cnn_transformer import CNNTransformer
from src.models.lstm_transformer import LSTMTransformer
from src.models.patchtst import PatchTST
from src.models.modern_tcn import ModernTCN
from src.models.dlinear import DLinear
from src.models.timemixer import TimeMixer

FEATURE_DIMS = {
    "price_only": 18,
    "price_funding": 28,
    "price_funding_fng": 40,
    "price_onchain": 40,
    "price_long_onchain": 48,
    "full": 64,
}


def count_params(model, name, input_dim):
    m = model(input_dim, cfg[name])
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


def main():
    models = [
        ("lstm", BiLSTM),
        ("transformer", Transformer),
        ("tcn", TCN),
        ("cnn_transformer", CNNTransformer),
        ("lstm_transformer", LSTMTransformer),
        ("patchtst", PatchTST),
        ("modern_tcn", ModernTCN),
        ("dlinear", DLinear),
        ("timemixer", TimeMixer),
    ]

    print(f"{'Model':<22s} ", end="")
    for ft in FEATURE_DIMS:
        print(f"{ft:>12s}", end=" ")
    print()
    print("-" * (23 + 13 * len(FEATURE_DIMS)))

    for model_name, model_cls in models:
        print(f"{model_name:<22s} ", end="")
        for ft, dim in FEATURE_DIMS.items():
            total, _ = count_params(model_cls, model_name, dim)
            if total >= 1_000_000:
                print(f"{total/1_000_000:>9.2f}M", end=" ")
            elif total >= 1_000:
                print(f"{total/1_000:>9.1f}K", end=" ")
            else:
                print(f"{total:>10d}", end="  ")
        print()

    print()
    print("XGBoost: tree-based model, 2000 trees, max_depth=5 (no differentiable parameters)")


if __name__ == "__main__":
    main()
