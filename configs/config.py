cfg = {
    "seed": 0,  # 全局随机种子，用于实验完全复现
    "lookback": 48,  # 回溯窗口长度 L
    "PERIODS_PER_YEAR": 2190,
    "pred_clip_min": -0.005,  # 预测值 clip 下限
    "pred_clip_max": 0.005,   # 预测值 clip 上限

    "data": {
        "root": "../features",
        "feature_type": "full",  # 切换实验只改这里
        # 可选：
        # "price_only"
        # "price_funding"
        # "price_funding_fng"
        # "price_long_onchain"
        # "price_onchain"
        # "full"
    },

    "train": {
        "batch_size": 64,
        "num_workers": 0,
        "pin_memory": False,

        "epochs": 200,
        "weight_decay": 1e-4,
        "patience": 20,
        # 注意：lr 已移到每个模型配置中，不再放在全局 train
    },

    "tcn": {
        "num_layers": 2,
        "kernel_size": 3,
        "hidden_dim": 96, #128严重过拟合
        "dropout": 0.13,
        "lr": 1e-4,
        "grad_clip": 1.0,
        "ranking_loss_weight": 0.13,
        "ranking_margin": 0.0005,
    },

    "lstm": {
        "hidden_dim": 192,
        "num_layers": 2,
        "dropout": 0.13, # 0.1存在较严重过拟合
        "bidirectional": False,
        "lr": 1e-4,
        "grad_clip": 1.0,
        "ranking_loss_weight": 0.13,  # 与 CNN-Transformer 保持一致
        "ranking_margin": 0.0005,
    },

    "transformer": {
        "d_model": 192,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 192,
        "dropout": 0.13,
        "lr": 1e-4,
        "grad_clip": 1.0,
        "ranking_loss_weight": 0.13,
        "ranking_margin": 0.0005,
    },

    "cnn_transformer": {
        "conv_channels": 192,
        "kernel_size": 5,
        "num_conv_layers": 3,
        "d_model": 192,
        "nhead": 12,
        "num_layers": 2,
        "dim_feedforward": 512,
        "dropout": 0.13,
        "max_len": 512,
        "use_residual": True,
        "pool": "last",
        "lr": 1e-4,
        "grad_clip": 1.0,
        "ranking_loss_weight": 0.13,
        "ranking_margin": 0.0005,
    },

    "lstm_transformer": {
        "hidden_dim": 192,
        "num_lstm_layers": 2,
        "lstm_dropout": 0.1,
        "bidirectional": False,
        "d_model": 192,
        "nhead": 4,
        "dim_feedforward": 512,
        "num_layers": 2,
        "dropout": 0.13,
        "lr": 1e-4,
        "max_len": 512,
        "grad_clip": 1.0,
        "ranking_loss_weight": 0.13,
        "ranking_margin": 0.0005,
    },

    "xgboost": {
        "learning_rate": 0.05,
        "max_depth":5,
        "subsample": 0.6,
        "colsample_bytree": 0.2,
        "min_child_weight": 10,
        "gamma": 0.02,
        "reg_alpha": 0.12,
        "reg_lambda": 1.0,
        "num_boost_round": 2000,
        "early_stopping_rounds": 100,
        "verbosity": 1
    },

    "patchtst": {
        "patch_len": 8,
        "stride": 2,  # overlap = patch_len - stride
        "d_model": 192,
        "nhead": 8,
        "num_layers": 2,
        "dim_feedforward": 384,
        "dropout": 0.13,
        "max_len": 512,
        "lr": 1e-4,
        "grad_clip": 1.0,
        "ranking_loss_weight": 0.13,  # 和 CNN-Transformer 一致
        "ranking_margin": 0.0,
    },

    "modern_tcn": {
        "hidden_dim": 96, #同TCN，128严重过拟合
        "num_layers": 3,  # ModernTCN 推荐 3~5 层
        "kernel_size": 31,  # 大核是 ModernTCN 的核心优势
        "dropout": 0.13,
        "lr": 1e-4,
        "grad_clip": 1.0,
        "ranking_loss_weight": 0.13,
        "ranking_margin": 0.0005,
    },

    "dlinear": {
        "moving_avg_kernel": 13,  # 约为 lookback=48 的 1/4
        "individual": True,       # 通道独立（原论文默认）
        "lr": 1e-4,
        "grad_clip": 1.0,
        "ranking_loss_weight": 0.13,
        "ranking_margin": 0.0005,
    },

    "timemixer": {
        "d_model": 128,
        "num_scales": 4,          # 多尺度层级数
        "num_blocks": 2,          # PDM 块数
        "decomp_kernel": 5,       # 序列分解的移动平均核大小
        "dropout": 0.13,
        "lr": 1e-4,
        "grad_clip": 1.0,
        "ranking_loss_weight": 0.13,
        "ranking_margin": 0.0005,
    },
}