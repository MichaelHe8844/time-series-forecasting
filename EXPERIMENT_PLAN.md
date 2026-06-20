# 补充实验执行方案

> 投《计算机科学》前的实验补充 —— 基于审稿人视角识别的 6 个实验缺口

---

## 执行总览

| 缺口 | 说明 | 新增训练 | GPU 时间 | 优先级 |
|------|------|---------|---------|--------|
| 1 | 特征消融多种子分析 | **0** (数据已存在) | 0 | 🔴 必须 |
| 2 | 滑动窗口 fng 集验证 | 16 次 | ~1-2h | 🔴 必须 |
| 3 | 损失函数消融 | 18 次 | ~1-2h | 🟡 强烈建议 |
| 4 | 架构组件消融 | 9 次 | ~0.5h | 🟡 强烈建议 |
| 5 | 超参数敏感性 | 16 次 | ~1h | 🟡 强烈建议 |
| 6 | XGBoost 失效分析 | **0** (分析) | 0 | 🟢 锦上添花 |

---

## 缺口 1: 特征消融多种子分析

### 现状
全部 10 模型 × 6 特征 × 5 种子已训练完毕 (checkpoint/ 目录中)。

### 执行
```bash
cd E:\Mycodes\Time_series_forecasting

# 生成跨种子特征消融表
python src/analysis/multi_seed_ablation.py --output all
```

### 输出文件
- `results/feature_ablation_multi_seed.csv` — 跨种子 Sharpe 均值±标准差
- `results/delta_sharpe_multi_seed.csv` — ΔSharpe 跨种子统计
- `results/feature_ablation_multi_seed.tex` — LaTeX 表格 (替换论文表3)
- `results/delta_sharpe_multi_seed.tex` — LaTeX 表格 (替换论文表5)

### 预期发现
- CNN-Transformer 的 ΔSharpe 在 5/5 种子均为正值 ✓ (三组分类验证)
- TimeMixer 的 ΔSharpe 仅 1/5 种子为正值 ✗ (负向干扰组稳定)
- 直接用此表替换论文中的 seed=1 单种子表

---

## 缺口 2: 滑动窗口 price_funding_fng 验证

### 现状
滑动窗口验证仅在 full 集上完成。论文的核心论点"fng→full 的 ΔSharpe"未在滑动窗口下验证。

### 执行
```bash
cd E:\Mycodes\Time_series_forecasting

# CNN-Transformer on price_funding_fng, 8 windows, seed=1
python src/analysis/run_sw_training.py \
    --model cnn_transformer \
    --feature price_funding_fng \
    --seed 1

# TimeMixer on price_funding_fng
python src/analysis/run_sw_training.py \
    --model timemixer \
    --feature price_funding_fng \
    --seed 1
```

### 输出
- `checkpoint/cnn_transformer_price_funding_fng_swW{i}_seed1/results.txt` (8个)
- `checkpoint/timemixer_price_funding_fng_swW{i}_seed1/results.txt` (8个)

### 分析
训练完成后，对比已有 `results/sliding_window.tex` (full集) 与新生成的 fng 集结果，计算每个窗口的 ΔSharpe = Sharpe(full) - Sharpe(fng)，验证跨窗口符号一致性。

### 预期发现
- CNN-Transformer: 8/8 窗口 ΔSharpe > 0 (链上数据在滑动窗口下持续贡献正增益)
- TimeMixer: 大部分窗口 ΔSharpe < 0 (跨尺度时间混合无法替代跨特征源建模)

---

## 缺口 3: 损失函数消融

### 执行
```bash
cd E:\Mycodes\Time_series_forecasting

# CNN-Transformer 和 TimeMixer 在 full 集上的损失函数消融
python src/analysis/run_ablation_experiments.py \
    --exp loss \
    --model all \
    --feature full

# 汇总结果
python src/analysis/aggregate_ablation_results.py --exp loss
```

### 消融变体
| 变体 | λ_rank | 损失 |
|------|--------|------|
| 纯 Huber | 0.0 | Huber(δ=0.3) |
| 纯 MSE | 0.0 | MSE |
| Huber+Ranking (当前) | 0.13 | Huber + MarginRanking |

### 输出
- `results/loss_ablation.tex` — LaTeX 表格

### 预期发现
- Huber+Ranking 优于纯 Huber 优于纯 MSE
- 三组分类规律在纯 Huber 下是否仍然成立（方法稳健性）

---

## 缺口 4: 架构组件消融

### 执行
```bash
cd E:\Mycodes\Time_series_forecasting

# CNN-Transformer 架构消融 (full 集)
python src/analysis/run_ablation_experiments.py \
    --exp arch \
    --feature full

# 汇总结果
python src/analysis/aggregate_ablation_results.py --exp arch
```

### 消融变体
| 变体 | 说明 |
|------|------|
| CNN-Transformer (完整) | 3层因果卷积 → Transformer (2层, 12头) |
| CNN-only | 去 Transformer，CNN 后直接 Pooling |
| Transformer-only | 去 CNN，原始输入直接进 Transformer |
| Transformer-only (参数量匹配) | 加深加宽至 ~1.75M 参数 |

### 输出
- `results/arch_ablation.tex` — LaTeX 表格

### 预期发现
- CNN-only 和 Transformer-only 的 Sharpe 均显著低于完整 CNN-Transformer
- 参数量匹配的 Transformer-only 仍不如混合架构 → "混合"本身是增益源，不是参数量带来的

---

## 缺口 5: 超参数敏感性

### 执行
```bash
cd E:\Mycodes\Time_series_forecasting

# CNN-Transformer / TimeMixer / Transformer 超参数敏感性
python src/analysis/run_ablation_experiments.py \
    --exp hparam \
    --model all \
    --feature full

# 汇总结果
python src/analysis/aggregate_ablation_results.py --exp hparam
```

### 敏感性网格
| 超参数 | 候选值 |
|--------|--------|
| Learning Rate | 5e-5, 1e-4, 2e-4 |
| Dropout | 0.10, 0.13, 0.20 |

### 输出
控制台输出各配置的 Sharpe 对比。

### 预期发现
- CNN-Transformer 在不同 LR/Dropout 下保持最高 Sharpe (稳健)
- TimeMixer 最优参数下的 Sharpe 仍无法超越 baseline CNN-Transformer
- 三组分化规律对超参数不敏感

---

## 缺口 6: XGBoost 失效分析 (纯分析，零训练)

### 问题
论文表3中 XGBoost 在 price_only/price_funding/price_onchain 等特征集上标注为 "--" (IC=0)，需要详细解释。

### 分析步骤
1. 检查 XGBoost 在这些特征集上的 training IC vs test IC
2. 如果 training IC 正常但 test IC≈0 → 讨论过拟合
3. 如果 training IC 也≈0 → 讨论树模型在时序扁平化后的结构局限

### 论文中建议增加的讨论
- XGBoost 将 L×F 矩阵展平为 LF 维向量，丢失全部时序结构
- 在单一数据源下，扁平化后特征维度低 (~18-48维)，XGBoost 贪心分裂策略在高噪声环境中选到噪声
- 在 price_funding_fng 和 full 上有效，是因为高维特征空间 (~40-64维) 中存在可被贪心搜索发现的强信号

---

## 一键运行全部实验

```bash
cd E:\Mycodes\Time_series_forecasting

# Step 1: 缺口 1 — 纯分析，无需训练
python src/analysis/multi_seed_ablation.py --output all

# Step 2: 缺口 2 — 滑动窗口 fng (约 1-2 小时)
python src/analysis/run_sw_training.py --model cnn_transformer --feature price_funding_fng --seed 1
python src/analysis/run_sw_training.py --model timemixer --feature price_funding_fng --seed 1

# Step 3: 缺口 3 — 损失函数消融 (约 1-2 小时)
python src/analysis/run_ablation_experiments.py --exp loss --model all --feature full
python src/analysis/aggregate_ablation_results.py --exp loss

# Step 4: 缺口 4 — 架构消融 (约 0.5 小时)
python src/analysis/run_ablation_experiments.py --exp arch --feature full
python src/analysis/aggregate_ablation_results.py --exp arch

# Step 5: 缺口 5 — 超参数敏感性 (约 1 小时)
python src/analysis/run_ablation_experiments.py --exp hparam --model all --feature full
python src/analysis/aggregate_ablation_results.py --exp hparam
```

---

## 预计总 GPU 时间

| 阶段 | 时间 |
|------|------|
| 缺口 1 (分析) | 即时 |
| 缺口 2 (训练) | ~1-2 小时 |
| 缺口 3 (训练) | ~1-2 小时 |
| 缺口 4 (训练) | ~0.5 小时 |
| 缺口 5 (训练) | ~1 小时 |
| **合计** | **~4-6 小时** |

---

## 文件清单

```
src/analysis/
├── multi_seed_ablation.py          # 缺口 1: 跨种子特征消融分析
├── run_sw_training.py              # 缺口 2: 滑动窗口训练编排
├── _sw_train_one.py                # 缺口 2: 单窗口训练 worker
├── run_ablation_experiments.py     # 缺口 3-5: 消融实验统一运行
└── aggregate_ablation_results.py   # 缺口 3-5: 消融结果汇总
```
