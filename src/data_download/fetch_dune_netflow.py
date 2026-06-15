import os
import logging
import pandas as pd
from dune_client.client import DuneClient
from dune_client.query import QueryBase

# ==================== 关闭 Dune 日志 ====================
logging.getLogger("dune_client").setLevel(logging.WARNING)

# ==================== 配置区 ====================
QUERY_ID = 7442642

# key 文件路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KEY_PATH = os.path.join(BASE_DIR, "../../configs/key")

# 输出路径
OUTPUT_DIR = os.path.join(BASE_DIR, "../../data/raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)
FINAL_FILE = os.path.join(OUTPUT_DIR, "exchange_netflow_1h.csv")

# ==================== 读取 API Key ====================
if not os.path.exists(KEY_PATH):
    raise FileNotFoundError(f"未找到 key 文件: {KEY_PATH}")

with open(KEY_PATH, "r") as f:
    DUNE_API_KEY = f.read().strip()

if not DUNE_API_KEY:
    raise ValueError("key 文件为空，请填入你的 Dune API Key")

# ==================== 执行 ====================
dune = DuneClient(api_key=DUNE_API_KEY)

print(f"🚀 正在执行 Dune Query ID: {QUERY_ID} ...")
print("（全量 2020-01-01 到 2026-05-07 小时级 BTC exchange netflow）")

query = QueryBase(query_id=QUERY_ID)
result = dune.run_query(query)

df = pd.DataFrame(result.result.rows)
print(f"✅ 本次获取到 {len(df):,} 行数据")

# ==================== 数据清洗 ====================
# 关键：把 hour 转为 datetime 类型（强烈推荐）
if 'hour' in df.columns:
    df['hour'] = pd.to_datetime(df['hour'])

# 列顺序调整（更直观）
df = df[['hour', 'inflow_btc', 'outflow_btc', 'netflow_btc']]

# ==================== 合并 & 去重 ====================
if os.path.exists(FINAL_FILE):
    existing_df = pd.read_csv(FINAL_FILE)
    # 同样转为 datetime 防止类型不一致
    if 'hour' in existing_df.columns:
        existing_df['hour'] = pd.to_datetime(existing_df['hour'])

    df = pd.concat([existing_df, df], ignore_index=True)
    df = (
        df.drop_duplicates(subset=["hour"])
          .sort_values("hour")           # 正序（2020 → 2026）
          .reset_index(drop=True)
    )
    print(f"📌 追加去重后总行数: {len(df):,}")
else:
    print("📌 首次保存")

# ==================== 保存 ====================
df.to_csv(FINAL_FILE, index=False)

print(f"\n🎉 数据已保存到: {FINAL_FILE}")
print(f"   当前总行数: {len(df):,} 行")
print(f"   时间范围: {df['hour'].min()} → {df['hour'].max()}")