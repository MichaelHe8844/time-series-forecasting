import os
import logging
import pandas as pd
from dune_client.client import DuneClient
from dune_client.query import QueryBase

# ==================== 关闭 Dune 日志 ====================
logging.getLogger("dune_client").setLevel(logging.WARNING)

# ==================== 配置区 ====================
QUERY_ID = 6924721

# 基础路径（以当前脚本为基准）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# key 文件路径
KEY_PATH = os.path.join(BASE_DIR, "../../configs/key")

# 输出路径（和 netflow 保持一致结构）
OUTPUT_DIR = os.path.join(BASE_DIR, "../../data/raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FINAL_FILE = os.path.join(OUTPUT_DIR, "active_addresses_1h.csv")

# ==================== 读取 API Key ====================
if not os.path.exists(KEY_PATH):
    raise FileNotFoundError(f"未找到 key 文件: {KEY_PATH}")

with open(KEY_PATH, "r") as f:
    DUNE_API_KEY = f.read().strip()

if not DUNE_API_KEY:
    raise ValueError("key.txt 为空，请填入你的 Dune API Key")

# ==================== 执行 ====================
dune = DuneClient(api_key=DUNE_API_KEY)

print(f"正在获取 Active Addresses 最新结果（Query ID: {QUERY_ID}） ...")

# 改用 get_latest_result 获取你在 Dune UI 上已经跑完的最新结果
result = dune.get_latest_result(QUERY_ID)

if result is None:
    raise RuntimeError("未找到该 Query 的最新执行结果，请先在 Dune 页面手动执行一次。")

df = pd.DataFrame(result.result.rows)
print(f"本次获取到 {len(df)} 行数据")
# ==================== 合并 & 去重 ====================
if os.path.exists(FINAL_FILE):
    existing_df = pd.read_csv(FINAL_FILE)

    df = pd.concat([existing_df, df], ignore_index=True)
    df = (
        df.drop_duplicates(subset=["hour"])
          .sort_values("hour")
          .reset_index(drop=True)
    )

    print(f"追加后总行数: {len(df)}")
else:
    print("首次保存")

# ==================== 保存 ====================
df.to_csv(FINAL_FILE, index=False)

print(f"\n✅ 数据已保存到: {FINAL_FILE}")
print(f"当前总行数: {len(df)}")