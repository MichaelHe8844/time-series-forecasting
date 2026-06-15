# ============================================================================
# 文件名：fetch_fear_greed_index.py
# 功能：获取指定时间段的恐惧与贪婪指数数据，插值至小时级别，并保留分类列
# 数据源：Alternative.me API
# ============================================================================

import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
import time


# ============================================================================
# 配置部分
# ============================================================================

class Config:
    """项目配置"""
    API_URL = "https://api.alternative.me/fng/"
    API_TIMEOUT = 10
    MAX_RETRIES = 3
    RETRY_DELAY = 2

    # 适配脚本新位置 src/data_download/，向上三级到项目根目录
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
    OUTPUT_FILE_HOURLY = "fear_greed_index_1h.csv"   # 小时数据文件名（不再保存日度原始数据）

    DEFAULT_START_DATE = "2020-01-01"
    DEFAULT_END_DATE = "2026-05-07"


# ============================================================================
# 数据获取类
# ============================================================================

class FearGreedIndexFetcher:
    """恐惧与贪婪指数数据获取器"""

    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f" 数据保存路径：{self.config.OUTPUT_DIR}")

    def fetch_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取指定时间段的恐惧与贪婪指数数据"""
        start_date = start_date or self.config.DEFAULT_START_DATE
        end_date = end_date or self.config.DEFAULT_END_DATE

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        days_to_fetch = (end_dt - start_dt).days + 30

        print(f"   开始获取恐惧与贪婪指数数据")
        print(f"   时间范围：{start_date} ~ {end_date}")

        all_data = []

        for attempt in range(self.config.MAX_RETRIES):
            try:
                response = self.session.get(
                    f"{self.config.API_URL}?limit={days_to_fetch}",
                    timeout=self.config.API_TIMEOUT
                )
                response.raise_for_status()
                data = response.json()

                if isinstance(data, dict) and 'data' in data:
                    all_data = data['data']
                elif isinstance(data, list):
                    all_data = data
                else:
                    raise Exception(f"未知的 API 响应格式：{type(data)}")

                if not all_data:
                    raise Exception("API 返回空数据")

                print(f"   成功获取 {len(all_data)} 条记录")
                break

            except requests.exceptions.RequestException as e:
                print(f"   请求失败 (尝试 {attempt + 1}/{self.config.MAX_RETRIES}): {e}")
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(self.config.RETRY_DELAY * (attempt + 1))
                else:
                    raise Exception("所有重试均失败")

        df = pd.DataFrame(all_data)
        df = self._process_dataframe(df, start_date, end_date)
        return df

    def _process_dataframe(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """处理 DataFrame，保留数值列和分类列"""
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['fng_value'] = pd.to_numeric(df['value'])

        if 'value_classification' in df.columns:
            df['fng_classification'] = df['value_classification']

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)]

        df = df.sort_values('datetime').reset_index(drop=True)
        df = df.set_index('datetime')

        # 保留 fng_value 和可能的 fng_classification 列
        keep_cols = ['fng_value']
        if 'fng_classification' in df.columns:
            keep_cols.append('fng_classification')

        return df[keep_cols]

    def interpolate_to_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将日度恐惧贪婪指数插值至小时级别：
          - fng_value 采用线性插值（一阶差分均匀分配到小时）
          - fng_classification 采用向前填充（当天分类保持不变）
        返回包含完整小时索引的 DataFrame。
        """
        # 确保索引为 DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame 索引必须是 DatetimeIndex")

        # 生成从最小日期到最大日期的完整小时索引
        hourly_index = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq='h'
        )

        # 重采样至小时（此时所有列为 NaN）
        df_hourly = df.reindex(hourly_index)

        # 对 fng_value 进行线性插值
        df_hourly['fng_value'] = df_hourly['fng_value'].interpolate(method='linear')

        # 对 fng_classification 向前填充（如果存在）
        if 'fng_classification' in df_hourly.columns:
            df_hourly['fng_classification'] = df_hourly['fng_classification'].ffill()

        print(f"   已将日度数据插值为小时数据，共 {len(df_hourly)} 条记录")
        return df_hourly

    def save_to_csv(self, df: pd.DataFrame, filename: str) -> Path:
        """保存数据到 CSV 文件"""
        filepath = self.config.OUTPUT_DIR / filename
        df.to_csv(filepath, index=True)
        print(f"  数据已保存至：{filepath}")
        return filepath


# ============================================================================
# 主函数
# ============================================================================

def main(start_date: str = None, end_date: str = None):
    """主执行函数：获取日度数据 -> 插值小时数据 -> 保存小时数据（不保留日度）"""
    print("=" * 80)
    print("   加密货币恐惧与贪婪指数数据获取工具（小时插值版）")
    print("=" * 80)

    config = Config()
    fetcher = FearGreedIndexFetcher(config)

    try:
        # 1. 获取日度数据
        df_daily = fetcher.fetch_data(start_date=start_date, end_date=end_date)

        # 2. 插值为小时数据（包含数值列和分类列）
        df_hourly = fetcher.interpolate_to_hourly(df_daily)

        # 3. 仅保存小时数据（日度数据不保留）
        fetcher.save_to_csv(df_hourly, config.OUTPUT_FILE_HOURLY)

        print("\n" + "=" * 80)
        print("   小时数据已生成并保存完成！（原始日度数据未保留）")
        print("=" * 80)

        return df_hourly

    except Exception as e:
        print(f"\n   错误：{e}")
        return None


if __name__ == "__main__":
    main(start_date="2020-01-01", end_date="2026-05-07")