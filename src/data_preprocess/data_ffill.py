import pandas as pd
from pathlib import Path


def preprocess_data():
    # 项目根目录下 data/ 与 src/ 同级，代码位于 src/data_preprocess/
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    raw_dir = project_root / 'data' / 'raw'
    processed_dir = project_root / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)

    file_configs = {
        'BTC_USDT_1h.csv': {'original_time_col': 'datetime', 'rename_to': None},
        'btc_funding_rate_1h.csv': {'original_time_col': 'datetime', 'rename_to': None},
        'fear_greed_index_1h.csv': {'original_time_col': 'Unnamed: 0', 'rename_to': 'datetime'},
        'exchange_netflow_1h.csv': {'original_time_col': 'hour', 'rename_to': 'datetime'},
        'total_btc_on_exchange_1h.csv': {'original_time_col': 'hour', 'rename_to': 'datetime'},
        'active_addresses_1h.csv': {'original_time_col': 'hour', 'rename_to': 'datetime'},
        'cdd_1h.csv': {'original_time_col': 'datetime', 'rename_to': None},
        'sopr_1h.csv': {'original_time_col': 'datetime', 'rename_to': None},
    }

    for filename, config in file_configs.items():
        input_file = raw_dir / filename
        if not input_file.exists():
            print(f"Skip: {filename} not found")
            continue

        # 读取文件
        df = pd.read_csv(input_file)

        # 去除列名两端空格（防止 CSV 导出时出现 'hour ' 等情况）
        df.columns = df.columns.str.strip()

        # 若需要，重命名时间列为标准 'datetime'
        if config['rename_to']:
            df = df.rename(columns={config['original_time_col']: config['rename_to']})

        time_col = 'datetime'

        # 若缺少时间列则跳过
        if time_col not in df.columns:
            print(f"Skip: {filename} missing '{time_col}' column")
            continue

        # 解析为 datetime
        # 对带 UTC 的时间格式同样兼容，然后去掉时区信息，方便后续统一合并
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce', utc=True).dt.tz_localize(None)
        df = df.dropna(subset=[time_col])

        # 设置索引并排序
        df = df.set_index(time_col).sort_index()

        # 移除重复索引
        df = df[~df.index.duplicated(keep='first')]

        # 前向填充缺失值
        df = df.ffill()

        # 保存清洗后的文件
        output_file = processed_dir / f'cleaned_{filename}'
        df.to_csv(output_file)
        print(f"Saved: {output_file}")


if __name__ == "__main__":
    preprocess_data()
