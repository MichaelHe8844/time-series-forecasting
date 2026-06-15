import requests
import pandas as pd
import time
from datetime import datetime, timezone
from pathlib import Path

SYMBOL = "BTCUSDT"
BASE_URL = "https://fapi.binance.com/fapi/v1/fundingRate"

# 修改保存路径：基于脚本位置，向上三级到达项目根目录，再进入 data/raw
SAVE_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "btc_funding_rate_1h.csv"
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_custom_date_timestamp():
    """获取 2020-01-01 到 2026-05-07 的时间戳 (UTC)"""
    start = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 7, 23, 59, 59, tzinfo=timezone.utc)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def fetch_funding_rate():
    """循环获取 binance funding rate，包含重试和防断连机制"""
    start_ts, end_ts = get_custom_date_timestamp()
    all_data = []
    current = start_ts

    session = requests.Session()

    while True:
        params = {
            "symbol": SYMBOL,
            "startTime": current,
            "limit": 1000
        }

        max_retries = 5
        data = None

        for attempt in range(max_retries):
            try:
                r = session.get(BASE_URL, params=params, timeout=10)
                r.raise_for_status()
                data = r.json()
                break

            except requests.exceptions.RequestException as e:
                wait_time = 2 ** attempt
                print(f"\n[网络波动] 抓取失败，原因: {e}")
                print(f"等待 {wait_time} 秒后进行第 {attempt + 1}/{max_retries} 次重试...")
                time.sleep(wait_time)
        else:
            print(f"\n[严重错误] 连续 {max_retries} 次请求失败。保存当前已抓取的数据并退出。")
            return all_data

        if not data:
            break

        # 过滤掉超过结束时间的数据
        valid_data = [d for d in data if d["fundingTime"] <= end_ts]
        all_data.extend(valid_data)

        last_time = data[-1]["fundingTime"]
        current = last_time + 1

        if last_time >= end_ts:
            break

        print(f"已抓取至: {pd.to_datetime(last_time, unit='ms')}", end='\r')

        time.sleep(0.5)

    print("\n抓取完成！")
    return all_data


def process_data(data):
    df = pd.DataFrame(data)
    df["timestamp"] = df["fundingTime"]
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["fundingRate"] = df["fundingRate"].astype(float)
    df = df[["timestamp", "datetime", "fundingRate"]]
    df = df.sort_values("datetime")
    df = df.set_index("datetime")

    # 重采样到小时级
    hourly_df = df.resample("1h").interpolate(method="linear")

    # timestamp 改为微秒级（16 位数）
    hourly_df["timestamp"] = hourly_df.index.astype("int64") // 10 ** 3
    hourly_df = hourly_df.reset_index()

    hourly_df = hourly_df[["datetime", "timestamp", "fundingRate"]]
    return hourly_df


def main():
    print("Downloading funding rate from 2020-01-01 to 2026-05-07...")
    raw_data = fetch_funding_rate()
    print("Total records fetched:", len(raw_data))

    if not raw_data:
        print("No data found for the specified period.")
        return

    df = process_data(raw_data)

    df.to_csv(
        SAVE_PATH,
        index=False,
        header=True,
        float_format='%.15e',
        lineterminator='\n'
    )

    print("Saved to:", SAVE_PATH)


if __name__ == "__main__":
    main()