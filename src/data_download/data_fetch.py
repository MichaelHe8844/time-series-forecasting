import ccxt
import pandas as pd
import os
import time
from datetime import datetime, timezone

def fetch_binance_data_by_range(symbol='BTC/USDT', timeframe='1h',
                                start_str='2020-01-01', end_str=None):
    """
    抓取指定日期范围内的 Binance 数据并保存至 ../data/raw
    """
    # 如果未指定结束日期，则自动使用当前 UTC 日期
    if end_str is None:
        end_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    # 1. 初始化交易所配置
    exchange = ccxt.binance({
        'timeout': 30000,
        'enableRateLimit': True,
        'proxies': {
            'http': 'http://127.0.0.1:7888',
            'https': 'http://127.0.0.1:7888',
        },
    })

    # 2. 转换日期为毫秒时间戳
    since = exchange.parse8601(f"{start_str}T00:00:00Z")
    end_timestamp = exchange.parse8601(f"{end_str}T23:59:59Z")

    all_ohlcv = []

    print(f"开始抓取 {symbol} 从 {start_str} 到 {end_str} 的数据...")

    # 3. 循环分页获取数据
    while since < end_timestamp:
        try:
            # fetch_ohlcv 默认每次返回 500-1000 条
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since)

            if not ohlcv:
                break

            # 更新下一次获取的时间点（最后一条数据的时间戳 + 1ms）
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 1
            all_ohlcv.extend(ohlcv)

            # 打印进度
            current_time_str = datetime.fromtimestamp(last_timestamp / 1000).strftime('%Y-%m-%d %H:%M')
            print(f"已同步至: {current_time_str} | 当前已累计: {len(all_ohlcv)} 条")

            # 稍微停顿，尊重频率限制
            time.sleep(exchange.rateLimit / 1000)

            # 终止条件：如果最后抓到的时间戳已超过设定的结束时间
            if last_timestamp >= end_timestamp:
                break

        except Exception as e:
            print(f"抓取过程中出错: {e}")
            break

    if not all_ohlcv:
        print("未获取到任何数据，请检查网络、代理设置或交易对符号。")
        return

    # 4. 转换为 DataFrame
    # 包含原始毫秒时间戳 (timestamp)
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # 添加可读的日期时间列 (datetime)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

    # 5. 过滤与重排
    # 确保只保留在指定范围内的数据
    df = df[df['timestamp'] <= end_timestamp]
    # 调整列顺序，将时间信息放在最前面，方便 CNN/Transformer 索引
    df = df[['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume']]

    # 6. 路径处理（适配当前脚本位于 src/data_download/ 下）
    # 获取当前脚本的绝对路径，向上三级到项目根目录
    current_script_path = os.path.abspath(__file__)
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
    target_dir = os.path.join(root_dir, 'data', 'raw')

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    file_name = f"{symbol.replace('/', '_')}_{timeframe}.csv"
    save_path = os.path.join(target_dir, file_name)

    # 7. 保存数据
    df.to_csv(save_path, index=False)
    print("\n任务完成")
    print(f"文件保存路径: {save_path}")
    print(f"有效数据总量: {len(df)} 行")


if __name__ == "__main__":
    fetch_binance_data_by_range(
        symbol='BTC/USDT',
        timeframe='1h',
        start_str='2020-01-01',
        end_str='2026-05-07'
    )