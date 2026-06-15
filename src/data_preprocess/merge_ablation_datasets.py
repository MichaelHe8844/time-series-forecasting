import pandas as pd
from pathlib import Path

def merge_ablation_datasets():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    processed_dir = project_root / 'data' / 'processed'
    merged_dir = processed_dir / 'merged'
    merged_dir.mkdir(parents=True, exist_ok=True)

    # 加载所有清洗后的文件
    price = pd.read_csv(processed_dir / 'cleaned_BTC_USDT_1h.csv', index_col=0, parse_dates=True)
    funding = pd.read_csv(processed_dir / 'cleaned_btc_funding_rate_1h.csv', index_col=0, parse_dates=True)
    fng = pd.read_csv(processed_dir / 'cleaned_fear_greed_index_1h.csv', index_col=0, parse_dates=True)
    netflow = pd.read_csv(processed_dir / 'cleaned_exchange_netflow_1h.csv', index_col=0, parse_dates=True)
    total = pd.read_csv(processed_dir / 'cleaned_total_btc_on_exchange_1h.csv', index_col=0, parse_dates=True)
    active = pd.read_csv(processed_dir / 'cleaned_active_addresses_1h.csv', index_col=0, parse_dates=True)
    sopr = pd.read_csv(processed_dir / 'cleaned_sopr_1h.csv', index_col=0, parse_dates=True)
    cdd = pd.read_csv(processed_dir / 'cleaned_cdd_1h.csv', index_col=0, parse_dates=True)

    # 统一所有索引为 tz-naive
    for df in [price, funding, fng, netflow, total, active, sopr, cdd]:
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

    # 统一 resample 到 4h
    price_4h = price.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    funding_4h = funding.resample('4h').last()[['fundingRate']]
    fng_4h = fng.resample('4h').last()[['fng_value']]
    netflow_4h = netflow.resample('4h').last()
    total_4h = total.resample('4h').last()
    active_4h = active.resample('4h').last()
    sopr_4h = sopr.resample('4h').last()[['sopr']]
    cdd_4h = cdd.resample('4h').last()[['cdd']]

    # 6个消融组合
    ablation_groups = {
        'price_only': price_4h,
        'price_funding': pd.concat([price_4h, funding_4h], axis=1),
        'price_funding_fng': pd.concat([price_4h, funding_4h, fng_4h], axis=1),
        'price_long_onchain': pd.concat([price_4h, active_4h, netflow_4h, total_4h], axis=1),
        'price_onchain': pd.concat([price_4h, sopr_4h, cdd_4h], axis=1),
        'full': pd.concat([price_4h, funding_4h, fng_4h, sopr_4h, cdd_4h], axis=1),
    }

    for name, df in ablation_groups.items():
        # 去重 + 前向填充
        df = df[~df.index.duplicated(keep='first')]
        df = df.ffill()

        # 保存
        output_file = merged_dir / f'merged_{name}_4h.csv'
        df.to_csv(output_file)
        print(f"Saved: {output_file}  shape: {df.shape}")

if __name__ == "__main__":
    merge_ablation_datasets()