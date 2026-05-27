#!/usr/bin/env python3
"""
真实市场数据加载器 (方案F)
支持从Binance/本地CSV加载K线数据用于策略回测

使用方式:
    # 从Binance加载 (需要 ccxt: pip install ccxt)
    python3 data_loader.py --source binance --symbol BTC/USDT --timeframe 15m --days 180 --output btc_15m.csv

    # 从本地CSV加载
    python3 data_loader.py --source csv --input data.csv

    # 查看数据摘要
    python3 data_loader.py --source binance --symbol ETH/USDT --timeframe 1h --days 30 --summary
"""

import argparse, sys, os, json
from datetime import datetime, timedelta
import csv

def load_from_csv(filepath):
    """从本地CSV加载K线数据"""
    if not os.path.exists(filepath):
        print(f"错误: 文件不存在 {filepath}")
        sys.exit(1)

    data = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "timestamp": row.get("timestamp") or row.get("datetime") or row.get("time"),
                "open": float(row.get("open", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "close": float(row.get("close", 0)),
                "volume": float(row.get("volume", 0)),
            })
    return data

def load_from_binance(symbol, timeframe, days):
    """从Binance加载K线数据 (需要ccxt)"""
    try:
        import ccxt
    except ImportError:
        print("错误: 需要安装 ccxt: pip install ccxt")
        sys.exit(1)

    exchange = ccxt.binance({"enableRateLimit": True})
    since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat() + "Z")
    limit = 1000

    all_ohlcv = []
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        if len(ohlcv) == 0:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < limit:
            break

    data = []
    for row in all_ohlcv:
        data.append({
            "timestamp": datetime.fromtimestamp(row[0] / 1000).isoformat(),
            "open": row[1], "high": row[2], "low": row[3],
            "close": row[4], "volume": row[5],
        })
    return data

def save_to_csv(data, filepath):
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        writer.writerows(data)
    print(f"已保存 {len(data)} 条数据到 {filepath}")

def print_summary(data):
    closes = [d["close"] for d in data]
    returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
    import math
    mean_ret = sum(returns) / len(returns) if returns else 0
    std_ret = (sum((r - mean_ret)**2 for r in returns) / len(returns))**0.5 if returns else 0

    print(f"\n数据摘要:")
    print(f"  时间范围: {data[0]['timestamp']} ~ {data[-1]['timestamp']}")
    print(f"  K线数量: {len(data)}")
    print(f"  价格范围: {min(closes):.2f} ~ {max(closes):.2f}")
    print(f"  年化波动: {std_ret * (252**0.5) * 100:.1f}%")
    print(f"  年化收益: {mean_ret * 252 * 100:.1f}%")
    sharpe = (mean_ret / std_ret) * (252**0.5) if std_ret > 1e-10 else 0
    print(f"  夏普比:   {sharpe:.2f}")

    # 市场状态分段
    n = len(closes)
    thirds = [closes[:n//3], closes[n//3:2*n//3], closes[2*n//3:]]
    for i, third in enumerate(thirds):
        vol = (sum((third[j] - third[j-1])**2 for j in range(1, len(third))) / len(third))**0.5
        print(f"  区间{i+1}波动: {vol * (252**0.5) * 100:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="真实市场数据加载器")
    parser.add_argument("--source", choices=["binance", "csv"], required=True)
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--input", help="CSV文件路径 (source=csv时)")
    parser.add_argument("--output", help="输出CSV文件路径")
    parser.add_argument("--summary", action="store_true", help="仅显示数据摘要")
    args = parser.parse_args()

    if args.source == "binance":
        print(f"从Binance加载: {args.symbol} {args.timeframe} {args.days}天...")
        data = load_from_binance(args.symbol, args.timeframe, args.days)
    else:
        if not args.input:
            print("错误: --source csv 需要 --input")
            sys.exit(1)
        data = load_from_csv(args.input)

    if not data:
        print("错误: 未加载到数据")
        sys.exit(1)

    print_summary(data)

    if args.output:
        save_to_csv(data, args.output)

if __name__ == "__main__":
    main()
