#!/usr/bin/env python3
"""
新实验 N4 (P0): 基准策略对比矩阵
验证: 星空策略是否在"含拐点轨迹"上优于简单基准策略

使用方式:
    python3 verify_benchmark_compare.py [--trials 300]

实验逻辑:
    在相同合成轨迹上运行 5 种策略:
    - TP/SL:    固定止盈+2%/止损-1% (经典 2:1)
    - MA_Cross:  5周期MA下穿20周期MA平仓
    - Bollinger: 价格回归中轨 (20周期, 2sigma) 平仓
    - Kalman:    卡尔曼滤波跟踪+新息检测平仓
    - StarrySky: 三次拟合+SIGMOID+动态阈值+铁律
    对比各策略在"含拐点轨迹"和"随机游走"两个子集上的夏普比
"""

import argparse, numpy as np
from collections import defaultdict

def sharpe(returns):
    if len(returns)<2 or np.std(returns)<1e-15: return -10
    return np.mean(returns)/np.std(returns)*np.sqrt(252)

def max_dd(returns):
    cum = np.cumprod(1+np.array(returns)); peak = np.maximum.accumulate(cum)
    return np.min((cum-peak)/peak)

def generate_inflection_trajectory(n_steps=100):
    """含拐点轨迹: 三次多项式+噪声"""
    t = np.arange(n_steps, dtype=float)*0.05
    signal = t**3 - 1.5*t**2 + 0.5*t + 100
    noise = np.random.normal(0, 0.3, n_steps)
    return signal + noise

def generate_randomwalk(n_steps=100):
    """随机游走 (无拐点)"""
    returns = np.random.normal(0, 0.02, n_steps)
    price = 100 * np.exp(np.cumsum(returns))
    return price

def strategy_tpsl(prices, entry_idx=20):
    """固定止盈止损"""
    rets = []; entry = prices[entry_idx]; pos = 1
    for i in range(entry_idx+1, len(prices)):
        ret = -(prices[i]-prices[i-1])/prices[i-1]
        pnl_pct = (entry-prices[i])/entry*100
        if pnl_pct > 2 or pnl_pct < -1: pos = 0
        rets.append(ret if pos else 0)
    return rets

def strategy_macross(prices, entry_idx=20):
    """移动平均交叉"""
    rets = []; pos = 1
    for i in range(entry_idx+1, len(prices)):
        if i >= 21:
            ma5 = np.mean(prices[i-5:i]); ma20 = np.mean(prices[i-20:i])
            if ma5 < ma20: pos = 0
        ret = -(prices[i]-prices[i-1])/prices[i-1]
        rets.append(ret if pos else 0)
    return rets

def strategy_bollinger(prices, entry_idx=20):
    """布林带回归"""
    rets = []; pos = 1
    for i in range(entry_idx+1, len(prices)):
        if i >= 21:
            ma20 = np.mean(prices[i-20:i]); std20 = np.std(prices[i-20:i])
            upper, lower = ma20+2*std20, ma20-2*std20
            if prices[i] < lower: pos = 0
        ret = -(prices[i]-prices[i-1])/prices[i-1]
        rets.append(ret if pos else 0)
    return rets

def strategy_kalman(prices, entry_idx=20):
    """简化卡尔曼滤波跟踪"""
    rets = []; pos = 1
    # 状态: [price, velocity]
    x = np.array([prices[entry_idx], -0.5])
    P = np.eye(2)*0.1; Q = np.eye(2)*0.01; R = np.array([[0.5]])
    for i in range(entry_idx+1, len(prices)):
        # predict
        dt = 1; F = np.array([[1,dt],[0,1]])
        x = F @ x; P = F @ P @ F.T + Q
        # update
        H = np.array([[1,0]]); y = prices[i] - H @ x
        S = H @ P @ H.T + R; K = P @ H.T / S[0,0]
        x = x + K.flatten()*y; P = P - K @ H @ P
        # anomaly detection
        if abs(y) > 3*np.sqrt(S[0,0]): pos = 0
        ret = -(prices[i]-prices[i-1])/prices[i-1]
        rets.append(ret if pos else 0)
    return rets

def strategy_starrysky(prices, entry_idx=20):
    """星空策略简化版: 三次拟合+SIGMOID+动态阈值"""
    rets = []; pos = 1
    t_fit = np.arange(entry_idx, dtype=float)
    coeffs = np.polyfit(t_fit, prices[:entry_idx], 3)
    lam = np.log(2)/5.0
    for i in range(entry_idx+1, len(prices)):
        pred = np.polyval(coeffs, float(i))
        resid = abs(prices[i]-pred)
        arg = -2*resid-0.74; alpha = 1/(1+np.exp(-max(min(arg,50),-50)))
        sigma_t = 0.5*np.exp(-lam*(i-entry_idx))
        if resid > 3*sigma_t or alpha < 0.1: pos = 0
        ret = -(prices[i]-prices[i-1])/prices[i-1]
        rets.append(ret if pos else 0)
    return rets

def main():
    parser = argparse.ArgumentParser(description="N4: 基准策略对比矩阵")
    parser.add_argument("--trials", type=int, default=300)
    args = parser.parse_args()

    strategies = {
        "TP/SL (2:1)": strategy_tpsl,
        "MA Cross": strategy_macross,
        "Bollinger": strategy_bollinger,
        "Kalman": strategy_kalman,
        "StarrySky": strategy_starrysky,
    }

    scenarios = {"含拐点轨迹": generate_inflection_trajectory, "随机游走": generate_randomwalk}

    print("="*72)
    print("  新实验 N4 (P0): 基准策略对比矩阵")
    print("="*72)
    print(f"  试验: {args.trials} 条轨迹/场景")
    print()

    for sc_name, gen_fn in scenarios.items():
        print(f"  [{sc_name}]")
        print(f"  {'策略':>18s}  {'夏普比':>10s}  {'最大回撤':>10s}  {'评级':>6s}")
        print(f"  {'-'*50}")
        results = {}
        for st_name, st_fn in strategies.items():
            all_rets = []
            for _ in range(args.trials):
                prices = gen_fn()
                all_rets.extend(st_fn(prices))
            s = sharpe(all_rets); d = max_dd(all_rets)
            results[st_name] = (s, d)
            grade = "A" if s>1 else "B" if s>0 else "C" if s>-1 else "D"
            print(f"  {st_name:>18s}  {s:>10.3f}  {d:>10.3f}  {grade:>6s}")

        starry_s = results["StarrySky"][0]
        best_other = max(v[0] for k,v in results.items() if k!="StarrySky")
        print(f"  {'':>18s}  {'星空 vs 最佳基准:':>22s} {starry_s-best_other:+.3f}")
        print()

    print("  报告引用: research-supplement-v1.md Part C - 新实验 N4")

if __name__ == "__main__":
    main()
