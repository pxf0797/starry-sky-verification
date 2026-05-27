#!/usr/bin/env python3
"""
新实验 N8 (P1): 多品种 tau 跨品种校准框架
验证: tau 的最优值是否因品种波动特征不同而变化

使用方式:
    python3 verify_cross_asset_tau.py [--trials 200]

实验逻辑:
    模拟不同波动特征的"品种":
    - BTC-like: 年化波动 ~60% (中等)
    - ETH-like: 年化波动 ~80% (偏高)
    - ALT-like: 年化波动 ~120% (高)
    对每个品种分别运行 tau 扫描
    对比 tau* 的品种间差异
"""

import argparse, numpy as np

def sharpe(r):
    if len(r)<2 or np.std(r)<1e-15: return -10
    return np.mean(r)/np.std(r)*np.sqrt(252)

def simulate_asset_prices(vol, n_steps=120):
    """生成不同波动水平的合成价格轨迹"""
    t = np.arange(n_steps, dtype=float)
    # 共同的拐点结构
    base_signal = 0.3*t**3 - 0.6*t**2 + 0.2*t + 100
    noise = np.random.normal(0, vol*0.5, n_steps)
    return base_signal + noise

def evaluate_tau(prices, tau, entry_idx=20):
    """评估给定 tau 的夏普比"""
    returns = []
    t_fit = np.arange(entry_idx, dtype=float)
    coeffs = np.polyfit(t_fit, prices[:entry_idx], 3)
    lam = np.log(2)/tau; pos = 1
    for i in range(entry_idx+1, len(prices)):
        pred = np.polyval(coeffs, float(i)); resid = abs(prices[i]-pred)
        sigma_t = 0.5*np.exp(-lam*(i-entry_idx))
        if resid > 3*sigma_t: pos = 0
        ret = -(prices[i]-prices[i-1])/prices[i-1]
        returns.append(ret if pos else 0)
    return sharpe(returns)

def main():
    parser = argparse.ArgumentParser(description="N8: 多品种 tau 跨品种校准")
    parser.add_argument("--trials", type=int, default=200)
    args = parser.parse_args()

    assets = {"BTC-like (vol=60%)": 0.6, "ETH-like (vol=80%)": 0.8, "ALT-like (vol=120%)": 1.2}
    taus = [1,2,3,5,7,10,15,20,30]

    print("="*72)
    print("  新实验 N8 (P1): 多品种 tau 跨品种校准")
    print("="*72)
    print(f"  试验: {args.trials} 条轨迹/品种")
    print()

    print(f"  {'品种':>22s}", end="")
    for tau in taus: print(f"  {'t='+str(tau):>8s}", end="")
    print(f"  {'t*':>6s}")
    print(f"  {'-'*(22+10*len(taus))}")

    for name, vol in assets.items():
        results = []
        for tau in taus:
            all_s = []
            for _ in range(args.trials):
                prices = simulate_asset_prices(vol)
                all_s.append(evaluate_tau(prices, tau))
            results.append(np.mean(all_s))

        best_idx = np.argmax(results)
        best_tau = taus[best_idx]

        print(f"  {name:>22s}", end="")
        for i, v in enumerate(results):
            marker = "*" if i==best_idx else " "
            print(f"  {marker}{v:>7.3f}", end="")
        print(f"  {best_tau:>4d}")

    print()
    print(f"  预期: 高波动品种 -> 更短的 tau (信息衰减更快)")
    print(f"        低波动品种 -> 稍长的 tau (信号更稳定)")
    print(f"  报告引用: research-supplement-v1.md Part C - 新实验 N8")

if __name__ == "__main__":
    main()
