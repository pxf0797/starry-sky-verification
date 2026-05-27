#!/usr/bin/env python3
"""
新实验 N6 (P1): 参数敏感度二维热力图
验证: 关键参数 (tau, sigma0) 的交互效应——是否存在"安全操作区"

使用方式:
    python3 verify_param_heatmap.py [--trials 100]

实验逻辑:
    在 tau x sigma0 的二维网格上
    每个格点运行蒙特卡洛模拟
    输出夏普比、最大回撤的 ASCII 热力图
    标注最优参数区域
"""

import argparse, numpy as np

def sharpe(r):
    if len(r)<2 or np.std(r)<1e-15: return -10
    return np.mean(r)/np.std(r)*np.sqrt(252)

def simulate_grid_point(tau, sigma0, n_trials=100):
    all_rets = []
    for _ in range(n_trials):
        n_steps = 80
        t_arr = np.arange(n_steps, dtype=float)
        signal = 0.5*t_arr**3-0.8*t_arr**2+0.3*t_arr+100
        noise = np.random.normal(0, 0.3, n_steps)
        prices = signal+noise
        t_fit = np.arange(20, dtype=float)
        coeffs = np.polyfit(t_fit, prices[:20], 3)
        lam = np.log(2)/tau; pos = 1
        for i in range(20, n_steps-1):
            pred = np.polyval(coeffs, float(i)); resid = abs(prices[i]-pred)
            sigma_t = sigma0*np.exp(-lam*(i-20))
            if resid > 3*sigma_t: pos = 0
            ret = -(prices[i+1]-prices[i])/prices[i]
            all_rets.append(ret if pos else 0)
    return sharpe(all_rets)

def main():
    parser = argparse.ArgumentParser(description="N6: 参数敏感度二维热力图")
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()

    taus = [1,2,3,5,7,10,15,20,30]
    sigma0s = [0.01,0.05,0.1,0.2,0.5,0.8,1.0,1.5,2.0]

    print("="*72)
    print("  新实验 N6 (P1): 参数敏感度二维热力图")
    print("="*72)
    print(f"  tau x sigma0 网格: {len(taus)}x{len(sigma0s)} = {len(taus)*len(sigma0s)} 点")
    print(f"  每点试验: {args.trials} | 正在计算...")
    print()

    grid = np.zeros((len(sigma0s), len(taus)))
    for i, s0 in enumerate(sigma0s):
        for j, tau in enumerate(taus):
            grid[i,j] = simulate_grid_point(tau, s0, args.trials)

    # ASCII heatmap
    print(f"  {'tau=':>8s}", end="")
    for tau in taus: print(f"  {tau:>5d}", end="")
    print()
    print(f"  {'-'* (8+6*len(taus))}")

    best_val, best_ij = -1e9, (0,0)
    for i, s0 in enumerate(sigma0s):
        print(f"  {'s0='+str(s0):>8s}", end="")
        for j, tau in enumerate(taus):
            v = grid[i,j]
            if v > best_val: best_val, best_ij = v, (i,j)
            if v > 1.0: c = "\033[92m"  # green
            elif v > 0: c = "\033[93m"   # yellow
            else: c = "\033[91m"          # red
            print(f"{c}  {v:>5.2f}\033[0m", end="")
        print()

    bi, bj = best_ij
    print()
    print(f"  最优参数: tau={taus[bj]}, sigma0={sigma0s[bi]} (夏普比={best_val:.3f})")
    # 安全操作区
    safe_count = np.sum(grid > 0)
    print(f"  安全操作区 (夏普比>0): {safe_count}/{grid.size} 格点 ({safe_count/grid.size:.0%})")
    print(f"  报告引用: research-supplement-v1.md Part C - 新实验 N6")

if __name__ == "__main__":
    main()
