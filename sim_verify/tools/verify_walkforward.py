#!/usr/bin/env python3
"""
新实验 N5 (P1): Walk-Forward 样本外验证
验证: 策略参数的时间鲁棒性——在未知数据上的表现

使用方式:
    python3 verify_walkforward.py [--windows 12] [--trials 50]

实验逻辑:
    将合成价格序列按时间顺序分为 N 个窗口
    窗口i 优化参数 -> 窗口i+1 做样本外测试
    报告样本外夏普比的均值/标准差/正收益窗口比例
"""

import argparse, numpy as np

def sharpe(r):
    if len(r)<2 or np.std(r)<1e-15: return -10
    return np.mean(r)/np.std(r)*np.sqrt(252)

def generate_window_data(n_steps=200):
    t = np.arange(n_steps, dtype=float)*0.1
    signal = 0.5*t**3 - 0.8*t**2 + 0.3*t + 100
    noise = np.random.normal(0, 0.5, n_steps)
    return signal + noise

def run_strategy(prices, tau=5, sigma0=0.5, k_sig=2.0, b_sig=-0.74, entry_idx=30):
    rets = []; pos = 1
    t_fit = np.arange(entry_idx, dtype=float)
    coeffs = np.polyfit(t_fit, prices[:entry_idx], 3)
    lam = np.log(2)/tau
    for i in range(entry_idx+1, len(prices)):
        pred = np.polyval(coeffs, float(i)); resid = abs(prices[i]-pred)
        alpha = 1/(1+np.exp(-(-k_sig*resid+b_sig)))
        sigma_t = sigma0*np.exp(-lam*(i-entry_idx))
        if resid > 3*sigma_t or alpha < 0.1: pos = 0
        ret = -(prices[i]-prices[i-1])/prices[i-1]
        rets.append(ret if pos else 0)
    return rets

def optimize_params(prices, param_grid):
    """在训练窗口上网格搜索最优参数"""
    best_s, best_p = -1e9, None
    mid = len(prices)//2
    train_prices = prices[:mid]
    for tau in param_grid["tau"]:
        for sigma0 in param_grid["sigma0"]:
            rets = run_strategy(train_prices, tau=tau, sigma0=sigma0, entry_idx=20)
            s = sharpe(rets)
            if s > best_s: best_s, best_p = s, (tau, sigma0)
    return best_p

def main():
    parser = argparse.ArgumentParser(description="N5: Walk-Forward 样本外验证")
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    param_grid = {"tau": [3,5,7,10], "sigma0": [0.1,0.5,1.0]}
    oos_sharpes = []

    print("="*72)
    print("  新实验 N5 (P1): Walk-Forward 样本外验证")
    print("="*72)
    print(f"  窗口数: {args.windows} | 重复: {args.trials} 次")
    print()

    for trial in range(args.trials):
        all_data = generate_window_data(args.windows*40)
        window_size = len(all_data)//args.windows
        trial_oos = []

        for w in range(args.windows-2):
            train = all_data[w*window_size:(w+2)*window_size]
            test = all_data[(w+2)*window_size:(w+3)*window_size]
            best_p = optimize_params(train, param_grid)
            rets = run_strategy(test, tau=best_p[0], sigma0=best_p[1], entry_idx=20)
            trial_oos.append(sharpe(rets))

        oos_sharpes.extend(trial_oos)
        if trial < 5 or trial == args.trials-1:
            print(f"  第{trial+1}轮: 样本外均值={np.mean(trial_oos):.3f}, 正比例={sum(1 for s in trial_oos if s>0)}/{len(trial_oos)}")

    oos_sharpes = np.array(oos_sharpes)
    pos_ratio = np.mean(oos_sharpes > 0)

    print()
    print(f"  总体样本外夏普比: 均值={np.mean(oos_sharpes):.3f}, 标准差={np.std(oos_sharpes):.3f}")
    print(f"  正收益窗口比例: {pos_ratio:.1%}")
    if pos_ratio > 0.67:
        print(f"  ✅ 策略具有时间鲁棒性 (>{2/3:.0%} 窗口为正)")
    else:
        print(f"  ⚠️ 策略时间鲁棒性不足——参数可能对数据生成过程过拟合")
    print(f"  报告引用: research-supplement-v1.md Part C - 新实验 N5")

if __name__ == "__main__":
    main()
