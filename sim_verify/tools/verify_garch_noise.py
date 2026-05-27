#!/usr/bin/env python3
"""
新实验 N2 (P0): GARCH 噪声下的三次最优性复验
验证: 当噪声具有波动率聚集特征 (GARCH) 时，三次是否仍为最优阶数

使用方式:
    python3 verify_garch_noise.py [--trials 300]

实验逻辑:
    生成 GARCH(1,1) 噪声: e_t = sigma_t * z_t, z_t ~ N(0,1)
    sigma^2_t = omega + alpha*e^2_{t-1} + beta*sigma^2_{t-1}
    在三次多项式信号 + GARCH 噪声上，重跑实验 1.1 的阶数对比
"""

import argparse, numpy as np

def generate_garch_noise(n, omega=0.01, alpha=0.1, beta=0.85):
    sigma2 = np.zeros(n); eps = np.zeros(n)
    sigma2[0] = omega/(1-alpha-beta)
    for t in range(1, n):
        sigma2[t] = omega + alpha*eps[t-1]**2 + beta*sigma2[t-1]
        eps[t] = np.sqrt(sigma2[t])*np.random.normal()
    return eps

def main():
    parser = argparse.ArgumentParser(description="N2: GARCH 噪声三次最优性复验")
    parser.add_argument("--trials", type=int, default=300)
    args = parser.parse_args()

    ks = [1,2,3,4,5]
    n_fit, n_ext = 20, 10
    t_fit = np.linspace(0,1,n_fit); t_ext = np.linspace(1,1.5,n_ext)
    t_all = np.concatenate([t_fit, t_ext])
    true_signal = t_all**3 - 1.5*t_all**2 + 0.5*t_all

    rmse_accum = {k: [] for k in ks}
    for _ in range(args.trials):
        noise = generate_garch_noise(len(t_all), omega=0.01, alpha=0.1, beta=0.85)
        y_obs = true_signal + noise
        y_fit = y_obs[:n_fit]; y_true_ext = true_signal[n_fit:]
        for k in ks:
            c = np.polyfit(t_fit, y_fit, k)
            pred = np.polyval(c, t_ext)
            rmse_accum[k].append(np.sqrt(np.mean((pred-y_true_ext)**2)))

    print("="*72)
    print("  新实验 N2 (P0): GARCH 噪声三次最优性复验")
    print("="*72)
    print(f"  噪声模型: GARCH(1,1) w=0.01 a=0.1 b=0.85")
    print(f"  试验: {args.trials} 次")
    print()

    avg = {}
    for k in ks:
        avg[k] = np.mean(rmse_accum[k])
        print(f"  k={k}: RMSE = {avg[k]:.4f}")

    best = min(avg, key=avg.get)
    r3 = avg[3]
    r4 = avg[4]/r3 if r3>0 else float('inf')
    r5 = avg[5]/r3 if r3>0 else float('inf')

    print()
    print(f"  最优阶数: k={best}")
    print(f"  相对比值: R4={r4:.1f}x, R5={r5:.1f}x")
    if best == 3:
        print(f"  ✅ 三次最优性在 GARCH 噪声下跨模型证实")
    else:
        print(f"  ⚠️ 三次最优性在 GARCH 下偏移至 k={best}——需进一步分析")
    print(f"  报告引用: research-supplement-v1.md Part C - 新实验 N2")

if __name__ == "__main__":
    main()
