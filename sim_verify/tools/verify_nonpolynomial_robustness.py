#!/usr/bin/env python3
"""
新实验 N1 (P0): 非多项式信号下的三次拟合鲁棒性
验证: 当真实信号不是多项式时，三次拟合是否仍然可用

使用方式:
    python3 verify_nonpolynomial_robustness.py [--trials 500] [--signal sine]

实验逻辑:
    在三种非多项式信号上测试 k=1~5 阶多项式的外推 RMSE:
    - sine:   正弦衰减 P(t)=A*sin(wt)*exp(-lt) (振荡收敛)
    - pwlinear: 分段线性 (阶梯式价格)
    - gbm:    几何布朗运动 P(t)=P0*exp((mu-s^2/2)t + s*W(t)) (无拐点)
    核心关注: 三次在非多项式信号上的 RMSE 是否仍在可用范围 (不超过最优阶 2x)
"""

import argparse, numpy as np, sys

def generate_sine_signal(t, noise_std=0.02):
    A, w, lam = 3.0, 4.0, 0.5
    signal = A * np.sin(w * t) * np.exp(-lam * t)
    return signal + np.random.normal(0, noise_std, len(t)), signal

def generate_pwlinear_signal(t, noise_std=0.02):
    breaks = [0, 0.3, 0.5, 0.7, 1.0]
    slopes = [2.0, -1.5, 0.5, -2.0]
    y0 = 0.0
    clean = np.zeros(len(t))
    for i, ti in enumerate(t):
        seg = max(0, min(len(breaks)-2, np.searchsorted(breaks, ti)-1))
        dt = ti - breaks[seg]
        y_val = y0
        for s in range(seg): y_val += slopes[s]*(breaks[s+1]-breaks[s])
        y_val += slopes[seg]*dt
        clean[i] = y_val
    return clean + np.random.normal(0, noise_std, len(t)), clean

def generate_gbm_signal(t, noise_std=0.02):
    dt = t[1]-t[0]; mu, sigma_gbm = 0.3, 0.4; p0 = 100.0
    W = np.random.randn(len(t)).cumsum()*np.sqrt(dt)
    signal = p0 * np.exp((mu - 0.5*sigma_gbm**2)*t + sigma_gbm*W)
    # observation noise
    noisy = signal + np.random.normal(0, noise_std*signal[0], len(t))
    return noisy, signal

def polyfit(x, y, deg):
    return np.polyfit(x, y, deg)

def polyval(c, x):
    return np.polyval(c, x)

def main():
    parser = argparse.ArgumentParser(description="N1: 非多项式信号拟合鲁棒性")
    parser.add_argument("--trials", type=int, default=300, help="Monte Carlo 次数")
    parser.add_argument("--signal", default="all", choices=["sine","pwlinear","gbm","all"])
    parser.add_argument("--noise", type=float, default=0.02)
    args = parser.parse_args()

    signals = {"sine": generate_sine_signal, "pwlinear": generate_pwlinear_signal, "gbm": generate_gbm_signal}
    if args.signal != "all": signals = {args.signal: signals[args.signal]}

    print("="*72)
    print("  新实验 N1 (P0): 非多项式信号下的三次拟合鲁棒性")
    print("="*72)
    print(f"  试验次数: {args.trials} | 噪声: {args.noise}")
    print()

    for name, gen_fn in signals.items():
        rmse_accum = {k: [] for k in range(1,6)}
        for _ in range(args.trials):
            n_pts = 60; t_all = np.linspace(0, 2, n_pts)
            y_all, clean = gen_fn(t_all, args.noise)
            t_fit, y_fit = t_all[:40], y_all[:40]
            t_ext, y_ext = t_all[40:], clean[40:]
            for k in range(1,6):
                try:
                    c = polyfit(t_fit, y_fit, k)
                    pred = polyval(c, t_ext)
                    rmse = np.sqrt(np.mean((pred-y_ext)**2))
                    rmse_accum[k].append(rmse)
                except: rmse_accum[k].append(1e9)

        avg_rmse = {k: np.mean(v) for k, v in rmse_accum.items()}
        best_k = min(avg_rmse, key=avg_rmse.get)
        best_rmse = avg_rmse[best_k]
        rmse3 = avg_rmse[3]
        ratio = rmse3/best_rmse if best_rmse > 1e-10 else 1.0

        print(f"  [{name.upper()} 信号]")
        for k in range(1,6):
            marker = " ★" if k==best_k else ""
            print(f"    k={k}: RMSE={avg_rmse[k]:.4f}{marker}")
        print(f"    最优: k={best_k}, 三次相对最优={ratio:.1f}x | {'鲁棒 OK' if ratio<2 else '需注意' if ratio<5 else '不可用'}")
        print()

    print("  报告引用: research-supplement-v1.md Part C - 新实验 N1")

if __name__ == "__main__":
    main()
