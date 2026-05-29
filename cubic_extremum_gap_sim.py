#!/usr/bin/env python3
"""
三次曲线极值差(Δy)数值模拟脚本

f(x) = ax³ + bx² + cx + d

核心公式: Δy = 4(b²-3ac)^(3/2) / (27a²)  [当 D_crit = b²-3ac > 0]

用法: python cubic_extremum_gap_sim.py
输出: 控制台统计摘要 + cubic_gap_analysis.png 图表

依赖: numpy, scipy, matplotlib (标准科学计算栈)
"""

import warnings
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")  # headless 后端, 须在 pyplot 导入之前
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 抑制 scipy KDE 计算中因近零值/重复值产生的数值警告 (不影响结果)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")

# ── 字体配置 (macOS 中文 fallback) ──────────────────────────────────────────
rcParams["font.family"] = ["Heiti TC", "Arial Unicode MS", "sans-serif"]
rcParams["axes.unicode_minus"] = False

try:
    plt.style.use("seaborn-v0_8-darkgrid")
except (OSError, ValueError):
    try:
        plt.style.use("ggplot")
    except (OSError, ValueError):
        pass  # 使用 matplotlib 默认样式


# ============================================================================
# 核心计算函数
# ============================================================================


def compute_gap(a, b, c):
    """
    计算 Δy = 4 * D_crit^(3/2) / (27 * a²)

    参数:
        a, b, c: 三次曲线系数 f(x)=ax³+bx²+cx+d (d 不影响 Δy)

    返回:
        Δy 的浮点值; D_crit ≤ 0 或 |a| ≈ 0 时返回 None
    """
    D = b * b - 3.0 * a * c
    if D <= 0.0 or abs(a) < 1e-15:
        return None
    return 4.0 * D ** 1.5 / (27.0 * a * a)


def critical_points(a, b, c, d=0.0):
    """
    计算两个极值点的 (x, y) 坐标

    返回:
        (x1, y1, x2, y2) 其中:
          a>0 → x1=极大值点, x2=极小值点
          a<0 → x1=极小值点, x2=极大值点
        D_crit ≤ 0 时返回 None
    """
    D = b * b - 3.0 * a * c
    if D <= 0.0 or abs(a) < 1e-15:
        return None
    sqrtD = np.sqrt(D)
    x1 = (-b - sqrtD) / (3.0 * a)
    x2 = (-b + sqrtD) / (3.0 * a)
    y1 = ((a * x1 + b) * x1 + c) * x1 + d
    y2 = ((a * x2 + b) * x2 + c) * x2 + d
    return (x1, y1, x2, y2)


def cardano_discriminant(a, b, c, d=0.0):
    """
    计算三次方程 f(x)=0 的 Cardano 判别式

    Δ = 18abcd - 4b³d + b²c² - 4ac³ - 27a²d²
    当 d=0: Δ = c²(b² - 4ac)
    """
    return (18.0 * a * b * c * d - 4.0 * b**3 * d
            + b * b * c * c - 4.0 * a * c**3 - 27.0 * a * a * d * d)


# ============================================================================
# 参数采样
# ============================================================================


def sample_params(n=30000, range_r=200.0, method="uniform", seed=42):
    """
    采样满足 D_crit > 0 的参数组合

    参数:
        n: 采样数量
        range_r: 参数范围 [-range_r, range_r]
        method: 'uniform' | 'loguniform' | 'importance'
        seed: 随机种子

    返回:
        params: (n_valid, 4) 数组 [a, b, c, D_crit]
        dy_values: (n_valid,) 数组
    """
    rng = np.random.RandomState(seed)

    if method == "uniform":
        a_raw = rng.uniform(-range_r, range_r, n)
        b_raw = rng.uniform(-range_r, range_r, n)
        c_raw = rng.uniform(-range_r, range_r, n)

    elif method == "loguniform":
        log_a = rng.uniform(-3.0, np.log10(range_r), n)
        log_b = rng.uniform(-3.0, np.log10(range_r), n)
        log_c = rng.uniform(-3.0, np.log10(range_r), n)
        a_raw = rng.choice([-1.0, 1.0], n) * (10.0 ** log_a)
        b_raw = rng.choice([-1.0, 1.0], n) * (10.0 ** log_b)
        c_raw = rng.choice([-1.0, 1.0], n) * (10.0 ** log_c)

    elif method == "importance":
        # 偏向小 |a|: a' = sign(a) * |a|³ / range_r²
        a_base = rng.uniform(-range_r, range_r, n)
        a_raw = np.sign(a_base) * (np.abs(a_base) ** 3) / (range_r ** 2)
        b_raw = rng.uniform(-range_r, range_r, n)
        c_raw = rng.uniform(-range_r, range_r, n)

    else:
        raise ValueError(f"Unknown sampling method: {method}")

    # 计算 D_crit 并过滤
    D_raw = b_raw * b_raw - 3.0 * a_raw * c_raw
    valid_mask = (D_raw > 0.0) & (np.abs(a_raw) > 1e-15)

    a_v = a_raw[valid_mask]
    b_v = b_raw[valid_mask]
    c_v = c_raw[valid_mask]
    D_v = D_raw[valid_mask]
    n_valid = len(a_v)

    params = np.column_stack([a_v, b_v, c_v, D_v])

    # 向量化计算 Δy
    dy_values = 4.0 * D_v ** 1.5 / (27.0 * a_v * a_v)

    return params, dy_values


# ============================================================================
# 统计分析
# ============================================================================


def analyze(params, dy_values):
    """
    对 Δy 分布进行统计描述

    参数:
        params: (n, 4) 数组 [a, b, c, D_crit]
        dy_values: (n,) 数组

    返回:
        dict: 含均值、中位数、分位数、Spearman 相关系数等
    """
    n = len(dy_values)
    r = {"n_valid": n}

    # 基础统计量
    r["mean"] = float(np.mean(dy_values))
    r["median"] = float(np.median(dy_values))
    r["std"] = float(np.std(dy_values))
    r["min"] = float(np.min(dy_values))
    r["max"] = float(np.max(dy_values))
    r["skewness"] = float(stats.skew(dy_values))

    # 分位数
    for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        r[f"q{q:02d}"] = float(np.percentile(dy_values, q))

    # Spearman 秩相关: log10(Δy) vs 各参数的对数绝对值
    log_dy = np.log10(np.clip(dy_values, 1e-30, None))
    abs_a = np.abs(params[:, 0])
    abs_b = np.abs(params[:, 1])
    abs_c = np.abs(params[:, 2])
    D_crit = params[:, 3]

    for name, vals in [("|a|", abs_a), ("|b|", abs_b),
                        ("|c|", abs_c), ("D_crit", D_crit)]:
        pos = vals > 1e-30
        if pos.sum() >= 5:
            rho, _ = stats.spearmanr(log_dy[pos], np.log10(vals[pos]))
            r[f"spearman_{name}"] = float(rho)
        else:
            r[f"spearman_{name}"] = float("nan")

    return r


# ============================================================================
# 可视化 (2×3 子图)
# ============================================================================


def plot_analysis(params, dy_values, save_path="cubic_gap_analysis.png", seed=42):
    """
    绘制 2×3 分析图组并保存到文件

    布局:
      (0,0) log10(Δy) 直方图 + KDE
      (0,1) Δy vs Cardano Δ 散点图 (色标 D_crit)
      (0,2) (a,b) 平面上 log10(Δy) 的 hexbin 热力图
      (1,0) Δy vs a (symlog x 轴)
      (1,1) Δy vs |b| (log-log)
      (1,2) Δy vs D_crit (log-log)
    """
    # ── 数据准备 ──
    valid = dy_values > 0
    p = params[valid]
    dy = dy_values[valid]
    n = len(dy)

    log_dy = np.log10(dy)
    abs_a = np.abs(p[:, 0])
    D_crit = p[:, 3]

    # 为散点图做子采样 (避免渲染过多点)
    if n > 5000:
        idx = np.random.RandomState(seed).choice(n, 5000, replace=False)
    else:
        idx = slice(None)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # ── 图 1: log10(Δy) 直方图 + KDE ──
    ax = axes[0, 0]
    ax.hist(log_dy, bins=60, density=True, alpha=0.6,
            color="steelblue", edgecolor="white", linewidth=0.3)
    try:
        kde = stats.gaussian_kde(log_dy)
        x_kde = np.linspace(log_dy.min(), log_dy.max(), 200)
        ax.plot(x_kde, kde(x_kde), "r-", lw=2, label="KDE")
    except Exception:
        pass
    # 分位线标注
    ylim = ax.get_ylim()
    for qv_pct, lbl, clr in [(25, "Q1", "green"), (50, "Median", "orange"),
                               (75, "Q3", "red")]:
        qv_val = np.percentile(log_dy, qv_pct)
        ax.axvline(qv_val, color=clr, ls="--", lw=1.2, alpha=0.7)
        ax.text(qv_val, ylim[1] * 0.88, f"{lbl}={10**qv_val:.1f}",
                color=clr, fontsize=8, ha="center")
    ax.set_xlabel(r"$\log_{10}(\Delta y)$")
    ax.set_ylabel("Probability Density")
    ax.set_title(fr"$\log_{{10}}(\Delta y)$ Distribution  (n={n:,})")
    ax.legend(fontsize=8)

    # ── 图 2: Δy vs Cardano Δ ──
    ax = axes[0, 1]
    # Cardano 判别式 (d=0)
    cdisc = p[:, 1]**2 * p[:, 2]**2 - 4.0 * p[:, 0] * p[:, 2]**3
    cdisc_abs = np.abs(cdisc) + 1e-30  # 避免 log(0)
    sc = ax.scatter(cdisc_abs[idx], dy[idx], c=D_crit[idx],
                    s=5, alpha=0.5, cmap="viridis",
                    norm=matplotlib.colors.LogNorm())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|{\rm Cardano}\ \Delta|$ (log scale)")
    ax.set_ylabel(r"$\Delta y$ (log scale)")
    ax.set_title(r"$\Delta y$ vs Cardano Discriminant")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r"$D_{\rm crit}$", fontsize=8)

    # ── 图 3: (a, b) hexbin ──
    ax = axes[0, 2]
    lo_a, hi_a = np.percentile(p[:, 0], [1, 99])
    lo_b, hi_b = np.percentile(p[:, 1], [1, 99])
    mask_ab = ((p[:, 0] >= lo_a) & (p[:, 0] <= hi_a) &
               (p[:, 1] >= lo_b) & (p[:, 1] <= hi_b))
    hb = ax.hexbin(p[mask_ab, 0], p[mask_ab, 1], C=log_dy[mask_ab],
                   gridsize=50, cmap="plasma", reduce_C_function=np.mean)
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_title(r"Mean $\log_{10}(\Delta y)$ on $(a, b)$ Plane")
    plt.colorbar(hb, ax=ax, label=r"Mean $\log_{10}(\Delta y)$")

    # ── 图 4: Δy vs a (symlog) ──
    ax = axes[1, 0]
    ax.scatter(p[idx, 0], dy[idx], s=5, alpha=0.4,
               c="coral", edgecolors="none")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("a (symlog scale)")
    ax.set_ylabel(r"$\Delta y$ (log scale)")
    ax.set_title(r"$\Delta y$ vs $a$")
    # 理论参考线: Δy ∝ 1/a²
    D_med = float(np.median(D_crit))
    a_th = np.logspace(-2, 2, 100)
    for sign in [1, -1]:
        ath_signed = sign * a_th
        dy_th = 4.0 * D_med**1.5 / (27.0 * ath_signed**2)
        ax.plot(ath_signed, dy_th, "b--", lw=1, alpha=0.6)
    ax.axvline(0, color="gray", ls=":", lw=1, alpha=0.5)

    # ── 图 5: Δy vs |b| ──
    ax = axes[1, 1]
    ax.scatter(np.abs(p[idx, 1]), dy[idx], s=5, alpha=0.4,
               c="mediumseagreen", edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|b|$ (log scale)")
    ax.set_ylabel(r"$\Delta y$ (log scale)")
    ax.set_title(r"$\Delta y$ vs $|b|$")
    # 理论参考线: Δy = 4|b|³/(27a²) 当 c=0
    b_th = np.logspace(0, 2.5, 100)
    dy_b_th = 4.0 * b_th**3 / 27.0
    ax.plot(b_th, dy_b_th, "b--", lw=1, alpha=0.6,
            label=r"$\sim|b|^3$ ($c=0, a=1$)")
    ax.legend(fontsize=8)

    # ── 图 6: Δy vs D_crit ──
    ax = axes[1, 2]
    ax.scatter(D_crit[idx], dy[idx], s=5, alpha=0.4,
               c="purple", edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$D_{\rm crit} = b^2 - 3ac$ (log scale)")
    ax.set_ylabel(r"$\Delta y$ (log scale)")
    ax.set_title(r"$\Delta y$ vs $D_{\rm crit}$")
    # 理论参考线: Δy ∝ D^(3/2)
    D_th = np.logspace(0, 5, 100)
    a_med = float(np.median(abs_a))
    dy_D_th = 4.0 * D_th**1.5 / (27.0 * a_med**2)
    ax.plot(D_th, dy_D_th, "b--", lw=1, alpha=0.6,
            label=fr"$\sim D^{{3/2}}$ ($|a|={a_med:.1f}$)")
    ax.legend(fontsize=8)

    # ── 保存 ──
    plt.tight_layout(pad=2.0)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved → {save_path}")


# ============================================================================
# 三阶段优化搜索
# ============================================================================


MIN_ABS_A = 1e-4  # |a| 下界, 避免 a→0 产生无意义的极端 Δy


def optimize_search(range_r=200.0, seed=42):
    """
    三阶段优化寻找最大 Δy (|a| 有下界约束避免退化)

    阶段 1: 粗网格 12×9×9 全局搜索
    阶段 2: Top-20 随机爬山 (各 200 步)
    阶段 3: 冠军邻域精细网格 20×20×20

    返回:
        champion: (a, b, c, D_crit, Δy) 全局最优
        top20: list of (a, b, c, Δy) 排序后的前 20
    """
    rng = np.random.RandomState(seed)

    # ── 阶段 1: 粗网格 ──
    print("\n  --- Stage 1: Coarse Grid (12×9×9) ---")

    # a: 对数间隔 (6 负 + 6 正 = 12)
    a_pos = np.logspace(-2.0, np.log10(range_r), 6)
    a_grid = np.concatenate([-a_pos[::-1], a_pos])
    b_grid = np.linspace(-range_r, range_r, 9)
    c_grid = np.linspace(-range_r, range_r, 9)

    candidates = []
    for a in a_grid:
        if abs(a) < MIN_ABS_A:      # 跳过 |a| 过小
            continue
        a2 = a * a
        for b in b_grid:
            b2 = b * b
            for c in c_grid:
                D = b2 - 3.0 * a * c
                if D <= 0.0:
                    continue
                dy = 4.0 * D**1.5 / (27.0 * a2)
                candidates.append((a, b, c, D, dy))

    candidates.sort(key=lambda x: x[4], reverse=True)
    print(f"    Grid points: {len(a_grid)*len(b_grid)*len(c_grid)}, "
          f"valid: {len(candidates)}")
    print(f"    Best Δy = {candidates[0][4]:.2f}  "
          f"(a={candidates[0][0]:.6f}, b={candidates[0][1]:.1f}, "
          f"c={candidates[0][2]:.1f})")

    # ── 阶段 2: 随机爬山 ──
    print("\n  --- Stage 2: Hill Climbing (Top 20, 200 steps) ---")

    top20_seeds = candidates[:20]
    hill_results = []

    for a0, b0, c0, D0, dy0 in top20_seeds:
        a_c, b_c, c_c = a0, b0, c0
        dy_c = dy0

        for _ in range(200):
            # 相对扰动 a (因为小 |a| 对 Δy 敏感)
            a_p = a_c * (1.0 + rng.uniform(-0.5, 0.5))
            if abs(a_p) < MIN_ABS_A or abs(a_p) > range_r:
                continue
            b_p = float(np.clip(b_c + rng.uniform(-20.0, 20.0),
                                -range_r, range_r))
            c_p = float(np.clip(c_c + rng.uniform(-20.0, 20.0),
                                -range_r, range_r))

            D_p = b_p * b_p - 3.0 * a_p * c_p
            if D_p <= 0.0:
                continue
            dy_p = 4.0 * D_p**1.5 / (27.0 * a_p * a_p)

            if dy_p > dy_c:
                a_c, b_c, c_c = a_p, b_p, c_p
                dy_c = dy_p

        hill_results.append((a_c, b_c, c_c, dy_c))

    hill_results.sort(key=lambda x: x[3], reverse=True)
    print(f"    Best Δy = {hill_results[0][3]:.2f}  "
          f"(a={hill_results[0][0]:.8f}, b={hill_results[0][1]:.4f}, "
          f"c={hill_results[0][2]:.4f})")

    # ── 阶段 3: 精细网格 ──
    print("\n  --- Stage 3: Fine Grid around Champion (20×20×20) ---")

    a_ch, b_ch, c_ch = hill_results[0][0], hill_results[0][1], hill_results[0][2]
    a_delta = max(abs(a_ch) * 0.5, 1e-4)
    b_delta = 10.0
    c_delta = 10.0

    a_f = np.linspace(a_ch - a_delta, a_ch + a_delta, 20)
    b_f = np.linspace(b_ch - b_delta, b_ch + b_delta, 20)
    c_f = np.linspace(c_ch - c_delta, c_ch + c_delta, 20)

    fine_best = None
    fine_best_dy = -1.0

    for a in a_f:
        if abs(a) < MIN_ABS_A:
            continue
        a2 = a * a
        for b in b_f:
            b2 = b * b
            for c in c_f:
                D = b2 - 3.0 * a * c
                if D <= 0.0:
                    continue
                dy = 4.0 * D**1.5 / (27.0 * a2)
                if dy > fine_best_dy:
                    fine_best_dy = dy
                    fine_best = (a, b, c, D, dy)

    print(f"    Best Δy = {fine_best_dy:.2f}  "
          f"(a={fine_best[0]:.10f}, b={fine_best[1]:.6f}, "
          f"c={fine_best[2]:.6f})")

    # 汇集 top-20
    combined = [(r[0], r[1], r[2], r[3]) for r in hill_results]
    combined.append((fine_best[0], fine_best[1], fine_best[2], fine_best[4]))
    combined.sort(key=lambda x: x[3], reverse=True)
    top20 = combined[:20]

    return fine_best, top20


# ============================================================================
# 主流程
# ============================================================================


def main():
    print("=" * 70)
    print("  Cubic Extremum Gap (Δy) Numerical Simulation")
    print("  f(x) = a x^3 + b x^2 + c x + d")
    print("=" * 70)

    # ── [1] 采样 ──
    print("\n[1/4] Sampling parameters (uniform, n=30,000, range=[-200,200]) ...")
    params, dy_values = sample_params(n=30000, range_r=200.0,
                                      method="uniform", seed=42)
    n_valid = len(dy_values)
    print(f"      Valid samples (D_crit > 0): {n_valid:,} / 30,000 "
          f"({100*n_valid/30000:.1f}%)")

    # ── [2] 统计分析 ──
    print("\n[2/4] Statistical analysis ...")
    stats_dict = analyze(params, dy_values)

    print("\n" + "-" * 70)
    print("  STATISTICAL SUMMARY")
    print("-" * 70)
    print(f"  Valid samples:         {stats_dict['n_valid']:>14,}")
    print(f"  Mean Δy:               {stats_dict['mean']:>14.2f}")
    print(f"  Median Δy:             {stats_dict['median']:>14.2f}")
    print(f"  Std Dev:               {stats_dict['std']:>14.2f}")
    print(f"  Min / Max:             {stats_dict['min']:>14.4f}  /  "
          f"{stats_dict['max']:>.2f}")
    print(f"  Skewness:              {stats_dict['skewness']:>14.2f}")

    print("\n  Percentiles:")
    for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        key = f"q{q:02d}"
        print(f"    {q:>2d}%  {stats_dict[key]:>18.2f}")

    print("\n  Spearman Rank Correlations  [log10(Δy) vs]:")
    for name in ["|a|", "|b|", "|c|", "D_crit"]:
        key = f"spearman_{name}"
        val = stats_dict.get(key, float("nan"))
        if not np.isnan(val):
            bar = "#" * int(abs(val) * 20)
            print(f"    {name:>10s}  {val:>+8.4f}  {bar}")

    # ── [3] 可视化 ──
    print("\n[3/4] Plotting 2×3 analysis figure ...")
    save_png = "/Users/xfpan/claude/research/cubic_gap_analysis.png"
    plot_analysis(params, dy_values, save_path=save_png, seed=42)

    # ── [4] 优化搜索 ──
    print("\n[4/4] Three-stage optimization search")
    champion, top20 = optimize_search(range_r=200.0, seed=42)

    print("\n" + "-" * 70)
    print("  TOP 10 PARAMETER COMBINATIONS  (largest Δy)")
    print("-" * 70)
    print(f"  {'Rank':<6} {'a':>14} {'b':>12} {'c':>12} "
          f"{'Δy':>18} {'D_crit':>14}")
    print("  " + "-" * 76)

    # 验证每组的 D_crit
    for i, (a_i, b_i, c_i, dy_i) in enumerate(top20[:10]):
        D_i = b_i * b_i - 3.0 * a_i * c_i
        print(f"  {i+1:<6} {a_i:>14.8f} {b_i:>12.4f} {c_i:>12.4f} "
              f"{dy_i:>18.2f} {D_i:>14.2f}")

    print("\n" + "=" * 70)
    print("  SIMULATION COMPLETE")
    print("  Output: /Users/xfpan/claude/research/cubic_gap_analysis.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
