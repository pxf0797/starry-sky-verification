#!/usr/bin/env python3
"""
星空策略模拟验证 — 综合可视化生成脚本
基于 deep-research-report-v3.md §6 的 10 项实验数据
生成 8 张 PNG 图表到 charts/ 目录
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.patches import FancyBboxPatch
import os

OUT = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(OUT, exist_ok=True)

# 注册 CJK 字体 (macOS)
_cjk_fonts = [
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
]
for _fp in _cjk_fonts:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Heiti SC", "Songti SC", "DejaVu Sans", "Arial"],
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

STYLE = {
    "blue":   "#2171b5",
    "red":    "#cb181d",
    "green":  "#238b45",
    "orange": "#e6550d",
    "purple": "#6a51a3",
    "grey":   "#636363",
    "bg":     "#f7f7f7",
}


# ═══════════════════════════════════════════════════════════
# 图 1: 实验 1.1 — 三次最优阶数验证
# ═══════════════════════════════════════════════════════════
def fig1_optimal_order():
    sigma_levels = ["0.001", "0.01", "0.05", "0.1", "0.2"]
    rmse = {
        "k=1": [9245, 9245, 9245, 9245, 9245],
        "k=2": [3393, 3393, 3393, 3393, 3394],
        "k=3": [0.0045, 0.0418, 0.2108, 0.4405, 0.8475],
        "k=4": [0.0138, 0.1408, 0.7355, 1.3894, 2.8793],
        "k=5": [0.0474, 0.4802, 2.4824, 4.8473, 10.1679],
    }
    colors = [STYLE["grey"], STYLE["grey"], STYLE["blue"], STYLE["orange"], STYLE["red"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左: 全部阶数 (线性)
    x = np.arange(len(sigma_levels))
    w = 0.15
    for i, (label, vals) in enumerate(rmse.items()):
        axes[0].bar(x + i * w, vals, w, label=label, color=colors[i], edgecolor="white", linewidth=0.5)

    axes[0].set_xticks(x + 2 * w)
    axes[0].set_xticklabels([f"σ={s}" for s in sigma_levels])
    axes[0].set_ylabel("外推 RMSE")
    axes[0].set_title("全部阶数 RMSE 对比")
    axes[0].legend(fontsize=9, ncols=3)
    axes[0].grid(axis="y", alpha=0.3)

    # 右: k=3,4,5 放大 (对数)
    for i, (label, vals) in enumerate(list(rmse.items())[2:]):
        axes[1].plot(sigma_levels, vals, "o-", label=label, color=colors[i + 2], linewidth=2, markersize=8)

    axes[1].set_yscale("log")
    axes[1].set_xlabel("噪声水平 σ")
    axes[1].set_ylabel("外推 RMSE (对数)")
    axes[1].set_title("k=3 vs k=4 vs k=5 (对数 Y 轴)")
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    # 标注倍率
    for i_s, s in enumerate(sigma_levels):
        r4 = rmse["k=4"][i_s] / rmse["k=3"][i_s]
        r5 = rmse["k=5"][i_s] / rmse["k=3"][i_s]
        axes[1].annotate(f"R₄={r4:.1f}×\nR₅={r5:.1f}×", (s, rmse["k=5"][i_s]),
                         textcoords="offset points", xytext=(12, -5), fontsize=7, color=STYLE["grey"])

    fig.suptitle("实验 1.1 — 三次最优阶数验证 (所有 σ 下 k=3 严格最优)", fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUT}/exp1_1_optimal_order.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# 图 2: 实验 1.2 — 维度鸿沟实证检验
# ═══════════════════════════════════════════════════════════
def fig2_dimension_gap():
    var_levels = ["10", "100", "1000"]
    ratios = [2.3, 6.2, 19.3]
    v2_expected = [5, 5, 5]  # v2 预期上限 5×

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(var_levels, ratios, color=[STYLE["blue"], STYLE["orange"], STYLE["red"]],
                  edgecolor="white", linewidth=1.5, width=0.5)

    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                f"{ratio}×", ha="center", fontsize=16, fontweight="bold", color=bar.get_facecolor())

    ax.axhline(y=5, color=STYLE["grey"], linestyle="--", linewidth=2, alpha=0.7, label="v2 预期上限 (5×)")
    ax.axhline(y=1, color="black", linestyle=":", linewidth=1, alpha=0.4, label="P 空间基准 (1×)")

    # 添加增长曲线
    x_smooth = np.linspace(0, 2, 100)
    ax.plot(x_smooth, 2.3 * np.exp(1.06 * np.array(x_smooth)), color=STYLE["red"], alpha=0.2, linewidth=6)

    ax.set_xlabel("开仓价格方差 σ²_P")
    ax.set_ylabel("RMSE_Z / RMSE_P (Z 空间误差倍数)")
    ax.set_title("实验 1.2 — 维度鸿沟实证: Z 空间 vs P 空间", fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # 含义标注
    annotations = ["低波动市场\n(Z 空间代价 2.3×)", "中等波动\n(代价急剧扩大 6.2×)", "高波动/跨品种\n(Z 空间预测几乎无意义 19.3×)"]
    for i, (bar, ann) in enumerate(zip(bars, annotations)):
        ax.annotate(ann, (bar.get_x() + bar.get_width() / 2, bar.get_height() - 4),
                    ha="center", fontsize=9, color="white", fontweight="bold")

    fig.tight_layout()
    fig.savefig(f"{OUT}/exp1_2_dimension_gap.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# 图 3: 实验 1.3 — 高阶信噪比衰减
# ═══════════════════════════════════════════════════════════
def fig3_snr_decay():
    noise_levels = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5])
    snr = {
        "d=1  (一阶导数)": np.array([6.9e7, 2.8e6, 6.9e5, 2.8e4, 6.9e3, 1.7e3, 2.8e2]),
        "d=2  (二阶导数)": np.array([1.1e3, 43, 10.8, 0.43, 0.11, 0.027, 4.3e-3]),
        "d=3  (三阶导数)": np.array([0.23, 9.2e-3, 2.3e-3, 9.2e-5, 2.3e-5, 5.8e-6, 9.2e-7]),
        "d=4  (四阶导数)": np.zeros(len(noise_levels)) + 1e-10,
    }
    colors = [STYLE["blue"], STYLE["green"], STYLE["orange"], STYLE["red"]]

    fig, ax = plt.subplots(figsize=(11, 7))

    for (label, vals), c in zip(snr.items(), colors):
        ls = "-" if "d=4" not in label else "--"
        ax.semilogy(noise_levels, np.maximum(vals, 1e-10), "o-", label=label, color=c, linewidth=2.5, markersize=9, linestyle=ls)

    ax.axhline(y=1, color="black", linestyle=":", linewidth=1.5, alpha=0.5)
    ax.annotate("SNR = 1 (可用性阈值)", (0.001, 1), textcoords="data", fontsize=9, color=STYLE["grey"],
                va="bottom", ha="left")

    # 可用性标注
    ax.fill_between([0.001, 0.5], 1, 1e10, alpha=0.05, color=STYLE["green"])
    ax.fill_between([0.001, 0.5], 1e-10, 1, alpha=0.05, color=STYLE["red"])
    ax.text(0.25, 3e8, "可用区 (SNR > 1)", fontsize=10, color=STYLE["green"], fontweight="bold", alpha=0.6)
    ax.text(0.25, 3e-2, "不可用区 (SNR < 1)", fontsize=10, color=STYLE["red"], fontweight="bold", alpha=0.6)

    ax.set_xlabel("噪声水平 σ")
    ax.set_ylabel("信噪比 SNR (对数)")
    ax.set_title("实验 1.3 — 高阶信噪比 (SNR) 指数衰减", fontweight="bold")
    ax.legend(fontsize=11, loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_xlim(0.0005, 0.6)

    fig.tight_layout()
    fig.savefig(f"{OUT}/exp1_3_snr_decay.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# 图 4: 实验 2.1 — 不重算 vs 重算 (分区)
# ═══════════════════════════════════════════════════════════
def fig4_frozen_vs_recalc():
    zones = ["Zone1 [0, τ)", "Zone2 [τ, 2τ)", "Zone3 [2τ, ∞)"]
    frozen = [-1.63, -2.16, 0.43]
    recalc = [0.01, 0.08, -0.02]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(zones))
    w = 0.32

    bars_f = ax.bar(x - w / 2, frozen, w, label="冻结模型 (Frozen)", color=STYLE["blue"], edgecolor="white", linewidth=1.2)
    bars_r = ax.bar(x + w / 2, recalc, w, label="重算模型 (Recalc)", color=STYLE["orange"], edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars_f, frozen):
        y_pos = val + 0.2 if val >= 0 else val - 0.35
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{val:+.2f}", ha="center", fontsize=12, fontweight="bold",
                color=STYLE["blue"])
    for bar, val in zip(bars_r, recalc):
        y_pos = val + 0.2 if val >= 0 else val - 0.35
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{val:+.2f}", ha="center", fontsize=12, fontweight="bold",
                color=STYLE["orange"])

    # 优胜标注
    winners = ["重算略优", "重算略优", "冻结反转!"]
    for i, (x_i, w_text) in enumerate(zip(x, winners)):
        ax.annotate(w_text, (x_i, 1.2), ha="center", fontsize=10,
                    color=STYLE["red"] if "反转" in w_text else STYLE["grey"],
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.axhline(y=0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(zones, fontsize=12)
    ax.set_ylabel("夏普比 (Sharpe Ratio)")
    ax.set_title("实验 2.1 — 不重算 vs 重算: 三区间结构性反转", fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    # 95% CI 标注
    ax.annotate("95% Bootstrap CI: [−2.39, 0.76] (含零 → 总体不显著)",
                (1, -2.8), ha="center", fontsize=10, color=STYLE["grey"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3cd", alpha=0.9))

    fig.tight_layout()
    fig.savefig(f"{OUT}/exp2_1_frozen_vs_recalc.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# 图 5: 实验 2.2 — 半衰期最优值搜索 ★
# ═══════════════════════════════════════════════════════════
def fig5_tau_loss():
    tau = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30, 60])
    loss = np.array([6.96, 6.91, 6.64, 5.20, 8.19, 94.2, 1315.9, 7156.5, 106753, 5210942])
    tau_star = 4.7
    ci_low, ci_high = 4.5, 4.8

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左: U 型曲线 (全范围, 对数 Y)
    axes[0].semilogy(tau, loss, "o-", color=STYLE["blue"], linewidth=2.5, markersize=10, markerfacecolor="white",
                     markeredgewidth=2)
    axes[0].axvline(x=tau_star, color=STYLE["red"], linestyle="--", linewidth=2, alpha=0.8, label=f"τ* = {tau_star} min")
    axes[0].axvspan(ci_low, ci_high, alpha=0.12, color=STYLE["red"], label=f"95% CI [{ci_low}, {ci_high}]")
    axes[0].axvline(x=15, color=STYLE["orange"], linestyle=":", linewidth=2, alpha=0.8, label="v2 默认 τ=15")
    axes[0].annotate(f"τ=15: 1,316\n(253× 最优值!)", (15, 1315.9), textcoords="offset points",
                     xytext=(20, -30), fontsize=9, color=STYLE["red"], fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=STYLE["red"], alpha=0.6))

    axes[0].set_xlabel("半衰期 τ (分钟)")
    axes[0].set_ylabel("损失函数 L(τ) = B² + V (对数)")
    axes[0].set_title("全范围 — L(τ) 指数爆炸 (τ>10)")
    axes[0].legend(fontsize=9, loc="upper left")
    axes[0].grid(alpha=0.3)

    # 右: U 型底部放大 (线性 Y)
    mask = tau <= 10
    axes[1].plot(tau[mask], loss[mask], "o-", color=STYLE["blue"], linewidth=2.5, markersize=10, markerfacecolor="white",
                 markeredgewidth=2)
    axes[1].axvline(x=tau_star, color=STYLE["red"], linestyle="--", linewidth=2, alpha=0.8)
    axes[1].axvspan(ci_low, ci_high, alpha=0.12, color=STYLE["red"])
    axes[1].annotate(f"τ*={tau_star}\nCI[{ci_low},{ci_high}]", (tau_star, 5.20),
                     textcoords="offset points", xytext=(25, -30), fontsize=10, color=STYLE["red"], fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=STYLE["red"], alpha=0.6))
    axes[1].annotate("L(5) = 5.20\n(近最优)", (5, 5.20), textcoords="offset points",
                     xytext=(-30, -40), fontsize=9, color=STYLE["green"], fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=STYLE["green"], alpha=0.6))

    axes[1].set_xlabel("半衰期 τ (分钟)")
    axes[1].set_ylabel("损失函数 L(τ) = B² + V (线性)")
    axes[1].set_title("U 型底部放大 — τ∈[1,10]")
    axes[1].grid(alpha=0.3)

    fig.suptitle("实验 2.2 ★ — 半衰期最优值搜索 (τ* = 4.7 min, v2 的 τ=15 被证伪)", fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUT}/exp2_2_tau_loss_curve.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# 图 6: 实验 2.5 — 动态 vs 静态阈值 (帕累托前沿)
# ═══════════════════════════════════════════════════════════
def fig6_dynamic_vs_static():
    np.random.seed(42)
    sigma0_vals = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0])

    dynamic_sharpe = np.array([0.96, 0.89, 0.82, 0.95, 1.08, 1.15, 1.19])
    dynamic_dd = np.array([-0.12, -0.10, -0.09, -0.13, -0.15, -0.22, -0.28])
    static_sharpe = np.array([-0.31, -0.18, -0.05, 0.04, 0.16, 0.24, 0.30])
    static_dd = np.array([-0.45, -0.38, -0.32, -0.28, -0.25, -0.30, -0.35])

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.scatter(static_dd, static_sharpe, c=STYLE["grey"], s=140, marker="s", edgecolors="white",
               linewidth=1.2, zorder=5, label="静态阈值 (各 σ₀)")
    ax.scatter(dynamic_dd, dynamic_sharpe, c=STYLE["blue"], s=160, marker="o", edgecolors="white",
               linewidth=1.2, zorder=6, label="动态阈值 (各 σ₀)")

    for i, s0 in enumerate(sigma0_vals):
        ax.annotate(f"σ₀={s0}", (static_dd[i] + 0.01, static_sharpe[i] + 0.02), fontsize=8, color=STYLE["grey"], alpha=0.8)
        ax.annotate(f"σ₀={s0}", (dynamic_dd[i] + 0.01, dynamic_sharpe[i] + 0.02), fontsize=8, color=STYLE["blue"], alpha=0.8)

    # Pareto 前沿 (动态)
    pareto_idx = [0, 2, 3, 5, 6]
    ax.plot(dynamic_dd[pareto_idx], dynamic_sharpe[pareto_idx], "--", color=STYLE["blue"], alpha=0.5, linewidth=2,
            label="动态 Pareto 前沿")

    # 优势标注
    ax.annotate("", xy=(dynamic_dd[-1], dynamic_sharpe[-1]), xytext=(static_dd[-1], static_sharpe[-1]),
                arrowprops=dict(arrowstyle="<->", color=STYLE["red"], linewidth=2.5, alpha=0.7))
    ax.annotate(f"4× Sharpe\n优势!", ((dynamic_dd[-1] + static_dd[-1]) / 2 - 0.02,
                                      (dynamic_sharpe[-1] + static_sharpe[-1]) / 2),
                ha="center", fontsize=12, color=STYLE["red"], fontweight="bold")

    # 区域标注
    ax.fill_between([-0.5, 0], -1, 0, alpha=0.04, color=STYLE["red"])
    ax.fill_between([-0.5, 0], 0, 2, alpha=0.04, color=STYLE["green"])
    ax.text(-0.15, 0.7, "理想区\n(正Sharpe, 低回撤)", fontsize=9, color=STYLE["green"], fontweight="bold", alpha=0.6)
    ax.text(-0.42, -0.6, "危险区\n(负Sharpe)", fontsize=9, color=STYLE["red"], fontweight="bold", alpha=0.6)

    ax.axhline(y=0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("最大回撤 (Max Drawdown)")
    ax.set_ylabel("夏普比 (Sharpe Ratio)")
    ax.set_title("实验 2.5 ★ — 动态 vs 静态阈值: 帕累托前沿对比", fontweight="bold")
    ax.legend(fontsize=11, loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.48, -0.05)
    ax.set_ylim(-0.5, 1.4)

    fig.tight_layout()
    fig.savefig(f"{OUT}/exp2_5_dynamic_vs_static_pareto.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# 图 7: 实验 3.2 — 结构性断点响应
# ═══════════════════════════════════════════════════════════
def fig7_break_response():
    deltas = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    trigger_mean = np.array([9.3, 7.7, 8.0, 4.0, 0.0])
    trigger_median = np.array([7.5, 4.0, 9.0, 4.0, 0.0])
    counts = np.array([12, 7, 5, 2, 4])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左: 触发时间 vs 断点幅度
    axes[0].scatter(deltas, trigger_mean, s=counts * 40, c=STYLE["blue"], alpha=0.7, edgecolors="white",
                    linewidth=1.2, zorder=5, label="平均触发时间")

    # 趋势线
    z = np.polyfit(deltas, trigger_mean, 1)
    x_fit = np.linspace(0, 11, 100)
    axes[0].plot(x_fit, np.polyval(z, x_fit), "--", color=STYLE["red"], linewidth=2, alpha=0.7,
                 label=f"线性趋势 (斜率={z[0]:.1f})")

    for d, t, cnt in zip(deltas, trigger_mean, counts):
        axes[0].annotate(f"{t}步\n({cnt}次)", (d, t), textcoords="offset points",
                         xytext=(10, 5), fontsize=8, color=STYLE["grey"])

    axes[0].set_xlabel("断点幅度 δ (%)")
    axes[0].set_ylabel("平均触发时间 (步)")
    axes[0].set_title("触发时间 vs 断点幅度 (气泡=触发次数)")
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    # 标注瞬时响应
    axes[0].annotate("10% 断点\n瞬时响应!", (10, 0), textcoords="offset points",
                     xytext=(-60, -50), fontsize=10, color=STYLE["red"], fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=STYLE["red"], linewidth=1.5),
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3cd", alpha=0.9))

    # 右: 触发率分解
    categories = ["总体断点\n触发率", "不利断点\n正确触发", "有利断点\n误触发", "未触发"]
    values = [6.0, 7.0, 4.9, 94.0]
    pie_colors = [STYLE["blue"], STYLE["green"], STYLE["orange"], STYLE["grey"]]

    wedges, texts, autotexts = axes[1].pie(
        values, labels=categories, autopct="%1.1f%%", colors=pie_colors,
        explode=(0.05, 0.05, 0.05, 0), startangle=90, textprops={"fontsize": 10}
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_fontsize(10)

    axes[1].set_title("断点触发事件分解", fontweight="bold")

    # 结论标注
    fig.text(0.5, 0.02, "dT_trigger/dδ = −97.4 < 0 ✓ (严格单调) | 10% 断点 0 步瞬时触发 | 误触发率 4.9% 可接受",
             ha="center", fontsize=10, color=STYLE["grey"],
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#d4edda", alpha=0.8))

    fig.suptitle("实验 3.2 — 结构性断点响应测试", fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(f"{OUT}/exp3_2_break_response.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# 图 8: 综合验证矩阵热力图
# ═══════════════════════════════════════════════════════════
def fig8_summary_matrix():
    claims = [
        "维度鸿沟 (P₀不可恢复)",
        "不重算 优于重算",
        "三次最低次数",
        "三次上界 (k>3过拟合)",
        "τ = 15 min",
        "ρ ≈ 0.5 (SNR衰减)",
        "SNR 指数衰减 (定性)",
        "SIGMOID 优于硬阈值",
        "动态阈值 优于静态",
        "SVD/polyfit 稳定性",
        "铁律系统 断点响应",
        "铁律系统 微观鲁棒性",
        "判别式-策略映射",
        "EMA平滑 假阳性降低",
        "SIGMOID k 收敛性",
    ]

    # 实验类型
    exp_types = ["层次一\n数学", "层次一\n数学", "层次一\n数学", "层次一\n数学",
                 "层次二\n统计", "层次一\n数学", "层次一\n数学", "层次二\n统计",
                 "层次二\n统计", "层次一\n数学", "层次三\n实战", "层次三\n实战",
                 "层次二\n统计", "层次二\n统计", "层次二\n统计"]

    # 判定: ✅=2, 🟡=1, 🔴=0, ⏳=-1
    # v3状态
    status_scores = [2, 1, 2, 2, 0, 0, 2, 1, 2, 2, 2, 1, -1, -1, -1]
    status_labels = ["OK", "~", "OK", "OK", "XX", "XX", "OK", "~", "OK", "OK", "OK", "~", "--", "--", "--"]

    # 置信度
    confidence = [3, 1, 3, 3, 3, 3, 3, 1, 3, 3, 3, 2, 0, 0, 0]  # 0=无,1=低,2=中,3=高
    conf_labels = ["高", "低", "高", "高", "高", "高", "高", "低", "高", "高", "高", "中", "—", "—", "—"]

    # 夏普比影响 (估计)
    sharpe_impact = [10, 0, 8, 8, 10, 3, 7, 1, 9, 3, 6, 4, 5, 2, 2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # 左: 验证状态矩阵
    data_status = np.array(status_scores).reshape(15, 1)
    im = axes[0].imshow(data_status, cmap=plt.cm.RdYlGn, aspect="auto", vmin=-1, vmax=2)
    axes[0].set_yticks(range(len(claims)))
    axes[0].set_yticklabels([f"{c[:18]}..." if len(c) > 18 else c for c in claims], fontsize=8)
    axes[0].set_xticks([])
    axes[0].set_title("验证状态", fontweight="bold")
    for i, (score, label) in enumerate(zip(status_scores, status_labels)):
        color = "white" if score == 2 else "black"
        axes[0].text(0, i, label, ha="center", va="center", fontsize=12, color=color, fontweight="bold")

    # 中: 置信度
    data_conf = np.array(confidence).reshape(15, 1)
    axes[1].imshow(data_conf, cmap=plt.cm.Blues, aspect="auto", vmin=0, vmax=3)
    axes[1].set_yticks(range(len(claims)))
    axes[1].set_yticklabels([])
    axes[1].set_xticks([])
    axes[1].set_title("置信度", fontweight="bold")
    for i, cl in enumerate(conf_labels):
        axes[1].text(0, i, cl, ha="center", va="center", fontsize=10, color="white" if confidence[i] >= 2 else "black",
                     fontweight="bold")

    # 右: 策略影响度
    data_impact = np.array(sharpe_impact).reshape(15, 1)
    axes[2].imshow(data_impact, cmap=plt.cm.OrRd, aspect="auto", vmin=0, vmax=10)
    axes[2].set_yticks(range(len(claims)))
    axes[2].set_yticklabels([])
    axes[2].set_xticks([])
    axes[2].set_title("策略影响度 (0-10)", fontweight="bold")
    for i, si in enumerate(sharpe_impact):
        axes[2].text(0, i, str(si), ha="center", va="center", fontsize=10, color="white" if si >= 5 else "black",
                     fontweight="bold")

    # 分隔线
    for ax in axes:
        for y in [3.5, 7.5, 9.5, 11.5]:
            ax.axhline(y=y, color="white", linewidth=1.5)

    # 层级标注
    fig.text(0.07, 0.92, "层次一\n数学验证", fontsize=8, fontweight="bold", color=STYLE["blue"], ha="center")
    fig.text(0.07, 0.68, "层次二\n统计验证", fontsize=8, fontweight="bold", color=STYLE["orange"], ha="center")
    fig.text(0.07, 0.52, "层次三\n实战模拟", fontsize=8, fontweight="bold", color=STYLE["green"], ha="center")
    fig.text(0.07, 0.18, "未完成\n实验", fontsize=8, fontweight="bold", color=STYLE["red"], ha="center")

    fig.suptitle("综合验证矩阵 — v2 假设验证状态全景", fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUT}/summary_verification_matrix.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# 附加图: 实验 3.1 — 市场微观结构摩擦分析
# ═══════════════════════════════════════════════════════════
def fig_appendix_microstructure():
    metrics = ["理想 P&L", "实际 P&L\n(含摩擦)", "限价单\n成交率", "铁律三\n触发比例"]
    values = [0.007, -0.025, 47.2, 16.0]
    colors_bar = [STYLE["green"], STYLE["red"], STYLE["orange"], STYLE["blue"]]
    y_labels = ["%", "%", "%", "%"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左: 关键指标柱状图
    bars = axes[0].bar(metrics, values, color=colors_bar, edgecolor="white", linewidth=1.2, width=0.5)
    axes[0].axhline(y=0, color="black", linewidth=1, alpha=0.5)
    for bar, val in zip(bars, values):
        y_pos = val + 1.5 if val >= 0 else val - 2.5
        axes[0].text(bar.get_x() + bar.get_width() / 2, y_pos, f"{val:+.3f}%" if abs(val) < 1 else f"{val:.1f}%",
                     ha="center", fontsize=13, fontweight="bold", color=bar.get_facecolor())

    axes[0].set_ylabel("百分比")
    axes[0].set_title("关键性能指标", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    # 右: 摩擦成本分解 (瀑布图风格)
    stages = ["理论 P&L\n(无摩擦)", "买卖价差\n成本", "滑点\n成本", "未成交\n等待成本", "实际 P&L"]
    stage_vals = [0.007, -0.008, -0.012, -0.012, -0.025]
    cumsum = np.cumsum(stage_vals)
    colors_wf = [STYLE["green"]] + [STYLE["red"]] * 3 + [STYLE["orange"]]

    for i in range(len(stages)):
        if i == 0:
            axes[1].bar(i, stage_vals[i], color=colors_wf[i], edgecolor="white", linewidth=1, width=0.5)
        else:
            axes[1].bar(i, stage_vals[i], bottom=cumsum[i - 1] if i > 0 else 0, color=colors_wf[i],
                        edgecolor="white", linewidth=1, width=0.5)

    axes[1].axhline(y=0, color="black", linewidth=1, alpha=0.5)
    axes[1].set_xticks(range(len(stages)))
    axes[1].set_xticklabels(stages, fontsize=9)
    axes[1].set_ylabel("P&L (%)")
    axes[1].set_title("摩擦成本瀑布分解", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("实验 3.1 — 市场微观结构压力测试: 策略处于盈亏临界状态", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/exp3_1_microstructure.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("生成可视化图表...")
    fig1_optimal_order()
    print("  [1/9] exp1_1_optimal_order.png")
    fig2_dimension_gap()
    print("  [2/9] exp1_2_dimension_gap.png")
    fig3_snr_decay()
    print("  [3/9] exp1_3_snr_decay.png")
    fig4_frozen_vs_recalc()
    print("  [4/9] exp2_1_frozen_vs_recalc.png")
    fig5_tau_loss()
    print("  [5/9] exp2_2_tau_loss_curve.png")
    fig6_dynamic_vs_static()
    print("  [6/9] exp2_5_dynamic_vs_static_pareto.png")
    fig7_break_response()
    print("  [7/9] exp3_2_break_response.png")
    fig8_summary_matrix()
    print("  [8/9] summary_verification_matrix.png")
    fig_appendix_microstructure()
    print("  [9/9] exp3_1_microstructure.png")
    print(f"\n全部图表已生成到: {OUT}/")
