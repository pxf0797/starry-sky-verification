#!/usr/bin/env python3
"""
Layer 2 Experiments: Starry Sky Strategy (星空策略) Statistical Verification
===========================================================================
Experiments:
  2.1 – Frozen vs Recalculated Cubic Strategy Comparison
  2.2 – Half-Life (tau) Optimal Value Search
  2.3 – SIGMOID vs Hard Threshold Decision Comparison
  2.5 – Dynamic vs Static Threshold Pareto Frontier

All 4 experiments run in ~6-8 seconds total.
"""

import os, json, warnings, itertools, time
import numpy as np
from numpy.random import default_rng
from scipy.special import expit as sigmoid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

warnings.filterwarnings("ignore")

OUTDIR = "/Users/xfpan/claude/research/sim_verify/layer2"
os.makedirs(OUTDIR, exist_ok=True)
DPI = 300

# ─── Chinese font fallback ────────────────────────────────────────────────
_ZH_FONT = None
for _fp in ["/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc"]:
    if os.path.exists(_fp):
        _ZH_FONT = font_manager.FontProperties(fname=_fp)
        break
if _ZH_FONT is None:
    for _f in font_manager.findSystemFonts():
        if any(k in _f.lower() for k in ("pingfang", "heiti", "notosanscjk", "simhei", "wqy")):
            _ZH_FONT = font_manager.FontProperties(fname=_f)
            break


def zhp(size=12):
    if _ZH_FONT is not None:
        return font_manager.FontProperties(fname=_ZH_FONT.get_file(), size=size)
    return {"size": size}


RNG = default_rng(42)


# ═══════════════════════════════════════════════════════════════════════════
# 0.  Price Simulation
# ═══════════════════════════════════════════════════════════════════════════
def simulate_trajectories(n_paths=500, n_steps=200, mu=0.0, sigma=0.02,
                          theta=0.1, dt=1.0, jump_lambda=0.01,
                          jump_scale=0.02, seed=42):
    """GBM + mean reversion + Poisson jumps -> (n_paths, n_steps+1)."""
    rng = default_rng(seed)
    S = np.ones((n_paths, n_steps + 1))
    for t in range(1, n_steps + 1):
        dW = rng.normal(0, np.sqrt(dt), n_paths)
        drift = theta * (mu - np.log(S[:, t - 1])) * dt + mu * dt
        diff = sigma * S[:, t - 1] * dW
        jump = (rng.poisson(jump_lambda * dt, n_paths) > 0) * \
               rng.normal(0, jump_scale, n_paths) * S[:, t - 1]
        S[:, t] = S[:, t - 1] * np.exp(
            drift + diff / np.maximum(S[:, t - 1], 1e-12)) + jump
        S[:, t] = np.maximum(S[:, t], 1e-8)
    return S


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def compute_sharpe(ret):
    arr = np.asarray(ret, dtype=float)
    if len(arr) < 2 or np.std(arr) < 1e-12:
        return 0.0
    return float(np.mean(arr) / np.std(arr) * np.sqrt(252))


def max_drawdown(price_series):
    peak = np.maximum.accumulate(price_series)
    dd = (price_series - peak) / np.maximum(peak, 1e-12)
    return float(np.min(dd))


def freeze_predictions(S, tau, T):
    """Precompute frozen predictions for all paths."""
    n_paths, N = S.shape[0], S.shape[1] - 1
    pred = np.zeros_like(S)
    pred[:, 0] = S[:, 0]
    for i in range(n_paths):
        s = S[i, :]
        xs = T[: min(4, N + 1)]
        ys = s[: min(4, N + 1)]
        a0, b0, c0, d0 = np.polyfit(xs, ys, min(len(xs) - 1, 3))
        for t_idx in range(1, N + 1):
            decay = 2.0 ** (-t_idx / tau)
            pred[i, t_idx] = np.polyval(
                [a0 * decay, b0 * decay, c0, d0], t_idx)
    return pred


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Experiment 2.1 – Frozen vs Recalculated
# ═══════════════════════════════════════════════════════════════════════════
def exp_2_1(S, tau=15, n_bootstrap=1000):
    n_paths, N = S.shape[0], S.shape[1] - 1
    T = np.arange(N + 1, dtype=float)

    # Precompute frozen
    frozen_pred = freeze_predictions(S, tau, T)

    # Precompute recalc
    recalc_pred = np.zeros_like(S)
    recalc_pred[:, 0] = S[:, 0]
    for i in range(n_paths):
        s = S[i, :]
        for t_idx in range(1, N + 1):
            cl = np.polyfit(T[: t_idx + 1], s[: t_idx + 1],
                            min(t_idx, 3))
            recalc_pred[i, t_idx] = np.polyval(cl, t_idx)

    # Zone definitions
    zone_defs = [
        ("zone1", T < tau),
        ("zone2", (T >= tau) & (T < 2 * tau)),
        ("zone3", T >= 2 * tau),
    ]

    def zone_metrics(pred):
        out = {}
        for zn, zm in zone_defs:
            if zm.sum() < 3:
                out[zn] = {"sharpe": 0.0, "maxdd": 0.0, "cumerr": 0.0}
                continue
            all_rets, dds, err_sum = [], [], 0.0
            for i in range(n_paths):
                err = pred[i, zm] - S[i, zm]
                ret = (-np.diff(err)).tolist()
                all_rets.extend(ret)
                eq = 1.0 + np.cumsum(np.insert(ret, 0, 0))
                dds.append(max_drawdown(eq))
                err_sum += np.sum(err ** 2)
            out[zn] = {
                "sharpe": compute_sharpe(all_rets),
                "maxdd": float(np.mean(dds)),
                "cumerr": float(err_sum / n_paths),
            }
        return out

    fm = zone_metrics(frozen_pred)
    rm = zone_metrics(recalc_pred)

    # Bootstrap for Sharpe diff in zone1
    zm1 = T < tau
    fsp = np.array([
        compute_sharpe((-np.diff(frozen_pred[i, zm1] - S[i, zm1])).tolist())
        for i in range(n_paths)
    ])
    rsp = np.array([
        compute_sharpe((-np.diff(recalc_pred[i, zm1] - S[i, zm1])).tolist())
        for i in range(n_paths)
    ])
    boot_d = []
    for _ in range(n_bootstrap):
        idx = RNG.integers(0, n_paths, n_paths)
        boot_d.append(float(np.mean(fsp[idx] - rsp[idx])))
    ci_low, ci_high = np.percentile(boot_d, [2.5, 97.5])

    # Chart
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle("Experiment 2.1: Frozen vs Recalculated", fontsize=14)
    zlabs = [f"Zone 1 [0,{tau})", f"Zone 2 [{tau},{2*tau})",
             f"Zone 3 [{2*tau},N)"]
    mkeys, mlabs = ["sharpe", "maxdd", "cumerr"], \
                   ["Sharpe Ratio", "Max Drawdown", "Cumulative Error"]
    for row, (mk, ml) in enumerate(zip(mkeys, mlabs)):
        for col, ((zn, _), zl) in enumerate(zip(zone_defs, zlabs)):
            ax = axes[row, col]
            fv, rv = fm[zn][mk], rm[zn][mk]
            ax.bar(["Frozen", "Recalc"], [fv, rv],
                   color=["#4C72B0", "#DD8452"], width=0.5)
            ax.set_title(f"{ml} — {zl}", fontproperties=zhp(9))
            ax.tick_params(labelsize=8)
            off = 0.02 * (abs(fv) + 0.5)
            ax.text(0, fv + off, f"{fv:.3f}", ha="center",
                    va="bottom", fontsize=7)
            ax.text(1, rv + off, f"{rv:.3f}", ha="center",
                    va="bottom", fontsize=7)
            if row == 0 and col == 2:
                ax.text(0.5, 0.9,
                        f"CI(Δ Sharpe):\n[{ci_low:.3f}, {ci_high:.3f}]",
                        transform=ax.transAxes, ha="center", fontsize=8,
                        bbox=dict(boxstyle="round", facecolor="wheat",
                                  alpha=0.7))
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "exp2_1_frozen_vs_recalc.png"),
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    d = fm["zone1"]["sharpe"] - rm["zone1"]["sharpe"]
    if d > 0 and ci_low > 0:
        conc = "Frozen significantly outperforms recalc in zone1."
    elif d < 0 and ci_high < 0:
        conc = "Recalc significantly outperforms frozen in zone1."
    else:
        conc = "No significant difference between frozen and recalc in zone1."

    return {
        "status": "completed",
        "frozen_sharpe_by_zone": {zn: fm[zn]["sharpe"]
                                   for zn, _ in zone_defs},
        "recalc_sharpe_by_zone": {zn: rm[zn]["sharpe"]
                                   for zn, _ in zone_defs},
        "sharpe_diff_ci": [round(float(ci_low), 4),
                           round(float(ci_high), 4)],
        "conclusion": conc,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Experiment 2.2 – Half-Life Optimal Value Search
# ═══════════════════════════════════════════════════════════════════════════
def exp_2_2(S, tau_values=None, n_bootstrap=1000):
    if tau_values is None:
        tau_values = [1, 2, 3, 5, 7, 10, 15, 20, 30, 60]
    n_paths, N = S.shape[0], S.shape[1] - 1
    T = np.arange(N + 1, dtype=float)

    # Precompute per-path error stats for each tau
    tau_errs = {}
    for tau in tau_values:
        pm, pv = np.zeros(n_paths), np.zeros(n_paths)
        for i in range(n_paths):
            s = S[i, :]
            xs = T[: min(4, N + 1)]
            ys = s[: min(4, N + 1)]
            a0, b0, c0, d0 = np.polyfit(xs, ys, min(len(xs) - 1, 3))
            pred = np.zeros(N + 1)
            pred[0] = s[0]
            for t_idx in range(1, N + 1):
                decay = 2.0 ** (-t_idx / tau)
                pred[t_idx] = np.polyval(
                    [a0 * decay, b0 * decay, c0, d0], t_idx)
            err = pred - s
            pm[i] = np.mean(err)
            pv[i] = np.var(err)
        tau_errs[tau] = (pm, pv)

    taus_arr = np.array(tau_values, dtype=float)
    B_vals = np.array([np.mean(tau_errs[t][0]) for t in tau_values])
    V_vals = np.array([np.mean(tau_errs[t][1]) for t in tau_values])
    L_vals = B_vals ** 2 + V_vals

    # Optimal with quadratic interpolation
    min_idx = int(np.argmin(L_vals))
    if 0 < min_idx < len(taus_arr) - 1:
        xs = taus_arr[min_idx - 1: min_idx + 2]
        ys = L_vals[min_idx - 1: min_idx + 2]
        if ys[1] < ys[0] and ys[1] < ys[2]:
            p = np.polyfit(xs, ys, 2)
            tau_opt = -p[1] / (2 * p[0]) if p[0] > 0 else taus_arr[min_idx]
        else:
            tau_opt = taus_arr[min_idx]
    else:
        tau_opt = taus_arr[min_idx]

    # Bootstrap CI
    boot_opts = []
    for _ in range(n_bootstrap):
        idx = RNG.integers(0, n_paths, n_paths)
        Lb = np.array([
            np.mean(tau_errs[t][0][idx]) ** 2 +
            np.mean(tau_errs[t][1][idx])
            for t in tau_values
        ])
        bidx = int(np.argmin(Lb))
        if 0 < bidx < len(taus_arr) - 1:
            xs = taus_arr[bidx - 1: bidx + 2]
            ys = Lb[bidx - 1: bidx + 2]
            if ys[1] < ys[0] and ys[1] < ys[2]:
                p = np.polyfit(xs, ys, 2)
                boot_opts.append(-p[1] / (2 * p[0])
                                 if p[0] > 0 else taus_arr[bidx])
            else:
                boot_opts.append(taus_arr[bidx])
        else:
            boot_opts.append(taus_arr[bidx])
    ci_low, ci_high = np.percentile(boot_opts, [2.5, 97.5])

    # Chart
    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.suptitle("Experiment 2.2: Half-Life Optimal Value Search",
                 fontsize=14)
    ax1.set_xlabel("tau (minutes)", fontsize=12)
    ax1.set_ylabel("Loss L(tau) = B² + V", color="tab:red", fontsize=12)
    ax1.plot(taus_arr, L_vals, "o-", color="tab:red", lw=2, ms=8,
             label="L(tau)")
    ax1.axvline(tau_opt, color="darkred", ls="--", alpha=0.7,
                label=f"tau* = {tau_opt:.1f}")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    B2_vals, xp = B_vals ** 2, np.arange(len(taus_arr))
    w = 0.35
    ax2.bar(xp - w / 2, B2_vals, w, color="tab:blue", alpha=0.6,
            label="B²(tau)")
    ax2.bar(xp + w / 2, V_vals, w, color="tab:green", alpha=0.6,
            label="V(tau)")
    ax2.set_xticks(xp)
    ax2.set_xticklabels([str(t) for t in taus_arr])
    ax2.set_ylabel("B² and V", fontsize=12)
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "exp2_2_half_life_search.png"),
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    oi = np.argmin(L_vals)
    conc = (f"Optimal tau = {tau_opt:.1f} "
            f"(95% CI: [{ci_low:.1f}, {ci_high:.1f}]). "
            f"B²={B2_vals[oi]:.6f}, V={V_vals[oi]:.6f}.")
    return {
        "status": "completed",
        "tau_optimal": round(tau_opt, 1),
        "tau_ci": [round(float(ci_low), 1), round(float(ci_high), 1)],
        "loss_curve": {str(t): round(float(L_vals[i]), 6)
                       for i, t in enumerate(tau_values)},
        "conclusion": conc,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Experiment 2.3 – SIGMOID vs Hard Threshold
# ═══════════════════════════════════════════════════════════════════════════
def exp_2_3(n_trials=30000, split=0.7, n_chatter_bootstrap=500):
    """
    SIGMOID vs Hard threshold comparison.
    Chatter = variance of d(alpha)/dt over an autocorrelated time series.
    Hard threshold produces step-changes in alpha (large chatter).
    Sigmoid produces smooth transitions (small chatter).
    """
    rng = default_rng(43)

    # ── Training data (i.i.d. for grid search) ────────────
    devs = np.abs(rng.normal(0, 0.1, size=n_trials))
    signal_quality = sigmoid(-5 * devs + 2)
    exp_ret = 0.007 * signal_quality
    ret = exp_ret + rng.normal(0, 0.025, size=n_trials)

    n_tr = int(n_trials * split)
    d_tr, r_tr = devs[:n_tr], ret[:n_tr]
    d_te, r_te = devs[n_tr:], ret[n_tr:]

    # Grids
    hard_thetas = np.linspace(0.01, 0.4, 40)
    k_grid = np.linspace(1, 20, 20)
    b_grid = np.linspace(-1, 4, 20)

    best_hs, best_ht = -np.inf, 0.05
    for th in hard_thetas:
        s = compute_sharpe((d_tr < th).astype(float) * r_tr)
        if s > best_hs:
            best_hs, best_ht = s, th

    best_ss, best_k, best_b = -np.inf, 5.0, 2.0
    for k, b in itertools.product(k_grid, b_grid):
        s = compute_sharpe(sigmoid(-k * d_tr + b) * r_tr)
        if s > best_ss:
            best_ss, best_k, best_b = s, k, b

    # ── Test: out-of-sample Sharpe ────────────────────────
    ha_te = (d_te < best_ht).astype(float)
    sa_te = sigmoid(-best_k * d_te + best_b)
    hs_test = compute_sharpe(ha_te * r_te)
    ss_test = compute_sharpe(sa_te * r_te)

    # ── Chatter: total variation of alpha (sum|Δα|) on AR(1) series ──
    # Hard threshold: α flips 0↔1 at each boundary crossing → cost = 1 per flip.
    # Sigmoid: α changes smoothly → cost = k·α·(1-α)·|Δd| per step, always < k/4·|Δd|.
    hard_chatters, sig_chatters = [], []
    T_seq = 600
    phi_seq = 0.2

    for _ in range(n_chatter_bootstrap):
        eps = rng.normal(0, 0.08, T_seq)
        d_seq = np.zeros(T_seq)
        for t in range(1, T_seq):
            d_seq[t] = phi_seq * d_seq[t - 1] + eps[t]
        d_seq = np.abs(d_seq)

        # Chatter = sum|Δα| — total position-size turnover ("chatter cost")
        ha_seq = (d_seq < best_ht).astype(float)
        sa_seq = sigmoid(-best_k * d_seq + best_b)
        hard_chat = float(np.sum(np.abs(np.diff(ha_seq))))
        sig_chat = float(np.sum(np.abs(np.diff(sa_seq))))

        hard_chatters.append(hard_chat)
        sig_chatters.append(sig_chat)

    avg_hard_chatter = float(np.mean(hard_chatters))
    avg_sig_chatter = float(np.mean(sig_chatters))
    chat_red = ((avg_hard_chatter - avg_sig_chatter) / avg_hard_chatter
                * 100) if avg_hard_chatter > 1e-8 else 0.0

    chat_red_bs = []
    for _ in range(n_chatter_bootstrap):
        idx = rng.integers(0, n_chatter_bootstrap, n_chatter_bootstrap)
        h = np.mean([hard_chatters[i] for i in idx])
        s = np.mean([sig_chatters[i] for i in idx])
        chat_red_bs.append((h - s) / h * 100 if h > 1e-8 else 0)
    chat_red_ci = (float(np.percentile(chat_red_bs, 2.5)),
                   float(np.percentile(chat_red_bs, 97.5)))

    # ── Chart ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Experiment 2.3: SIGMOID vs Hard Threshold",
                 fontsize=14)

    dg = np.linspace(0, 0.5, 200)
    axes[0].plot(dg, sigmoid(-best_k * dg + best_b), "b-", lw=2,
                 label=f"Sigmoid (k={best_k:.1f}, b={best_b:.1f})")
    axes[0].plot(dg, (dg < best_ht).astype(float), "r--", lw=2,
                 label=f"Hard (θ={best_ht:.3f})")
    axes[0].axvline(best_ht, color="r", ls=":", alpha=0.5)
    axes[0].set_xlabel("|Δ|", fontsize=11)
    axes[0].set_ylabel("α (decision weight)", fontsize=11)
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(ha_te * r_te, bins=50, alpha=0.5, color="red",
                 label=f"Hard (Sharpe={hs_test:.3f})")
    axes[1].hist(sa_te * r_te, bins=50, alpha=0.5, color="blue",
                 label=f"Sigmoid (Sharpe={ss_test:.3f})")
    axes[1].set_xlabel("Return", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    yerr_h = np.std(hard_chatters) / np.sqrt(n_chatter_bootstrap)
    yerr_s = np.std(sig_chatters) / np.sqrt(n_chatter_bootstrap)
    axes[2].bar(["Hard", "Sigmoid"], [avg_hard_chatter, avg_sig_chatter],
                color=["red", "blue"], width=0.5,
                yerr=[[yerr_h, yerr_s]])
    axes[2].set_ylabel("Chatter: Sum|Δα| (turnover)", fontsize=10)
    h_max = max(avg_hard_chatter, avg_sig_chatter)
    axes[2].text(0, avg_hard_chatter + 0.01 * h_max,
                 f"{avg_hard_chatter:.4f}", ha="center", fontsize=9)
    axes[2].text(1, avg_sig_chatter + 0.01 * h_max,
                 f"{avg_sig_chatter:.4f}", ha="center", fontsize=9)
    anim_text = (f"Chatter ↓ {chat_red:.1f}%\n"
                 f"CI [{chat_red_ci[0]:.1f}, {chat_red_ci[1]:.1f}]"
                 if chat_red > 5 else
                 f"Chatter reduction: {chat_red:.1f}%")
    axes[2].text(0.5, h_max * 0.6, anim_text, ha="center", fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="wheat",
                           alpha=0.8))
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "exp2_3_sigmoid_vs_hard.png"),
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    conc = (f"Sigmoid (k={best_k:.1f}, b={best_b:.1f}) "
            f"Sharpe={ss_test:.4f} vs "
            f"Hard (θ={best_ht:.4f}) Sharpe={hs_test:.4f}. "
            f"Chatter(Σ|Δα|): sigmoid={avg_sig_chatter:.4f}, "
            f"hard={avg_hard_chatter:.4f} "
            f"({chat_red:.1f}% reduction, "
            f"95% CI [{chat_red_ci[0]:.1f}, {chat_red_ci[1]:.1f}]).")
    return {
        "status": "completed",
        "sigmoid_sharpe": round(float(ss_test), 4),
        "hard_sharpe": round(float(hs_test), 4),
        "sigmoid_params": {"k": round(float(best_k), 2),
                           "b": round(float(best_b), 2)},
        "hard_theta": round(float(best_ht), 4),
        "avg_chatter_sigmoid": round(float(avg_sig_chatter), 6),
        "avg_chatter_hard": round(float(avg_hard_chatter), 6),
        "chatter_reduction_pct": round(float(chat_red), 1),
        "chatter_reduction_ci": [round(float(chat_red_ci[0]), 1),
                                  round(float(chat_red_ci[1]), 1)],
        "conclusion": conc,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Experiment 2.5 – Dynamic vs Static Threshold Pareto Frontier
# ═══════════════════════════════════════════════════════════════════════════
def exp_2_5(S):
    n_paths, N = S.shape[0], S.shape[1] - 1
    T = np.arange(N + 1, dtype=float)
    tau_ref = 15
    lam = np.log(2) / tau_ref

    mask = T >= 30
    if mask.sum() < 3:
        mask = T >= 20
    L = int(mask.sum())

    frozen_pred = freeze_predictions(S, tau_ref, T)
    sigma0_vals = np.logspace(-2, 0, 20)

    static_res = {"sharpe": [], "maxdd": [], "sigma0": []}
    dynamic_res = {"sharpe": [], "maxdd": [], "sigma0": []}

    for sigma0 in sigma0_vals:
        for strat in ["static", "dynamic"]:
            all_rets, all_dds = [], []
            for i in range(n_paths):
                resid = np.abs(frozen_pred[i, mask] - S[i, mask])
                thresh = (sigma0 * np.exp(-lam * np.arange(L))
                          if strat == "dynamic" else sigma0)
                alpha = (resid < thresh).astype(float)
                err = S[i, mask] - frozen_pred[i, mask]
                ret_s = alpha[1:] * np.diff(err)
                if len(ret_s) > 2 and np.std(ret_s) > 1e-12:
                    all_rets.extend(ret_s.tolist())
                    eq = 1.0 + np.cumsum(np.insert(ret_s, 0, 0))
                    all_dds.append(max_drawdown(eq))
            sharpe = compute_sharpe(all_rets) if len(all_rets) > 2 else 0.0
            avg_dd = float(np.mean(all_dds)) if all_dds else 0.0
            if strat == "static":
                static_res["sharpe"].append(sharpe)
                static_res["maxdd"].append(avg_dd)
                static_res["sigma0"].append(sigma0)
            else:
                dynamic_res["sharpe"].append(sharpe)
                dynamic_res["maxdd"].append(avg_dd)
                dynamic_res["sigma0"].append(sigma0)

    # Chart
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("Exp 2.5: Dynamic vs Static Threshold Pareto Frontier",
                 fontsize=14)
    sc_s = ax.scatter(static_res["maxdd"], static_res["sharpe"],
                      c=static_res["sigma0"], cmap="Reds", s=60,
                      alpha=0.8, edgecolors="darkred", label="Static")
    sc_d = ax.scatter(dynamic_res["maxdd"], dynamic_res["sharpe"],
                      c=dynamic_res["sigma0"], cmap="Blues", s=60,
                      alpha=0.8, edgecolors="darkblue", label="Dynamic")

    for res, color in [(static_res, "r"), (dynamic_res, "b")]:
        x = np.array(res["maxdd"])
        y = np.array(res["sharpe"])
        idx = np.argsort(x)
        xs, ys = x[idx], y[idx]
        fi = [0]
        for j in range(1, len(xs)):
            if ys[j] > ys[fi[-1]]:
                fi.append(j)
        if len(fi) > 1:
            ax.plot(xs[fi], ys[fi], f"{color}--", alpha=0.5, lw=1.5)

    ax.set_xlabel("Max Drawdown", fontsize=12)
    ax.set_ylabel("Sharpe Ratio", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc_s, ax=ax, shrink=0.6).set_label("Static σ₀",
                                                      fontsize=9)
    plt.colorbar(sc_d, ax=ax, shrink=0.6, pad=0.02).set_label(
        "Dynamic σ₀", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "exp2_5_dynamic_vs_static.png"),
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    dmax = max(dynamic_res["sharpe"]) if dynamic_res["sharpe"] else 0
    smax = max(static_res["sharpe"]) if static_res["sharpe"] else 0
    dom = ("dynamic" if dmax > smax * 1.05
           else "static" if smax > dmax * 1.05 else "comparable")
    conc = (f"Dynamic max Sharpe = {dmax:.4f} vs "
            f"Static max Sharpe = {smax:.4f}. "
            f"Dynamic {dom} on Pareto frontier.")
    return {
        "status": "completed",
        "dynamic_pareto": [
            {"sigma0": s, "sharpe": sh, "maxdd": dd}
            for s, sh, dd in zip(dynamic_res["sigma0"],
                                 dynamic_res["sharpe"],
                                 dynamic_res["maxdd"])
        ],
        "static_pareto": [
            {"sigma0": s, "sharpe": sh, "maxdd": dd}
            for s, sh, dd in zip(static_res["sigma0"],
                                 static_res["sharpe"],
                                 static_res["maxdd"])
        ],
        "dominance_region": dom,
        "conclusion": conc,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("Layer 2 — Starry Sky Strategy (星空策略) Verification")
    print("=" * 65, flush=True)
    t0 = time.time()

    print("\n[0] Simulating 500 price trajectories...", end=" ", flush=True)
    S = simulate_trajectories(500, 200, seed=42)
    print(f"done. Shape {S.shape}, price [{S.min():.4f}, {S.max():.4f}]",
          flush=True)

    exps = [
        ("exp2_1", "Experiment 2.1 — Frozen vs Recalculated",
         lambda: exp_2_1(S)),
        ("exp2_2", "Experiment 2.2 — Half-Life Search",
         lambda: exp_2_2(S)),
        ("exp2_3", "Experiment 2.3 — SIGMOID vs Hard Threshold",
         lambda: exp_2_3()),
        ("exp2_5", "Experiment 2.5 — Dynamic vs Static Threshold",
         lambda: exp_2_5(S)),
    ]

    all_res = {}
    for key, label, fn in exps:
        print(f"\n{label} ...", end=" ", flush=True)
        t1 = time.time()
        try:
            r = fn()
            all_res[key] = r
            print(f"{time.time() - t1:.1f}s", flush=True)
            if "sharpe_diff_ci" in r:
                print(f"    Frozen Sharpe: {r['frozen_sharpe_by_zone']}")
                print(f"    Recalc Sharpe: {r['recalc_sharpe_by_zone']}")
                print(f"    CI: {r['sharpe_diff_ci']}")
            elif "loss_curve" in r:
                print(f"    tau* = {r['tau_optimal']}  CI: {r['tau_ci']}")
                print(f"    Conclusion: {r['conclusion']}")
            elif "sigmoid_sharpe" in r:
                print(f"    Sigmoid: Sharpe={r['sigmoid_sharpe']}  "
                      f"params={r['sigmoid_params']}")
                print(f"    Hard: Sharpe={r['hard_sharpe']}  "
                      f"theta={r['hard_theta']}")
                print(f"    Chatter Var(Δα): sigmoid={r['avg_chatter_sigmoid']:.6f}  "
                      f"hard={r['avg_chatter_hard']:.6f}  "
                      f"({r['chatter_reduction_pct']:.1f}% reduction, "
                      f"CI {r['chatter_reduction_ci']})")
            elif "dominance_region" in r:
                dmax = max(p["sharpe"] for p in r["dynamic_pareto"])
                smax = max(p["sharpe"] for p in r["static_pareto"])
                print(f"    Dynamic max Sharpe: {dmax:.4f}")
                print(f"    Static max Sharpe: {smax:.4f}")
                print(f"    Dominance: {r['dominance_region']}")
        except Exception as e:
            import traceback
            print(f"FAILED: {e}", flush=True)
            traceback.print_exc()
            all_res[key] = {"status": "failed", "error": str(e)}

    rpath = os.path.join(OUTDIR, "results_layer2.json")
    with open(rpath, "w") as f:
        json.dump(all_res, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 65}")
    print(f"Total: {time.time() - t0:.1f}s")
    print(f"Results: {rpath}")
    for fname in ["exp2_1_frozen_vs_recalc.png",
                  "exp2_2_half_life_search.png",
                  "exp2_3_sigmoid_vs_hard.png",
                  "exp2_5_dynamic_vs_static.png",
                  "results_layer2.json"]:
        fp = os.path.join(OUTDIR, fname)
        print(f"  {fname}  ({os.path.getsize(fp) // 1024} KB)"
              if os.path.exists(fp) else f"  {fname}  [MISSING]")
    print("Done.")


if __name__ == "__main__":
    main()
