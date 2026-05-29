#!/usr/bin/env python3
"""
Cubic State Analyzer
====================
Interactive tool for analyzing cubic curves f(x) = ax^3 + bx^2 + cx + d.

Features:
  - Four sliders controlling coefficients a, b, c, d
  - Real-time cubic curve plot with annotated key points
    (roots, inflection point, local extrema)
  - p-q bifurcation phase diagram showing the discriminant boundary
  - State classification info panel (discriminant, root count, type)
  - Three preset buttons for quick parameter switching

Usage:
    python cubic_state_analyzer.py

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.gridspec as gridspec

# =========================================================================
# Configuration
# =========================================================================
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti SC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

EPS = 1e-10          # numerical zero threshold
SLIDER_LIM = (-5.0, 5.0)
SLIDER_STEP = 0.01

DEFAULT_A, DEFAULT_B, DEFAULT_C, DEFAULT_D = 1.0, 0.0, -3.0, 0.0

# =========================================================================
# State Computation
# =========================================================================

def _f(x, a, b, c, d):
    """Evaluate the cubic f(x) = ax^3 + bx^2 + cx + d."""
    return a * x ** 3 + b * x ** 2 + c * x + d


def cubic_state(a, b, c, d):
    """
    Compute full analytical state of the cubic f(x) = ax^3 + bx^2 + cx + d.

    Returns a dict with keys:
        kind:       'cubic' | 'degenerate'
        p, q, delta, delta_c, D_crit: discriminant parameters
        roots:      sorted list of real roots (float) or marker strings
        inflection: (x, y) tuple or None
        extrema:    list of (x, y, 'min'|'max')
        label:      Chinese classification string (5-type)
    """
    # --- Degenerate: a ≈ 0 -------------------------------------------------
    if abs(a) < EPS:
        return _degenerate_state(b, c, d)

    # --- Normalized parameters ---------------------------------------------
    # Depressed cubic coefficients: f(x) = a[(x + b/3a)^3 + p(x + b/3a) + q]
    p = (3.0 * a * c - b * b) / (3.0 * a * a)
    q = (2.0 * b ** 3 - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a ** 3)
    delta = (q / 2.0) ** 2 + (p / 3.0) ** 3      # Cardano discriminant
    delta_c = -4.0 * p ** 3 - 27.0 * q ** 2       # Cusp discriminant
    D_crit = 4.0 * b * b - 12.0 * a * c           # discriminant of f'(x)

    # --- Real roots --------------------------------------------------------
    all_roots = np.roots([a, b, c, d])
    roots = sorted(r.real for r in all_roots if abs(r.imag) < EPS)

    # --- Inflection point: f''(x) = 6ax + 2b = 0 --------------------------
    xi = -b / (3.0 * a)
    yi = _f(xi, a, b, c, d)

    # --- Local extrema: f'(x) = 3ax^2 + 2bx + c = 0 -----------------------
    extrema = []
    if D_crit > EPS:
        sd = np.sqrt(D_crit)
        # x = (-b +/- sqrt(D_crit)) / (3a)
        # f''(x) at these points: 6a*x + 2b = +/- 2*sqrt(D_crit)
        x_min = (-b + sd) / (3.0 * a)   # f'' = +2*sqrt(D_crit) > 0  => min
        x_max = (-b - sd) / (3.0 * a)   # f'' = -2*sqrt(D_crit) < 0  => max
        extrema = [
            (x_min, _f(x_min, a, b, c, d), 'min'),
            (x_max, _f(x_max, a, b, c, d), 'max'),
        ]

    # --- Classification (5-type: Δ_c convention, see doc Sec 2.3) ----------
    # Use Δ_c (Cusp convention) for classification: >0 → S-shaped, <0 → monotonic
    if abs(a) < EPS:
        label = "V: 降次退化型"
    elif delta_c > EPS:
        # Δ_c > 0 → three real roots → S-shaped or bistable
        label = "II: S形双稳型(三实根)"
    elif delta_c < -EPS:
        # Δ_c < 0 → one real root → monotonic
        label = "I: 单调型(单实根,无局部极值)" if D_crit < EPS else "I: 单调型(单实根,有局部结构)"
    else:
        # Δ_c ≈ 0 → critical / bifurcation boundary
        if abs(p) < EPS and abs(q) < EPS:
            label = "IV: 拐点退化型(Cusp尖点)"
        elif p < -EPS:
            label = "III: 临界折叠型(鞍结分岔)"
        else:
            label = "III/IV 退化型(p≥0临界)"

    return dict(
        kind='cubic', p=p, q=q, delta=delta, delta_c=delta_c, D_crit=D_crit,
        roots=roots, inflection=(xi, yi), extrema=extrema, label=label,
    )


def _degenerate_state(b, c, d):
    """State when a = 0 (cubic degenerates to quadratic, linear, or constant)."""
    st = dict(
        kind='degenerate',
        p=float('nan'), q=float('nan'),
        delta=float('nan'), delta_c=float('nan'), D_crit=float('nan'),
        roots=[], inflection=None, extrema=[], label='V: 降次退化型',
    )

    if abs(b) > EPS:
        # Quadratic: b x^2 + c x + d = 0
        all_r = np.roots([b, c, d])
        st['roots'] = sorted(r.real for r in all_r if abs(r.imag) < EPS)
        # Vertex
        xv = -c / (2.0 * b)
        yv = b * xv * xv + c * xv + d
        st['extrema'] = [(xv, yv, 'min' if b > 0 else 'max')]
        st['label'] = 'V: 降次退化型→二次曲线'
    elif abs(c) > EPS:
        # Linear: c x + d = 0
        st['roots'] = [-d / c]
        st['label'] = 'V: 降次退化型→一次曲线'
    elif abs(d) > EPS:
        # Non-zero constant: no roots
        st['roots'] = []
        st['label'] = 'V: 降次退化型→常数(无根)'
    else:
        # f(x) = 0 identically
        st['roots'] = []
        st['label'] = 'V: 降次退化型→零函数(全体实数根)'

    return st


# =========================================================================
# Adaptive Plot Range
# =========================================================================

def _compute_x_range(state):
    """Determine x-axis limits covering all key points with padding."""
    xs = []
    for r in state['roots']:
        if isinstance(r, str):
            continue
        xs.append(r)
    if state['inflection'] is not None:
        xs.append(state['inflection'][0])
    for x, _, _ in state['extrema']:
        xs.append(x)

    if not xs:
        return -5.0, 5.0

    lo, hi = min(xs), max(xs)
    span = max(hi - lo, 0.5)
    pad = span * 0.4
    return lo - pad, hi + pad


# =========================================================================
# Main Application
# =========================================================================

def main():
    # ----- Figure & Layout ------------------------------------------------
    fig = plt.figure(figsize=(14, 10))
    fig.canvas.manager.set_window_title("三次曲线状态分析器")

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.07, right=0.95, bottom=0.22, top=0.95,
        hspace=0.35, wspace=0.3, height_ratios=[1.6, 1],
    )

    ax_curve = fig.add_subplot(gs[0, :])   # top:  cubic curve (full width)
    ax_phase = fig.add_subplot(gs[1, 0])   # left: p-q phase diagram
    ax_info = fig.add_subplot(gs[1, 1])    # right: info panel

    # ----- Sliders ---------------------------------------------------------
    slider_defs = [
        ('a', 'a = ', DEFAULT_A),
        ('b', 'b = ', DEFAULT_B),
        ('c', 'c = ', DEFAULT_C),
        ('d', 'd = ', DEFAULT_D),
    ]

    sw, sg, sh = 0.15, 0.03, 0.03          # slider width, gap, height
    total_slider_width = 4 * sw + 3 * sg
    slider_x0 = (1.0 - total_slider_width) / 2.0

    sliders = {}
    for i, (name, label, val) in enumerate(slider_defs):
        left = slider_x0 + i * (sw + sg)
        s_ax = fig.add_axes([left, 0.08, sw, sh])
        sliders[name] = Slider(
            s_ax, label, SLIDER_LIM[0], SLIDER_LIM[1],
            valinit=val, valstep=SLIDER_STEP,
            facecolor='#aaaaaa',
        )

    # ----- Preset Buttons --------------------------------------------------
    presets = [
        ('S-Shape (三实根)',  1.0, 0.0, -3.0, 0.0),
        ('Monotonic (单实根)', 1.0, 0.0,  2.0, 0.0),
        ('Critical (重根)',   1.0, 0.0,  0.0, 0.0),
    ]

    bw, bg, bh = 0.16, 0.025, 0.04         # button width, gap, height
    total_btn_width = 3 * bw + 2 * bg
    btn_x0 = (1.0 - total_btn_width) / 2.0

    # ----- Mutable state for closures --------------------------------------
    _busy = [False]       # guard against re-entrant updates
    _annots = []          # list of annotation Text objects on ax_curve

    # Slider batch-setter (defined before buttons so lambdas can reference it)
    def set_sliders(a, b, c, d):
        """Update all four sliders atomically (single redraw)."""
        _busy[0] = True
        sliders['a'].set_val(a)
        sliders['b'].set_val(b)
        sliders['c'].set_val(c)
        sliders['d'].set_val(d)
        _busy[0] = False
        update(None)

    # Create preset buttons
    for i, (label_text, av, bv, cv, dv) in enumerate(presets):
        left = btn_x0 + i * (bw + bg)
        b_ax = fig.add_axes([left, 0.14, bw, bh])
        btn = Button(b_ax, label_text, color='#e8e8e8', hovercolor='#c0c0c0')
        btn.on_clicked(
            lambda _, a=av, b=bv, c=cv, d=dv: set_sliders(a, b, c, d)
        )

    # ----- Curve Plot Setup ------------------------------------------------
    ax_curve.axhline(0, color='gray', lw=0.5, ls='--')
    ax_curve.axvline(0, color='gray', lw=0.5, ls='--')
    ax_curve.set_title('三次曲线  f(x) = ax³ + bx² + cx + d', fontsize=12)
    ax_curve.set_xlabel('x')
    ax_curve.set_ylabel('f(x)')
    ax_curve.grid(True, alpha=0.25)

    curve_line, = ax_curve.plot([], [], 'b-', lw=2, zorder=2)

    # Scatter artists for key points (data updated via set_offsets)
    root_marker = ax_curve.scatter([], [], c='red',   marker='o', s=60, zorder=5,
                                   label='实根')
    infl_marker = ax_curve.scatter([], [], c='blue',  marker='^', s=80, zorder=5,
                                   label='拐点')
    extr_marker = ax_curve.scatter([], [], c='green', marker='s', s=60, zorder=5,
                                   label='极值点')

    ax_curve.legend(fontsize=9, loc='upper right')

    # ----- Phase Diagram Setup ---------------------------------------------
    # Bifurcation curve: 4p^3 + 27q^2 = 0  =>  q = +/- (2/3)*sqrt(-p^3/3)
    p_bif = np.linspace(-5.0, 0.0, 800)
    sqrt_arg = np.maximum(0.0, -p_bif ** 3 / 3.0)
    q_bif = (2.0 / 3.0) * np.sqrt(sqrt_arg)

    ax_phase.plot(p_bif, q_bif, 'r-', lw=1.2, alpha=0.7)
    ax_phase.plot(p_bif, -q_bif, 'r-', lw=1.2, alpha=0.7)
    ax_phase.fill_between(p_bif, -q_bif, q_bif,
                          alpha=0.12, color='red', label='Δ_c>0 (三实根)')
    ax_phase.axhline(0, color='gray', lw=0.5, ls='--')
    ax_phase.axvline(0, color='gray', lw=0.5, ls='--')
    ax_phase.set_xlim(-5, 5)
    ax_phase.set_ylim(-10, 10)
    ax_phase.set_title('p-q 相位图  (分岔边界: 4p³+27q²=0)', fontsize=11)
    ax_phase.set_xlabel('p')
    ax_phase.set_ylabel('q')
    ax_phase.grid(True, alpha=0.2)
    ax_phase.legend(fontsize=8, loc='upper right')

    phase_point, = ax_phase.plot([], [], 'ko', ms=8, zorder=10)

    # =====================================================================
    # Update Functions
    # =====================================================================

    def _clear_annots():
        """Remove all annotation text objects from the curve axes."""
        for t in _annots:
            try:
                t.remove()
            except Exception:
                pass
        _annots.clear()

    def _add_annot(x, y, text, color):
        """Add an annotation text near a data point on the curve."""
        t = ax_curve.annotate(
            text, xy=(x, y), xytext=(8, 8),
            textcoords='offset points', fontsize=8,
            color=color, fontweight='bold', zorder=6,
        )
        _annots.append(t)

    def update(val=None):
        """Main update: recompute everything and refresh all plots."""
        if _busy[0]:
            return

        # Read current slider values
        a = sliders['a'].val
        b = sliders['b'].val
        c = sliders['c'].val
        d = sliders['d'].val

        state = cubic_state(a, b, c, d)

        # --- Update curve --------------------------------------------------
        xr = _compute_x_range(state)
        x_vals = np.linspace(xr[0], xr[1], 2000)
        y_vals = _f(x_vals, a, b, c, d)

        curve_line.set_data(x_vals, y_vals)
        ax_curve.set_xlim(xr)

        # Auto-scale Y range with 20 % padding
        ymin, ymax = y_vals.min(), y_vals.max()
        if ymax - ymin < 1e-12:
            ymin, ymax = -1.0, 1.0
        ypad = (ymax - ymin) * 0.2
        ax_curve.set_ylim(ymin - ypad, ymax + ypad)

        # --- Update scatter markers ----------------------------------------
        # Roots (red circles)
        root_data = []
        for r in state['roots']:
            if isinstance(r, str):
                continue
            root_data.append([r, _f(r, a, b, c, d)])
        root_marker.set_offsets(root_data if root_data else np.empty((0, 2)))

        # Inflection (blue triangle)
        if state['inflection'] is not None:
            infl_marker.set_offsets([[state['inflection'][0], state['inflection'][1]]])
        else:
            infl_marker.set_offsets(np.empty((0, 2)))

        # Extrema (green squares)
        extr_data = [(x, y) for x, y, _ in state['extrema']]
        extr_marker.set_offsets(extr_data if extr_data else np.empty((0, 2)))

        # --- Update annotations --------------------------------------------
        _clear_annots()

        for r in state['roots']:
            if isinstance(r, str):
                continue
            yr = _f(r, a, b, c, d)
            _add_annot(r, yr, f'({r:.2f}, {yr:.2f})', 'red')

        if state['inflection'] is not None:
            xi, yi = state['inflection']
            _add_annot(xi, yi, f'({xi:.2f}, {yi:.2f})', 'blue')

        for xe, ye, _ in state['extrema']:
            _add_annot(xe, ye, f'({xe:.2f}, {ye:.2f})', 'green')

        # --- Update phase diagram ------------------------------------------
        if state['kind'] == 'cubic':
            phase_point.set_data([state['p']], [state['q']])
        else:
            phase_point.set_data([], [])

        # --- Update info panel ---------------------------------------------
        ax_info.clear()
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)
        ax_info.axis('off')
        ax_info.set_title('状态信息', fontsize=11, pad=8)

        # Build info lines
        if state['kind'] == 'cubic':
            info_lines = [
                f"Δ_cardano = {state['delta']:.6f}",
                f"Δ_c (Cusp) = {state['delta_c']:.6f}",
                f"Δ_c 符号: {'Δ_c>0 (三实根)' if state['delta_c'] > EPS else ('Δ_c≈0 (临界)' if abs(state['delta_c']) < EPS else 'Δ_c<0 (单实根)')}",
            ]
        else:
            info_lines = [
                f"Δ_cardano = {state['delta']}",
                f"Δ_c (Cusp) = {state['delta_c']}",
                "Δ 符号: 无定义 (退化)",
            ]

        info_lines.extend([
            f"p = {state['p']:.6f}",
            f"q = {state['q']:.6f}",
            f"D_crit = b²-3ac = {state['D_crit']:.6f}",
            "",
            f"分类: {state['label']}",
            "",
        ])

        # Root listing
        if state['roots']:
            root_strs = []
            for r in state['roots']:
                if isinstance(r, str):
                    root_strs.append(r)
                else:
                    root_strs.append(f"x = {r:.4f}")
            info_lines.append(f"实根 ({len(state['roots'])}):")
            info_lines.extend(f"    {s}" for s in root_strs)
        else:
            info_lines.append("实根: 无")

        if state['kind'] == 'degenerate':
            info_lines.append("")
            info_lines.append("⚠ a ≈ 0, 函数已退化")

        # Render text lines
        y_cursor = 0.95
        for line in info_lines:
            ax_info.text(
                0.05, y_cursor, line, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                transform=ax_info.transAxes,
            )
            y_cursor -= 0.055

        # --- Schedule redraw -----------------------------------------------
        fig.canvas.draw_idle()

    # ----- Wire callbacks --------------------------------------------------
    for s in sliders.values():
        s.on_changed(update)

    # ----- Initial render --------------------------------------------------
    update(None)

    # ----- Show (blocks until window is closed) ----------------------------
    plt.show()


# =========================================================================
# Entry Point
# =========================================================================
if __name__ == '__main__':
    main()
