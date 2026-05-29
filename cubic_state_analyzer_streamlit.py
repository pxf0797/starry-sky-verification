#!/usr/bin/env python3
"""
Cubic State Analyzer -- Streamlit Web Version
==============================================
Interactive web tool for analyzing cubic curves f(x) = ax^3 + bx^2 + cx + d.

Usage:
    streamlit run cubic_state_analyzer_streamlit.py

Dependencies: streamlit, numpy, plotly
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

EPS = 1e-10


# =========================================================================
# Computation Functions
# =========================================================================

def _f(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


def compute_state(a, b, c, d):
    """Compute full analytical state of the cubic f(x) = ax^3 + bx^2 + cx + d."""
    if abs(a) < EPS:
        return _degenerate_state(b, c, d)

    p = (3.0 * a * c - b * b) / (3.0 * a * a)
    q = (2.0 * b ** 3 - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a ** 3)
    delta = (q / 2.0) ** 2 + (p / 3.0) ** 3
    delta_c = -4.0 * p ** 3 - 27.0 * q ** 2
    D_crit = 4.0 * b * b - 12.0 * a * c

    all_roots = np.roots([a, b, c, d])
    roots = sorted(r.real for r in all_roots if abs(r.imag) < EPS)

    xi = -b / (3.0 * a)
    yi = _f(xi, a, b, c, d)

    extrema = []
    if D_crit > EPS:
        sd = np.sqrt(D_crit)
        x_min = (-b + sd) / (3.0 * a)
        x_max = (-b - sd) / (3.0 * a)
        extrema = [
            (x_min, _f(x_min, a, b, c, d), 'min'),
            (x_max, _f(x_max, a, b, c, d), 'max'),
        ]

    return dict(
        kind='cubic', p=p, q=q, delta=delta, delta_c=delta_c, D_crit=D_crit,
        roots=roots, inflection=(xi, yi), extrema=extrema,
    )


def _degenerate_state(b, c, d):
    """State when a = 0 (cubic degenerates to quadratic / linear / constant)."""
    st_dict = dict(
        kind='degenerate',
        p=float('nan'), q=float('nan'),
        delta=float('nan'), delta_c=float('nan'), D_crit=float('nan'),
        roots=[], inflection=None, extrema=[],
    )
    if abs(b) > EPS:
        all_r = np.roots([b, c, d])
        st_dict['roots'] = sorted(r.real for r in all_r if abs(r.imag) < EPS)
        xv = -c / (2.0 * b)
        st_dict['extrema'] = [(xv, b * xv * xv + c * xv + d, 'min' if b > 0 else 'max')]
    elif abs(c) > EPS:
        st_dict['roots'] = [-d / c]
    return st_dict


def classify_label(a, delta, D_crit):
    """5-type classification based on Cardano discriminant delta."""
    if abs(a) < EPS:
        return '退化'
    if delta > EPS:
        return 'I: 单实根·起伏单调' if D_crit > EPS else 'II: 单实根·平坦单调'
    if delta < -EPS:
        return 'III: 三实根·S型(双极值)'
    return 'IV: 临界·切触型(含重根)' if D_crit > EPS else 'V: 临界·拐点退化(三重根)'


def _compute_x_range(state):
    """Determine x-axis limits covering all key points with 40 % padding."""
    xs = list(state['roots'])
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
# Plotting Functions
# =========================================================================

def make_curve_figure(state, a, b, c, d):
    """Plotly cubic curve with annotated roots, inflection, and extrema."""
    x_range = _compute_x_range(state)
    x_vals = np.linspace(x_range[0], x_range[1], 2000)
    y_vals = _f(x_vals, a, b, c, d)
    fig = go.Figure()

    # Main curve
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals, mode='lines', name='f(x)',
        line=dict(color='blue', width=2),
        hovertemplate='x=%{x:.4f}<br>f(x)=%{y:.4f}<extra></extra>',
    ))
    # Roots (red circles)
    root_pts = [(r, _f(r, a, b, c, d)) for r in state['roots']]
    if root_pts:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in root_pts], y=[p[1] for p in root_pts],
            mode='markers', name='实根',
            marker=dict(color='red', size=10, symbol='circle'),
            hovertemplate='根: x=%{x:.4f}<br>f(x)=%{y:.4f}<extra></extra>',
        ))
    # Inflection point (blue triangle)
    if state['inflection'] is not None:
        xi, yi = state['inflection']
        fig.add_trace(go.Scatter(
            x=[xi], y=[yi], mode='markers', name='拐点',
            marker=dict(color='blue', size=12, symbol='triangle-up'),
            hovertemplate='拐点: x=%{x:.4f}<br>f(x)=%{y:.4f}<extra></extra>',
        ))
    # Extrema (green squares)
    if state['extrema']:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in state['extrema']],
            y=[p[1] for p in state['extrema']],
            mode='markers', name='极值点',
            marker=dict(color='green', size=10, symbol='square'),
            hovertemplate='极值: x=%{x:.4f}<br>f(x)=%{y:.4f}<extra></extra>',
        ))

    ymin, ymax = float(y_vals.min()), float(y_vals.max())
    if ymax - ymin < 1e-12:
        ymin, ymax = -1.0, 1.0
    ypad = (ymax - ymin) * 0.2
    fig.update_layout(
        title=dict(text='三次曲线  f(x) = ax³ + bx² + cx + d'),
        xaxis_title='x', yaxis_title='f(x)',
        xaxis=dict(range=list(x_range), zeroline=False),
        yaxis=dict(range=[ymin - ypad, ymax + ypad], zeroline=False),
        height=420, hovermode='closest', showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.add_hline(y=0, line=dict(color='gray', width=0.5, dash='dash'))
    fig.add_vline(x=0, line=dict(color='gray', width=0.5, dash='dash'))
    return fig


def make_phase_figure(state):
    """Plotly p-q phase diagram with bifurcation boundary 4p^3+27q^2=0."""
    fig = go.Figure()
    p_bif = np.linspace(-5.0, 0.0, 800)
    sqrt_arg = np.maximum(0.0, -p_bif ** 3 / 3.0)
    q_bif = (2.0 / 3.0) * np.sqrt(sqrt_arg)

    fig.add_trace(go.Scatter(x=p_bif, y=q_bif, mode='lines', name='分岔边界',
                              line=dict(color='red', width=1.2), showlegend=True))
    fig.add_trace(go.Scatter(x=p_bif, y=-q_bif, mode='lines',
                              line=dict(color='red', width=1.2), showlegend=False))
    fig.add_trace(go.Scatter(
        x=np.concatenate([p_bif, p_bif[::-1]]),
        y=np.concatenate([q_bif, -q_bif[::-1]]),
        fill='toself', fillcolor='rgba(255, 0, 0, 0.12)',
        line=dict(width=0), name='Δ_c>0 (三实根区域)',
    ))
    if state['kind'] == 'cubic' and not np.isnan(state['p']):
        fig.add_trace(go.Scatter(
            x=[state['p']], y=[state['q']], mode='markers',
            marker=dict(color='black', size=10, symbol='circle'),
            name=f'当前 ({state["p"]:.3f}, {state["q"]:.3f})',
            hovertemplate='p=%{x:.4f}<br>q=%{y:.4f}<extra></extra>',
        ))
    fig.update_layout(
        title=dict(text='p-q 相位图 (分岔边界: 4p³+27q²=0)'),
        xaxis_title='p', yaxis_title='q',
        xaxis=dict(range=[-5, 5], zeroline=False),
        yaxis=dict(range=[-10, 10], zeroline=False),
        height=360, hovermode='closest', showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.add_hline(y=0, line=dict(color='gray', width=0.5, dash='dash'))
    fig.add_vline(x=0, line=dict(color='gray', width=0.5, dash='dash'))
    return fig


# =========================================================================
# Info Panel
# =========================================================================

def render_info_card(state, label):
    """Render color-coded state information card with metrics and roots."""
    if label.startswith('III'):
        bg, fg = '#d4edda', '#155724'
    elif label.startswith('I') or label.startswith('II'):
        bg, fg = '#f8d7da', '#721c24'
    elif '退化' in label:
        bg, fg = '#e2e3e5', '#383d41'
    else:
        bg, fg = '#fff3cd', '#856404'

    st.markdown(
        f'<div style="background:{bg};padding:12px;border-radius:8px;'
        f'margin-bottom:12px;">'
        f'<h4 style="color:{fg};margin:0 0 4px 0;">状态分类</h4>'
        f'<p style="color:{fg};font-size:1.2em;font-weight:bold;margin:0;">'
        f'{label}</p></div>',
        unsafe_allow_html=True,
    )

    def _fmt(v):
        return f'{v:.6f}' if not np.isnan(v) else 'N/A'

    delta_val = state['delta']
    delta_disp = _fmt(delta_val)
    if not np.isnan(delta_val):
        delta_disp += '  (Δ > 0)' if delta_val > EPS else ('  (Δ < 0)' if delta_val < -EPS else '  (Δ ≈ 0)')
    st.metric('Δ (Cardano)', delta_disp)
    st.metric('p', _fmt(state['p']))
    st.metric('q', _fmt(state['q']))
    st.metric('D_crit = 4b²-12ac', _fmt(state['D_crit']))
    st.markdown('##### 实根')
    if state['roots']:
        for i, r in enumerate(state['roots'], 1):
            st.markdown(f'{i}. x = {r:.4f}')
    else:
        st.markdown('无')


# =========================================================================
# Main Application
# =========================================================================

def main():
    st.set_page_config(page_title='三次曲线状态分析器', page_icon=':chart_with_upwards_trend:', layout='wide')

    defaults = {'a': 1.0, 'b': 0.0, 'c': -3.0, 'd': 0.0}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    with st.sidebar:
        st.title('三次曲线状态分析器')
        st.markdown('### 预设')
        bcols = st.columns(3)
        presets = [
            ('S型(三实根)', 1.0, 0.0, -3.0, 0.0),
            ('单调(单实根)', 1.0, 0.0, 2.0, 0.0),
            ('临界(重根)', 1.0, 0.0, 0.0, 0.0),
        ]
        icons = ['\U0001f535', '\U0001f7e2', '\U0001f7e0']
        for col, (plabel, av, bv, cv, dv), icon in zip(bcols, presets, icons):
            with col:
                if st.button(f'{icon} {plabel}', use_container_width=True, key=f'preset_{plabel}'):
                    st.session_state.a = av
                    st.session_state.b = bv
                    st.session_state.c = cv
                    st.session_state.d = dv
        st.markdown('---')
        a = st.slider('a', -200.0, 200.0, value=st.session_state.a, key='a', step=1.0)
        b = st.slider('b', -200.0, 200.0, value=st.session_state.b, key='b', step=1.0)
        c = st.slider('c', -200.0, 200.0, value=st.session_state.c, key='c', step=1.0)
        d = st.slider('d', -200.0, 200.0, value=st.session_state.d, key='d', step=1.0)

    state = compute_state(a, b, c, d)
    label = classify_label(a, state['delta'], state['D_crit'])

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(make_curve_figure(state, a, b, c, d), use_container_width=True)
    with col2:
        render_info_card(state, label)

    st.plotly_chart(make_phase_figure(state), use_container_width=True)


if __name__ == '__main__':
    main()
