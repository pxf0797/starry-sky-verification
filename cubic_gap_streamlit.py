#!/usr/bin/env python3
"""
cubic_gap_streamlit.py — 三次曲线极值差(Δy)交互式探索工具
核心: Δy = 4*(b²-3ac)^(3/2) / (27*a²), D_crit = b²-3ac > 0, f(x) = ax³+bx²+cx+d
运行: streamlit run cubic_gap_streamlit.py | 依赖: streamlit, numpy, plotly
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

# ── 核心计算 ──

def compute_gap(a, b, c):
    """Δy = 4*D_crit^(3/2)/(27*a²); 返回 None 若无极值"""
    D = b*b - 3.0*a*c
    return None if (D <= 0.0 or abs(a) < 1e-12) else 4.0*D**1.5/(27.0*a*a)


def critical_points(a, b, c, d=0.0):
    """返回状态字典含roots/inflection/extrema"""
    result = {'roots': [], 'inflection': None, 'extrema': []}
    # 实根
    all_roots = np.roots([a, b, c, d])
    result['roots'] = sorted(r.real for r in all_roots if abs(r.imag) < 1e-10)
    if abs(a) < 1e-12:
        return result
    # 拐点 x = -b/(3a)
    xi = -b / (3.0*a)
    yi = ((a*xi + b)*xi + c)*xi + d
    result['inflection'] = (xi, yi)
    # 极值点
    D = b*b - 3.0*a*c
    if D > 0.0:
        s = np.sqrt(D)
        x1 = (-b - s) / (3.0*a)   # 极大值点 (f''(x1) = -2s < 0)
        x2 = (-b + s) / (3.0*a)   # 极小值点 (f''(x2) = 2s > 0)
        y1 = ((a*x1 + b)*x1 + c)*x1 + d
        y2 = ((a*x2 + b)*x2 + c)*x2 + d
        result['extrema'] = [(x1, y1, 'max'), (x2, y2, 'min')]
    return result


# ── 曲线可视化 ──

def _f(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d


def _compute_x_range(state, a, b, c, d):
    """自动计算x轴范围（含40% padding）"""
    xs = list(state.get('roots', []))
    if state.get('inflection'):
        xs.append(state['inflection'][0])
    for x, _, _ in state.get('extrema', []):
        xs.append(x)
    if not xs:
        return -5.0, 5.0
    lo, hi = min(xs), max(xs)
    span = max(hi - lo, 0.5)
    pad = span * 0.4
    return lo - pad, hi + pad


def make_curve_figure(a, b, c, d, state):
    """绘制三次曲线 plotly figure"""
    xr = _compute_x_range(state, a, b, c, d)
    x_vals = np.linspace(xr[0], xr[1], 1000)
    y_vals = _f(x_vals, a, b, c, d)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines',
        name='f(x)', line=dict(color='blue', width=2)))

    # 实根
    roots = state.get('roots', [])
    if roots:
        fig.add_trace(go.Scatter(x=roots, y=[_f(r,a,b,c,d) for r in roots],
            mode='markers', name='实根', marker=dict(color='red', size=10)))

    # 拐点
    inf = state.get('inflection')
    if inf:
        fig.add_trace(go.Scatter(x=[inf[0]], y=[inf[1]],
            mode='markers', name='拐点', marker=dict(color='blue', size=12, symbol='triangle-up')))

    # 极值点
    for x, y, t in state.get('extrema', []):
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers',
            name=f'{"极大" if t=="max" else "极小"}', marker=dict(color='green', size=10, symbol='square')))

    fig.update_layout(height=300, margin=dict(l=20,r=20,t=30,b=20),
        xaxis_title='x', yaxis_title='f(x)')
    return fig


# ── 参数采样 ──

def sample_params(n=5000, R=200.0, method="uniform", seed=42):
    """采样满足 D_crit > 0 的参数组合, 返回 (params, dy_values)"""
    rng = np.random.RandomState(seed)
    if method in ("均匀采样", "uniform"):
        a_raw, b_raw, c_raw = rng.uniform(-R, R, (3, n))
    elif method in ("对数均匀采样", "loguniform"):
        la, lb, lc = rng.uniform(-3.0, np.log10(R), (3, n))
        a_raw = rng.choice([-1., 1.], n)*(10.**la)
        b_raw = rng.choice([-1., 1.], n)*(10.**lb)
        c_raw = rng.choice([-1., 1.], n)*(10.**lc)
    else:
        raise ValueError(f"Unknown method: {method}")
    D = b_raw*b_raw - 3.0*a_raw*c_raw
    ok = (D > 0.0) & (np.abs(a_raw) > 1e-15)
    return (np.column_stack([a_raw[ok], b_raw[ok], c_raw[ok], D[ok]]),
            4.0 * D[ok]**1.5 / (27.0 * a_raw[ok]**2))

# ── 三阶段优化搜索 (轻量版) ──

MIN_ABS_A = 1e-4

def optimize_search(R=200.0, seed=42, cb=None):
    """
    阶段1: 粗网格 9×7×7
    阶段2: Top5 随机爬山 50 步
    阶段3: 冠军邻域精细网格 10×10×10
    cb(phase, pct, msg): 进度回调
    """
    rng = np.random.RandomState(seed)

    # 阶段1: 粗网格
    if cb:
        cb(1, 5, "阶段1: 粗网格 (9×7×7)...")
    ap = np.logspace(-2.0, np.log10(R), 5)
    ag = np.concatenate([-ap[1:][::-1], ap])  # 4负 + 5正 = 9
    bg = np.linspace(-R, R, 7)
    cg = np.linspace(-R, R, 7)
    cand = []
    for a in ag:
        if abs(a) < MIN_ABS_A:
            continue
        a2 = a*a
        for b in bg:
            b2 = b*b
            for c in cg:
                D = b2 - 3.0*a*c
                if D <= 0.0:
                    continue
                cand.append((a, b, c, D, 4.0*D**1.5/(27.0*a2)))
    cand.sort(key=lambda x: x[4], reverse=True)
    if cb:
        cb(1, 30, f"阶段1: {len(cand)}有效点, 最佳Δy={cand[0][4]:.2f}")

    # 阶段2: 随机爬山
    if cb:
        cb(2, 30, "阶段2: 随机爬山 (Top5 × 50步)...")
    hill = []
    for a0, b0, c0, _, dy0 in cand[:5]:
        ac, bc, cc, dyc = a0, b0, c0, dy0
        for _ in range(50):
            ap_ = ac * (1.0 + rng.uniform(-0.5, 0.5))
            if abs(ap_) < MIN_ABS_A or abs(ap_) > R:
                continue
            bp_ = float(np.clip(bc + rng.uniform(-20., 20.), -R, R))
            cp_ = float(np.clip(cc + rng.uniform(-20., 20.), -R, R))
            Dp = bp_*bp_ - 3.0*ap_*cp_
            if Dp <= 0.0:
                continue
            dyp = 4.0*Dp**1.5 / (27.0*ap_*ap_)
            if dyp > dyc:
                ac, bc, cc, dyc = ap_, bp_, cp_, dyp
        hill.append((ac, bc, cc, dyc))
    hill.sort(key=lambda x: x[3], reverse=True)
    if cb:
        cb(2, 65, f"阶段2完成, 最佳Δy={hill[0][3]:.2f}")

    # 阶段3: 冠军邻域精细网格
    if cb:
        cb(3, 65, "阶段3: 冠军邻域精细网格 (10×10×10)...")
    ach, bch, cch = hill[0][0], hill[0][1], hill[0][2]
    ad = max(abs(ach)*0.5, MIN_ABS_A)
    af = np.linspace(ach-ad, ach+ad, 10)
    bf = np.linspace(bch-10, bch+10, 10)
    cf = np.linspace(cch-10, cch+10, 10)
    best3, best3_dy = None, -1.0
    for a in af:
        if abs(a) < MIN_ABS_A:
            continue
        a2 = a*a
        for b in bf:
            b2 = b*b
            for c in cf:
                D = b2 - 3.0*a*c
                if D <= 0.0:
                    continue
                dy = 4.0*D**1.5/(27.0*a2)
                if dy > best3_dy:
                    best3_dy, best3 = dy, (a, b, c, D, dy)

    # 汇集 Top20 (hill + 阶段3冠军 + 补充阶段1候选)
    seen = set()
    combined = []
    for src in [hill, [(best3[0], best3[1], best3[2], best3[4])],
                [(c[0], c[1], c[2], c[4]) for c in cand]]:
        for item in src:
            key = (round(item[0], 8), round(item[1], 4), round(item[2], 4))
            if key not in seen:
                seen.add(key)
                combined.append(item)
    combined.sort(key=lambda x: x[3], reverse=True)

    champion = combined[0]  # (a, b, c, dy)
    cD = champion[1]*champion[1] - 3.0*champion[0]*champion[2]
    if cb:
        cb(3, 100, f"搜索完成! 最佳Δy={champion[3]:.2f}")
    return (champion[0], champion[1], champion[2], cD, champion[3]), combined[:20]

# ── UI 辅助 ──

def dy_html(dy):
    """带颜色 HTML (绿<10, 橙10~1000, 红>1000)"""
    if dy is None:
        return '<span style="color:#999;">Δy = N/A (无极值)</span>'
    c, t = ("#2ecc71","小") if dy<10 else (("#f39c12","中") if dy<1000 else ("#e74c3c","大"))
    return (f'<span style="color:{c};font-weight:bold;font-size:22px;">Δy={dy:.6f}</span><br>'
            f'<span style="color:#999;font-size:12px;">{t} Δy</span>')


def subsample(arr, n, seed=42):
    """若数组超过 n 则随机子采样"""
    if len(arr) > n:
        return arr[np.random.RandomState(seed).choice(len(arr), n, replace=False)]
    return arr


def mk_scatter(x, y, c, cm, xl, yl, tt, xt="linear", yt="log", h=320):
    """创建统一风格的 plotly 散点图"""
    fig = go.Figure(go.Scattergl(x=x, y=y, mode="markers",
        marker=dict(size=3, color=c, colorscale=cm,
                    colorbar=dict(title=cm), showscale=True)))
    fig.update_xaxes(type=xt, title=xl)
    fig.update_yaxes(type=yt, title=yl)
    fig.update_layout(title=tt, height=h, margin=dict(l=40,r=20,t=40,b=40))
    return fig

# ── 页面配置与 session_state 初始化 ──

st.set_page_config(page_title="三次曲线极值差(Δy)动态探索", page_icon="📈", layout="wide")

for k, v in dict(a=1., b=0., c=-3., d=0., sampled=False, params=None,
                 dy_values=None, search_done=False, champion=None, top20=None,
                 n_sampled=0, mean_dy=0., median_dy=0., p99_dy=0., pending_load=None).items():
    if k not in st.session_state:
        st.session_state[k] = v

# 预计算小数据集 (Tab2 备用)
if "default_params" not in st.session_state:
    st.session_state["default_params"], st.session_state["default_dy"] = \
        sample_params(1000, 200., "uniform", 0)

# 处理"加载"按钮的延迟同步（避免widget后修改session_state冲突）
if st.session_state.get("pending_load"):
    ai, bi, ci = st.session_state["pending_load"]
    st.session_state.update(a=float(ai), b=float(bi), c=float(ci))
    st.session_state["pending_load"] = None

# ── 标题 ──

st.title(":chart_with_upwards_trend: 三次曲线极值差 (Δy) 动态探索")
st.markdown(r"$f(x)=ax^3+bx^2+cx+d$　$\displaystyle\Delta y=\frac{4(b^2-3ac)^{3/2}}{27a^2}$")

# ═══════════════════════════════ 侧边栏 ═══════════════════════════════

with st.sidebar:
    st.subheader(":triangular_ruler: 预设参数")

    # 预设按钮 (在滑块之前, 避免 session_state 冲突)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("标准三次", use_container_width=True):
            st.session_state.update(a=1., b=0., c=-3., d=0.)
    with c2:
        if st.button("对称双峰", use_container_width=True):
            st.session_state.update(a=1., b=-3., c=0., d=0.)
    with c3:
        if st.button("不对称型", use_container_width=True):
            st.session_state.update(a=1., b=2., c=-5., d=0.)

    st.divider()
    st.subheader(":control_knobs: 系数滑块")

    a_val = st.slider("a (三次项)", -200., 200.,
                       st.session_state["a"], step=1., key="a")
    b_val = st.slider("b (二次项)", -200., 200.,
                       st.session_state["b"], step=1., key="b")
    c_val = st.slider("c (一次项)", -200., 200.,
                       st.session_state["c"], step=1., key="c")
    d_val = st.slider("d (常数项)", -200., 200.,
                       st.session_state["d"], step=1., key="d")

    st.divider()
    st.subheader(":bar_chart: 当前参数分析")

    dy_c = compute_gap(a_val, b_val, c_val)
    cp = critical_points(a_val, b_val, c_val, d_val)
    Dc = b_val*b_val - 3.0*a_val*c_val

    st.markdown(dy_html(dy_c), unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    m1.metric("D_crit", f"{Dc:.4f}")
    m2.metric("存在极值", "是" if dy_c else "否")

    if cp['extrema']:
        x1,y1,t1 = cp['extrema'][0]
        x2,y2,t2 = cp['extrema'][1]
        s1,s2 = "极大","极小"
        st.caption(f"极值点① ({s1}): x={x1:.4f}, y={y1:.4f}")
        st.caption(f"极值点② ({s2}): x={x2:.4f}, y={y2:.4f}")
    else:
        st.info("D_crit ≤ 0: 无极值点 (单调曲线)", icon="ℹ️")

    st.divider()
    st.subheader(":curly_loop: 三次曲线")
    cfig = make_curve_figure(a_val, b_val, c_val, d_val, cp)
    st.plotly_chart(cfig, use_container_width=True)

# ═══════════════════════════════ 主区域 ═══════════════════════════════

t1, t2, t3 = st.tabs([":microscope: 采样分析", ":link: 参数关系", ":rocket: 优化搜索"])

# ── Tab1: 采样分析 ──

with t1:
    st.subheader("参数空间采样与 Δy 分布分析")

    cc, cb_ = st.columns([3, 1])
    with cc:
        method = st.selectbox("采样方案", ["均匀采样","对数均匀采样"], key="sampling_method")
        n_samp = st.slider("采样数量", 1000, 30000, 5000, step=1000, key="n_samples")
        rr = st.slider("参数范围 R [-R,R]", 50, 500, 200, step=10, key="range_r")
    with cb_:
        st.write("\n"*2)
        start = st.button(":rocket: 开始采样", type="primary", use_container_width=True)

    if start:
        with st.status("采样进行中...", expanded=True) as s:
            st.write("正在过滤 D_crit > 0 ...")
            pb = st.progress(0, text="初始化...")
            mk = "uniform" if method == "均匀采样" else "loguniform"
            pb.progress(20, text="采样中...")
            time.sleep(0.05)
            p, dy = sample_params(n=n_samp, R=rr, method=mk, seed=42)
            pb.progress(70, text="计算统计量...")
            nv = len(dy)
            md = float(np.mean(dy))
            med = float(np.median(dy))
            p99 = float(np.percentile(dy, 99))
            st.session_state.update(sampled=True, params=p, dy_values=dy,
                n_sampled=nv, mean_dy=md, median_dy=med, p99_dy=p99)
            pb.progress(100, text="完成!")
            s.update(label=f"采样完成: 有效 {nv}/{n_samp}", state="complete", expanded=True)

    if st.session_state.get("sampled"):
        p, dy = st.session_state["params"], st.session_state["dy_values"]
        nv, md, med, p99 = (st.session_state[k] for k in
                            ("n_sampled","mean_dy","median_dy","p99_dy"))

        st.subheader("统计摘要")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("有效样本数", f"{nv:,}")
        c2.metric("均值 Δy", f"{md:.4f}")
        c3.metric("中位数 Δy", f"{med:.4f}")
        c4.metric("99分位 Δy", f"{p99:.4f}")

        si = subsample(np.arange(len(dy)), 5000)
        ps, dys = p[si], dy[si]

        # log10(Δy) 直方图
        st.subheader("log₁₀(Δy) 分布")
        fh = go.Figure(go.Histogram(
            x=np.log10(np.maximum(dy,1e-30)), nbinsx=30,
            marker=dict(color="steelblue", line=dict(color="white", width=.5))))
        fh.update_layout(title="log₁₀(Δy) 分布", xaxis_title="log₁₀(Δy)",
                         yaxis_title="频数", height=350, bargap=.05,
                         margin=dict(l=40,r=20,t=40,b=40))
        st.plotly_chart(fh, use_container_width=True)

        # 散点图: Δy vs a, Δy vs |b|
        st.subheader("Δy 与参数关系")
        s1, s2 = st.columns(2)
        with s1:
            fa = go.Figure(go.Scattergl(x=ps[:,0], y=dys, mode="markers",
                marker=dict(size=3, color=ps[:,3], colorscale="Viridis",
                            colorbar=dict(title="D_crit"), showscale=True)))
            fa.update_xaxes(type="linear", title="a (linear)")
            fa.update_yaxes(type="log", title="Δy (log)")
            fa.update_layout(title="Δy vs a", height=350,
                             margin=dict(l=40,r=20,t=40,b=40))
            st.plotly_chart(fa, use_container_width=True)
        with s2:
            fb = go.Figure(go.Scattergl(x=np.abs(ps[:,1]), y=dys, mode="markers",
                marker=dict(size=3, color="mediumseagreen")))
            fb.update_xaxes(type="log", title="|b| (log)")
            fb.update_yaxes(type="log", title="Δy (log)")
            fb.update_layout(title="Δy vs |b|", height=350,
                             margin=dict(l=40,r=20,t=40,b=40))
            st.plotly_chart(fb, use_container_width=True)
    else:
        st.info("点击 **:rocket: 开始采样** 按钮进行参数空间探索", icon="💡")

# ── Tab2: 参数关系 ──

with t2:
    st.subheader("Δy 与各参数的依赖关系")

    if st.session_state.get("sampled"):
        p, dy = st.session_state["params"], st.session_state["dy_values"]
        st.caption(":bar_chart: 数据来源: Tab1 采样数据")
    else:
        p, dy = st.session_state["default_params"], st.session_state["default_dy"]
        st.caption(":bar_chart: 数据来源: 预计算数据集 (1000 组)")

    si = subsample(np.arange(len(dy)), 5000)
    p, dy = p[si], dy[si]
    aa = np.abs(p[:,0])

    # 三个散点图垂直排列
    st.plotly_chart(mk_scatter(p[:,0], dy, p[:,3], "Viridis",
        "a (linear)", "Δy (log)", "Δy vs a (色标 = D_crit)"),
        use_container_width=True)

    st.plotly_chart(mk_scatter(p[:,1], dy, aa, "Plasma",
        "b", "Δy (log)", "Δy vs b (色标 = |a|)", xt="linear"),
        use_container_width=True)

    st.plotly_chart(mk_scatter(p[:,3], dy, aa, "Plasma",
        "D_crit = b²-3ac (log)", "Δy (log)", "Δy vs D_crit (log-log)"),
        use_container_width=True)

# ── Tab3: 优化搜索 ──

with t3:
    st.subheader("三阶段搜索: 寻找最大 Δy")
    st.markdown("**策略:** ① 粗网格 9×7×7 ② 随机爬山 Top5×50步 ③ 精细网格 10×10×10")

    if st.button(":rocket: 搜索最大 Δy", type="primary"):
        st.session_state.update(search_done=False, champion=None, top20=None)
        sp = st.empty()
        pb = st.progress(0, text="准备...")

        def cb(_, pct, msg):
            pb.progress(pct, text=msg)
            sp.caption(msg)

        champ, top20 = optimize_search(R=200., seed=42, cb=cb)
        st.session_state.update(search_done=True, champion=champ, top20=top20)
        pb.empty()
        sp.empty()

    if st.session_state.get("search_done"):
        ch = st.session_state["champion"]
        top20 = st.session_state["top20"]
        a_o,b_o,c_o,D_o,dy_o = ch

        st.success(
            f":trophy: **最优参数** "
            f"a={a_o:.10f} | b={b_o:.6f} | "
            f"c={c_o:.6f} | D_crit={D_o:.4f} | Δy={dy_o:.2f}",
            icon="🎯",
        )

        # Top20 表格 (columns 布局, 每行含加载按钮)
        st.subheader("Top 20 参数组合")
        hc = st.columns([.6,1.8,1.8,1.8,1.8,2.,2.,1.2])
        for c_, l in zip(hc, ["排名","a","b","c","D_crit","Δy","log₁₀(Δy)","操作"]):
            c_.markdown(f"**{l}**")
        st.markdown("<hr style='margin:0'>", unsafe_allow_html=True)

        for i, (ai,bi,ci,dyi) in enumerate(top20):
            Di = bi*bi - 3.*ai*ci
            rc = st.columns([.6,1.8,1.8,1.8,1.8,2.,2.,1.2])
            rc[0].write(f"{i+1}")
            rc[1].write(f"{ai:.8f}")
            rc[2].write(f"{bi:.4f}")
            rc[3].write(f"{ci:.4f}")
            rc[4].write(f"{Di:.2f}")
            rc[5].write(f"{dyi:.2f}")
            rc[6].write(f"{np.log10(max(dyi,1e-30)):.4f}")
            if rc[7].button("加载", key=f"lr_{i}", use_container_width=True):
                st.session_state["pending_load"] = (float(ai), float(bi), float(ci))
                st.rerun()
    else:
        st.info("点击 **:rocket: 搜索最大 Δy** 开始三阶段优化", icon="💡")
