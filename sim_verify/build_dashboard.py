#!/usr/bin/env python3
"""
Build a self-contained HTML dashboard for 星空策略 (Starry Sky Strategy)
simulation verification experiments.

Reads 10 PNG chart images + 3 JSON result files and produces a single
self-contained HTML file with all images embedded as base64 data URIs.
"""

import base64
import json
import os
from pathlib import Path

BASE_DIR = Path("/Users/xfpan/claude/research/sim_verify")

# ── Load JSON results ──────────────────────────────────────────────────────

with open(BASE_DIR / "layer1" / "results_layer1.json") as f:
    L1 = json.load(f)
with open(BASE_DIR / "layer2" / "results_layer2.json") as f:
    L2 = json.load(f)
with open(BASE_DIR / "layer3" / "results_layer3.json") as f:
    L3 = json.load(f)

# ── Base64-encode images ───────────────────────────────────────────────────

def b64_img(path):
    p = BASE_DIR / path
    with open(p, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{data}"

IMGS = {
    "exp1_1": b64_img("layer1/exp1_1_optimal_order.png"),
    "exp1_2": b64_img("layer1/exp1_2_dimensional_gap.png"),
    "exp1_3": b64_img("layer1/exp1_3_snr_decay.png"),
    "exp1_4": b64_img("layer1/exp1_4_svd_stability.png"),
    "exp2_1": b64_img("layer2/exp2_1_frozen_vs_recalc.png"),
    "exp2_2": b64_img("layer2/exp2_2_half_life_search.png"),
    "exp2_3": b64_img("layer2/exp2_3_sigmoid_vs_hard.png"),
    "exp2_5": b64_img("layer2/exp2_5_dynamic_vs_static.png"),
    "exp3_1": b64_img("layer3/exp3_1_market_stress.png"),
    "exp3_2": b64_img("layer3/exp3_2_breakpoint_response.png"),
}

# ── Experiment data ────────────────────────────────────────────────────────

EXP1_1_RMSE = L1["exp1_1"]["rmse_table"]
EXP1_1_CONC = L1["exp1_1"]["conclusion"]

EXP1_2_RATIOS = L1["exp1_2"]["rmse_ratios"]
EXP1_2_CONC = L1["exp1_2"]["conclusion"]

EXP1_3_RHO = L1["exp1_3"]["estimated_rho"]
EXP1_3_CI = L1["exp1_3"]["rho_confidence_interval"]
EXP1_3_CONC = L1["exp1_3"]["conclusion"]

EXP1_4_ERR_POLY = L1["exp1_4"]["polyfit_max_error"]
EXP1_4_ERR_SVD = L1["exp1_4"]["svd_max_error"]
EXP1_4_CONC = L1["exp1_4"]["conclusion"]

EXP2_1_FROZEN = L2["exp2_1"]["frozen_sharpe_by_zone"]
EXP2_1_RECALC = L2["exp2_1"]["recalc_sharpe_by_zone"]
EXP2_1_CI = L2["exp2_1"]["sharpe_diff_ci"]
EXP2_1_CONC = L2["exp2_1"]["conclusion"]

EXP2_2_TAU = L2["exp2_2"]["tau_optimal"]
EXP2_2_CI = L2["exp2_2"]["tau_ci"]
EXP2_2_LOSS = L2["exp2_2"]["loss_curve"]
EXP2_2_CONC = L2["exp2_2"]["conclusion"]

EXP2_3_SIG = L2["exp2_3"]["sigmoid_sharpe"]
EXP2_3_HARD = L2["exp2_3"]["hard_sharpe"]
EXP2_3_CONC = L2["exp2_3"]["conclusion"]

EXP2_5_DYN = L2["exp2_5"]["dynamic_pareto"]
EXP2_5_STATIC = L2["exp2_5"]["static_pareto"]
EXP2_5_CONC = L2["exp2_5"]["conclusion"]

EXP3_1_SLIPPAGE = L3["exp3_1"]["slippage_cost_bps"]
EXP3_1_FILL = L3["exp3_1"]["fill_rate_limit_orders"]
EXP3_1_R3 = L3["exp3_1"]["iron_rule_3_triggers"]
EXP3_1_R5 = L3["exp3_1"]["iron_rule_5_triggers"]
EXP3_1_CONC = L3["exp3_1"]["conclusion"]

EXP3_2_SLOPE = L3["exp3_2"]["dT_dDelta_slope"]
EXP3_2_FP = L3["exp3_2"]["false_positive_rate"]
EXP3_2_TRIGGER = L3["exp3_2"]["trigger_times_by_delta"]
EXP3_2_CONC = L3["exp3_2"]["conclusion"]

DYNAMIC_MAX_SHARPE = max(p["sharpe"] for p in EXP2_5_DYN)
STATIC_MAX_SHARPE = max(p["sharpe"] for p in EXP2_5_STATIC)

# ── Build HTML ─────────────────────────────────────────────────────────────

HTML = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>星空策略模拟验证仪表盘</title>
<style>
  :root {{
    --bg: #0a0e27;
    --card: #1a1f3a;
    --card-hover: #222850;
    --accent: #4fc3f7;
    --accent-dim: #2a5f8a;
    --text: #e0e6f0;
    --text-dim: #8892b0;
    --green: #4caf50;
    --yellow: #ff9800;
    --red: #f44336;
    --border: #2a2f4a;
    --radius: 12px;
    --shadow: 0 4px 24px rgba(0,0,0,0.4);
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", "Helvetica Neue", sans-serif;
    line-height: 1.6;
    min-height: 100vh;
  }}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar {{ width: 6px; }}
  ::-webkit-scrollbar-track {{ background: var(--bg); }}
  ::-webkit-scrollbar-thumb {{ background: var(--accent-dim); border-radius: 3px; }}

  /* ── Header ── */
  header {{
    background: linear-gradient(135deg, #0d1338 0%, #162050 40%, #1a2d6a 100%);
    border-bottom: 1px solid var(--border);
    padding: 32px 40px 24px;
    position: relative;
    overflow: hidden;
  }}
  header::before {{
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(79,195,247,0.06) 0%, transparent 60%),
                radial-gradient(circle at 70% 30%, rgba(79,195,247,0.04) 0%, transparent 50%);
    pointer-events: none;
  }}
  header h1 {{
    font-size: 28px; font-weight: 700; letter-spacing: 2px;
    position: relative;
  }}
  header h1 span {{ color: var(--accent); }}
  header .subtitle {{
    color: var(--text-dim); font-size: 15px; margin-top: 6px;
    display: flex; align-items: center; gap: 12px; position: relative;
  }}
  header .subtitle .dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
  .dot.green {{ background: var(--green); box-shadow: 0 0 8px var(--green); }}

  /* ── Container ── */
  .container {{ max-width: 1440px; margin: 0 auto; padding: 24px 32px; }}

  /* ── Key Metrics ── */
  .metrics-row {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 28px;
  }}
  .metric-card {{
    background: var(--card); border-radius: var(--radius); padding: 20px 24px;
    border: 1px solid var(--border); position: relative; overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
  }}
  .metric-card:hover {{ transform: translateY(-2px); box-shadow: var(--shadow); }}
  .metric-card .label {{
    font-size: 12px; text-transform: uppercase; letter-spacing: 1.5px;
    color: var(--text-dim); margin-bottom: 6px;
  }}
  .metric-card .value {{
    font-size: 26px; font-weight: 700; color: var(--accent); font-variant-numeric: tabular-nums;
  }}
  .metric-card .sub {{
    font-size: 13px; color: var(--text-dim); margin-top: 4px;
  }}
  .metric-card .glow {{
    position: absolute; top: -40px; right: -40px; width: 120px; height: 120px;
    border-radius: 50%; opacity: 0.08; pointer-events: none;
  }}
  .metric-card:nth-child(1) .glow {{ background: var(--accent); }}
  .metric-card:nth-child(2) .glow {{ background: var(--green); }}
  .metric-card:nth-child(3) .glow {{ background: var(--yellow); }}
  .metric-card:nth-child(4) .glow {{ background: var(--accent); }}

  /* ── Tab Navigation ── */
  .tabs {{
    display: flex; gap: 4px; margin-bottom: 0; background: var(--card);
    border-radius: var(--radius) var(--radius) 0 0; border: 1px solid var(--border);
    border-bottom: none; overflow: hidden;
  }}
  .tab-btn {{
    flex: 1; padding: 14px 20px; background: transparent; color: var(--text-dim);
    border: none; font-size: 14px; font-weight: 600; cursor: pointer;
    transition: all 0.25s; position: relative; font-family: inherit;
    letter-spacing: 0.5px;
  }}
  .tab-btn:hover {{ color: var(--text); background: rgba(79,195,247,0.05); }}
  .tab-btn.active {{
    color: var(--accent); background: rgba(79,195,247,0.1);
  }}
  .tab-btn.active::after {{
    content: ''; position: absolute; bottom: 0; left: 20%; width: 60%;
    height: 2px; background: var(--accent); border-radius: 1px 1px 0 0;
  }}

  /* ── Tab Content ── */
  .tab-content {{
    background: var(--card); border: 1px solid var(--border); border-top: none;
    border-radius: 0 0 var(--radius) var(--radius); padding: 28px;
    display: none;
  }}
  .tab-content.active {{ display: block; }}

  /* ── Experiment Grid ── */
  .exp-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
    gap: 20px;
  }}
  .exp-card {{
    background: rgba(255,255,255,0.03); border: 1px solid var(--border);
    border-radius: 10px; overflow: hidden; transition: all 0.25s;
    cursor: pointer;
  }}
  .exp-card:hover {{ border-color: var(--accent-dim); background: rgba(255,255,255,0.05); }}
  .exp-card-header {{
    padding: 16px 20px; display: flex; align-items: flex-start;
    justify-content: space-between; gap: 12px;
  }}
  .exp-card-header .info {{ flex: 1; min-width: 0; }}
  .exp-card-header .info h3 {{
    font-size: 14px; font-weight: 600; margin-bottom: 2px; color: var(--text);
  }}
  .exp-card-header .info .exp-id {{
    font-size: 11px; color: var(--text-dim); letter-spacing: 0.5px;
  }}
  .exp-card-header .info .key-val {{
    margin-top: 6px; font-size: 15px; font-weight: 600; color: var(--accent);
  }}
  .exp-card-header .badge {{
    flex-shrink: 0; padding: 4px 12px; border-radius: 20px; font-size: 12px;
    font-weight: 700; letter-spacing: 1px;
  }}
  .badge.confirmed {{ background: rgba(76,175,80,0.15); color: var(--green); border: 1px solid rgba(76,175,80,0.3); }}
  .badge.partial {{ background: rgba(255,152,0,0.15); color: var(--yellow); border: 1px solid rgba(255,152,0,0.3); }}
  .badge.rejected {{ background: rgba(244,67,54,0.15); color: var(--red); border: 1px solid rgba(244,67,54,0.3); }}
  .badge.major {{ background: rgba(79,195,247,0.15); color: var(--accent); border: 1px solid rgba(79,195,247,0.3); }}

  .exp-card-body {{
    max-height: 0; overflow: hidden; transition: max-height 0.4s ease;
  }}
  .exp-card.expanded .exp-card-body {{ max-height: 2000px; }}
  .exp-card-body-inner {{ padding: 0 20px 20px; }}
  .exp-card-body-inner img {{
    width: 100%; border-radius: 8px; margin-bottom: 12px;
    border: 1px solid var(--border); background: #0a0e27;
  }}
  .exp-card-body-inner .conclusion {{
    font-size: 13px; color: var(--text-dim); line-height: 1.7;
    background: rgba(0,0,0,0.2); border-radius: 8px; padding: 12px 16px;
    border-left: 3px solid var(--accent-dim);
  }}
  .exp-card .expand-icon {{
    font-size: 14px; color: var(--text-dim); transition: transform 0.3s;
    margin-left: auto; flex-shrink: 0;
  }}
  .exp-card.expanded .expand-icon {{ transform: rotate(180deg); }}

  /* ── Conclusion Panel ── */
  .conclusion-panel {{
    background: var(--card); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 28px 32px; margin-top: 28px;
  }}
  .conclusion-panel h2 {{
    font-size: 18px; margin-bottom: 20px; color: var(--text);
    display: flex; align-items: center; gap: 10px;
  }}
  .conclusion-panel h2::before {{
    content: '◆'; color: var(--accent); font-size: 14px;
  }}
  .conc-grid {{
    display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;
  }}
  .conc-section {{
    background: rgba(255,255,255,0.03); border-radius: 10px; padding: 18px 20px;
    border: 1px solid var(--border);
  }}
  .conc-section h3 {{
    font-size: 14px; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;
  }}
  .conc-section ul {{ list-style: none; padding: 0; }}
  .conc-section li {{
    font-size: 13px; color: var(--text-dim); padding: 5px 0;
    padding-left: 18px; position: relative; line-height: 1.6;
  }}
  .conc-section li::before {{
    content: ''; position: absolute; left: 0; top: 12px;
    width: 6px; height: 6px; border-radius: 50%;
  }}
  .conc-section.confirmed li::before {{ background: var(--green); }}
  .conc-section.recalibrate li::before {{ background: var(--yellow); }}
  .conc-section.inconclusive li::before {{ background: var(--text-dim); }}
  .conc-section.recommend li::before {{ background: var(--accent); }}

  .conc-section.confirmed h3 {{ color: var(--green); }}
  .conc-section.recalibrate h3 {{ color: var(--yellow); }}
  .conc-section.inconclusive h3 {{ color: var(--text-dim); }}
  .conc-section.recommend h3 {{ color: var(--accent); }}

  /* ── Animations ── */
  @keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(12px); }}
    to {{ opacity: 1; transform: translateY(0); }}
  }}
  .tab-content.active .exp-card {{
    animation: fadeInUp 0.4s ease forwards; opacity: 0;
  }}
  .tab-content.active .exp-card:nth-child(1) {{ animation-delay: 0.05s; }}
  .tab-content.active .exp-card:nth-child(2) {{ animation-delay: 0.10s; }}

  /* ── Responsive ── */
  @media (max-width: 900px) {{
    .metrics-row {{ grid-template-columns: repeat(2, 1fr); }}
    .exp-grid {{ grid-template-columns: 1fr; }}
    .conc-grid {{ grid-template-columns: 1fr; }}
    .container {{ padding: 16px; }}
    header {{ padding: 24px 20px; }}
    .tab-content {{ padding: 16px; }}
  }}
  @media (max-width: 500px) {{
    .metrics-row {{ grid-template-columns: 1fr; }}
    header h1 {{ font-size: 20px; }}
    .tab-btn {{ font-size: 12px; padding: 10px 8px; }}
  }}

  /* ── Counter animation ── */
  .count-up {{ display: inline-block; }}
</style>
</head>
<body>

<!-- ═══ HEADER ═══ -->
<header>
  <h1>星空策略 <span>模拟验证仪表盘</span></h1>
  <div class="subtitle">
    <span class="dot green"></span>
    10 / 10 实验已完成 &middot; 数学验证层 &middot; 统计验证层 &middot; 实战模拟层
  </div>
</header>

<div class="container">

<!-- ═══ KEY METRICS ═══ -->
<div class="metrics-row">
  <div class="metric-card">
    <div class="glow"></div>
    <div class="label">最优半衰期 &tau;*</div>
    <div class="value"><span class="count-up" data-target="4.7" data-decimals="1">0</span> min</div>
    <div class="sub">95% CI [4.5, 4.8] &mdash; 原假设 15min 被推翻</div>
  </div>
  <div class="metric-card">
    <div class="glow"></div>
    <div class="label">多项式最优阶数</div>
    <div class="value"><span class="count-up" data-target="3" data-decimals="0">0</span></div>
    <div class="sub">三次最优，k=4 误差 3-4x，k=5 误差 10-12x</div>
  </div>
  <div class="metric-card">
    <div class="glow"></div>
    <div class="label">维度鸿沟 (P<sub>0</sub> 方差 1000)</div>
    <div class="value"><span class="count-up" data-target="19.3" data-decimals="1">0</span>x</div>
    <div class="sub">P-space vs Z-space RMSE 比值</div>
  </div>
  <div class="metric-card">
    <div class="glow"></div>
    <div class="label">动态 vs 静态阈值 Sharpe</div>
    <div class="value"><span class="count-up" data-target="1.19" data-decimals="2">0</span> / <span class="count-up" data-target="0.30" data-decimals="2">0</span></div>
    <div class="sub">动态阈值完全主导 Pareto 前沿</div>
  </div>
</div>

<!-- ═══ TABS ═══ -->
<div class="tabs" id="tabNav">
  <button class="tab-btn active" data-tab="layer1">层次一：数学验证</button>
  <button class="tab-btn" data-tab="layer2">层次二：统计验证</button>
  <button class="tab-btn" data-tab="layer3">层次三：实战模拟</button>
  <button class="tab-btn" data-tab="conclusion">综合结论</button>
</div>

<!-- ═══ TAB 1: LAYER 1 ═══ -->
<div class="tab-content active" id="tab-layer1">
  <div class="exp-grid">
    <!-- Exp 1.1 -->
    <div class="exp-card" onclick="toggleExp(this)">
      <div class="exp-card-header">
        <div class="info">
          <div class="exp-id">EXP 1.1 &middot; 最优多项式阶数</div>
          <h3>三次多项式在所有噪声水平下均最优</h3>
          <div class="key-val">&sigma;=0.001: k=3 RMSE=0.0045 &middot; k=5 RMSE=47.38</div>
        </div>
        <span class="badge confirmed">&#10004; 证实</span>
        <span class="expand-icon">&#9660;</span>
      </div>
      <div class="exp-card-body">
        <div class="exp-card-body-inner">
          <img src="{IMGS['exp1_1']}" alt="Exp 1.1 — Optimal Order">
          <div class="conclusion">{EXP1_1_CONC}</div>
        </div>
      </div>
    </div>

    <!-- Exp 1.2 -->
    <div class="exp-card" onclick="toggleExp(this)">
      <div class="exp-card-header">
        <div class="info">
          <div class="exp-id">EXP 1.2 &middot; 维度鸿沟</div>
          <h3>P<sub>0</sub> 方差越大，维度鸿沟越显著</h3>
          <div class="key-val">&sigma;²=10: 2.3x &middot; &sigma;²=100: 6.2x &middot; &sigma;²=1000: 19.3x</div>
        </div>
        <span class="badge confirmed">&#10004; 证实</span>
        <span class="expand-icon">&#9660;</span>
      </div>
      <div class="exp-card-body">
        <div class="exp-card-body-inner">
          <img src="{IMGS['exp1_2']}" alt="Exp 1.2 — Dimensional Gap">
          <div class="conclusion">{EXP1_2_CONC}</div>
        </div>
      </div>
    </div>

    <!-- Exp 1.3 -->
    <div class="exp-card" onclick="toggleExp(this)">
      <div class="exp-card-header">
        <div class="info">
          <div class="exp-id">EXP 1.3 &middot; SNR 衰减</div>
          <h3>SNR 随阶数指数衰减，&rho; &asymp; 9.76</h3>
          <div class="key-val">&rho; = {EXP1_3_RHO} &middot; 95% CI [{EXP1_3_CI[0]}, {EXP1_3_CI[1]}]</div>
        </div>
        <span class="badge partial">&#9888; 待校准</span>
        <span class="expand-icon">&#9660;</span>
      </div>
      <div class="exp-card-body">
        <div class="exp-card-body-inner">
          <img src="{IMGS['exp1_3']}" alt="Exp 1.3 — SNR Decay">
          <div class="conclusion">{EXP1_3_CONC}</div>
        </div>
      </div>
    </div>

    <!-- Exp 1.4 -->
    <div class="exp-card" onclick="toggleExp(this)">
      <div class="exp-card-header">
        <div class="info">
          <div class="exp-id">EXP 1.4 &middot; SVD vs 正规方程</div>
          <h3>SVD 略优于正规方程，两者均在可用范围内</h3>
          <div class="key-val">SVD 最大误差 {EXP1_4_ERR_SVD} &middot; 正规方程 {EXP1_4_ERR_POLY}</div>
        </div>
        <span class="badge confirmed">&#10004; 证实</span>
        <span class="expand-icon">&#9660;</span>
      </div>
      <div class="exp-card-body">
        <div class="exp-card-body-inner">
          <img src="{IMGS['exp1_4']}" alt="Exp 1.4 — SVD Stability">
          <div class="conclusion">{EXP1_4_CONC}</div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ═══ TAB 2: LAYER 2 ═══ -->
<div class="tab-content" id="tab-layer2">
  <div class="exp-grid">
    <!-- Exp 2.1 -->
    <div class="exp-card" onclick="toggleExp(this)">
      <div class="exp-card-header">
        <div class="info">
          <div class="exp-id">EXP 2.1 &middot; Frozen vs Recalc</div>
          <h3>整体无显著差异，但 Frozen 在 Zone 3 更优</h3>
          <div class="key-val">Sharpe diff CI [{EXP2_1_CI[0]}, {EXP2_1_CI[1]}] &middot; Zone3 Frozen {EXP2_1_FROZEN['zone3']:.2f} vs Recalc {EXP2_1_RECALC['zone3']:.2f}</div>
        </div>
        <span class="badge partial">&#9888; 待校准</span>
        <span class="expand-icon">&#9660;</span>
      </div>
      <div class="exp-card-body">
        <div class="exp-card-body-inner">
          <img src="{IMGS['exp2_1']}" alt="Exp 2.1 — Frozen vs Recalc">
          <div class="conclusion">{EXP2_1_CONC}</div>
        </div>
      </div>
    </div>

    <!-- Exp 2.2 -->
    <div class="exp-card" onclick="toggleExp(this)">
      <div class="exp-card-header">
        <div class="info">
          <div class="exp-id">EXP 2.2 &middot; 半衰期搜索</div>
          <h3>&tau;* = 4.7 min — 原假设 15min 被推翻</h3>
          <div class="key-val">&tau;* = {EXP2_2_TAU} min &middot; 95% CI [{EXP2_2_CI[0]}, {EXP2_2_CI[1]}]</div>
        </div>
        <span class="badge major">&#9650; 重要发现</span>
        <span class="expand-icon">&#9660;</span>
      </div>
      <div class="exp-card-body">
        <div class="exp-card-body-inner">
          <img src="{IMGS['exp2_2']}" alt="Exp 2.2 — Half-Life Search">
          <div class="conclusion">{EXP2_2_CONC}</div>
        </div>
      </div>
    </div>

    <!-- Exp 2.3 -->
    <div class="exp-card" onclick="toggleExp(this)">
      <div class="exp-card-header">
        <div class="info">
          <div class="exp-id">EXP 2.3 &middot; SIGMOID vs Hard</div>
          <h3>合成数据上性能接近，真实数据可能不同</h3>
          <div class="key-val">SIGMOID Sharpe {EXP2_3_SIG} vs Hard {EXP2_3_HARD}</div>
        </div>
        <span class="badge partial">&#9888; 待校准</span>
        <span class="expand-icon">&#9660;</span>
      </div>
      <div class="exp-card-body">
        <div class="exp-card-body-inner">
          <img src="{IMGS['exp2_3']}" alt="Exp 2.3 — Sigmoid vs Hard">
          <div class="conclusion">{EXP2_3_CONC}</div>
        </div>
      </div>
    </div>

    <!-- Exp 2.5 -->
    <div class="exp-card" onclick="toggleExp(this)">
      <div class="exp-card-header">
        <div class="info">
          <div class="exp-id">EXP 2.5 &middot; 动态 vs 静态阈值</div>
          <h3>动态阈值完全主导静态</h3>
          <div class="key-val">动态最高 Sharpe {DYNAMIC_MAX_SHARPE:.2f} vs 静态 {STATIC_MAX_SHARPE:.2f}</div>
        </div>
        <span class="badge confirmed">&#10004; 证实</span>
        <span class="expand-icon">&#9660;</span>
      </div>
      <div class="exp-card-body">
        <div class="exp-card-body-inner">
          <img src="{IMGS['exp2_5']}" alt="Exp 2.5 — Dynamic vs Static">
          <div class="conclusion">{EXP2_5_CONC}</div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ═══ TAB 3: LAYER 3 ═══ -->
<div class="tab-content" id="tab-layer3">
  <div class="exp-grid">
    <!-- Exp 3.1 -->
    <div class="exp-card" onclick="toggleExp(this)">
      <div class="exp-card-header">
        <div class="info">
          <div class="exp-id">EXP 3.1 &middot; 市场摩擦压力测试</div>
          <h3>摩擦成本 ~3bps，限价单成交率 47%</h3>
          <div class="key-val">Slippage {EXP3_1_SLIPPAGE:.2f} bps &middot; Fill rate {EXP3_1_FILL*100:.1f}% &middot; IR3 {EXP3_1_R3} 次 &middot; IR5 {EXP3_1_R5} 次</div>
        </div>
        <span class="badge confirmed">&#10004; 证实</span>
        <span class="expand-icon">&#9660;</span>
      </div>
      <div class="exp-card-body">
        <div class="exp-card-body-inner">
          <img src="{IMGS['exp3_1']}" alt="Exp 3.1 — Market Stress">
          <div class="conclusion">{EXP3_1_CONC}</div>
        </div>
      </div>
    </div>

    <!-- Exp 3.2 -->
    <div class="exp-card" onclick="toggleExp(this)">
      <div class="exp-card-header">
        <div class="info">
          <div class="exp-id">EXP 3.2 &middot; 断点响应测试</div>
          <h3>dT<sub>trigger</sub>/d&delta; &lt; 0 严格成立</h3>
          <div class="key-val">斜率 {EXP3_2_SLOPE:.1f} &middot; 假阳性率 {EXP3_2_FP*100:.1f}%</div>
        </div>
        <span class="badge confirmed">&#10004; 证实</span>
        <span class="expand-icon">&#9660;</span>
      </div>
      <div class="exp-card-body">
        <div class="exp-card-body-inner">
          <img src="{IMGS['exp3_2']}" alt="Exp 3.2 — Breakpoint Response">
          <div class="conclusion">{EXP3_2_CONC}</div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ═══ TAB 4: CONCLUSION ═══ -->
<div class="tab-content" id="tab-conclusion">
  <div class="conclusion-panel" style="margin-top:0;">
    <h2>验证结论汇总</h2>
    <div class="conc-grid">
      <div class="conc-section confirmed">
        <h3>&#10004; 强确认</h3>
        <ul>
          <li><strong>Exp 1.1</strong> — 三次多项式为最优阶数，高阶导致过拟合</li>
          <li><strong>Exp 1.2</strong> — 维度鸿沟真实存在，P<sub>0</sub> 无法从 Z-space 恢复</li>
          <li><strong>Exp 1.4</strong> — SVD 求解略优于正规方程</li>
          <li><strong>Exp 2.5</strong> — 动态阈值完全主导静态阈值 Pareto 前沿</li>
          <li><strong>Exp 3.2</strong> — 断点响应机制正常，dT<sub>trigger</sub>/d&delta; &lt; 0 严格成立</li>
        </ul>
      </div>
      <div class="conc-section recalibrate">
        <h3>&#9888; 需重新校准</h3>
        <ul>
          <li><strong>Exp 2.2 — 关键</strong>：&tau; 应从 15min 下调至 ~5min。&tau;=15 时损失爆炸 260 倍</li>
          <li><strong>Exp 1.3</strong>：&rho;&asymp;9.76 远高于假设的 0.5，取决于采样频率 &Delta;t</li>
        </ul>
      </div>
      <div class="conc-section inconclusive">
        <h3>&#8212; 尚无定论（需真实数据）</h3>
        <ul>
          <li><strong>Exp 2.1</strong> — Frozen vs Recalc：整体无显著差异，合成数据不足以区分</li>
          <li><strong>Exp 2.3</strong> — SIGMOID vs Hard：合成数据上 Sharpe 几乎相同（3.54 vs 3.53），SIGMOID 优势可能在噪声真实数据上才显现</li>
        </ul>
      </div>
      <div class="conc-section recommend">
        <h3>&#9733; 建议</h3>
        <ul>
          <li><strong>(a)</strong> 按交易对 & 市场状态分别校准 &tau;</li>
          <li><strong>(b)</strong> Iron Rule 5 改用动态阈值</li>
          <li><strong>(c)</strong> 三次拟合数学基础稳固，工程精力放在其他模块</li>
          <li><strong>(d)</strong> 在真实 BTC/USDT 数据上重跑 Exp 2.1 和 Exp 2.3</li>
        </ul>
      </div>
    </div>
  </div>

  <!-- Expanded loss curve table for exp2.2 -->
  <div class="conclusion-panel" style="margin-top:16px;">
    <h2>&tau; 敏感性分析</h2>
    <div style="overflow-x:auto;">
      <table style="width:100%; border-collapse:collapse; font-size:13px;">
        <thead>
          <tr style="border-bottom:1px solid var(--border);">
            <th style="text-align:left; padding:8px 12px; color:var(--text-dim);">&tau; (min)</th>
            {''.join(f'<th style="text-align:right; padding:8px 12px; color:var(--text-dim);">{t}</th>' for t in sorted(EXP2_2_LOSS.keys(), key=float))}
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style="padding:8px 12px; color:var(--text-dim);">Loss</td>
            {''.join(f'<td style="text-align:right; padding:8px 12px; font-variant-numeric:tabular-nums;{" color:var(--yellow); font-weight:600;" if float(t)>=10 else ""}">{v:.1f}</td>' for t,v in sorted(EXP2_2_LOSS.items(), key=lambda x:float(x[0])))}
          </tr>
        </tbody>
      </table>
    </div>
    <p style="margin-top:12px; font-size:13px; color:var(--text-dim);">
      从 &tau;=5 到 &tau;=15，Loss 放大 <strong style="color:var(--yellow);">{EXP2_2_LOSS['15']/EXP2_2_LOSS['5']:.0f}x</strong>，
      到 &tau;=60 放大 <strong style="color:var(--red);">{EXP2_2_LOSS['60']/EXP2_2_LOSS['5']:.0f}x</strong>。
      正确设置 &tau; 对策略性能至关重要。
    </p>
  </div>
</div>

</div><!-- .container -->

<script>
// ── Tab switching ──
document.querySelectorAll('.tab-btn').forEach(btn => {{
  btn.addEventListener('click', function() {{
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    this.classList.add('active');
    document.getElementById('tab-' + this.dataset.tab).classList.add('active');
  }});
}});

// ── Expand/collapse experiment cards ──
function toggleExp(el) {{
  el.classList.toggle('expanded');
}}

// ── Animated counters ──
function animateCounters() {{
  document.querySelectorAll('.count-up').forEach(el => {{
    const target = parseFloat(el.dataset.target);
    const decimals = parseInt(el.dataset.decimals) || 0;
    const duration = 1200;
    const start = performance.now();
    function step(now) {{
      const t = Math.min((now - start) / duration, 1);
      // ease-out cubic
      const eased = 1 - Math.pow(1 - t, 3);
      const current = eased * target;
      el.textContent = current.toFixed(decimals);
      if (t < 1) requestAnimationFrame(step);
    }}
    requestAnimationFrame(step);
  }});
}}

// ── Counter animation on load ──
document.addEventListener('DOMContentLoaded', animateCounters);

// ── Re-trigger counter animation when switching to conclusion tab (it has counters) ──
// (no numeric counters in conclusion, so this is just a placeholder for future use)
</script>

</body>
</html>
"""

# ── Write dashboard.html ────────────────────────────────────────────────────
OUTPUT = BASE_DIR / "dashboard.html"
with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write(HTML)

print(f"Done. Dashboard written to {OUTPUT} ({os.path.getsize(OUTPUT):,} bytes)")
