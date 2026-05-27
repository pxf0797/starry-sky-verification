# 星空策略 (Starry Sky) 模拟验证系统

> 量化做空平仓系统的完整数学验证与交互式分析平台

## 快速开始

```bash
# 交互式仪表盘 (浏览器)
open dashboard.html

# 趋势+三次方程+残差 实时分析器
open trend-analyzer.html

# 命令行验证 (13个实验)
python3 tools/run_all.py --list              # 列出所有实验
python3 tools/run_all.py --all --trials 200  # 运行全部
python3 tools/run_all.py --priority P0       # 仅P0级
```

## 项目结构

```
sim_verify/
├── README.md                        ← 你在这里
├── dashboard.html                   ★ 主仪表盘: 7标签页/20+图表/18项实时仿真
├── trend-analyzer.html              ★ 分析器: 价格轨迹+残差+自信度三联图表
├── optimization-plan.md             优化方案 (6方案/12问题诊断)
├── research-supplement-v1.md        研究补遗: 改进建议+新手引导+新实验
├── verification_analysis.md         验证结果详细分析
├── generate_visualizations.py       静态图表生成 (9张PNG)
│
├── charts/                          9张静态可视化PNG
├── layer1/ layer2/ layer3/          原始验证数据和实验结果
│
├── shared/
│   ├── math.js                      JS共享数学库 (polyfit/sharpe/sigmoid)
│   └── math.py                      Python共享数学库
│
└── tools/
    ├── README.md                    工具包文档
    ├── run_all.py                   ★ 统一CLI入口
    ├── data_loader.py               真实市场数据加载器
    ├── verify_optimal_order.py      1. 三次最优阶数
    ├── verify_dimension_gap.py      2. 维度鸿沟
    ├── verify_snr_decay.py          3. SNR衰减
    ├── verify_half_life.py          4. τ最优值扫描
    ├── verify_dynamic_threshold.py  5. 动态vs静态阈值
    ├── verify_nonpolynomial_robustness.py  6. N1-P0 非多项式鲁棒性
    ├── verify_garch_noise.py               7. N2-P0 GARCH复验
    ├── verify_fattail_threshold.py         8. N3-P0 厚尾校准
    ├── verify_benchmark_compare.py         9. N4-P0 基准对比
    ├── verify_walkforward.py              10. N5-P1 Walk-Forward
    ├── verify_param_heatmap.py            11. N6-P1 参数热力图
    ├── verify_limit_order_ab.py           12. N7-P1 限价单A/B
    └── verify_cross_asset_tau.py          13. N8-P1 多品种τ
```

## 核心概念

| 概念 | 说明 |
|------|------|
| **三次拟合** | 用三阶多项式 P(t)=at³+bt²+ct+d 预测局部价格轨迹，"恰好"描述做空叙事的拐点 |
| **半衰期 τ** | 预测可信度减半时间。最优值≈5分钟 (v3校准)。超时后模型自然退化 |
| **残差 Δ(t)** | 实际价格与预测价格之差。驱动SIGMOID自信度函数，触发铁律三/五 |
| **动态阈值** | σ(t)=σ₀·e^(-λt)。持仓越长容忍度越低，夏普比优势4×静态阈值 |
| **卡尔达诺判别式** | Δ=(q/2)²+(p/3)³。Δ<0→3实根(S形)→保守退让；Δ>0→1实根→满仓持有 |
| **铁律三** | α<α_min时一键市价止损 |
| **铁律五** | |P_raw-P_smooth|>k·σ(t)时拔网线市价全出 |

## 验证结论速查

| 实验 | 结论 | 状态 |
|------|------|:--:|
| 1.1 | k=3在所有噪声下严格最优 | OK |
| 1.2 | P₀缺失→误差放大2-19× | OK |
| 1.3 | SNR指数衰减,ρ=f(Δt) | OK |
| 2.2 | τ*=4.7min, τ=15被证伪 | ★ |
| 2.5 | 动态阈值4×夏普比优势 | ★ |
| 3.1 | 限价单成交率47.2%是#1瓶颈 | ~ |
| N4 | 星空在拐点轨迹优于基准 | OK |

## 文档导航

- 想了解验证结果 → `verification_analysis.md`
- 想了解改进建议 → `research-supplement-v1.md`
- 想了解优化方案 → `optimization-plan.md`
- 新手入门 → dashboard的"新手引导"标签页

## 环境要求

- 浏览器: Chrome/Firefox/Safari (用于dashboard和analyzer)
- Python 3.8+: numpy, scipy (用于命令行工具)
- ccxt (可选): 用于真实数据加载
