# 星空策略 — 手动验证工具包

本目录包含 **13 个**可直接运行的 Python 验证脚本——覆盖原始 15 项实验中的核心实验（工具 1-5），以及研究补遗中提出的 8 项新实验（工具 6-13）。

## 环境要求

```bash
pip install numpy scipy
```

## 原始验证工具（v3 报告 §6）

### 1. 多项式拟合阶数对比 (`verify_optimal_order.py`)
验证命题 3.1 + 工程判断 3.1补
```bash
python3 verify_optimal_order.py --noise 0.05 --trials 200
```

### 2. 维度鸿沟实证检验 (`verify_dimension_gap.py`)
验证定理 2.1 应用推论
```bash
python3 verify_dimension_gap.py --var-p0 100 --trials 500
```

### 3. 高阶信噪比衰减分析 (`verify_snr_decay.py`)
验证假设 3.2 — SNR 指数衰减
```bash
python3 verify_snr_decay.py --noise 0.05 --dt 0.1
```

### 4. 半衰期参数扫描 (`verify_half_life.py`)
验证设计原则 3.1 — τ* 最优值搜索
```bash
python3 verify_half_life.py --trials 100
```

### 5. 动态阈值 vs 静态阈值 (`verify_dynamic_threshold.py`)
验证设计原则 4.1
```bash
python3 verify_dynamic_threshold.py --sigma0 0.5 --tau 5 --trials 200
```

---

## P0 新增实验（填补方法论空白）

### 6. 非多项式信号鲁棒性 (`verify_nonpolynomial_robustness.py`) ★ N1
验证：当真实信号不是多项式时，三次拟合是否仍可用
```bash
python3 verify_nonpolynomial_robustness.py --trials 300 --signal all
```
**预期**: 在包含拐点的分段线性信号上三次仍最优；在正弦衰减/GBM 上三次非最优但可用。

### 7. GARCH 噪声三次最优性复验 (`verify_garch_noise.py`) ★ N2
验证：波动率聚集 (GARCH) 噪声下三次是否仍最优
```bash
python3 verify_garch_noise.py --trials 300
```
**预期**: GARCH 下波动率聚集可能使三次优势收窄——需真实数据进一步验证。

### 8. 厚尾分布动态阈值重校准 (`verify_fattail_threshold.py`) ★ N3
验证：Student's t 噪声下铁律五的 k 最优值
```bash
python3 verify_fattail_threshold.py --nu 3 --trials 300
```
**预期**: 厚尾下 3σ 阈值过松——最优 k 降至 2-2.5。

### 9. 基准策略对比矩阵 (`verify_benchmark_compare.py`) ★ N4
验证：星空策略是否优于简单基准（TP/SL, MA Cross, Bollinger, Kalman）
```bash
python3 verify_benchmark_compare.py --trials 100
```
**预期**: 星空在含拐点轨迹上优于所有基准；随机游走上无明显优势。

---

## P1 新增实验（工程就绪化）

### 10. Walk-Forward 样本外验证 (`verify_walkforward.py`) N5
验证：参数的时间鲁棒性
```bash
python3 verify_walkforward.py --windows 12 --trials 50
```
**预期**: >67% 窗口样本外夏普比为正。

### 11. 参数敏感度二维热力图 (`verify_param_heatmap.py`) N6
验证：tau × sigma0 的交互效应——寻找"安全操作区"
```bash
python3 verify_param_heatmap.py --trials 100
```
**预期**: 存在 tau∈[3,7], sigma0∈[0.2,1.0] 的安全操作区。

### 12. 限价单策略 A/B 测试 (`verify_limit_order_ab.py`) N7
验证：三段分层挂单是否优于单一限价单
```bash
python3 verify_limit_order_ab.py --trials 300
```
**预期**: B 组成交率 +20%~30%，夏普比改善。

### 13. 多品种 tau 跨品种校准 (`verify_cross_asset_tau.py`) N8
验证：tau* 是否因品种波动特征而不同
```bash
python3 verify_cross_asset_tau.py --trials 200
```
**预期**: 高波动品种 → 更短 tau；低波动品种 → 稍长 tau。

---

## 快速运行全部

```bash
cd /Users/xfpan/claude/research/sim_verify/tools

echo "=== 原始工具 (快) ==="
python3 verify_optimal_order.py --trials 100
python3 verify_dimension_gap.py --trials 200
python3 verify_snr_decay.py

echo "=== P0 新实验 (快) ==="
python3 verify_nonpolynomial_robustness.py --trials 50
python3 verify_garch_noise.py --trials 100
python3 verify_fattail_threshold.py --trials 100
python3 verify_benchmark_compare.py --trials 50

echo "=== P1 新实验 (快) ==="
python3 verify_walkforward.py --trials 20
python3 verify_param_heatmap.py --trials 50
python3 verify_limit_order_ab.py --trials 100
python3 verify_cross_asset_tau.py --trials 100

echo "=== 计算密集型 (慢) ==="
python3 verify_half_life.py --trials 200
python3 verify_dynamic_threshold.py --trials 200
```

> 工具 4 (半衰期扫描) 和工具 5 (动态阈值) 计算量最大。P0/P1 工具可用 `--trials` 参数控制精度。

## 完整参数对照表

| # | 实验 | 报告位置 | 脚本 | 验证对象 | 优先级 |
|:--|------|:--|------|------|:--|
| 1 | 1.1 | 报告 §6.2.1 | `verify_optimal_order.py` | 三次最优阶数 | — |
| 2 | 1.2 | 报告 §6.2.2 | `verify_dimension_gap.py` | 维度鸿沟 | — |
| 3 | 1.3 | 报告 §6.2.3 | `verify_snr_decay.py` | SNR 衰减 | — |
| 4 | 2.2 | 报告 §6.3.2 | `verify_half_life.py` | τ 最优值 | — |
| 5 | 2.5 | 报告 §6.3.4 | `verify_dynamic_threshold.py` | 动态阈值 | — |
| 6 | N1 | 补遗 Part C | `verify_nonpolynomial_robustness.py` | 非多项式鲁棒性 | 🔴 P0 |
| 7 | N2 | 补遗 Part C | `verify_garch_noise.py` | GARCH 复验 | 🔴 P0 |
| 8 | N3 | 补遗 Part C | `verify_fattail_threshold.py` | 厚尾重校准 | 🔴 P0 |
| 9 | N4 | 补遗 Part C | `verify_benchmark_compare.py` | 基准对比 | 🔴 P0 |
| 10 | N5 | 补遗 Part C | `verify_walkforward.py` | Walk-Forward | 🟡 P1 |
| 11 | N6 | 补遗 Part C | `verify_param_heatmap.py` | 参数热力图 | 🟡 P1 |
| 12 | N7 | 补遗 Part C | `verify_limit_order_ab.py` | 限价单 A/B | 🟡 P1 |
| 13 | N8 | 补遗 Part C | `verify_cross_asset_tau.py` | 多品种 τ | 🟡 P1 |
