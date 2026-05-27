# 星空策略系统 — 深度审查与优化方案

> 审查日期：2026-05-27
> 审查范围：dashboard.html、trend-analyzer.html、13个验证工具、9份报告/文档、9张静态图表
> 总代码量：~8435行 (HTML/JS/Python/Markdown)

---

## 一、现状诊断

### 1.1 系统全景

| 组件 | 规模 | 状态 |
|------|:--|------|
| dashboard.html | 1549行 / 31函数 / 7标签页 / 20图表 | 功能完整，但**单体膨胀** |
| trend-analyzer.html | 503行 / 3图表联动 | 刚创建，基础功能可用 |
| Python验证工具 | 13个脚本 / ~1500行 | 覆盖完整，**无统一入口** |
| 静态PNG图表 | 9张 | **已被dashboard替代**，可归档 |
| 研究文档 | 9份MD文件 | **内容重叠严重**，需整合 |
| generate_visualizations.py | 571行 | 生成已被替代的PNG |

### 1.2 识别到的 12 个问题

#### P0 — 影响可用性

| # | 问题 | 位置 | 影响 |
|:--|------|------|------|
| 1 | **UI阻塞**：18个模拟函数在`runAllSimulations()`中同步执行，200次试验时阻塞UI 3-8秒 | dashboard.html:1426 | 用户操作卡顿，体验差 |
| 2 | **无进度反馈**：长时间模拟中仅显示"...模拟中"，用户不知道进度、剩余时间 | dashboard.html:1430 | 用户焦虑，可能误以为卡死 |
| 3 | **JavaScript与Python逻辑重复**：polyfit/sharpe/仿真逻辑在两套代码中独立实现，已出现不一致迹象 | dashboard.html + tools/*.py | Bug风险，维护成本翻倍 |

#### P1 — 影响可维护性

| # | 问题 | 位置 | 影响 |
|:--|------|------|------|
| 4 | **dashboard单体膨胀**：1549行单文件，18个simulate函数，修改任一函数需定位到海量代码中 | dashboard.html | 定位困难，合并冲突风险 |
| 5 | **文档碎片化**：9份MD文件内容重叠（verification_analysis.md与research-supplement-v1.md的Part B大量重复） | sim_verify/*.md | 读者困惑，更新时遗漏 |
| 6 | **工具无统一接口**：13个Python脚本各自解析参数、各自格式化输出，无法批量运行或聚合结果 | tools/verify_*.py | 无法自动化回归测试 |
| 7 | **静态PNG残留**：9张PNG已完全被dashboard替代，但仍占据~1MB空间 | charts/*.png | 磁盘浪费，版本控制负担 |
| 8 | **dashboard无错误恢复**：任一simulate函数抛出异常，后续所有图表保持空白 | dashboard.html:runAllSimulations | 一个实验失败影响全局 |

#### P2 — 影响功能深度

| # | 问题 | 位置 | 影响 |
|:--|------|------|------|
| 9 | **trend-analyzer无对比模式**：一次只能看一组参数，无法对比k=3 vs k=5的差异 | trend-analyzer.html | 分析能力受限 |
| 10 | **无参数持久化**：刷新页面后所有滑块回到默认值 | dashboard/analyzer | 无法保存探索结果 |
| 11 | **dashboard实验选择不可控**：用户无法选择只运行特定实验，每次必须全部重算 | dashboard.html | 浪费时间，特别是不关心的Tab |
| 12 | **真实数据集成为零**：整个系统完全依赖合成数据，无任何真实市场数据接口 | 全局 | 最核心的方法论缺陷 |

---

## 二、优化方案

### 方案 A：dashboard 性能与架构优化 (P0, 预计3-4小时)

**目标**：消除UI阻塞，增加进度反馈，改善错误恢复

**具体改动**：

1. **分Tab延迟加载** — `runAllSimulations()` 不再一次执行全部18个函数。改为：
   - 初始仅加载当前激活的Tab（4个实验）
   - 切换Tab时按需加载（首次切换运行该Tab的模拟，结果缓存）
   - 减少初始加载时间 75%

2. **进度条** — 在控制面板增加进度指示器：
   ```
   [████████░░░░░░░░░░░░] 8/18 实验完成 | 当前: 实验2.2 τ扫描
   ```
   每完成一个实验更新一次，给用户实时反馈。

3. **错误隔离** — 每个simulate函数用try/catch包裹：
   ```javascript
   try { const r = simulate1_1(p); Plotly.react(...); }
   catch(e) { showErrorCard('chart1_1', e.message); }
   ```
   一个实验失败不影响其他图表渲染。

4. **requestAnimationFrame分片** — 将18个模拟函数分片到多个帧中执行：
   ```javascript
   async function runChunked(tasks, chunkSize=3) {
     for (let i=0; i<tasks.length; i+=chunkSize) {
       await new Promise(r => requestAnimationFrame(async () => {
         tasks.slice(i, i+chunkSize).forEach(t => t());
         updateProgress(i+chunkSize);
         r();
       }));
     }
   }
   ```
   每帧执行3个模拟 → 6帧完成 → UI始终响应。

**预期效果**：初始加载从3-8s降至<1s，切换Tab流畅，错误不扩散。

---

### 方案 B：代码去重与模块化 (P1, 预计2-3小时)

**目标**：消除JS/Python间的重复逻辑，dashboard代码可拆分

**具体改动**：

1. **提取共享数学库** — 创建 `sim_verify/shared/math.js`：
   ```javascript
   // 一份实现，被dashboard和analyzer共享
   export function polyfit(x, y, degree) { ... }
   export function sharpe(returns) { ... }
   export function sigmoid(resid, k, b) { ... }
   ```

2. **提取共享数学库 (Python)** — 创建 `sim_verify/shared/math.py`：
   ```python
   # 一份实现，被所有verify_*.py工具共享
   def polyfit(x, y, degree): ...
   def sharpe(returns): ...
   def sigmoid(resid, k, b): ...
   ```

3. **dashboard模块化** — 将18个simulate函数拆分为独立JS文件：
   ```
   sim_verify/dashboard/
   ├── index.html          (布局+标签页+控制面板 ~300行)
   ├── core.js             (共享函数 ~80行)
   ├── tab1_math.js        (simulate1_1~1_4 ~180行)
   ├── tab2_stats.js       (simulate2_1~2_5 ~200行)
   ├── tab3_practical.js   (simulate3_1~3_2 ~100行)
   ├── tab_p0.js           (simulateN1~N4 ~250行)
   └── tab_p1.js           (simulateN5~N8 ~250行)
   ```
   index.html 通过 `<script type="module">` 加载，每个Tab按需import。

4. **Python工具公共化** — 重构13个工具，共享 `shared/math.py` 和 `shared/sim.py`：
   ```python
   # tools/verify_optimal_order.py (重构后)
   import sys; sys.path.insert(0, '..')
   from shared.math import polyfit, sharpe, randn
   from shared.sim import generate_cubic_signal, run_monte_carlo
   ```

**预期效果**：核心数学逻辑只有一份实现，dashboard代码可读性大幅提升，Python工具间不再重复造轮子。

---

### 方案 C：Python工具统一化 (P1, 预计1-2小时)

**目标**：一键运行所有验证、统一输出格式、支持回归测试

**具体改动**：

1. **统一CLI入口** — 创建 `tools/run_all.py`：
   ```bash
   # 批量运行
   python3 tools/run_all.py --all --trials 200
   
   # 按优先级运行
   python3 tools/run_all.py --priority P0 --trials 300
   
   # 仅运行原始实验
   python3 tools/run_all.py --original --trials 200
   
   # 输出JSON结果
   python3 tools/run_all.py --all --trials 100 --output results.json
   ```

2. **统一输出格式** — 所有工具输出JSON行：
   ```json
   {
     "experiment": "N1",
     "priority": "P0",
     "timestamp": "2026-05-27T10:30:00",
     "params": {"trials": 300, "noise": 0.05},
     "verdict": "ok",
     "key_metrics": {"k3_optimal": true, "r4_ratio": 3.4, "r5_ratio": 11.5},
     "raw_data": {...}
   }
   ```

3. **CI就绪** — 添加 `tools/run_all.py --ci` 模式：
   - 固定参数（trials=500，确保可复现）
   - 退出码 0=全部通过, 1=有证伪, 2=有失败
   - 输出JUnit XML兼容格式

**预期效果**：一键回归测试，结果可机器解析，支持CI/CD流水线集成。

---

### 方案 D：文档整合 (P1, 预计1-2小时)

**目标**：消除重复，建立清晰的文档层次

**当前问题**：9份MD文件中，以下三对存在显著内容重叠：
- `verification_analysis.md` ↔ `research-supplement-v1.md` Part B
- `v3_chapters_1_5.md` + `v3_chapters_6_8.md` ↔ 原始报告 `deep-research-report-v3.md`

**具体改动**：

1. **合并为3层文档结构**：
   ```
   sim_verify/
   ├── README.md                    # 项目总览 (新建)
   ├── ANALYSIS.md                  # 验证结果分析 (合并verification_analysis + supplement PartB)
   ├── IMPROVEMENTS.md              # 改进建议+新实验 (supplement PartA + PartC)
   └── archive/                     # 归档旧版
       ├── verification_analysis_report.md
       ├── v3_chapters_1_5.md
       ├── v3_chapters_6_8.md
       └── analysis_gap_and_plan.md
   ```

2. **README.md 作为入口**：
   - 项目结构图
   - 快速开始（dashboard打开方式、工具运行方式）
   - 指向详细文档的链接

**预期效果**：读者从README进入，3个文档覆盖全部内容，无重复。

---

### 方案 E：trend-analyzer 功能增强 (P2, 预计2-3小时)

**目标**：增加对比模式、导出功能

**具体改动**：

1. **A/B对比模式** — 在analyzer中增加"对比"开关，开启后：
   - 左侧显示参数组A（如k=3, τ=5）
   - 右侧显示参数组B（如k=5, τ=15）
   - 同时展示两组拟合曲线、残差、自信度
   - 视觉上直接对比过拟合效果

2. **添加卡尔达诺判别式可视化** — 在侧边栏增加：
   ```
   判别式 Δ = (q/2)² + (p/3)³ = -0.042
   Δ < 0 → 3个实根 → S形曲线 (具有局部极值)
   策略建议: 保守退让
   ```
   实时计算并显示拟合多项式的 p, q, Δ 值及几何含义。

3. **导出功能** — 添加"导出分析"按钮：
   - 导出当前参数 + 图表为PNG截图
   - 导出关键指标为CSV
   - 导出完整参数配置为可分享的URL hash

4. **交互式数据点** — 在价格图表上：
   - 鼠标悬停显示该点的实际值、拟合值、残差
   - 点击任意数据点可将其设为"开仓点"，重新拟合

**预期效果**：analyzer从"看一眼"工具升级为"深入研究"工具。

---

### 方案 F：真实数据集成 (P2, 预计3-4小时)

**目标**：为系统接入真实市场数据建立基础框架

**具体改动**：

1. **数据加载器** — 创建 `tools/data_loader.py`：
   ```python
   # 支持从Binance/CCXT加载真实K线数据
   python3 tools/data_loader.py --symbol BTC/USDT --timeframe 15m --days 180 --output data/btc_15m.csv
   ```

2. **真实数据回测工具** — 创建 `tools/verify_real_data.py`：
   ```bash
   python3 tools/verify_real_data.py --data data/btc_15m.csv --tau 5 --sigma0 0.5
   ```
   输出：夏普比、最大回撤、铁律触发统计、分段P&L

3. **dashboard增加"真实数据"数据源选项** — 在控制面板增加数据源下拉框：
   - "合成数据 (三次多项式)"
   - "合成数据 (GARCH噪声)"
   - "真实数据 (如有)"

**预期效果**：策略可在真实BTC/USDT历史上验证，弥补最核心的方法论缺陷。

---

## 三、优先级与排期

```
第1周 (P0): 方案A dashboard性能优化
  ├── Day 1-2: 分Tab延迟加载 + 进度条
  └── Day 3-4: requestAnimationFrame分片 + 错误隔离

第2周 (P1): 方案B+C 代码质量
  ├── Day 1-2: 提取共享数学库 (JS + Python)
  ├── Day 2-3: 统一CLI入口 run_all.py
  └── Day 3-4: 文档整合

第3周 (P2): 方案E+F 功能深化
  ├── Day 1-2: trend-analyzer对比模式 + 判别式可视化
  ├── Day 2-3: 真实数据加载器 + 回测工具
  └── Day 3-4: URL参数持久化 + 导出
```

## 四、预期收益

| 维度 | 当前 | 优化后 |
|------|------|------|
| dashboard初始加载 | 3-8s阻塞 | <1s，UI始终响应 |
| 错误时影响范围 | 一个失败→全局空白 | 错误隔离，仅影响单个图表 |
| JS/Python代码重复 | 2套独立实现 | 1套共享库 |
| 工具运行方式 | 13个独立命令 | 1个统一入口 |
| 文档数量 | 9份重叠MD | 3份清晰层次 |
| 真实数据支持 | 零 | 完整加载+回测流水线 |
| trend-analyzer能力 | 单参数组观察 | 对比+判别式+导出 |
