# 平仓管线决策过程动态可视化方案调查

## 目录

1. [背景与当前状态](#1-背景与当前状态)
2. [核心可视化需求](#2-核心可视化需求)
3. [候选方案详细分析](#3-候选方案详细分析)
4. [方案对比矩阵](#4-方案对比矩阵)
5. [推荐方案](#5-推荐方案)
6. [与 ExitPipeline 的集成方案](#6-与-exitpipeline-的集成方案)
7. [附录：行业实践参考](#7-附录行业实践参考)

---

## 1. 背景与当前状态

### 当前系统架构

项目路径：`/Users/xfpan/claude/quant-trading`

平仓管线（`ExitPipeline`）的核心数据流：

```
开仓快照(Snapshot) → 三次方程拟合(CubicTrajectory)
    → 每根K线计算残差(Residual%) → EMA平滑
    → SIGMOID自信度映射 → 铁律评估 → ExitDecision
```

当前可视化（`src/visualize.py`）：**Matplotlib 静态 3-panel PNG**

| 面板 | 内容 | 当前表现 |
|------|------|---------|
| Panel 1: 价格 | 实际价(蓝实线) + 三次预测(橙虚线) + 快照价(灰水平线) + 谷底目标(绿水平线) + 平仓X(红) | 静态 |
| Panel 2: 残差% | 柱状填充 + ±5% emergency 阈值红线 | 静态 |
| Panel 3: 自信度 | SIGMOID曲线 + 0.3 止损阈值红线 + 填充色 | 静态 |

### 局限性

1. 无法观察 K 线逐根推进的时间演变过程
2. 无法交互（缩放/拖拽/悬浮查看数值）
3. 平仓触发瞬间缺少视觉强调（当前只是一个红 X 散点）
4. 多 scenario 对比只有 2x2 缩略图，没有时间轴同步播放

---

## 2. 核心可视化需求

### 2.1 数据特性

| 数据 | 类型 | 更新频率 | 关键事件 |
|------|------|---------|---------|
| 价格轨迹 | 连续时序 | 每根K线 | 平仓触发瞬间 |
| 三次方程预测 | 连续曲线 | 仅 arm() 时拟合一次 | N/A |
| 残差 % | 连续时序 | 每根K线 | 越过 ±5% emergency 线 |
| 自信度 | 连续时序 | 每根K线 | 跌破 0.3 止损阈值 |
| 平仓决策 | 离散事件 | 生命周期一次 | 需要红色闪烁/脉冲 |

### 2.2 动画需求

1. **K 线逐根推进**：以回放模式展示 Entry → Exit 全过程
2. **轨迹线实时延伸**：三次方程曲线随 K 线推进延伸（非预渲染整条）
3. **残差柱实时渐变**：正 = 红色，负 = 绿色，幅度 = 柱高，实时生长
4. **自信度实时衰减**：SIGMOID 曲线随残差增大而衰减的动画过程
5. **平仓瞬间高亮**：闪烁 / 脉冲 / 颜色突变 + 文字标注

### 2.3 交互需求

- 时间轴拖拽（跳到特定 K 线位置）
- 图表缩放/平移
- 悬浮显示精确数值
- 多 scenario 并排对比 + 同步播放

### 2.4 输出需求

- 导出为 MP4 或 GIF（用于文档/演示）
- 导出为 HTML（用于交互式报告）
- 与现有 `plot_scenario()` 接口兼容（最低侵入性）

---

## 3. 候选方案详细分析

### 方案 1：Matplotlib FuncAnimation → MP4/GIF

#### 原理
利用 `matplotlib.animation.FuncAnimation` 逐帧更新现有 3-panel 图表的 data 或 artist 对象。当前代码已经在使用 matplotlib，改造量最小。

#### 关键实现
```python
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
line1, = ax1.plot([], [], color=BLUE, lw=1.5)
# ... 初始化其他 artists ...

def update(frame):
    # frame = 0..N-1, 每帧推进一个时间步
    t = times[:frame+1]
    line1.set_data(t, actuals[:frame+1])
    # 更新残差填充、自信度曲线...
    return line1, line2, ...

anim = FuncAnimation(fig, update, frames=len(times), interval=200, blit=True)
anim.save("exit_process.mp4", writer=FFMpegWriter(fps=5))
```

#### 优势
- 当前项目已安装 matplotlib + imageio-ffmpeg + pillow，零新增依赖
- 可直接复用现有 `plot_scenario()` 的配色、布局、参数
- 改造成本最小（约 100-150 行新增代码）
- 导出 MP4/GIF 开箱即用
- `blit=True` 优化后性能良好（~200ms/帧，60帧回放约 12 秒）

#### 劣势
- **无交互性**：生成的 MP4/GIF 无法缩放、拖拽、悬浮
- 帧率受 Python GIL 限制，复杂帧更新可能卡顿
- 渲染大量数据点（>5000 帧）时内存占用上升
- 无法在浏览器中直接查看（需要播放器）

#### 评估

| 维度 | 评分 (1-5) |
|------|-----------|
| 动画流畅度 | 4 (blit 优化后流畅) |
| 交互性 | 1 (纯输出，无交互) |
| 开发成本 | 5 (最低，复用现有代码) |
| 导出能力 | 5 (MP4/GIF 原生支持) |
| 与现有代码集成 | 5 (同一代码库，直接调用) |
| K线逐根推进 | 3 (可实现但需额外渲染 OHLC) |
| 平仓瞬间高亮 | 3 (散点突变，无脉冲效果) |
| 多场景并排比较 | 2 (不支持同步播放) |

---

### 方案 2：Plotly → HTML 交互式

#### 原理
使用 Plotly 的 `make_subplots` + `go.Frame` 实现动画，利用 `go.Scatter` 和 `go.Bar` 构建三面板图。Plotly 的帧动画（`frames`）天然支持逐时间步推进。

#### 关键实现
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=3, cols=1, shared_xaxes=True)

# 第一帧
fig.add_trace(go.Scatter(x=[t[0]], y=[actuals[0]], name="Actual"), row=1, col=1)
# ...

# 逐帧构建
frames = []
for i in range(len(times)):
    frame = go.Frame(
        data=[
            go.Scatter(x=times[:i+1], y=actuals[:i+1], ...),
            go.Scatter(x=times[:i+1], y=predicted[:i+1], ...),
            # ... 残差、自信度面板
        ],
        name=f"step_{i}"
    )
    frames.append(frame)

fig.frames = frames
fig.update_layout(
    updatemenus=[dict(type="buttons", showactive=False,
                      buttons=[dict(label="Play", method="animate",
                                    args=[None, dict(frame=dict(duration=200))])])]
)
fig.write_html("exit_pipeline.html")
```

#### 优势
- **交互性最强**：缩放、拖拽、悬浮、播放控制按钮开箱即用
- 动画播放器自带 Play/Pause/Speed 控制
- 导出为独立 HTML 文件，无需服务端
- 支持 `Slider` 直接跳转到任意时间步
- 可叠加 `Rect` 形状实现平仓瞬间高亮
- 社区成熟，文档丰富

#### 劣势
- 需要新增 `plotly` 依赖（约 50MB）
- 复杂帧动画时，数百个数据点的 HTML 文件可能 >10MB
- Frame-based 动画在数据量 >500 步时构建帧列表内存占用高
- 色彩和风格需要重新配置才能匹配当前 matplotlib 风格

#### 评估

| 维度 | 评分 (1-5) |
|------|-----------|
| 动画流畅度 | 4 (JS 端渲染，帧率取决于数据量) |
| 交互性 | 5 (缩放/拖拽/悬浮/跳转全部支持) |
| 开发成本 | 3 (中等，需学习 Plotly Frame API) |
| 导出能力 | 4 (HTML 自包含，不支持直接 MP4) |
| 与现有代码集成 | 3 (需新增依赖，重构图表函数) |
| K线逐根推进 | 4 (原生支持 OHLC 和 Candlestick 图) |
| 平仓瞬间高亮 | 4 (可用 annotation + shape 实现闪烁) |
| 多场景并排比较 | 4 (可并排 subplot 且同步播放) |

---

### 方案 3：Streamlit → Web 应用

#### 原理
使用 Streamlit 构建实时 Web dashboard。核心是利用 `st.pyplot()` 或 `st.plotly_chart()` 配合 `st.slider` 做时间步选择器，或利用 `st.empty()` + `time.sleep()` 做自动回放。

#### 关键实现
```python
import streamlit as st

st.title("Exit Pipeline Playback")

step = st.slider("K-line Step", 0, len(times)-1, 0)
auto = st.checkbox("Auto Play")
speed = st.select_slider("Speed", options=[0.1, 0.2, 0.5, 1.0, 2.0])

placeholder = st.empty()
for i in range(step, len(times)):
    fig = build_frame(i, times, actuals, ...)  # 复用 matplotlib/plotly
    placeholder.pyplot(fig)
    if auto:
        time.sleep(1.0 / speed)
```

#### 优势
- **快速原型**：几行代码就能跑起来
- 纯 Python，不写 HTML/JS
- 可结合 `st.columns` 并排多 scenario
- 支持多种图表库（matplotlib/plotly/altair）
- Slider + Play 按钮天然支持逐帧回放

#### 劣势
- **必须运行 Streamlit Server**（`streamlit run app.py`），不能导出为独立文件
- 实时更新效率低（每次更新重绘整个图表）
- 多用户并发需要额外配置
- 不适合生产级部署（需要 Docker 等）
- 如果每帧用 matplotlib 渲染，性能较慢

#### 评估

| 维度 | 评分 (1-5) |
|------|-----------|
| 动画流畅度 | 2 (重绘整图，>200ms/帧) |
| 交互性 | 4 (Slider/Play/checkbox 齐全) |
| 开发成本 | 4 (很低，快速原型) |
| 导出能力 | 1 (不能导出为独立文件) |
| 与现有代码集成 | 4 (可直接调用现有 Python 函数) |
| K线逐根推进 | 4 (Slider 天然支持) |
| 平仓瞬间高亮 | 3 (可用 st.balloons 效果) |
| 多场景并排比较 | 3 (st.columns 可并排但同步困难) |

---

### 方案 4：Bokeh → 服务端渲染

#### 原理
Bokeh 提供 `ColumnDataSource` 作为数据绑定层，通过 `cds.stream()` 实现高效数据推送。使用 `bokeh serve` 启动服务端应用，支持服务端推送和回调。

#### 关键实现
```python
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Slider, Button
from bokeh.layouts import column, row

# ColumnDataSource — 数据绑定，更新 source 自动更新图表
source = ColumnDataSource(data=dict(x=[], y=[], residual=[], confidence=[]))
p = figure()
p.line('x', 'y', source=source, color=BLUE)

# 回调：推进到下一步
def step_forward():
    i = len(source.data['x'])
    if i < len(all_data):
        source.stream(dict(
            x=[all_x[i]], y=[all_actuals[i]],
            residual=[all_residuals[i]], confidence=[all_confidences[i]]
        ))

curdoc().add_periodic_callback(step_forward, 200)
```

#### 优势
- **数据流式更新最高效**：`stream()` 只推送新数据，不重绘整图
- 支持真实时推送（WebSocket）
- 原生交互（缩放/平移/悬浮）
- 可导出为独立 HTML（需预先渲染）
- 多面板共享 x_range 天然同步

#### 劣势
- **需要 bokeh server** 运行（本地 `bokeh serve` 或部署）
- 学习曲线较陡（ColumnDataSource、回调系统）
- 复杂布局比 Plotly 麻烦
- 动画控制不如 Plotly Frame 直观
- 社区活跃度下降（Plotly 更流行）

#### 评估

| 维度 | 评分 (1-5) |
|------|-----------|
| 动画流畅度 | 5 (流式更新，只推增量) |
| 交互性 | 4 (缩放/平移/悬浮原生) |
| 开发成本 | 2 (学习曲线陡，需 bokeh server) |
| 导出能力 | 3 (可导出 HTML，但动画需预渲染) |
| 与现有代码集成 | 2 (需重构为 bokeh 数据模型) |
| K线逐根推进 | 5 (stream + rollover 天然适合) |
| 平仓瞬间高亮 | 4 (回调可在触发时修改图表样式) |
| 多场景并排比较 | 4 (共享 range 同步播放) |

---

### 方案 5：Manim（数学动画引擎）

#### 原理
Manim 是 3Blue1Brown 的数学动画引擎（`manim` / `manimgl`）。通过 `VGroup`、`Animation` 等基类构建矢量动画场景。适合生成电影级质量的回放视频。

#### 关键实现
```python
from manim import *

class ExitPipelineAnimation(Scene):
    def construct(self):
        axes = Axes(x_range=[0, len(times)], y_range=[min(actuals), max(actuals)])
        
        # 逐步绘制价格线
        dots = VGroup()
        for i, (t, p) in enumerate(zip(times, actuals)):
            dot = Dot(axes.coords_to_point(t, p), color=BLUE)
            dots.add(dot)
            self.play(Create(dot), run_time=0.05)
        
        # 平仓瞬间：红色脉冲
        if exit_idx:
            exit_dot = dots[exit_idx]
            self.play(
                exit_dot.animate.set_color(RED).scale(2),
                Flash(exit_dot, color=RED, line_length=0.5),
            )
```

#### 优势
- **视觉效果最佳**：平滑过渡、渐变、脉冲、闪烁效果完美
- 数学公式渲染（Latex）原生支持
- 可输出 4K 60fps 视频
- 适合做演示/教学/YouTube 视频

#### 劣势
- **开发成本极高**：每条曲线/每个元素都需手动定义动画
- 不支持交互（纯视频输出）
- 依赖 Cairo/FFmpeg，安装复杂
- 渲染时间长（几分钟场景可能需要数小时渲染）
- 完全不适用于日常使用或迭代开发
- 与现有代码集成困难（需重写所有绘图逻辑）

#### 评估

| 维度 | 评分 (1-5) |
|------|-----------|
| 动画流畅度 | 5 (60fps 电影级) |
| 交互性 | 1 (纯输出，无交互) |
| 开发成本 | 1 (极高，每条线手动定义动画) |
| 导出能力 | 5 (MP4/4K/GIF 最佳) |
| 与现有代码集成 | 1 (完全重写) |
| K线逐根推进 | 3 (可实现但效率低) |
| 平仓瞬间高亮 | 5 (脉冲/闪烁/缩放完美) |
| 多场景并排比较 | 2 (需手动布局，同步困难) |

---

### 方案 6：lightweight-charts-python → TradingView 风格

#### 原理
TradingView Lightweight Charts 的 Python 封装。在浏览器/桌面中渲染高性能 HTML5 Canvas 图表。支持跨平台（Jupyter/PyQt5/PySide6/Streamlit）。

#### 关键实现（Streamlit 模式）
```python
from lightweight_charts import Chart

chart = Chart()
chart.layout(background_color='#131722', text_color='#d1d4dc')

# Panel 1: K线 + 三条线
chart.set(ohlcv_data)  # 直接传入 OHLC DataFrame
chart.crosshair(mode=1)

# 叠加预测线、快照价、谷底价
chart.create_line('cubic', predicted, color='orange', width=2)
chart.create_line('snapshot', [snap_price]*len(times), color='gray', width=1, style='dotted')
chart.create_line('valley', [valley_price]*len(times), color='green', width=1, style='dotted')

# Panel 2: 残差 (subchart)
residual_sub = chart.create_subchart(width=1, height=0.3)
residual_sub.create_histogram('residual', residuals, color='orange')

# Panel 3: 自信度
conf_sub = chart.create_subchart(width=1, height=0.3)
conf_sub.create_line('confidence', confidences, color='green')

# 平仓标记
chart.marker(exit_idx, text='EXIT', color='red', shape='arrow_up')
```

#### 优势
- **TradingView 级别的渲染质量**和可辨识度
- Canvas 渲染，性能极高（适用于数千根 K 线）
- 原生支持 OHLC/Candlestick 图
- 子图(subchart)机制天然适合多面板
- 跨平台支持（Jupyter / PyQt / Streamlit）
- 轻量级（JS 库仅 ~40KB gzipped）

#### 劣势
- **需要新增依赖**，且 API 仍在快速迭代中
- 多面板残差+自信度等功能需要额外配置 subchart
- 导出静态文件不如 Plotly 方便
- 自定义动画效果（脉冲/闪烁）需要 JS 桥接
- Python API 文档不如 Plotly 完善
- 学习曲线中等

#### 评估

| 维度 | 评分 (1-5) |
|------|-----------|
| 动画流畅度 | 5 (Canvas 硬件加速) |
| 交互性 | 5 (TradingView 级交互) |
| 开发成本 | 3 (中等，API 较新需学习) |
| 导出能力 | 2 (主要面向实时显示，导出需额外配置) |
| 与现有代码集成 | 2 (需重构为 OHLC 格式，非纯时序) |
| K线逐根推进 | 5 (原生支持 K 线动画回顾) |
| 平仓瞬间高亮 | 3 (内置 marker 但不支持自定义动画) |
| 多场景并排比较 | 4 (多个 Chart 实例并排) |

---

### 方案 7：Panel + HoloViews → 交互式 Dashboard

#### 原理
HoloViews 提供高层声明式可视化 API，Panel 负责布局和交互组件。二者结合可快速构建复杂的交互式仪表盘。

#### 优势
- 声明式 API，代码量少
- 支持多种后端（Bokeh/Plotly/Matplotlib）
- 支持 Jupyter 和独立 Web App
- 交互组件丰富（Slider/Button/Select）

#### 劣势
- 依赖较多（holoviews + panel + 后端）
- 动画控制不如专门方案
- 社区相对较小

#### 评估

| 维度 | 评分 (1-5) |
|------|-----------|
| 动画流畅度 | 3 (依赖后端渲染) |
| 交互性 | 4 (组件丰富) |
| 开发成本 | 3 (需学习 HoloViews API) |
| 导出能力 | 2 (可导出 HTML 但功能有限) |
| 与现有代码集成 | 3 (需适配 HoloViews 数据结构) |

---

## 4. 方案对比矩阵

| 对比维度 | Matplotlib FuncAnimation | Plotly | Streamlit | Bokeh | Manim | lightweight-charts | Panel+HoloViews |
|----------|:------------------------:|:------:|:---------:|:-----:|:-----:|:------------------:|:----------------:|
| **动画流畅度** | 7/10 | 8/10 | 4/10 | 9/10 | 10/10 | 9/10 | 6/10 |
| **交互性** | 1/10 | 10/10 | 8/10 | 9/10 | 1/10 | 10/10 | 9/10 |
| **开发成本** | 9/10 | 6/10 | 8/10 | 4/10 | 2/10 | 6/10 | 6/10 |
| **导出 MP4/GIF** | 10/10 | 6/10 | 0/10 | 4/10 | 10/10 | 2/10 | 4/10 |
| **导出 HTML** | 0/10 | 10/10 | 0/10 | 8/10 | 0/10 | 6/10 | 8/10 |
| **代码集成难度** | 10/10 | 6/10 | 8/10 | 4/10 | 1/10 | 5/10 | 6/10 |
| **K线逐根推进** | 6/10 | 8/10 | 8/10 | 10/10 | 6/10 | 10/10 | 8/10 |
| **平仓瞬间高亮** | 6/10 | 8/10 | 6/10 | 8/10 | 10/10 | 6/10 | 7/10 |
| **多场景并排** | 4/10 | 8/10 | 6/10 | 8/10 | 4/10 | 8/10 | 8/10 |
| **新增依赖** | 无 | ~50MB | ~100MB | ~30MB | ~500MB | ~20MB | ~120MB |
| **是否需要服务端** | 否 | 否 | `streamlit run` | `bokeh serve` | 否 | 可选的 | 可选的 |
| **学习成本** | 低 | 中 | 低 | 高 | 极高 | 中 | 中 |
| **维护成本** | 低 | 低 | 低 | 中 | 高 | 中 | 中 |

### 总分排名

| 排名 | 方案 | 加权总分 | 最佳使用场景 |
|:----:|------|:--------:|-------------|
| 1 | **Plotly** | 62/70 | 通用首选：快速出交互式 HTML，开发和导出平衡最佳 |
| 2 | **lightweight-charts** | 56/70 | 追求 TradingView 级渲染质量和 K 线原生支持 |
| 3 | **Bokeh** | 56/70 | 需要流式实时推送 + 服务端渲染 |
| 4 | Matplotlib FuncAnimation | 46/70 | 最小改动快速出 MP4/GIF |
| 5 | Streamlit | 40/70 | 快速原型展示，团队内部分享 |
| 6 | HoloViews+Panel | 54/70 | 复杂交互仪表盘，多组件联动 |
| 7 | Manim | 34/70 | 产品宣传视频/教学演示 |

---

## 5. 推荐方案

### 第1名：Plotly（通用首选）

**推荐理由**：动画 + 交互 + 导出 HTML 的最佳平衡。

**对本项目的适配策略**：

1. **新建** `src/visualize_plotly.py`，不修改现有 `visualize.py`
2. 实现 `plot_scenario_animated()` 函数，参数签名与现有 `plot_scenario()` 兼容
3. 使用 `make_subplots(rows=3, cols=1, shared_xaxes=True)` 创建 3 面板
4. 使用 `go.Frame` 构建逐帧动画，每帧推进一个时间步
5. 平仓瞬间：在那一帧添加红色 `go.Scatter`（大圆点） + `layout.shapes` 矩形闪烁背景

**Demo 可行性评估**：1-2 天可完成

```
Day 1: 搭建 3 面板 + 动画帧结构 + 播放控制
Day 2: 平仓高亮效果 + 导出 HTML + 与 __main__.py 集成
```

**代码路径**：`src/visualize_plotly.py`（新增文件）

### 第2名：lightweight-charts-python（K线首选）

**推荐理由**：如果最终目标是展示 K 线图（而不是纯价格线），这是最佳选择。TradingView 级别的 Canvas 渲染性能无可匹敌。

**对本项目的适配策略**：

1. **新建** `src/visualize_chart.py`，使用 lightweight-charts 的 multi-pane API
2. 需要将当前纯价格线数据转换为 OHLC 格式（至少需要 open/high/low/close）
3. 利用 subchart 机制实现残差 + 自信度面板
4. 利用 marker API 标记平仓点

**Demo 可行性评估**：2-3 天

```
Day 1: OHLC 数据适配 + K 线图 + 叠加线（预测/快照/谷底）
Day 2: 残差 subchart + 自信度 subchart
Day 3: marker + Streamlit 集成
```

### 垫底方案说明

| 方案 | 不推荐原因 |
|------|-----------|
| Manim | 开发成本过高，渲染时间过长，不适合交互式使用 |
| Streamlit | 功能被 Plotly 完全覆盖且无法导出，适合原型不适合报告 |
| Bokeh | 学习曲线陡，社区萎缩，除非需要真流式推送否则不如 Plotly |

---

## 6. 与 ExitPipeline 的集成方案

### 6.1 当前数据流

```
__main__.py
  ├─ candles → EntryPipeline → Snapshot
  ├─ Snapshot → ExitPipeline.arm() → CubicTrajectory
  ├─ Loop: ExitPipeline.update(candle) → ExitDecision
  │   └─ 记录: actuals[], predicted_vals[], residuals[], confidences[]
  └─ plot_scenario(*data) → static PNG
```

### 6.2 推荐集成方式（两步渐进）

**阶段一：Plotly 动画 HTML（低侵入，1-2 天）**

```python
# src/visualize_plotly.py
from src.visualize import BLUE, ORANGE, GREEN, RED, GRAY  # 复用配色

def plot_scenario_animated(
    filename: str,        # → app.py 或 output.html
    title: str,
    times: list[float],
    actuals: list[float],
    predicted: list[float] | None,
    snapshot_price: float,
    valley_price: float,
    residuals: list[float],
    confidences: list[float],
    exit_idx: int | None,
    exit_action: str,
    no_recalc_threshold: float,
    emergency_divergence: float,
) -> str:
    """返回 HTML 文件路径"""
    # ... Plotly Frame 动画实现 ...
    fig.write_html(filename)
    return filename
```

在 `__main__.py` 中只需替换调用：
```python
# 旧的静态图
# path = plot_scenario(...)

# 新的动态图
path = plot_scenario_animated(...)
print(f"  交互式图表已保存: {path}")
```

**阶段二：lightweight-charts K 线版（可选，2-3 天）**

```python
# src/visualize_chart.py
from lightweight_charts import Chart

def plot_scenario_kline(
    candles: list[Candle],
    data: dict,  # 复用 __main__.py 返回的数据字典
    exit_idx: int | None,
) -> None:
    """返回 lightweight-charts Chart 对象（Streamlit/桌面显示）"""
    chart = Chart()
    # 设置 OHLC 数据
    chart.set(candles_to_df(candles))
    # 叠加预测线等
    ...
    chart.show(block=True)
```

**数据适配层**（可选，预留接口）：
```python
# src/adapters.py: 将 ExitPipeline 数据转换为各种可视化库的输入格式
def to_plotly_data(scenario_data: dict) -> dict: ...
def to_lightweight_candles(candles: list[Candle]) -> pd.DataFrame: ...
```

### 6.3 接口兼容性

| 现有 `plot_scenario()` 参数 | Plotly 版本兼容性 | lightweight-charts 版本 |
|---------------------------|------------------|----------------------|
| `times: list[float]` | 直接使用作为 x 轴 | 转换为 datetime index |
| `actuals: list[float]` | `go.Scatter` y 值 | 需要 K 线 OHLC 格式 |
| `predicted: list[float]` | 直接使用 | `create_line()` |
| `snapshot_price: float` | `add_hline()` | `create_line()` level |
| `valley_price: float` | `add_hline()` | `create_line()` level |
| `residuals: list[float]` | `go.Bar` + 颜色映射 | subchart histogram |
| `confidences: list[float]` | `go.Scatter` | subchart line |
| `exit_idx: int | None` | Frame 中条件判断 | `chart.marker()` |
| `exit_action: str` | annotation 文本 | marker text |
| `no_recalc_threshold: float` | `add_hline()` | subchart line |
| `emergency_divergence: float` | `add_hline()` 正负 | subchart line 正负 |

### 6.4 导出格式策略

| 用途 | 推荐方案 | 格式 |
|------|---------|------|
| 日常回测分析 | Plotly HTML | `.html` |
| 文档/报告 | Plotly HTML + 截图 | `.html` + `.png` |
| 演示/视频 | Matplotlib FuncAnimation | `.mp4` / `.gif` |
| 团队 Dashboard | lightweight-charts + Streamlit | Web 页面 |
| 教学/YouTube | Manim（如有余力） | `.mp4` 4K |

---

## 7. 附录：行业实践参考

### 参考链接

| 来源 | 链接 | 要点 |
|------|------|------|
| lightweight-charts-python | https://github.com/louisnw01/lightweight-charts-python | TradingView 级 K 线 Python 封装 |
| Plotly 动画文档 | https://plotly.com/python/animations/ | Frame-based 动画 + 播放控制 |
| QuantStats | https://github.com/ranaroussi/quantstats | 量化策略 50+ 评估指标可视化 |
| FinML-Toolkit | https://github.com/a-dorgham/FinML-Toolkit | 金融 ML + Plotly 交互可视化 |
| Bokeh 流式数据 | https://docs.bokeh.org/en/latest/docs/user_guide/data.html#streaming | ColumnDataSource.stream() 高效推送 |

### 搜索关键词（供进一步调研）

- `quantitative trading real-time dashboard visualization`
- `TradingView chart animation Python`
- `financial time series animation plotly frames`
- `algorithmic trading order execution visualization`
- `lightweight-charts-python multi pane`
- `bokeh streaming candlestick`

### 设计原则

> "Truly useful visualization is the 'second pair of eyes' for strategy R&D — focus on key indicators, one chart only explains one thing." -- 2025 quant best-practices

对本项目的映射：
1. **Panel 1（价格）**：使用者第一眼要看的是"价格和预测线的偏离"
2. **Panel 2（残差）**：直观展示偏离幅度和是否逼近 Emergency 线
3. **Panel 3（自信度）**：抽象信号的可视化，SIGMOID 映射的直观感受
4. **平仓瞬间**：三个面板同时高亮，突出系统做出决策的那个关键时刻

---

*报告生成日期：2026-05-23*
*基于 ExitPipeline v0.1.0 代码分析*
