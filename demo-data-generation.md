# 演示金融数据生成方法研究

## 目录

1. [核心方法论](#1-核心方法论)
2. [经典形态生成算法](#2-经典形态生成算法)
3. [精确触发控制](#3-精确触发控制)
4. [过拟合陷阱数据对](#4-过拟合陷阱数据对)
5. [市场微观结构噪声](#5-市场微观结构噪声)
6. [形态 + 噪声组装流水线](#6-组装流水线)

---

## 1. 核心方法论

### 设计原则

```
价格(t) = 骨架形态(t) + 微观结构噪声(t)
           ↑                   ↑
        可控骨架           真实感填充
        精确触发           风格化纹理
```

**骨架形态**: 分段函数 / 样条插值 / 贝塞尔曲线生成精确走势。
**微观结构噪声**: GARCH 波动、价格簇、买卖价差反弹、肥尾叠加在骨架之上。

两层解耦的好处：
- 形态和噪声可独立开发和测试
- 同一个形态可以叠加不同的噪声，观察策略在不同市场微观结构下的表现
- 触发条件在骨架层精确满足，不受噪声层干扰

### 整体架构

```
┌──────────────────────────────┐
│   Config: 形态类型 + 参数     │
│   (头肩顶, V形, 阶梯, 插针)   │
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│   骨架生成器                  │
│   - 样条插值 (scipy.interp)   │
│   - 分段线性                   │
│   - Bezier 曲线               │
│   输出: 每日 "理想" 价格序列    │
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│   微观结构噪声叠加器           │
│   - GARCH 波动聚集            │
│   - 价格簇 (整数位偏好)        │
│   - Bid-Ask Bounce           │
│   - Student-t 肥尾            │
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│   OHLC 生成器                 │
│   骨架+噪声 → Open/High/Low/Close│
│   精确触发条件校验             │
└──────────────────────────────┘
```

---

## 2. 经典形态生成算法

### 2.1 头肩顶 (Head & Shoulders Top)

**数学描述**：将价格走势看作一维函数 f(t)，需求满足：

- 左肩: 局部极大值 L (t1)
- 头部: 全局最大值 H (t2 > t1, H > L)
- 右肩: 局部极大值 R (t3 > t2, R < H, R ≈ L)
- 颈线: 连接左肩底部和右肩底部的直线
- 跌破: 价格跌破颈线

**方法**: 使用 5 个控制点的三次样条插值，强制指定关键点坐标。

```python
import numpy as np
from scipy.interpolate import CubicSpline

def head_shoulders(length=60, lhs_level=100, head_level=115,
                   rhs_level=100, neck_level=93):
    """
    生成头肩顶形态的骨架价格序列。

    控制点:
      t=0        左肩起点
      t=12       左肩峰 (lhs_level)
      t=20       左肩谷 (颈线位)
      t=30       头峰 (head_level)
      t=40       右肩峰 (rhs_level)
      t=50       右肩谷 (跌破颈线)
      t=59       终点 (继续下跌)

    返回: (t, price) — 可以直接绘制的骨架
    """
    t_ctrl = np.array([0, 12, 20, 30, 40, 50, 59])
    # 前两个点上升至左肩; 滑落至颈线; 升至头部; 降至右肩; 跌破颈线; 继续跌
    base = neck_level - 3  # 起跌点略高于颈线
    p_ctrl = np.array([
        base,                    # 起点
        lhs_level,               # 左肩
        neck_level,              # 左肩回落至颈线
        head_level,              # 头部
        rhs_level,               # 右肩
        neck_level - 2,          # 跌破颈线
        neck_level - 8           # 继续下跌
    ])
    cs = CubicSpline(t_ctrl, p_ctrl, bc_type='natural')
    t = np.arange(length)
    return t, cs(t)

# --- 使用 ---
t, p = head_shoulders()
# import matplotlib.pyplot as plt
# plt.plot(t, p)
# plt.axhline(neck_level, color='r', linestyle='--', label='颈线')
```

**关键参数**：`head_level` 必须显著高于 `lhs_level` 和 `rhs_level`，右肩谷必须低于颈线。

### 2.2 V 形反转 (V-shaped Reversal)

**数学描述**：对称或非对称的急跌-急涨。用两段线性函数或分段整函数拼接。

```python
def v_shape(length=60, start=100, trough=80, end=110,
            trough_at=0.4, symmetry=0.6):
    """
    V 形反转骨架。

    参数:
      length    : 总长度
      start     : 起点价格
      trough    : 谷底价格
      end       : 终点价格
      trough_at : 谷底位置占比 (0~1)
      symmetry  : >0.5 下跌更快, <0.5 上涨更快,
                  0.5 对称

    使用两段 power-law: f(t) = a * (t - t0)^k + b
    k 控制弯曲程度。
    """
    t = np.arange(length)
    n1 = int(length * trough_at)
    n2 = length - n1
    k_down = 2.0 * (1 - symmetry) + 0.5  # 下跌段弯曲
    k_up = 2.0 * symmetry + 0.5           # 上涨段弯曲

    t1 = np.linspace(0, 1, n1)
    t2 = np.linspace(0, 1, n2)

    seg1 = start - (start - trough) * (t1 ** k_down)
    seg2 = trough + (end - trough) * (t2 ** k_up)

    return t, np.concatenate([seg1, seg2])

# --- 使用 ---
t, p = v_shape(symmetry=0.7)  # 急跌缓涨
```

**变体**: 将 `k_down` 设得很小（如 0.3）可模拟"尖底"，设为很大（如 3.0）则模拟"圆弧底"。

### 2.3 阶梯式下跌 (Staircase Decline)

**数学描述**：下跌段 → 横盘段交替出现。每一级台阶的价格中枢逐级下移。

```python
def staircase(length=120, steps=4, start=100, total_drop=30):
    """
    阶梯式下跌骨架。

    参数:
      steps      : 阶梯级数
      start      : 起始价格
      total_drop : 总跌幅

    每一级: 下跌段 (占 30%) + 横盘段 (占 70%)
    横盘带小幅正弦波动。
    """
    step_len = length // steps
    drop_per_step = total_drop / steps
    t = np.arange(length)
    price = np.zeros(length)

    for i in range(steps):
        s = i * step_len
        e = s + step_len if i < steps - 1 else length
        n = e - s
        decline_n = max(1, int(n * 0.3))
        flat_n = n - decline_n

        base = start - i * drop_per_step
        # 下跌
        seg1 = np.linspace(base, base - drop_per_step * 0.8, decline_n)
        # 横盘 (带小幅波动)
        seg2 = base - drop_per_step * 0.8 + \
               np.sin(np.linspace(0, np.pi, flat_n)) * 0.3

        price[s:e] = np.concatenate([seg1, seg2[:n - decline_n]]) if n == len(seg1) + len(seg2[:n - decline_n]) else np.concatenate([seg1, seg2[:n - decline_n]])

    # 由于边界处理，用更稳健的方式:
    price = np.zeros(length)
    idx = 0
    for i in range(steps):
        n = step_len if i < steps - 1 else length - idx
        decline_n = max(1, int(n * 0.3))
        flat_n = n - decline_n
        base = start - i * drop_per_step
        seg1 = np.linspace(base, base - drop_per_step * 0.8, decline_n)
        seg2 = base - drop_per_step * 0.8 + \
               np.sin(np.linspace(0, np.pi * 0.5, flat_n)) * 0.3
        price[idx:idx+decline_n] = seg1
        price[idx+decline_n:idx+n] = seg2
        idx += n
    return t, price
```

### 2.4 插针行情 (Wick/Spike)

**数学描述**：正常走势中插入一个瞬间暴跌并快速收回的单根 K 线（或连续 2-3 根）。

插针的精髓在 OHLC 结构：**收盘价接近开盘价**（针身短），但最低价（或最高价）远离实体。这在 OHLC 生成阶段完成。

```python
def spike(length=60, base_price=100, spike_depth=0.06, spike_pos=0.5):
    """
    在价格序列中插入瞬间暴跌并反弹的"插针"。

    spike_depth: 暴跌幅度 (比例, 如 0.05 = 5%)
    spike_pos  : 插针位置占比

    返回的 price 是收盘价序列，插针点是突变的。
    真正的"长下影线"在 OHLC 构建时完成——插针点处 Low 远低于 Close。
    """
    t = np.arange(length)
    # 基础走势: 缓慢上升
    trend = base_price * (1 + 0.05 * t / length)
    noise = np.random.randn(length) * 0.3
    price = trend + noise

    # 插入暴跌
    spike_idx = int(length * spike_pos)
    price[spike_idx] = base_price * (1 - spike_depth)

    # 下一根反弹
    if spike_idx + 1 < length:
        price[spike_idx + 1] = base_price * (1 + 0.01)

    return t, price
```

**插针的 OHLC 渲染**：仅在价格序列层面看不出"长下影线"效果。真正的插针需要这样构建 OHLC：

```python
def build_ohlc_with_spike(close_prices, spike_idx, spike_low):
    """在插针点构造 Long Lower Shadow。"""
    ohlc = []
    for i, c in enumerate(close_prices):
        if i == spike_idx:
            ohlc.append({
                'open': c,
                'high': c,
                'low': spike_low,   # 远低于收盘
                'close': c * 0.995  # 小阴线
            })
        elif i == spike_idx + 1:
            ohlc.append({
                'open': close_prices[i-1] * 0.995,
                'high': c * 1.01,
                'low': c * 0.99,
                'close': c
            })
        else:
            # 正常 K 线
            o = c * (1 + np.random.randn() * 0.002)
            ohlc.append({
                'open': o,
                'high': max(o, c) * 1.005,
                'low': min(o, c) * 0.995,
                'close': c
            })
    return ohlc
```

---

## 3. 精确触发控制

设计原则：**先定触发条件，后反向构造价格序列**。不要先随机生成再筛选——对于教学场景，必须保证特定铁律 100% 触发。

### 3.1 铁律一：利润门槛精确到达

**需求**：价格必须精确到达某个目标价（如到达谷底），偏离 < 阈值。

**方法**：三次方程反推

```python
def ensure_profit_target(length=60, start=100, target_low=85,
                         target_high=115, target_pos=0.7):
    """
    构造一个先跌后涨的 V 形，保证：
      1. 最低价精确 = target_low
      2. 最终价格精确到达 target_high
      3. 最低价出现在 target_pos 位置

    使用约束构造三次多项式 p(t) = a*t^3 + b*t^2 + c*t + d。
    """
    # 约束:
    # p(0) = start, p(target_pos*L) = target_low,
    # p(L-1) = target_high, 且在 target_pos*L 处导数为 0 (极值点)
    L = length - 1
    tm = target_pos * L

    # 构建线性方程组 A * x = b
    # x = [a, b, c, d]
    A = np.array([
        [0, 0, 0, 1],                          # p(0) = start
        [tm**3, tm**2, tm, 1],                # p(tm) = target_low
        [L**3, L**2, L, 1],                   # p(L) = target_high
        [3*tm**2, 2*tm, 1, 0]                # p'(tm) = 0
    ])
    b_vec = np.array([start, target_low, target_high, 0])

    coeffs = np.linalg.solve(A, b_vec)
    t = np.arange(length)
    price = np.polyval(coeffs[::-1], t)

    # 验证
    assert abs(np.min(price) - target_low) < 0.01, "谷底未到达目标"
    assert abs(price[-1] - target_high) < 0.01, "终点未到达目标"
    return t, price
```

**扩展**: 如果需要更复杂的形态（头肩顶到达某个精确水平），可以用分段三次样条，在关键点处强制函数值 + 一阶导数值。

### 3.2 铁律三：自信度止损触发

**需求**: 预测残差持续增大，自信度跌破 0.3。

**方法**: 在"预测线"基础上叠加递增的偏移量，保证残差单调增长（且可微）。

```python
def ensure_confidence_stop(length=60, start=100,
                           pred_slope=0.02, max_residual=15.0):
    """
    构造数据使得自信度（基于残差的倒数）逐步下降。

    设计:
      - 预测线: 缓慢上升直线 (代表交易者乐观预测)
      - 实际价格: 预测线 + 递增的偏移量
        offset(t) = A * (t / T)^k, k > 1 保证残差加速增长
      - 自信度: confidence = clamp(1 - residual/max_residual, 0, 1)
      - 当 offset > max_residual 时自信度 = 0

    参数:
      pred_slope  : 预测线斜率
      max_residual: 自信度降至 0 的残差阈值
    """
    t = np.arange(length)
    T = length - 1

    # 预测线 (线性)
    prediction = start + pred_slope * t

    # 偏移: 前期缓慢增长, 后期加速
    # 用指数 k=2.5 确保残差"看似随机但单调递增"
    k = 2.5
    offset = max_residual * 1.2 * (t / T) ** k

    # 实际价格 = 预测线 + 偏移 (实际低于预期)
    actual = prediction - offset

    # 残差 = 预测 - 实际
    residual = prediction - actual

    # 自信度
    confidence = np.clip(1.0 - residual / max_residual, 0.0, 1.0)

    # 验证: 自信度跌破 0.3 的点存在
    cross_idx = np.where(confidence < 0.3)[0]
    assert len(cross_idx) > 0, "自信度未跌破 0.3"
    return t, actual, prediction, confidence
```

### 3.3 铁律五：拔网线触发

**需求**: 单根或连续 K 线跌幅 > 5%。

```python
def ensure_network_cut(length=60, base=100, crash_depth=0.06,
                       crash_pos=0.8, recovery=True):
    """
    在精确位置插入 >5% 的暴跌。

    参数:
      crash_depth: 跌幅 (如 0.06 = 6%)
      crash_pos  : 暴跌位置占比
      recovery   : 是否在暴跌后反弹

    返回的骨架序列中 crash_idx 处下跌 crash_depth。
    对应的 OHLC 需要把 Low 设为 crash_low 来体现"插针"或"大阴线"。
    """
    t = np.arange(length)
    crash_idx = int(length * crash_pos)

    # 正常走势
    trend = base * (1 + 0.03 * np.sin(2 * np.pi * t / length))
    noise = np.random.randn(length) * 0.2
    price = trend + noise

    # 记录暴跌前的价格
    pre_crash = price[crash_idx]

    # 暴跌
    price[crash_idx] = pre_crash * (1 - crash_depth)

    if recovery:
        # 下一根 K 线反弹
        if crash_idx + 1 < length:
            bounce = pre_crash * 0.98
            price[crash_idx + 1] = bounce
            # 恢复到正常走势
            if crash_idx + 2 < length:
                price[crash_idx + 2:] = trend[crash_idx + 2:] + \
                    np.random.randn(length - crash_idx - 2) * 0.2

    # 验证
    daily_return = np.diff(price) / price[:-1]
    min_ret = np.min(daily_return)
    assert min_ret < -crash_depth + 0.005, \
        f"最大日跌幅 {min_ret:.2%} 未达目标 {-crash_depth:.0%}"
    return t, price, crash_idx
```

### 3.4 铁律组合触发

教学场景经常需要同时触发多条铁律，可以流水线式组合：

```python
def multi_rule_scenario(length=120, rules=None):
    """
    在一个序列中按顺序触发多条铁律。

    rules = [
        (rule_type, params, trigger_position)
    ]

    示例：先触发自信度止损，再触发拔网线
    """
    if rules is None:
        rules = [
            ('confidence_stop', {'max_residual': 10}, (0, 0.5)),
            ('network_cut', {'crash_depth': 0.07}, (0.7, 0.9)),
            ('profit_target', {'target': 95}, (0.9, 1.0))
        ]
    # 分段构造，每段调用对应的生成函数
    # 然后在段间用平滑过渡连接
    segments = []
    for rule_type, params, (start_p, end_p) in rules:
        seg_len = int(length * (end_p - start_p))
        # 调用对应的生成器
        # ...
    return np.concatenate(segments)
```

---

## 4. 过拟合陷阱数据对

核心逻辑：训练集和测试集**来自同一个骨架形态**，但**噪声模型完全不同**。

GA 在训练集上学到的噪声模式，在测试集上不仅无用，还会带来负收益。

### 4.1 生成逻辑

```python
def overfitting_trap_pair(length=200, seed_train=42, seed_test=99):
    """
    生成 (train, test) 数据对，两者骨架相同但噪声不同。

    训练集策略: 低频周期噪声 (GA 容易学到)
    测试集策略: 趋势反转 + 高频随机噪声 (GA 所学低频模式完全失效)

    返回:
      train: DataFrame 骨架 + 低频噪声
      test : DataFrame 骨架 + 高频噪声 + 趋势反转
    """
    base_length = length

    # === 公共骨架: 温和上升后进入震荡 ===
    t = np.arange(base_length)
    skeleton = 100 + 10 * np.sin(2 * np.pi * t / base_length * 0.5) + \
               5 * np.sin(2 * np.pi * t / base_length * 2)

    # === 训练集噪声: 低频周期模式 ===
    rng_train = np.random.RandomState(seed_train)
    # 明显的低频正弦噪声
    low_freq_noise = (
        3 * np.sin(2 * np.pi * t / 60) +
        2 * np.sin(2 * np.pi * t / 35) +
        1.5 * np.sin(2 * np.pi * t / 22)
    )
    # 极少量高斯噪声
    gauss_train = rng_train.randn(base_length) * 0.3
    train_price = skeleton + low_freq_noise + gauss_train

    # === 测试集噪声: 高频随机噪声 + 趋势反转 ===
    rng_test = np.random.RandomState(seed_test)
    # 高频噪声 (白噪声 + 小幅尖峰)
    high_freq_noise = rng_test.randn(base_length) * 2.0
    # 随机尖峰 (模拟突发事件)
    n_spikes = 5
    spike_positions = rng_test.choice(base_length, n_spikes, replace=False)
    for pos in spike_positions:
        high_freq_noise[pos] += rng_test.randn() * 8.0

    # 趋势反转: 后半段骨架下沉
    reversal = np.zeros(base_length)
    reversal[base_length//2:] = -np.linspace(0, 8, base_length - base_length//2)
    test_price = skeleton + high_freq_noise + reversal

    return {
        'train': {'t': t, 'price': train_price, 'noise_type': '低频周期'},
        'test': {'t': t, 'price': test_price, 'noise_type': '高频随机+反转'}
    }
```

### 4.2 GA 过拟合演示

为了让学生看到"训练集回测完美，测试集崩溃"，需要一个简单的 GA 框架：

```python
def moving_average_strategy(price, short_window, long_window):
    """简单的双均线策略。返回信号序列 1=做多, -1=做空, 0=持有。"""
    short_ma = np.convolve(price, np.ones(short_window)/short_window, mode='same')
    long_ma = np.convolve(price, np.ones(long_window)/long_window, mode='same')
    signal = np.zeros_like(price)
    signal[short_ma > long_ma] = 1
    signal[short_ma < long_ma] = -1
    return signal

def evaluate(price, short_window, long_window):
    """评估双均线策略的收益。"""
    signal = moving_average_strategy(price, int(short_window), int(long_window))
    returns = np.diff(price) / price[:-1]
    strategy_returns = signal[:-1] * returns
    return np.prod(1 + strategy_returns) - 1  # 累计收益率

# === GA 在训练集上优化 ===
# 学生对 (short, long) 做网格搜索或简单 GA
# 训练集噪声有规律，GA 会学到 22/35/60 周期的规律
# 最佳参数: short=11, long=27 (在训练集表现极好)

# === 测试集上验证 ===
# 同一个参数在测试集上: 高频噪声打乱信号，趋势反转导致持续亏损
# 结果: 收益率由正转负
```

### 4.3 扩展：多组数据对

可以生成多组"过拟合陷阱"数据对，对应不同噪声模式组合：

| 训练集噪声 | 测试集噪声 | 教学要点 |
|---|---|---|
| 低频正弦 (周期 20-60) | 白噪声 + 随机尖峰 | GA 学到的周期模式在真实随机环境中失效 |
| GARCH 波动聚集 | 恒定波动率 | 策略过度押注波动率预测 |
| 价格簇 (整数位支撑) | 无价格簇 | 策略"神奇"地依赖整数位 |
| 趋势跟随友好 | 趋势反转 | 追涨杀跌策略在反转行情中血亏 |

---

## 5. 市场微观结构噪声

### 5.1 GARCH(1,1) 波动率聚集

GARCH(1,1) 的递归公式：

```
σ²(t) = ω + α * ε²(t-1) + β * σ²(t-1)
```

其中 ω > 0, α, β >= 0, α + β < 1。

```python
def garch_noise(length, omega=0.01, alpha=0.15, beta=0.80,
                seed=None, dist='normal', df_t=5):
    """
    用 GARCH(1,1) 生成条件异方差噪声。

    参数:
      omega, alpha, beta : GARCH 参数
      dist  : 'normal' 或 't' (Student-t)
      df_t  : Student-t 自由度 (越小尾部越肥)

    返回: (epsilon, sigma) 噪声序列和条件波动率序列
    """
    rng = np.random.RandomState(seed)
    sigma = np.zeros(length)
    epsilon = np.zeros(length)

    # 初始波动率
    sigma[0] = np.sqrt(omega / (1 - alpha - beta))

    for t in range(1, length):
        # 生成标准化残差
        if dist == 'normal':
            z = rng.randn()
        elif dist == 't':
            # Student-t 标准化 (方差 = df/(df-2))
            scale = np.sqrt((df_t - 2) / df_t)
            z = rng.standard_t(df_t) * scale
        else:
            z = rng.randn()

        epsilon[t] = z * sigma[t-1]
        sigma[t] = np.sqrt(omega + alpha * epsilon[t]**2 + beta * sigma[t-1]**2)

    return epsilon, sigma

# --- 使用 ---
# noise, vol = garch_noise(1000, dist='t', df_t=4)
# 可以观察到波动率聚集: 大幅波动后往往跟随大幅波动
```

### 5.2 价格簇 (Price Clustering)

价格倾向在整数位、0.5、0.25 等"心理价位"附近聚集。

```python
def price_clustering(price, cluster_levels=None, strength=0.3):
    """
    对价格序列施加价格簇效应。

    价格会被吸引到最近的 cluster_levels 上。

    参数:
      price          : 原始价格序列
      cluster_levels : 价格簇位置 (默认等间隔整数)
      strength       : 吸引力强度 (0~1)

    示例: cluster_levels = [整数, 整数+0.5]
    """
    if cluster_levels is None:
        # 默认: 所有整数 + 0.5 位置
        base = np.floor(np.min(price) // 5 * 5)
        top = np.ceil(np.max(price) // 5 * 5) + 5
        cluster_levels = np.arange(base, top, 0.5)

    clustered = np.copy(price)
    for i in range(len(price)):
        # 找到最近的价格簇
        distances = np.abs(cluster_levels - price[i])
        nearest = cluster_levels[np.argmin(distances)]
        # 以 strength 概率向最近簇偏移
        if np.random.random() < strength:
            clustered[i] = nearest * (1 - strength * 0.01) + \
                           price[i] * strength * 0.01
        else:
            clustered[i] = price[i] * (1 - 0.001) + nearest * 0.001

    return clustered
```

### 5.3 买卖价差反弹 (Bid-Ask Bounce)

理论：即使在"真实价格"不变的情况下，成交价在买一和卖一之间交替，也会产生微小振荡。

```python
def bid_ask_bounce(mid_price, spread=0.02, bounce_prob=0.4):
    """
    在中间价基础上模拟买卖价差反弹。

    参数:
      mid_price   : "真实"中间价序列
      spread      : 买卖价差 (绝对数值)
      bounce_prob : 反弹概率 (每次成交有概率从 bid 切换到 ask)

    返回: observed_price, 交替在 mid +/- spread/2 附近
    """
    observed = np.copy(mid_price)
    side = 1  # 1 = ask, -1 = bid
    for i in range(1, len(mid_price)):
        # 有 bounce_prob 的概率切换买卖方向
        if np.random.random() < bounce_prob:
            side *= -1
        observed[i] = mid_price[i] + side * spread / 2
    return observed
```

### 5.4 肥尾 (Fat Tail) — Student-t 分布

从正态分布切换到 Student-t 分布即可引入肥尾。

```python
def fat_tail_noise(length, df=3, scale=0.5, seed=None):
    """
    用 Student-t 分布生成肥尾噪声。

    df=3   : 非常肥的尾部 (存在尖峰风险)
    df=5   : 中等肥尾
    df=10  : 接近正态
    df=30+ : 近似正态

    Student-t 的方差 = df/(df-2) for df > 2
    需要缩放到目标方差。
    """
    rng = np.random.RandomState(seed)
    raw = rng.standard_t(df, size=length)
    # 缩放到目标标准差
    target_var = scale ** 2
    t_var = df / (df - 2) if df > 2 else 10  # df <= 2 时方差无限
    return raw * np.sqrt(target_var / t_var)
```

### 5.5 完整噪声叠加器

将所有噪声层组合起来，叠加到骨架上：

```python
def add_microstructure_noise(skeleton, config=None):
    """
    在价格骨架叠加多层微观结构噪声。

    噪声层叠加顺序:
      1. GARCH 异方差波动
      2. Student-t 肥尾冲击
      3. 价格簇偏移
      4. Bid-Ask Bounce
    """
    if config is None:
        config = {
            'garch': {'omega': 0.005, 'alpha': 0.12, 'beta': 0.85},
            'fat_tail': {'df': 4, 'scale': 0.3},
            'clustering': {'strength': 0.2},
            'bounce': {'spread': 0.01, 'prob': 0.3}
        }

    length = len(skeleton)

    # 1. GARCH + 肥尾 (合并成一个噪声)
    garch_eps, _ = garch_noise(
        length,
        **config['garch'],
        dist='t',
        df_t=config['fat_tail']['df']
    )

    # 2. 叠加 GARCH 噪声到骨架
    noisy = skeleton + garch_eps

    # 3. 价格簇
    noisy = price_clustering(noisy, strength=config['clustering']['strength'])

    # 4. Bid-Ask Bounce
    noisy = bid_ask_bounce(noisy, **config['bounce'])

    return noisy
```

---

## 6. 组装流水线

### 6.1 从骨架到 OHLC 的完整流程

```python
import pandas as pd

def generate_demo_ohlc(length=120, pattern='v_shape',
                       noise_config=None, params=None):
    """
    从形态骨架到 OHLC 数据帧的完整流水线。

    参数:
      pattern     : 'head_shoulders' | 'v_shape' | 'staircase' | 'spike'
      noise_config: 噪声配置字典
      params      : 形态特定参数
    """
    # Step 1: 生成骨架
    if pattern == 'head_shoulders':
        t, skeleton = head_shoulders(length=length, **(params or {}))
    elif pattern == 'v_shape':
        t, skeleton = v_shape(length=length, **(params or {}))
    elif pattern == 'staircase':
        t, skeleton = staircase(length=length, **(params or {}))
    elif pattern == 'spike':
        t, skeleton, _ = ensure_network_cut(length=length, **(params or {}))
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    # Step 2: 叠加噪声
    noisy_close = add_microstructure_noise(skeleton, noise_config)

    # Step 3: 生成 OHLC
    ohlc = []
    for i in range(length):
        c = noisy_close[i]
        o = c * (1 + np.random.randn() * 0.003)
        h = max(o, c) * (1 + abs(np.random.randn()) * 0.004)
        l_val = min(o, c) * (1 - abs(np.random.randn()) * 0.004)
        ohlc.append({
            'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(days=i),
            'open': round(o, 2),
            'high': round(h, 2),
            'low': round(l_val, 2),
            'close': round(c, 2),
            'volume': int(np.random.randint(5000, 20000))
        })

    df = pd.DataFrame(ohlc)

    # Step 4: 触发条件校验 (日志输出)
    _validate_triggers(df, skeleton, pattern, params)

    return df

def _validate_triggers(df, skeleton, pattern, params):
    """校验关键触发条件是否满足。"""
    close = df['close'].values
    low = df['low'].values
    max_drop = np.min(np.diff(close) / close[:-1])
    min_price = np.min(low)
    max_price = np.max(df['high'].values)

    if pattern == 'v_shape' and params:
        print(f"[校验] 最低价: {min_price:.2f} (目标: {params.get('trough', 'N/A')})")
    if pattern == 'spike':
        print(f"[校验] 最大日跌幅: {max_drop:.2%} (目标: {params.get('crash_depth', 'N/A')})")
```

### 6.2 使用示例

```python
# 生成一个 60 天的 V 形反转 + 完整噪声
df = generate_demo_ohlc(
    length=60,
    pattern='v_shape',
    params={'start': 100, 'trough': 85, 'end': 110, 'symmetry': 0.6},
    noise_config={
        'garch': {'omega': 0.005, 'alpha': 0.12, 'beta': 0.85},
        'fat_tail': {'df': 4, 'scale': 0.3},
        'clustering': {'strength': 0.2},
        'bounce': {'spread': 0.01, 'prob': 0.3}
    }
)
# df 可以直接用于回测系统
```

### 6.3 与现有系统的集成

当前系统的线性趋势 + 高斯噪声生成器可以用新的流水线替换——只需保持输出接口一致（`pd.DataFrame` 含 `open/high/low/close/volume` 列）。

```python
# 现有接口兼容:
def generate_price_series_enhanced(length, config):
    """替换现有 generate_price_series 的增强版本。"""
    pattern = config.get('pattern', 'v_shape')
    pattern_params = config.get('pattern_params', {})
    noise_config = config.get('noise_config', {})

    df = generate_demo_ohlc(
        length=length,
        pattern=pattern,
        params=pattern_params,
        noise_config=noise_config
    )
    return df['close'].values  # 兼容仅需要 close 的旧接口
```

---

## 总结

| 能力 | 方法 | 代码 |
|---|---|---|
| 头肩顶 | 5 控制点三次样条 | `CubicSpline` |
| V 形反转 | 两段 power-law 拼接 | `t**k` |
| 阶梯下跌 | 下跌+横盘交替分段 | 循环分段 |
| 插针行情 | 单点突变 + OHLC 长下影线 | `Low << Close` |
| 铁律一 (利润) | 三次方程反推 | `linalg.solve` |
| 铁律三 (止损) | 递增偏移量 | `(t/T)**k` |
| 铁律五 (拔网线) | 精确位置暴跌 | 直接赋值 |
| 过拟合陷阱 | 骨架相同噪声不同 | 低频 vs 高频 |
| 波动聚集 | GARCH(1,1) | `arch` 库或手写 |
| 价格簇 | 向整数位吸引 | 距离最近簇 |
| Bid-Ask Bounce | 买卖方向交替 | ±spread/2 |
| 肥尾 | Student-t | `standard_t` |

### 注意事项

1. **骨架和噪声必须解耦** — 触发条件只在骨架层保证，噪声层只负责"看起来真实"
2. **Seed 管控** — 每个数据集的每个噪声层都使用固定 seed，保证可重现
3. **逐层验证** — 每一步都断言关键指标（最低价、跌幅、残差等）满足要求
4. **教学标注** — 在返回的数据中包含 `metadata`，标注形态类型、关键点位置、噪声参数等信息，方便后续分析
