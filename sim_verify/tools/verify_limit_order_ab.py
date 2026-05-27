#!/usr/bin/env python3
"""
新实验 N7 (P1): 限价单策略 A/B 测试
验证: 三段分层挂单是否优于单一限价单

使用方式:
    python3 verify_limit_order_ab.py [--trials 300]

实验逻辑:
    A组 (当前): 单一限价单，挂 P_target*(1-gamma*(1-alpha))
    B组 (新方案): 三层分段挂单
      Tier1(30s) -> P_target*(1+gamma1), gamma1=f(alpha)
      Tier2(30s) -> P_target*(1+gamma2)
      Tier3(30s) -> P_target*(1+gamma3)
      90s后 -> 铁律三市价全出
    对比: 成交率、实际成交价与理想价偏差、总时间成本
"""

import argparse, numpy as np

def sharpe(r):
    if len(r)<2 or np.std(r)<1e-15: return -10
    return np.mean(r)/np.std(r)*np.sqrt(252)

def simulate_single_trial(n_steps=100):
    t = np.arange(n_steps, dtype=float)
    signal = 0.3*t**3 - 0.6*t**2 + 0.2*t + 100
    noise = np.random.normal(0, 0.2, n_steps)
    prices = signal + noise
    # 订单簿模拟: 每步有 bid/ask spread
    spread = 0.0003 + abs(np.random.normal(0, 0.0002))
    return prices, spread

def strategy_single_limit(prices, spread, entry_idx=20):
    """A组: 单一限价单"""
    returns = []; fills, orders = 0, 0
    t_fit = np.arange(entry_idx, dtype=float)
    coeffs = np.polyfit(t_fit, prices[:entry_idx], 3)
    pos = 1

    for i in range(entry_idx+1, len(prices)):
        if pos <= 0: break
        pred = np.polyval(coeffs, float(i)); resid = abs(prices[i]-pred)
        arg = -2*resid-0.74; alpha = 1/(1+np.exp(-max(min(arg,50),-50)))
        gamma = 0.001*(1-alpha)
        target = pred*(1+gamma)

        orders += 1
        # 限价单成交条件: 价格触及挂单价
        if prices[i] <= target: fills += 1
        else: pos = 0  # 未成交->等待下步

        ret = -(prices[i]-prices[i-1])/prices[i-1] - spread
        returns.append(ret if pos else 0)

    fill_rate = fills/max(orders,1)*100
    return returns, fill_rate

def strategy_tiered_limit(prices, spread, entry_idx=20):
    """B组: 三层分段挂单"""
    returns = []; fills, orders = 0, 0
    t_fit = np.arange(entry_idx, dtype=float)
    coeffs = np.polyfit(t_fit, prices[:entry_idx], 3)
    pos = 1; tier = 0
    tier_gammas = [0.0002, 0.0005, 0.0010]  # 逐层让步
    tier_start_step = entry_idx

    for i in range(entry_idx+1, len(prices)):
        if pos <= 0: break
        steps_in_tier = i - tier_start_step
        if steps_in_tier > 10 and tier < 2:
            tier += 1; tier_start_step = i  # 超时升级

        pred = np.polyval(coeffs, float(i)); resid = abs(prices[i]-pred)
        arg = -2*resid-0.74; alpha = 1/(1+np.exp(-max(min(arg,50),-50)))
        gamma = tier_gammas[tier]*(1-alpha)
        target = pred*(1+gamma)

        orders += 1
        if prices[i] <= target: fills += 1
        else: pos = 0

        ret = -(prices[i]-prices[i-1])/prices[i-1] - spread
        returns.append(ret if pos else 0)

    fill_rate = fills/max(orders,1)*100
    return returns, fill_rate

def main():
    parser = argparse.ArgumentParser(description="N7: 限价单 A/B 测试")
    parser.add_argument("--trials", type=int, default=300)
    args = parser.parse_args()

    a_returns, b_returns = [], []
    a_fills, b_fills = [], []

    print("="*72)
    print("  新实验 N7 (P1): 限价单策略 A/B 测试")
    print("="*72)
    print(f"  A组: 单一限价单 | B组: 三层分段挂单")
    print(f"  试验: {args.trials}")
    print()

    for _ in range(args.trials):
        prices, spread = simulate_single_trial()
        ret_a, fr_a = strategy_single_limit(prices, spread)
        ret_b, fr_b = strategy_tiered_limit(prices, spread)
        a_returns.extend(ret_a); b_returns.extend(ret_b)
        a_fills.append(fr_a); b_fills.append(fr_b)

    s_a, s_b = sharpe(a_returns), sharpe(b_returns)
    fr_a_avg, fr_b_avg = np.mean(a_fills), np.mean(b_fills)

    print(f"  {'指标':>20s}  {'A组(单一)':>12s}  {'B组(三段)':>12s}  {'改善':>10s}")
    print(f"  {'-'*60}")
    print(f"  {'夏普比':>20s}  {s_a:>12.3f}  {s_b:>12.3f}  {s_b-s_a:>+9.3f}")
    print(f"  {'限价单成交率':>20s}  {fr_a_avg:>11.1f}%  {fr_b_avg:>11.1f}%  {fr_b_avg-fr_a_avg:>+9.1f}%")

    print()
    if fr_b_avg > fr_a_avg + 10:
        print(f"  ✅ 三层分段挂单显著改善成交率 ({fr_b_avg-fr_a_avg:+.1f}%)")
    else:
        print(f"  ⚠️ 改善幅度有限——需调整分层参数或增加真实订单簿模拟")

    if s_b > s_a:
        print(f"  ✅ 三层分段挂单同时提升夏普比 ({s_b-s_a:+.3f})")
    print(f"  报告引用: research-supplement-v1.md Part C - 新实验 N7")

if __name__ == "__main__":
    main()
