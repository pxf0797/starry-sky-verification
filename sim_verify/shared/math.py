"""
星空策略共享数学库 (Python)
所有 verify_*.py 工具的统一数学基础
"""
import numpy as np

def polyval(coeffs, x):
    """多项式求值"""
    return np.polyval(coeffs, x)

def polyfit(x, y, degree):
    """最小二乘多项式拟合"""
    return np.polyfit(x, y, degree)

def sharpe(returns, annualize=True):
    """夏普比 (年化)"""
    if len(returns) < 2:
        return -10.0
    mean = np.mean(returns)
    std = np.std(returns)
    if std < 1e-15:
        return -10.0
    s = mean / std
    return s * np.sqrt(252) if annualize else s

def sortino(returns, annualize=True):
    """Sortino比 (仅惩罚下行波动)"""
    if len(returns) < 2:
        return -10.0
    mean = np.mean(returns)
    downside = np.std([r for r in returns if r < 0]) if any(r < 0 for r in returns) else 1e-10
    s = mean / max(downside, 1e-10)
    return s * np.sqrt(252) if annualize else s

def max_drawdown(returns):
    """最大回撤"""
    if len(returns) == 0:
        return -10.0
    cumulative = np.cumprod(1 + np.array(returns))
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    return float(np.min(drawdowns))

def sigmoid(residual, k=2.0, b=-0.74):
    """SIGMOID自信度函数: α(Δ) = σ(-k|Δ|+b)"""
    arg = -k * abs(residual) + b
    arg = max(min(arg, 50), -50)
    return 1.0 / (1.0 + np.exp(-arg))

def dynamic_threshold(holding_time, sigma0=0.5, tau=5.0):
    """动态阈值: σ(t) = σ₀·e^(-λt)"""
    lam = np.log(2) / tau
    return sigma0 * np.exp(-lam * holding_time)

def randn():
    """标准正态随机数"""
    return np.random.normal()

def generate_garch_noise(n, omega=0.01, alpha=0.1, beta=0.85):
    """生成 GARCH(1,1) 噪声序列"""
    sigma2 = np.zeros(n)
    eps = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
        eps[t] = np.sqrt(sigma2[t]) * randn()
    return eps

def generate_student_t_noise(n, nu=3):
    """生成 Student's t 分布噪声"""
    z = np.random.normal(size=n)
    chi2 = np.random.chisquare(nu, size=n)
    return z * np.sqrt(nu / chi2)

def cubic_discriminant(coeffs):
    """计算三次多项式的卡尔达诺判别式
    coeffs = [a, b, c, d] 对应 P(t) = a·t³ + b·t² + c·t + d
    返回 (p, q, delta, root_type)
    """
    a, b, c, d = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
    if abs(a) < 1e-15:
        return 0, 0, 0, "degenerate"
    p = (3*a*c - b**2) / (3*a**2)
    q = (2*b**3 - 9*a*b*c + 27*a**2*d) / (27*a**3)
    delta = (q/2)**2 + (p/3)**3
    if delta > 0:
        return p, q, delta, "1_real_root"
    elif abs(delta) < 1e-10:
        return p, q, delta, "multiple_roots"
    else:
        return p, q, delta, "3_real_roots"
