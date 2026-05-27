/**
 * 星空策略共享数学库 (JavaScript)
 * 被 dashboard.html 和 trend-analyzer.html 共享
 */

function polyval(coeffs, x) {
  let y = 0;
  for (let i = 0; i < coeffs.length; i++) y = y * x + coeffs[i];
  return y;
}

function polyfit(x, y, degree) {
  const n = x.length, m = degree + 1;
  const A = Array.from({ length: n }, (_, i) =>
    Array.from({ length: m }, (_, j) => Math.pow(x[i], j))
  );
  const AT = Array.from({ length: m }, (_, j) =>
    Array.from({ length: n }, (_, i) => A[i][j])
  );
  const ATA = Array.from({ length: m }, (_, j) =>
    Array.from({ length: m }, (_, k) =>
      AT[j].reduce((s, v, i) => s + v * A[i][k], 0)
    )
  );
  const ATy = Array.from({ length: m }, (_, j) =>
    AT[j].reduce((s, v, i) => s + v * y[i], 0)
  );
  const aug = ATA.map((row, j) => [...row, ATy[j]]);
  for (let col = 0; col < m; col++) {
    let pivot = col;
    for (let r = col + 1; r < m; r++)
      if (Math.abs(aug[r][col]) > Math.abs(aug[pivot][col])) pivot = r;
    [aug[col], aug[pivot]] = [aug[pivot], aug[col]];
    const piv = aug[col][col];
    if (Math.abs(piv) < 1e-14) continue;
    for (let c = col; c <= m; c++) aug[col][c] /= piv;
    for (let r = 0; r < m; r++) {
      if (r === col) continue;
      const f = aug[r][col];
      for (let c = col; c <= m; c++) aug[r][c] -= f * aug[col][c];
    }
  }
  return aug.map(r => r[m]).reverse();
}

function randn() {
  let u = 0;
  while (u === 0) u = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
}

function sharpe(returns) {
  if (returns.length < 2) return -10;
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((s, r) => s + (r - mean) ** 2, 0) / returns.length;
  if (variance < 1e-15) return -10;
  return (mean / Math.sqrt(variance)) * Math.sqrt(252);
}

function maxDrawdown(returns) {
  let cum = 1, peak = 1, maxDD = 0;
  for (const r of returns) {
    cum *= (1 + r);
    if (cum > peak) peak = cum;
    const dd = (cum - peak) / peak;
    if (dd < maxDD) maxDD = dd;
  }
  return maxDD;
}

function sigmoid(residual, k = 2.0, b = -0.74) {
  const arg = Math.max(Math.min(-k * Math.abs(residual) + b, 50), -50);
  return 1 / (1 + Math.exp(-arg));
}

function dynamicThreshold(holdingTime, sigma0 = 0.5, tau = 5.0) {
  const lam = Math.log(2) / tau;
  return sigma0 * Math.exp(-lam * holdingTime);
}

function Cnk(n, k) {
  if (k > n || k < 0) return 0;
  if (k === 0 || k === n) return 1;
  let r = 1;
  for (let i = 1; i <= k; i++) r = r * (n - i + 1) / i;
  return Math.round(r);
}

function cubicDiscriminant(coeffs) {
  const [a, b, c, d] = coeffs;
  if (Math.abs(a) < 1e-15) return { p: 0, q: 0, delta: 0, type: 'degenerate' };
  const p = (3 * a * c - b * b) / (3 * a * a);
  const q = (2 * b * b * b - 9 * a * b * c + 27 * a * a * d) / (27 * a * a * a);
  const delta = (q / 2) ** 2 + (p / 3) ** 3;
  let type;
  if (delta > 1e-10) type = '1_real_root';
  else if (Math.abs(delta) < 1e-10) type = 'multiple_roots';
  else type = '3_real_roots';
  return { p, q, delta, type };
}

// Zone determination
function getZone(holdingTime, tau) {
  if (holdingTime < tau) return 1;
  if (holdingTime < 2 * tau) return 2;
  return 3;
}

// EMA smoothing
function emaSmooth(values, tau) {
  const alpha = 1 - Math.pow(2, -1 / tau);
  const smoothed = [values[0]];
  for (let i = 1; i < values.length; i++)
    smoothed.push(alpha * values[i] + (1 - alpha) * smoothed[i - 1]);
  return smoothed;
}
