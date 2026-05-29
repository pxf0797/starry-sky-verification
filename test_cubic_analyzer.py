#!/usr/bin/env python3
"""
Test script for cubic_state_analyzer.py
Verifies core computation functions against expected values.
Non-interactive; no GUI.
"""

import sys
import os
import math

# Add the research directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cubic_state_analyzer import cubic_state, _f

EPS = 1e-10

def check(condition, msg):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {msg}")
    if not condition:
        print(f"         (expected opposite)")
    return condition

def test_case(name, a, b, c, d, expected):
    """Run a test case and compare against expected values."""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"  Coefficients: a={a}, b={b}, c={c}, d={d}")
    print(f"  Equation: {a}x³ + {b}x² + {c}x + {d} = 0")

    state = cubic_state(a, b, c, d)

    all_pass = True

    # Print computed values
    p, q, delta, D_crit = state['p'], state['q'], state['delta'], state['D_crit']
    print(f"  Computed: p={p:.6f}, q={q:.6f}, delta={delta:.6f}, D_crit={D_crit:.6f}")
    print(f"  Classification: {state['label']}")
    print(f"  Roots: {state['roots']}")
    print(f"  Extrema: {state['extrema']}")
    print(f"  Inflection: {state['inflection']}")

    # Check p
    if 'p' in expected:
        all_pass &= check(abs(p - expected['p']) < 1e-8,
                          f"p = {p:.6f} (expected {expected['p']:.6f})")

    # Check q
    if 'q' in expected:
        all_pass &= check(abs(q - expected['q']) < 1e-8,
                          f"q = {q:.6f} (expected {expected['q']:.6f})")

    # Check delta
    if 'delta' in expected:
        all_pass &= check(abs(delta - expected['delta']) < 1e-8,
                          f"delta = {delta:.6f} (expected {expected['delta']:.6f})")

    # Check classification
    if 'label' in expected:
        all_pass &= check(state['label'] == expected['label'],
                          f"label = '{state['label']}' (expected '{expected['label']}')")

    # Check root count
    if 'n_roots' in expected:
        actual_n = len([r for r in state['roots'] if not isinstance(r, str)])
        all_pass &= check(actual_n == expected['n_roots'],
                          f"n_real_roots = {actual_n} (expected {expected['n_roots']})")

    # Check extrema count
    if 'n_extrema' in expected:
        all_pass &= check(len(state['extrema']) == expected['n_extrema'],
                          f"n_extrema = {len(state['extrema'])} (expected {expected['n_extrema']})")

    # Check kind
    if 'kind' in expected:
        all_pass &= check(state['kind'] == expected['kind'],
                          f"kind = '{state['kind']}' (expected '{expected['kind']}')")

    return all_pass


def test_delta_convention():
    """
    Verify the sign convention of the Cardano discriminant.
    Δ = (q/2)² + (p/3)³
    - Δ > 0 → one real root
    - Δ < 0 → three real roots (casus irreducibilis)
    """
    print(f"\n{'='*60}")
    print("Test: Delta Sign Convention Verification")

    # x³ - 3x = 0 → p=-3, q=0 → Δ = 0 + (-1)³ = -1 → Δ < 0
    # This has 3 real roots: 0, ±√3
    state1 = cubic_state(1, 0, -3, 0)
    n1 = len([r for r in state1['roots'] if not isinstance(r, str)])
    delta1 = state1['delta']
    print(f"  x³-3x=0: p={state1['p']}, delta={delta1:.6f}, roots={state1['roots']}")

    # x³ + 2x + 1 = 0 → p=2, q=1 → Δ = 1/4 + 8/27 > 0
    # This has 1 real root
    state2 = cubic_state(1, 0, 2, 1)
    n2 = len([r for r in state2['roots'] if not isinstance(r, str)])
    delta2 = state2['delta']
    print(f"  x³+2x+1=0: p={state2['p']}, delta={delta2:.6f}, roots={state2['roots']}")

    all_pass = True
    all_pass &= check(delta1 < 0, f"x³-3x: delta < 0 → casus irreducibilis → 3 real roots")
    all_pass &= check(n1 == 3, f"x³-3x: has 3 real roots, found {n1}")
    all_pass &= check(delta2 > 0, f"x³+2x+1: delta > 0 → 1 real root")
    all_pass &= check(n2 == 1, f"x³+2x+1: has 1 real root, found {n2}")

    return all_pass


def test_delta_c_conversion():
    """
    Verify the relationship: Δ_c = -4p³ - 27q² = -108 * Δ
    """
    print(f"\n{'='*60}")
    print("Test: Δ_c = -108 * Δ Conversion")

    test_pairs = [
        (1, 0, -3, 0, "x³-3x"),
        (1, 0, 2, 1, "x³+2x+1"),
        (1, 0, -3, 2, "x³-3x+2"),
        (1, 0, 0, 0, "x³"),
    ]

    all_pass = True
    for a, b, c, d, name in test_pairs:
        state = cubic_state(a, b, c, d)
        p, q, delta = state['p'], state['q'], state['delta']
        delta_c_computed = -4 * p**3 - 27 * q**2
        delta_c_from_conversion = -108 * delta

        all_pass &= check(abs(delta_c_computed - delta_c_from_conversion) < 1e-9,
                          f"{name}: -4p³-27q² = {delta_c_computed:.6f} == -108Δ = {delta_c_from_conversion:.6f}")

    return all_pass


def test_classification_table():
    """
    Verify that the tool's classification matches the 5-type system
    from the document for representative examples.
    """
    print(f"\n{'='*60}")
    print("Test: 5-Type Classification Mapping")

    # Type I: Monotonic (D_crit < 0 → no extrema, delta > 0 → 1 root)
    # Example: x³ + 2x + 1
    s = cubic_state(1, 0, 2, 1)
    all_pass = True
    all_pass &= check(s['D_crit'] < 0 and s['delta'] > 0,
                      f"Type I (x³+2x+1): D_crit={s['D_crit']}<0, delta={s['delta']:.6f}>0 → monotonic, 1 root")
    all_pass &= check(len(s['extrema']) == 0,
                      f"Type I: 0 extrema (found {len(s['extrema'])})")
    all_pass &= check(len([r for r in s['roots'] if not isinstance(r, str)]) == 1,
                      f"Type I: 1 real root")

    # Type II: S-shaped (D_crit > 0, delta < 0 → 3 real roots)
    # Example: x³ - 3x
    s = cubic_state(1, 0, -3, 0)
    all_pass &= check(s['D_crit'] > 0 and s['delta'] < 0,
                      f"Type II (x³-3x): D_crit={s['D_crit']}>0, delta={s['delta']:.6f}<0 → S-shaped, 3 roots")
    all_pass &= check(len(s['extrema']) == 2,
                      f"Type II: 2 extrema (found {len(s['extrema'])})")

    # Type III: Critical fold (delta ≈ 0, D_crit > 0)
    # Example: x³ - 3x + 2 = 0 (q=2, p=-3, delta=0)
    s = cubic_state(1, 0, -3, 2)
    all_pass &= check(abs(s['delta']) < EPS and s['D_crit'] > 0,
                      f"Type III (x³-3x+2): delta≈0, D_crit={s['D_crit']}>0 → critical fold")

    # Also: x³ - 3x - 2 = 0 (q=-2)
    s = cubic_state(1, 0, -3, -2)
    all_pass &= check(abs(s['delta']) < EPS,
                      f"Type III (x³-3x-2): delta={s['delta']:.6f} ≈ 0 → critical fold")

    # Type IV: Cusp singularity (p=q=0)
    # Example: x³ = 0
    s = cubic_state(1, 0, 0, 0)
    all_pass &= check(abs(s['p']) < EPS and abs(s['q']) < EPS,
                      f"Type IV (x³=0): p=0, q=0 → Cusp singularity")
    all_pass &= check(abs(s['D_crit']) < EPS,
                      f"Type IV: D_crit=0 (degenerate extrema)")

    # Type V: Degenerate (a ≈ 0)
    # Example: 0x³ + 2x + 1 = 0
    s = cubic_state(0, 0, 2, 1)
    all_pass &= check(s['kind'] == 'degenerate',
                      f"Type V (a=0): kind='{s['kind']}' → degenerate")

    return all_pass


def test_preset_consistency():
    """
    Verify preset parameters match document Appendix B.1.
    """
    print(f"\n{'='*60}")
    print("Test: Preset Parameters vs Document Appendix B.1")

    presets = [
        # (a, b, c, d, doc_p, doc_q, doc_delta_c, doc_type)
        (1, 0, 2, 1, 2, 1, -59, "I"),
        (1, 0, -3, 0, -3, 0, 108, "II"),
        (1, 0, -3, 2, -3, 2, 0, "III"),
        (1, 0, -3, -2, -3, -2, 0, "III"),
        (1, 0, 0, 0, 0, 0, 0, "IV"),
    ]

    all_pass = True
    for a, b, c, d, doc_p, doc_q, doc_delta_c, doc_type in presets:
        s = cubic_state(a, b, c, d)

        # Check p
        all_pass &= check(abs(s['p'] - doc_p) < 1e-8,
                          f"{a}x³+{b}x²+{c}x+{d}: p={s['p']:.4f} (doc: {doc_p})")

        # Check q
        all_pass &= check(abs(s['q'] - doc_q) < 1e-8,
                          f"{a}x³+{b}x²+{c}x+{d}: q={s['q']:.4f} (doc: {doc_q})")

        # Compute Δ_c from p,q and compare
        delta_c = -4 * s['p']**3 - 27 * s['q']**2
        all_pass &= check(abs(delta_c - doc_delta_c) < 1e-8,
                          f"{a}x³+{b}x²+{c}x+{d}: Δ_c={delta_c:.4f} (doc: {doc_delta_c})")

    return all_pass


def test_boundary_cases():
    """
    Test edge cases: very small coefficients, zero coefficients, etc.
    """
    print(f"\n{'='*60}")
    print("Test: Boundary Cases")

    all_pass = True

    # Very small but non-zero a (should still be cubic, not degenerate with EPS=1e-10)
    s = cubic_state(1e-5, 0, 2, 1)
    all_pass &= check(s['kind'] == 'cubic',
                      f"a=1e-5 (a > EPS=1e-10): kind='{s['kind']}' (should be 'cubic')")

    # Exactly EPS
    s = cubic_state(1e-10, 0, 2, 1)
    # abs(1e-10) > 1e-10 ? No, 1e-10 > 1e-10 is False.
    # Actually: if abs(a) < EPS where EPS=1e-10, then 1e-10 is NOT < 1e-10, it's equal.
    # So 1e-10 > 1e-10 → False, so it's treated as cubic. Actually abs(a) = 1e-10, EPS = 1e-10,
    # abs(a) < EPS → 1e-10 < 1e-10 → False. So it's still cubic. Good.
    print(f"  a=1e-10: kind={s['kind']} (boundary, should still be cubic)")

    # Zero b and zero c (Type IV - Cusp)
    s = cubic_state(1, 0, 0, 0)
    all_pass &= check(len(s['extrema']) == 0,
                      f"x³=0: extrema count = {len(s['extrema'])} (degenerate)")

    # b ≠ 0, c = 0, d ≠ 0
    s = cubic_state(1, 1, 0, 1)
    print(f"  x³+x²+1=0: p={s['p']:.4f}, q={s['q']:.4f}, delta={s['delta']:.6f}")

    # All zeros
    s = cubic_state(0, 0, 0, 0)
    all_pass &= check(s['label'] == 'V: 降次退化型→零函数(全体实数根)',
                      f"f(x)=0: label='{s['label']}'")

    return all_pass


def print_delta_table():
    """
    Print a table comparing document Δ_c with tool's Cardano Δ for clarity.
    """
    print(f"\n{'='*60}")
    print("Reference: Δ (Cardano) vs Δ_c (Cusp) vs Root Count")
    print(f"{'='*60}")
    print(f"{'Equation':<20} {'Δ_cardano':>12} {'Δ_c':>12} {'Roots':>8} {'Doc Type':>8}")
    print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*8} {'-'*8}")

    cases = [
        (1, 0, 2, 1, "x³+2x+1"),
        (1, 0, -3, 0, "x³-3x"),
        (1, 0, -3, 2, "x³-3x+2"),
        (1, 0, -3, -2, "x³-3x-2"),
        (1, 0, 0, 0, "x³"),
    ]

    for a, b, c, d, name in cases:
        s = cubic_state(a, b, c, d)
        delta_cardano = s['delta']
        delta_c = -4 * s['p']**3 - 27 * s['q']**2
        n = len([r for r in s['roots'] if not isinstance(r, str)])

        # Document type mapping based on Δ_c
        if abs(delta_c) < 1e-8:
            if abs(s['p']) < 1e-8 and abs(s['q']) < 1e-8:
                doc_type = "IV"
            else:
                doc_type = "III"
        elif delta_c > 0:
            doc_type = "II"
        else:
            doc_type = "I"

        print(f"{name:<20} {delta_cardano:>12.6f} {delta_c:>12.6f} {n:>8} {doc_type:>8}")


if __name__ == '__main__':
    print("=" * 60)
    print("Cubic State Analyzer - Verification Suite")
    print("=" * 60)

    results = {}

    # Test 1: Delta sign convention
    results['delta_convention'] = test_delta_convention()

    # Test 2: Δ_c = -108Δ relationship
    results['delta_c_conversion'] = test_delta_c_conversion()

    # Test 3: Classification table
    results['classification'] = test_classification_table()

    # Test 4: Preset consistency
    results['presets'] = test_preset_consistency()

    # Test 5: Boundary cases
    results['boundary'] = test_boundary_cases()

    # Print reference table
    print_delta_table()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_pass = all(results.values())
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")

    if all_pass:
        print(f"\n  ALL TESTS PASSED ✓")
    else:
        print(f"\n  SOME TESTS FAILED ✗")

    # Diagnostics
    print(f"\n{'='*60}")
    print("DIAGNOSTICS: Delta Convention Note")
    print(f"{'='*60}")
    print("""
The tool uses Cardano discriminant:  Δ = (q/2)² + (p/3)³
The document primarily uses Cusp convention: Δ_c = -4p³ - 27q²
Relationship: Δ_c = -108 × Δ

Sign convention:
  Cardano Δ > 0  →  ONE real root (monotonic)     ← tool classification
  Cardano Δ < 0  →  THREE real roots (S-shaped)   ← tool classification
  Cusp Δ_c  > 0  →  THREE real roots (S-shaped)   ← doc classification
  Cusp Δ_c  < 0  →  ONE real root (monotonic)     ← doc classification

IMPORTANT: The sign conventions are OPPOSITE. Section 1.2 of the document
treats Δ and Δ_c as "equivalent" but they differ by -108×.
""")

    sys.exit(0 if all_pass else 1)
