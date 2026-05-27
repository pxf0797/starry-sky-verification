#!/usr/bin/env python3
"""
统一验证入口 — 一键运行所有或选定实验
使用方式:
    python3 run_all.py --all --trials 200          # 全部13个实验
    python3 run_all.py --priority P0 --trials 300  # 仅P0级
    python3 run_all.py --original --trials 200     # 仅原始5个实验
    python3 run_all.py --list                      # 列出所有实验
    python3 run_all.py --all --trials 200 --json results.json  # 输出JSON
"""

import argparse, sys, os, json, subprocess, time

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = {
    # 原始实验
    "1_optimal_order":        {"script": "verify_optimal_order.py",        "priority": "—",   "layer": 1, "desc": "三次最优阶数"},
    "2_dimension_gap":        {"script": "verify_dimension_gap.py",        "priority": "—",   "layer": 1, "desc": "维度鸿沟"},
    "3_snr_decay":            {"script": "verify_snr_decay.py",            "priority": "—",   "layer": 1, "desc": "SNR衰减"},
    "4_half_life":            {"script": "verify_half_life.py",            "priority": "—",   "layer": 2, "desc": "τ最优值"},
    "5_dynamic_threshold":    {"script": "verify_dynamic_threshold.py",    "priority": "—",   "layer": 2, "desc": "动态vs静态阈值"},
    # P0 新增
    "N1_nonpolynomial":       {"script": "verify_nonpolynomial_robustness.py", "priority": "P0", "layer": "new", "desc": "非多项式鲁棒性"},
    "N2_garch":               {"script": "verify_garch_noise.py",              "priority": "P0", "layer": "new", "desc": "GARCH复验"},
    "N3_fattail":             {"script": "verify_fattail_threshold.py",        "priority": "P0", "layer": "new", "desc": "厚尾重校准"},
    "N4_benchmark":           {"script": "verify_benchmark_compare.py",        "priority": "P0", "layer": "new", "desc": "基准对比"},
    # P1 新增
    "N5_walkforward":         {"script": "verify_walkforward.py",              "priority": "P1", "layer": "new", "desc": "Walk-Forward"},
    "N6_param_heatmap":       {"script": "verify_param_heatmap.py",            "priority": "P1", "layer": "new", "desc": "参数热力图"},
    "N7_limit_order":         {"script": "verify_limit_order_ab.py",           "priority": "P1", "layer": "new", "desc": "限价单A/B"},
    "N8_cross_asset":         {"script": "verify_cross_asset_tau.py",          "priority": "P1", "layer": "new", "desc": "多品种τ"},
}

def run_experiment(name, info, trials):
    script_path = os.path.join(TOOLS_DIR, info["script"])
    if not os.path.exists(script_path):
        return {"name": name, "status": "MISSING", "error": f"Script not found: {script_path}"}

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script_path, "--trials", str(trials)],
            capture_output=True, text=True, timeout=120,
            cwd=TOOLS_DIR
        )
        elapsed = time.time() - start
        # Parse verdict from output
        output = result.stdout
        verdict = "ok" if "✅" in output else "warn" if "⚠️" in output or "🟡" in output else "bad" if "🔴" in output else "unknown"

        return {
            "name": name, "desc": info["desc"], "priority": info["priority"],
            "status": "OK" if result.returncode == 0 else "FAIL",
            "verdict": verdict, "elapsed": round(elapsed, 1),
            "output": output[-500:] if len(output) > 500 else output,
            "stderr": result.stderr[-200:] if result.stderr else ""
        }
    except subprocess.TimeoutExpired:
        return {"name": name, "status": "TIMEOUT", "elapsed": 120}
    except Exception as e:
        return {"name": name, "status": "ERROR", "error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="星空策略统一验证入口")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="运行全部13个实验")
    g.add_argument("--priority", choices=["P0","P1"], help="按优先级运行")
    g.add_argument("--original", action="store_true", help="仅运行原始5个实验")
    g.add_argument("--list", action="store_true", help="列出所有实验")
    parser.add_argument("--trials", type=int, default=200, help="试验次数 (default: 200)")
    parser.add_argument("--json", help="输出JSON结果到文件")
    args = parser.parse_args()

    if args.list:
        print(f"{'ID':<22s} {'优先级':<6s} {'层次':<6s} {'描述'}")
        print("-" * 55)
        for name, info in EXPERIMENTS.items():
            print(f"  {name:<20s}  {info['priority']:<6s}  L{str(info['layer']):<5s}  {info['desc']}")
        return

    # Select experiments
    if args.all:
        selected = EXPERIMENTS
    elif args.priority:
        selected = {k: v for k, v in EXPERIMENTS.items() if v["priority"] == args.priority}
    elif args.original:
        selected = {k: v for k, v in EXPERIMENTS.items() if v["priority"] == "—"}

    print("=" * 72)
    print(f"  星空策略统一验证 | {len(selected)} 个实验 | trials={args.trials}")
    print("=" * 72)

    results = []
    passed, failed, warned = 0, 0, 0
    for i, (name, info) in enumerate(selected.items(), 1):
        print(f"  [{i}/{len(selected)}] {name}: {info['desc']} ... ", end="", flush=True)
        r = run_experiment(name, info, args.trials)
        results.append(r)
        status_icon = "✅" if r["status"] == "OK" and r.get("verdict") == "ok" else \
                      "⚠️" if r["status"] == "OK" and r.get("verdict") in ("warn","unknown") else \
                      "🔴" if r["status"] == "FAIL" or r.get("verdict") == "bad" else "⏳"
        print(f"{status_icon} {r['elapsed']}s")
        if r["status"] == "OK" and r.get("verdict") == "ok": passed += 1
        elif r["status"] == "FAIL" or r.get("verdict") == "bad": failed += 1
        else: warned += 1

    print()
    print(f"  结果: {passed} 通过 | {warned} 需关注 | {failed} 失败")
    print(f"  总计: {len(selected)} 实验, {sum(r.get('elapsed',0) for r in results):.0f}s")

    if args.json:
        with open(os.path.join(TOOLS_DIR, args.json), 'w') as f:
            json.dump({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "config": {"trials": args.trials},
                       "summary": {"passed": passed, "warned": warned, "failed": failed},
                       "results": results}, f, indent=2, ensure_ascii=False)
        print(f"  结果已保存到: {args.json}")

if __name__ == "__main__":
    main()
