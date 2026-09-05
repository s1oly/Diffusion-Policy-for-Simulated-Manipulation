"""
Summarize the 12-cell ablation from the eval JSONs into a table + a success-vs-N curve.

    python peg_insertion/analysis/analyze_ablation.py
    python peg_insertion/analysis/analyze_ablation.py --results-dir eval_results --out peg_insertion/analysis/ablation.png

Reads every eval_results/<controller>_<N>.json (written by eval.py), so it runs the same
whether the evals were done on the M2 or on the GCP L4. Each JSON carries success_rate,
success_se (binomial SE over n_rollouts), controller, n_demos, and the per_rollout list
(seed, spawn_yaw, success, steps).

Prints:
  1. the 3 (controller) x 4 (N) success-rate table, rate +/- SE per cell;
  2. a success-vs-N curve per controller (PNG), if matplotlib is available;
  3. a spawn-yaw breakdown -- eval yaw is uniform over 360deg while the training set was
     yaw-biased (only successful demos kept), so success falls off away from yaw~0. This
     pools all rollouts per controller into |yaw| terciles to make that gap visible.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

# Controller keys in increasing-degradation order; DISPLAY just prettifies for output.
CONTROLLER_ORDER = ["pd", "hybrid", "pure_noise"]
DISPLAY = {"pd": "pd (clean)", "hybrid": "hybrid (PD+noise)", "pure_noise": "pure noise"}


def load_results(results_dir):
    """Return {controller: {n_demos: record}} plus the sorted list of N seen."""
    cells = defaultdict(dict)
    ns = set()
    files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    if not files:
        raise SystemExit(f"no eval JSONs found in {results_dir!r} -- run/fetch the evals first")
    for f in files:
        d = json.load(open(f))
        ctrl = d["controller"]
        n = d["n_demos"]
        if n is None:
            print(f"  (skip {os.path.basename(f)}: no n_demos in cfg label)")
            continue
        cells[ctrl][n] = d
        ns.add(n)
    return cells, sorted(ns)


def print_table(cells, ns):
    print("\n=== success rate (rate +/- SE, 50 rollouts/cell) ===\n")
    W = 18
    header = f"{'controller':<{W}s}" + "".join(f"| N={n:<4d}      " for n in ns)
    print(header)
    print("-" * len(header))
    for ctrl in CONTROLLER_ORDER:
        if ctrl not in cells:
            continue
        row = f"{DISPLAY.get(ctrl, ctrl):<{W}s}"
        for n in ns:
            rec = cells[ctrl].get(n)
            if rec is None:
                row += "| --          "
            else:
                row += f"| {rec['success_rate']*100:4.0f}% +/-{rec['success_se']*100:3.0f}% "
        print(row)
    # also flag the best cell overall
    best = max((rec for c in cells.values() for rec in c.values()),
               key=lambda r: r["success_rate"])
    print(f"\nbest cell: {DISPLAY.get(best['controller'], best['controller'])} "
          f"@ N={best['n_demos']}  {best['success_rate']:.1%} +/- {best['success_se']:.1%}")


def print_yaw_breakdown(cells):
    print("\n=== success by |spawn yaw| tercile (pooled over N, per controller) ===")
    print("(eval yaw is uniform over +/-180deg; training kept only successful, yaw-biased demos)\n")
    edges = [0, 60, 120, 180]
    labels = ["|yaw|<60", "60-120", "120-180"]
    print(f"{'controller':<18s}" + "".join(f"| {l:<10s}" for l in labels))
    print("-" * 54)
    for ctrl in CONTROLLER_ORDER:
        if ctrl not in cells:
            continue
        buckets = [[0, 0] for _ in labels]   # [n_success, n_total]
        for rec in cells[ctrl].values():
            for r in rec["per_rollout"]:
                b = min(np.searchsorted(edges, abs(r["spawn_yaw"]), side="right") - 1, len(labels) - 1)
                buckets[b][1] += 1
                buckets[b][0] += int(r["success"])
        row = f"{DISPLAY.get(ctrl, ctrl):<18s}"
        for succ, tot in buckets:
            row += f"| {(succ/tot*100 if tot else 0):4.0f}% ({tot:>3d}) " if tot else "|  --        "
        print(row)


def make_plot(cells, ns, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed -- skipping the curve; `pip install matplotlib` to enable)")
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for ctrl in CONTROLLER_ORDER:
        if ctrl not in cells:
            continue
        xs = [n for n in ns if n in cells[ctrl]]
        ys = [cells[ctrl][n]["success_rate"] for n in xs]
        es = [cells[ctrl][n]["success_se"] for n in xs]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=DISPLAY.get(ctrl, ctrl))
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("# demos (N)")
    ax.set_ylabel("eval success rate")
    ax.set_ylim(0, 1)
    ax.set_title("Diffusion-policy success vs. demo count, by demonstrator")
    ax.grid(True, alpha=0.3)
    ax.legend(title="controller")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nwrote curve -> {out}")


def main():
    HERE = os.path.dirname(os.path.abspath(__file__))
    default_results = os.path.normpath(os.path.join(HERE, "..", "..", "eval_results"))
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=default_results,
                   help=f"dir of eval JSONs (default: {default_results})")
    p.add_argument("--out", default=os.path.join(HERE, "ablation.png"),
                   help="PNG path for the success-vs-N curve")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    cells, ns = load_results(args.results_dir)
    print(f"loaded {sum(len(v) for v in cells.values())} cells from {args.results_dir}")
    print_table(cells, ns)
    print_yaw_breakdown(cells)
    if not args.no_plot:
        make_plot(cells, ns, args.out)


if __name__ == "__main__":
    main()
