#!/usr/bin/env python3
"""
Aggregate all per-(psi1,seed) results.json into one combined.json:
for every sector / axion / field, the mean and std over seeds at each psi1.

Usage:  python moduli_scan/model1/aggregate.py [--runs-dir moduli_scan/model1/data/runs] [--out moduli_scan/model1/data/combined.json]
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--runs-dir", default="moduli_scan/model1/data/runs")
ap.add_argument("--out", default="moduli_scan/model1/data/combined.json")
args = ap.parse_args()

files = sorted(glob.glob(os.path.join(args.runs_dir, "*", "results.json")))
if not files:
    raise SystemExit(f"no results.json under {args.runs_dir}")

# raw[(sector, axion)][psi1] = list over seeds of [per-field coupling]
raw = defaultdict(lambda: defaultdict(list))
forms_of = {}
n_eaten_seen = set()
n_used = 0
for f in files:
    d = json.load(open(f))
    if float(d.get("psi1_im", 0.0)) != 0.0:
        continue                      # real-axis aggregation: skip complex-psi1 runs
    n_used += 1
    psi1 = float(d["psi1"])
    n_eaten_seen.add(int(d.get("n_eaten", 0)))
    for sec, sd in d["sectors"].items():
        forms_of[sec] = sd["forms"]
        for axion, vals in sd["coupling_diag"].items():
            raw[(sec, axion)][psi1].append(np.asarray(vals, float))

psi1_grid = sorted({p for (_, _), m in raw.items() for p in m})
print(f"real-axis runs used: {n_used}  psi1 grid: {psi1_grid}  n_eaten seen: {sorted(n_eaten_seen)}")

sectors = defaultdict(lambda: {"forms": None, "by_axion": {}})
for (sec, axion), per_psi in raw.items():
    sectors[sec]["forms"] = forms_of[sec]
    mean, std, ns, grid = [], [], [], []
    for p in psi1_grid:
        if p not in per_psi:
            continue
        stack = np.stack(per_psi[p], 0)          # (n_seeds, n_fields)
        grid.append(p)
        mean.append(stack.mean(0).tolist())
        std.append(stack.std(0, ddof=1).tolist() if stack.shape[0] > 1
                   else [0.0]*stack.shape[1])
        ns.append(stack.shape[0])
    sectors[sec]["by_axion"][axion] = {"psi1": grid, "mean": mean, "std": std, "n": ns}

out = {"psi1_grid": psi1_grid, "n_runs": n_used, "sectors": sectors}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(out, open(args.out, "w"), indent=2)
print(f"wrote {args.out}  ({len(sectors)} sectors)")
for sec, sd in sectors.items():
    axions = list(sd["by_axion"])
    ns = sd["by_axion"][axions[0]]["n"] if axions else []
    print(f"  {sec:7s} forms={sd['forms']}  axions={axions}  seeds/psi1={ns}")
