#!/usr/bin/env python
"""Model 2: per-run training-convergence QC, and a quality gate for aggregation.

WHY. The sigma nets do not converge equally well across the t3 grid. Convergence degrades
systematically as t3 grows (the D-flat point becomes strongly anisotropic: at t3=14,
t=(180,1,14,12.86)), and a small number of nets diverge outright. A diverged net silently
corrupts its (t3, seed) run, and averaging that run into its t3 point corrupts the point.

`M_sigma` and `M_HYM` are the training losses normalised by the mean |source|, so
  ~0    = the harmonic correction is fully resolved
  ~1    = the net has learnt nothing (sigma == 0 is as good as it gets)
  >1    = the net has DIVERGED (worse than predicting zero)
A run is rejected here if any of its nets exceeds --max-loss.

Reads the SLURM logs by default (cheap). The authoritative record is `sigma_histories` /
`hym_histories` inside each run's model2_run_*.pkl, but those pickles are ~233 MB each and
carry the full training set, so --from-pkl is offered but not the default.

Writes moduli_scan/model2/data/convergence.json: {run_dir: {net: final_loss}, plus a verdict}.
moduli_scan/model2/aggregate.py consumes it via --quality to exclude rejected runs.

Covers BOTH the main scan array (scan.sbatch, logs m2scan-*) and the replacement arrays
(rerun.sbatch, logs m2rerun-*), so a re-run point is gated exactly like an original one.
This matters because aggregate.py filters on a DENY list: a run absent from
convergence.json is aggregated WITHOUT having passed the gate. Every (t3, seed) that
reaches the averages must therefore appear here.

The (t3, seed) identity is parsed from each log's own header line
  === scan|rerun|point t3=<t3> seed=<seed> host=... ===
NOT from the array task id: the rerun arrays carry their own (t3, seed) pair list, so the
task-id arithmetic that works for the main grid is wrong for them.

Run:
  python moduli_scan/model2/check_convergence.py [--job <jobid>[,<jobid>]] [--max-loss 1.0]
Exit 0 if every completed run passes, 1 if any run is rejected.
"""
import argparse
import glob
import json
import os
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--job", default="",
                help="comma-separated SLURM array job ids whose logs to read "
                     "(main scan arrays and replacement arrays alike)")
ap.add_argument("--logdir", default="logs/slurm")
ap.add_argument("--max-loss", type=float, default=1.0,
                help="reject a run if any net's final loss exceeds this (>1 = diverged)")
ap.add_argument("--warn-loss", type=float, default=0.6,
                help="warn (do not reject) above this")
ap.add_argument("--out", default="moduli_scan/model2/data/convergence.json")
args = ap.parse_args()

_SIG = re.compile(r'^sigma (\S+) \(bundle (\w+), eps=[+-]\d\):\n((?:  ep.*\n)+)', re.M)
_HYM = re.compile(r'^HYM (\S+):\n((?:  ep.*\n)+)', re.M)
_VAL = re.compile(r'M_(?:sigma|HYM)=([\d.eE+-]+)')
# run_point.sh's callers echo this header first; it is the authoritative (t3, seed) of the
# log, whereas the array task id means different things in scan.sbatch and rerun.sbatch.
# EVERY caller's tag must appear in this alternation and its log prefix in LOG_PREFIXES
# below -- a caller missing from either is INVISIBLE to this gate, and aggregate.py filters
# on a DENY list, so an ungated run would be averaged in silently. That is exactly the bug
# fixed on 2026-08-13 (rerun.sbatch was invisible here); do not reintroduce it by adding a
# submit script without adding it here.
_HDR = re.compile(r'^=== (?:scan|rerun|point) t3=(\S+) seed=(\d+) ', re.M)
LOG_PREFIXES = ("m2scan", "m2rerun", "m2pt")

jobs = [j.strip() for j in args.job.split(",") if j.strip()]
logfiles = []
for job in jobs:
    for prefix in LOG_PREFIXES:
        logfiles += glob.glob(os.path.join(args.logdir, f"{prefix}-{job}_*.out"))

runs = {}
for f in sorted(logfiles):
    m = re.search(r"_(\d+)\.out$", f)
    if not m:
        continue
    tid = int(m.group(1))
    txt = open(f).read()
    hdr = _HDR.search(txt)
    if not hdr:
        print(f"  WARNING no '=== scan/rerun/point t3=... seed=... ' header in {f} -- SKIPPED")
        continue
    t3, seed = float(hdr.group(1)), int(hdr.group(2))
    nets = {}
    for name, _bnd, body in _SIG.findall(txt):
        v = [float(x) for x in _VAL.findall(body)]
        if v:
            nets[f"sigma_{name}"] = v[-1]
    for name, body in _HYM.findall(txt):
        v = [float(x) for x in _VAL.findall(body)]
        if v:
            nets[f"hym_{name}"] = v[-1]
    if nets:
        runs[f"t3={t3}_seed={seed}"] = {"t3": t3, "seed": seed, "task": tid, "nets": nets}

if not runs:
    raise SystemExit(f"no training records found for job {args.job} in {args.logdir}")

print(f"{'run':22} {'#nets':>6} {'worst net':>12} {'worst':>10}  verdict")
rejected, warned = [], []
for key in sorted(runs, key=lambda k: (runs[k]["t3"], runs[k]["seed"])):
    r = runs[key]
    worst_net = max(r["nets"], key=lambda n: r["nets"][n])
    worst = r["nets"][worst_net]
    if worst > args.max_loss:
        verdict = "REJECT (diverged)"
        rejected.append(key)
    elif worst > args.warn_loss:
        verdict = "warn (poor)"
        warned.append(key)
    else:
        verdict = "ok"
    r["worst_net"], r["worst"], r["verdict"] = worst_net, worst, verdict
    print(f"{key:22} {len(r['nets']):6d} {worst_net:>12} {worst:10.4f}  {verdict}")

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, "w") as fh:
    json.dump({"job": args.job, "max_loss": args.max_loss, "warn_loss": args.warn_loss,
               "runs": runs, "rejected": rejected, "warned": warned}, fh, indent=2)

print(f"\n  runs with training records : {len(runs)}")
print(f"  rejected (any net > {args.max_loss}) : {len(rejected)}  {rejected}")
print(f"  warned  (any net > {args.warn_loss}) : {len(warned)}")
print(f"  wrote {args.out}")
if rejected:
    print("\n  NOTE a rejected (t3, seed) should be RE-RUN with a different seed rather than")
    print("  silently dropped -- dropping it narrows that t3 point's seed sample and biases")
    print("  its error bar. Aggregation is failure-tolerant but reports n per point.")
sys.exit(1 if rejected else 0)
