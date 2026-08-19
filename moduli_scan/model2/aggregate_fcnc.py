#!/usr/bin/env python3
"""Model 2: aggregate the OFF-DIAGONAL (flavour-changing) axion couplings vs t3.

Companion to moduli_scan/model2/aggregate.py, which keeps only the diagonals. Mirrors model 1's
scan/aggregate_fcnc.py; the scan variable is the D-flat Kahler shape modulus t3 rather
than the complex-structure modulus psi1.

THE PHYSICS (draft main.tex:371,490, model 1's Fig. 3)
  The EATEN axions are not flavour-aligned, so their off-diagonal entries M_IJ mediate
  flavour-changing effects through the massive U(1) gauge bosons that ate them. The LIGHT
  axions are expected to be far more aligned. Model 2 lets that contrast be traced along a
  Kahler direction for the first time.

WHICH SECTORS CAN SAY ANYTHING. Only Q, u^c and e^c (bundle L5) are genuine 3x3. d^c and L
are U(3)-universal, ONE field each (moduli_scan/model2/stage2_matter.py:241-247), so their coupling
matrix is a multiple of the identity by symmetry and has no off-diagonal to measure -- not
"suppressed", absent by construction. They are reported as skipped rather than as zero.

A FREE SECOND BENCHMARK. The dilation identity forces the VOLUME-axion coupling to be
proportional to the identity in every sector at every point, so its off-diagonals must
VANISH -- an analytic zero, independent of the diagonal volume-axion benchmark, and testing a
different component of the same machinery. It is reported separately under "benchmark_offdiag". NOTE: the
diagonal version of this benchmark is biased because the reference (1,1)-forms are the
Fubini-Study representatives, not g-harmonic ones, so a nonzero result here is an error
bar on the pipeline and must
NOT be read as physics.

CP. M is Hermitian, so an off-diagonal entry carries a phase; its imaginary part is the
CP-violating proxy model 1 studied on the complex-psi1 plane. Model 2 fixes psi0=2,
psi1=1 -- the complex structure is NOT scanned -- so Im M here is expected to be a noise
floor, and is reported only as such.

Usage:
  python moduli_scan/model2/aggregate_fcnc.py [--runs-dir moduli_scan/model2/data/runs]
                                 [--out moduli_scan/model2/data/combined_fcnc.json]
                                 [--parity corrected]
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fcnc_lib import coupling_matrices, reconstruct_and_check  # noqa: E402
from labels import LABEL_CONVENTION, convention_of, sectors_in_mssm_labels  # noqa: E402
from runsel import (DEFAULT_MIN_EPOCHS, DEFAULT_MIN_POINTS, accept, find_runs,  # noqa: E402
                    load_rejected, parity_of)

ap = argparse.ArgumentParser()
ap.add_argument("--runs-dir", dest="runs_dir", default="moduli_scan/model2/data/runs")
ap.add_argument("--out", default="moduli_scan/model2/data/combined_fcnc.json")
ap.add_argument("--parity", default="corrected", choices=["corrected", "legacy"])
ap.add_argument("--quality", default="moduli_scan/model2/data/convergence.json")
ap.add_argument("--min-epochs", type=int, default=DEFAULT_MIN_EPOCHS)
ap.add_argument("--min-points", type=int, default=DEFAULT_MIN_POINTS)
args = ap.parse_args()

# Axes carried into the aggregate. light_raw* are the notebook's own light basis, kept
# comparable against the notebook's embedded outputs; they are NOT plotted (an
# arbitrary basis of the light plane can rotate between adjacent t3).
AX = ["volume", "shape", "eaten0", "eaten1", "light_raw0", "light_raw1"]

rejected = load_rejected(args.quality)
files = [p for p in find_runs(args.runs_dir) if parity_of(p) == args.parity]

# raw[sector][axion][t3] -> list over seeds of (max|off|, |M01|, |M02|, |M12|,
#                                               max|Im off|, mean|diag|)
raw = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
fields_of, skipped_1x1 = {}, {}
t3s, n_used, labels_seen, n_relabelled = set(), 0, set(), 0
worst_recon = 0.0

for p in files:
    with open(p) as f:
        d = json.load(f)
    if not accept(p, d, rejected, args.min_epochs, args.min_points):
        continue

    # The reconstruction is checked against the stored matrices on EVERY run rather than
    # in a separate pass, so a corrupted or half-written results.json cannot reach the
    # aggregate silently.
    _, w_full, _, _ = reconstruct_and_check(d)
    worst_recon = max(worst_recon, w_full)

    labels_seen.add(convention_of(d))
    sectors = d["sectors"]
    if convention_of(d) != LABEL_CONVENTION:
        sectors = sectors_in_mssm_labels(d, where=p, allclose=np.allclose)
        n_relabelled += 1

    _, M = coupling_matrices(d, sectors=sectors)
    t3 = float(d["t3"])
    t3s.add(t3)
    n_used += 1

    for sec, sd in sectors.items():
        n = len(sd["fields"])
        if n < 2:
            skipped_1x1[sec] = sd["fields"]          # U(3)-universal: no off-diagonal
            continue
        fields_of[sec] = sd["fields"]
        for ax in AX:
            Mx = M[sec][ax]
            off = Mx - np.diag(np.diag(Mx))
            raw[sec][ax][t3].append((
                float(np.abs(off).max()),
                float(abs(Mx[0, 1])), float(abs(Mx[0, 2])), float(abs(Mx[1, 2])),
                float(np.abs(np.imag(off)).max()),
                float(np.abs(np.real(np.diag(Mx))).mean()),
            ))

if not n_used:
    raise SystemExit(f"no usable runs under {args.runs_dir} (parity={args.parity})")

grid = sorted(t3s)
KEYS = ["max_abs", "abs01", "abs02", "abs12", "max_abs_im", "mean_abs_diag"]

out = {
    "t3_grid": grid,
    "n_runs": n_used,
    "parity": args.parity,
    "label_convention": LABEL_CONVENTION,
    "label_conventions_of_inputs": sorted(labels_seen),
    "n_runs_relabelled": n_relabelled,
    "worst_recon_vs_stored": worst_recon,
    "sectors_without_offdiagonals": skipped_1x1,
    "sectors": {},
}

for sec in sorted(raw):
    entry = {"fields": fields_of[sec], "offdiag": {}}
    for ax in AX:
        rec = {"t3": []}
        for k in KEYS:
            rec[k + "_mean"] = []
            rec[k + "_std"] = []
        rec["rel_mean"] = []                 # max|off| / mean|diag|, per run then averaged
        for t3 in grid:
            vals = raw[sec][ax].get(t3)
            if not vals:
                continue
            a = np.asarray(vals, float)      # (n_seeds, len(KEYS))
            rec["t3"].append(t3)
            for i, k in enumerate(KEYS):
                rec[k + "_mean"].append(float(a[:, i].mean()))
                rec[k + "_std"].append(float(a[:, i].std(ddof=1)) if len(a) > 1 else 0.0)
            rec["rel_mean"].append(float((a[:, 0] / a[:, 5]).mean()))
        entry["offdiag"][ax] = rec
    out["sectors"][sec] = entry

# The analytic zero: volume-axion off-diagonals, worst over sectors at each t3.
bench = {"t3": grid, "max_abs_mean": [], "rel_mean": []}
for t3 in grid:
    m = [out["sectors"][s]["offdiag"]["volume"] for s in out["sectors"]]
    idx = [r["t3"].index(t3) for r in m if t3 in r["t3"]]
    bench["max_abs_mean"].append(max(r["max_abs_mean"][i] for r, i in zip(m, idx)))
    bench["rel_mean"].append(max(r["rel_mean"][i] for r, i in zip(m, idx)))
out["benchmark_offdiag"] = bench

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(out, f, indent=2)

print(f"\nwrote {args.out}: {n_used} runs, t3 grid {grid}")
print(f"  reconstruction vs stored coupling_full: worst {worst_recon:.2e} (must be ~1e-15)")
print(f"  sectors with off-diagonals: {sorted(out['sectors'])}")
print(f"  U(3)-universal, no off-diagonal to measure: "
      f"{ {k: v for k, v in skipped_1x1.items()} }")

print("\nmax|off-diagonal| per axion, averaged over the grid (3x3 sectors):")
for sec in sorted(out["sectors"]):
    row = "  ".join(f"{ax}:{np.mean(out['sectors'][sec]['offdiag'][ax]['max_abs_mean']):.4f}"
                    for ax in AX[:4])
    print(f"  {sec:3s} {row}")

print("\nBENCHMARK -- volume-axion off-diagonals must VANISH (analytic zero):")
print(f"  {'t3':>6} {'max|off|':>10} {'/ mean|diag|':>13}")
for i, t3 in enumerate(bench["t3"]):
    print(f"  {t3:6.2f} {bench['max_abs_mean'][i]:10.5f} {bench['rel_mean'][i]:12.2%}")
