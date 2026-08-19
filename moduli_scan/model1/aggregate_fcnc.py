#!/usr/bin/env python3
"""
FCNC aggregation: from each run's reconstructed canonical M = -i*xi matrices
(moduli_scan/model1/fcnc_lib.py), extract the OFF-DIAGONAL (flavour-changing) entries and run the
third-generation-decoupling consistency check, aggregated over seeds vs psi1.

Physics:
  * The LIGHT (volume) axion is flavour-aligned -> its off-diagonals ~ 0 (no FCNC).
  * The EATEN (heavy) axions are NOT aligned -> non-zero 1<->2 off-diagonal = FCNC;
    the imaginary part is a CP-violating phase.
  * The 3rd generation (L2(x)L4 origin) decouples from gens 1-2 (L4(x)L5 origin):
    the cross-block entries vanish. NOTE: in this implementation that decoupling is
    ENFORCED by the block-zeroing in stage3 (sector_integrals, lines 50-52), so the
    check below confirms the implementation is consistent with the bundle selection
    rule rather than discovering it emergently.

Within-block FCNC entry = M[0,1] (gens 1<->2). Cross-block (3rd-gen) entries for the
3x3 sectors = M[0,2], M[1,2] (expected ~0).

Usage: python moduli_scan/model1/aggregate_fcnc.py [--runs-dir moduli_scan/model1/data/runs]
                                     [--out moduli_scan/model1/data/combined_fcnc.json]
"""
import argparse, glob, json, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fcnc_lib import coupling_matrices

ap = argparse.ArgumentParser()
ap.add_argument("--runs-dir", default="moduli_scan/model1/data/runs")
ap.add_argument("--out", default="moduli_scan/model1/data/combined_fcnc.json")
args = ap.parse_args()

# locate this script's dir so `from fcnc_lib` works regardless of CWD
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

files = sorted(glob.glob(os.path.join(args.runs_dir, "*", "results.json")))
if not files:
    raise SystemExit(f"no results.json under {args.runs_dir}")

AX = ["light", "eaten0", "eaten1", "eaten2"]

# raw[sector][axion][psi1] = list over seeds of complex M[0,1]
raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
forms_of = {}
# decoupling: cross[sector][psi1] = list over seeds of max|cross-block entry| over all axions
cross = defaultdict(lambda: defaultdict(list))
worst_decoupling = 0.0
worst_loc = None

n_used = 0
for f in files:
    run = json.load(open(f))
    if float(run.get("psi1_im", 0.0)) != 0.0:
        continue                       # real-axis aggregation: skip complex-psi1 runs
    n_used += 1
    psi1 = float(run["psi1"])
    _, M = coupling_matrices(run)
    for sec, sd in run["sectors"].items():
        forms = sd["forms"]; n = len(forms)
        forms_of[sec] = forms
        if n < 2:
            continue                       # no off-diagonals in a 1x1 sector
        for ax in AX:
            raw[sec][ax][psi1].append(complex(M[sec][ax][0, 1]))
        # 3rd-generation decoupling (only the 3x3 sectors have a cross block: col/row 2)
        if n == 3:
            cmax = 0.0
            for ax in AX:
                Mx = M[sec][ax]
                cmax = max(cmax, abs(Mx[0, 2]), abs(Mx[1, 2]))
            cross[sec][psi1].append(cmax)
            if cmax > worst_decoupling:
                worst_decoupling = cmax; worst_loc = f"{sec} psi1={psi1}"

psi1_grid = sorted({p for sec in raw for ax in raw[sec] for p in raw[sec][ax]})


def stat(arr):
    a = np.asarray(arr)
    return a.mean(), (a.std(ddof=1) if a.size > 1 else 0.0)


out = {"psi1_grid": psi1_grid, "n_runs": n_used,
       "decoupling": {"worst_abs_cross_entry": worst_decoupling, "worst_at": worst_loc,
                      "by_sector": {}},
       "sectors": {}}

for sec in sorted(raw):
    out["sectors"][sec] = {"forms": forms_of[sec], "off01": {}}
    for ax in AX:
        grid, absmean, absstd, remean, immean, imstd = [], [], [], [], [], []
        for p in psi1_grid:
            if p not in raw[sec][ax]:
                continue
            vals = np.asarray(raw[sec][ax][p])      # complex, over seeds
            am, asd = stat(np.abs(vals))
            rm, _ = stat(vals.real)
            im, isd = stat(vals.imag)
            grid.append(p); absmean.append(am); absstd.append(asd)
            remean.append(rm); immean.append(im); imstd.append(isd)
        out["sectors"][sec]["off01"][ax] = {
            "psi1": grid, "abs_mean": absmean, "abs_std": absstd,
            "re_mean": remean, "im_mean": immean, "im_std": imstd}

for sec in sorted(cross):
    by = {"psi1": [], "max_cross_mean": []}
    for p in psi1_grid:
        if p not in cross[sec]:
            continue
        by["psi1"].append(p)
        by["max_cross_mean"].append(float(np.mean(cross[sec][p])))
    out["decoupling"]["by_sector"][sec] = by

os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(out, open(args.out, "w"), indent=2)

print(f"wrote {args.out}")
print(f"sectors with off-diagonals: {list(out['sectors'])}")
print(f"\n3rd-gen DECOUPLING CHECK (3x3 sectors): worst |cross-block entry| over the "
      f"whole scan = {worst_decoupling:.2e}  at {worst_loc}")
print("  -> " + ("PASS (consistent with bundle selection rule / block-zeroing)"
                  if worst_decoupling < 1e-9 else
                  f"NOTE: nonzero at {worst_decoupling:.1e} (expected ~0 by construction)"))
print("\nFCNC |M[0,1]| (gen 1<->2), gen-avg over psi1, per axion:")
for sec in out["sectors"]:
    row = []
    for ax in AX:
        am = out["sectors"][sec]["off01"][ax]["abs_mean"]
        row.append(f"{ax}:{np.mean(am):.3f}")
    print(f"  {sec:7s} {'  '.join(row)}")
