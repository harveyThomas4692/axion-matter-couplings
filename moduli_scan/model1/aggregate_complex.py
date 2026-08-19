#!/usr/bin/env python3
"""
Aggregate the complex-psi1 scan: for each complex psi1 = re + i*im, reconstruct the
canonical couplings M=-i*xi (moduli_scan/model1/fcnc_lib.py) and extract the CP-violating signal,
i.e. the IMAGINARY part of the eaten-axion off-diagonal (1<->2) entries -- which is
~0 on the real axis (CP-protected) and is what may turn on for Im(psi1)!=0.

Also reads the real-axis runs (Im=0; legacy results.json have no psi1_im -> treated
as 0) so the complex map includes the real baseline.

Output (moduli_scan/model1/data/combined_complex.json):
  points[(re,im)] = { cp_phase, fcnc_mag, n_seeds, by_sector{...}, M01{sector:{axion:[re,im]}} }
The per-point complex M01 is kept so plot_complex.py can verify the CP (psi->psibar)
and psi->-psi symmetry probes.

Usage: python moduli_scan/model1/aggregate_complex.py [--runs-dir moduli_scan/model1/data/runs]
                                        [--out moduli_scan/model1/data/combined_complex.json]
"""
import argparse, glob, json, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fcnc_lib import coupling_matrices

ap = argparse.ArgumentParser()
ap.add_argument("--runs-dir", default="moduli_scan/model1/data/runs")
ap.add_argument("--out", default="moduli_scan/model1/data/combined_complex.json")
args = ap.parse_args()

AX = ["eaten0", "eaten1", "eaten2"]    # CP / FCNC lives in the eaten (heavy) axions

files = sorted(glob.glob(os.path.join(args.runs_dir, "*", "results.json")))
# raw[(re,im)] = list over seeds of dict(cp, fcnc, M01{sector:{ax:complex}})
raw = defaultdict(list)
for f in files:
    d = json.load(open(f))
    re = float(d.get("psi1_re", d.get("psi1", 0.0)))
    im = float(d.get("psi1_im", 0.0))
    _, M = coupling_matrices(d)
    cp = fcnc = 0.0
    m01 = {}
    for sec, sd in M.items():
        for ax in AX:
            mat = sd.get(ax)
            if mat is None or mat.shape[0] < 2:
                continue
            z = complex(mat[0, 1])
            m01.setdefault(sec, {})[ax] = [z.real, z.imag]
            cp = max(cp, abs(z.imag))
            fcnc = max(fcnc, abs(z))
    raw[(re, im)].append(dict(cp=cp, fcnc=fcnc, m01=m01))

points = []
for (re, im), seeds in sorted(raw.items()):
    cp = float(np.mean([s["cp"] for s in seeds]))
    fcnc = float(np.mean([s["fcnc"] for s in seeds]))
    # average the complex M01 over seeds, per sector/axion
    m01_avg = {}
    for sec in seeds[0]["m01"]:
        m01_avg[sec] = {}
        for ax in seeds[0]["m01"][sec]:
            zs = [complex(*s["m01"][sec][ax]) for s in seeds if sec in s["m01"] and ax in s["m01"][sec]]
            z = np.mean(zs)
            m01_avg[sec][ax] = [float(z.real), float(z.imag)]
    points.append(dict(re=re, im=im, cp_phase=cp, fcnc_mag=fcnc,
                       n_seeds=len(seeds), M01=m01_avg))

out = {"n_points": len(points), "points": points}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(out, open(args.out, "w"), indent=2)
print(f"wrote {args.out}  ({len(points)} distinct psi1 points)")
print(f"{'Re':>5} {'Im':>5} {'CP=|Im M12|':>12} {'FCNC=|M12|':>11} {'seeds':>6}")
for p in sorted(points, key=lambda p: (p["im"], p["re"])):
    print(f"{p['re']:5.1f} {p['im']:5.1f} {p['cp_phase']:12.4f} {p['fcnc_mag']:11.4f} {p['n_seeds']:6d}")
