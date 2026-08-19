#!/usr/bin/env python
"""Model 2: validate every generated point cloud against the exact CY volume.

WHY. Model 2 generates one cloud PER t3 (kmoduli enters basis.pickle as KMODULI and
KAPPA, and KAPPA feeds sigma_loss directly), and t3 ranges over a grid on which the CY
volume spans four orders of magnitude. A single wrong cloud would silently corrupt one
scan point.

THE CHECK. kappa is the normalisation relating the Fubini-Study volume to the
intersection-number volume, so it must be PROPORTIONAL to
    V_CY = (1/6) d_ijk t^i t^j t^k,   d_ijk = 2 for distinct i,j,k (tetraquadric)
across the whole grid. A constant kappa/V_CY over a 10^4 range in V is therefore a strong
simultaneous validation of every cloud: an error in any single one shows up as an outlier
in the ratio, and an error common to all would have to conspire to stay proportional.
The residual spread is Monte-Carlo noise at the cloud's n.

This reads the stored basis.pickle rather than the SLURM logs, so it also confirms what is
actually ON DISK and will be consumed by stage 1.

Cross-check of the same constant from an independent route: the pre-flight measured
kappa(t=(4,1,3,4/3)) / kappa(t=(1,1,1,1)) = 9.305 against the exact volume ratio
448/48 = 9.3333 (0.3% at n=3000).

Run: python moduli_scan/model2/check_clouds.py [--clouds-dir results/moduli_scan/model2/clouds] [--tol 0.01]
Exit 0 if every cloud's kappa/V_CY agrees with the grid mean within --tol (relative).
"""
import argparse
import glob
import os
import pickle
import re
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--clouds-dir", default="results/moduli_scan/model2/clouds")
ap.add_argument("--tol", type=float, default=0.01, help="max relative deviation of kappa/V_CY")
args = ap.parse_args()

D3 = np.zeros((4, 4, 4))
for _i in range(4):
    for _j in range(4):
        for _k in range(4):
            if len({_i, _j, _k}) == 3:
                D3[_i, _j, _k] = 2.0


def t_of(t3):
    """The D-flat point; verified in moduli_scan/model2/test_dflat_locus.py."""
    t1 = (t3 - 2.0) * (t3 + 1.0)
    return np.array([t1, 1.0, t3, t1 / t3])


def volume(t):
    return float((1.0 / 6.0) * np.einsum('ijk,i,j,k->', D3, t, t, t))


rows = []
for d in sorted(glob.glob(os.path.join(args.clouds_dir, "t3=*"))):
    m = re.search(r"t3=([\d.]+)$", d)
    bp = os.path.join(d, "basis.pickle")
    ds = os.path.join(d, "dataset.npz")
    if not (m and os.path.exists(bp)):
        print(f"  !! {d}: no basis.pickle")
        continue
    t3 = float(m.group(1))
    with open(bp, "rb") as f:
        B = pickle.load(f)
    kap = float(np.asarray(B["KAPPA"]).reshape(-1)[0])
    km = np.asarray(B["KMODULI"], dtype=float)
    t = t_of(t3)
    n = int(np.load(ds)["X_train"].shape[0]) if os.path.exists(ds) else -1
    km_ok = bool(np.allclose(km, t, rtol=1e-6))
    rows.append((t3, volume(t), kap, km_ok, n))

if not rows:
    raise SystemExit(f"no clouds under {args.clouds_dir}")

rows.sort(key=lambda r: r[0])
ratios = np.array([k / v for _, v, k, _, _ in rows])
mean = ratios.mean()
dev = np.abs(ratios - mean) / mean

print(f"{'t3':>7} {'n_train':>9} {'V_CY':>13} {'kappa':>13} {'kappa/V':>11} {'dev':>9} {'kmod':>6}")
bad = []
for (t3, v, k, km_ok, n), r, dv in zip(rows, ratios, dev):
    flag = "" if (dv <= args.tol and km_ok) else "  <-- FAIL"
    if flag:
        bad.append(t3)
    print(f"{t3:7.2f} {n:9d} {v:13.4f} {k:13.5f} {r:11.6f} {dv*100:8.3f}% "
          f"{'ok' if km_ok else 'BAD':>6}{flag}")

print(f"\n  clouds: {len(rows)}   kappa/V_CY mean {mean:.6f}  std {ratios.std(ddof=1):.2e}")
_vmin = min(r[1] for r in rows); _vmax = max(r[1] for r in rows)
print(f"  V_CY spans {_vmin:.3f} .. {_vmax:.1f} (factor {_vmax/_vmin:.3g})")
print(f"  worst relative deviation {dev.max()*100:.3f}%  (tolerance {args.tol*100:.1f}%)")
if bad:
    print(f"\nRESULT: FAIL for t3 = {bad}")
    sys.exit(1)
print("\nRESULT: PASS -- kappa proportional to V_CY across the grid; "
      "every basis.pickle carries its own KMODULI.")
