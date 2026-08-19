#!/usr/bin/env python3
"""
Generate the shared per-psi1 point cloud (one per psi1; reused by all seeds).

Builds the polynomial, point generator and prepared dataset/basis, so the cloud is
identical to what stage1 would produce on its own. Stage1 then finds basis.pickle
and skips generation. Running this as a separate phase avoids multiple seeds racing
to generate the same cloud.
"""
import argparse, os, itertools
import numpy as np
from cymetric.pointgen.pointgen_cicy import CICYPointGenerator  # numpy generator (no Mathematica)

ap = argparse.ArgumentParser()
ap.add_argument("--psi1", type=complex, required=True)  # complex-capable (e.g. "1+1j")
ap.add_argument("--n-points", dest="n_points", type=int, default=500000)
ap.add_argument("--cloud-dir", dest="cloud_dir", required=True)
args = ap.parse_args()

psi0 = 2.0                                  # fixed reference value (psi0 = 2)
psi1 = args.psi1                            # scan variable
kmoduli = np.array([1., 1., 1., 1.])        # SUSY locus
ambient = np.array([1, 1, 1, 1])
combinations = list(itertools.product([0, 1], repeat=4))

# Defining polynomial
_mons, _cfs = [], []
for c in combinations:
    _mons.append([2*(1-c[0]), 2*c[0], 2*(1-c[1]), 2*c[1],
                  2*(1-c[2]), 2*c[2], 2*(1-c[3]), 2*c[3]])
    _cfs.append(1. if sum(c) % 2 == 0 else psi0)
_mons.append([1, 1, 1, 1, 1, 1, 1, 1])
_cfs.append(psi1)
monomials = np.array(_mons, dtype=np.int64)
coefficients = np.array(_cfs)

basis_path = os.path.join(args.cloud_dir, "basis.pickle")
ds_path = os.path.join(args.cloud_dir, "dataset.npz")
# Skip ONLY if a cloud of the RIGHT size already exists. Guards against silently
# reusing a reduced-resolution (e.g. smoke-test) cloud: keying on basis.pickle
# presence alone would reuse a 3k-point cloud where 500k was intended. [audit C1]
_present = False
if os.path.exists(basis_path) and os.path.exists(ds_path):
    _n_have = int(np.load(ds_path)["X_train"].shape[0])
    _n_want = int(0.9 * args.n_points)            # prepare_dataset uses a 90/10 train/val split
    if _n_have >= 0.8 * _n_want:                   # tolerant: same-order-of-magnitude => trust it
        print(f"[gen_cloud] cloud present at {args.cloud_dir} ({_n_have} train pts) — skip.")
        _present = True
    else:
        print(f"[gen_cloud] existing cloud too small ({_n_have} train pts, want ~{_n_want}) "
              f"-> REGENERATING at {args.cloud_dir}")
if not _present:
    os.makedirs(args.cloud_dir, exist_ok=True)
    pg = CICYPointGenerator([monomials], [coefficients], kmoduli, ambient)
    print(f"[gen_cloud] psi1={psi1}  generating {args.n_points} points -> {args.cloud_dir}")
    kappa = pg.prepare_dataset(args.n_points, args.cloud_dir)
    pg.prepare_basis(args.cloud_dir, kappa)
    print(f"[gen_cloud] done. kappa={kappa:.6f}")
