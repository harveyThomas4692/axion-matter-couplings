#!/usr/bin/env python3
"""Model 2 Phase A: generate one point cloud at the D-flat Kahler point t(t3).

Standalone and NON-JAX on purpose, mirroring moduli_scan/model1/gen_cloud.py: the numpy point generator
forks, and doing that in a process that has already started JAX threads deadlocks.

Mirrors model2_train.ipynb cells t3-t4 EXACTLY (D-flat solve, polynomial,
generator, prepare_dataset/prepare_basis), so the cloud is identical to what stage 1 would
produce on its own.

UNLIKE MODEL 1, the cloud is per-t3, not shared across the scan: kmoduli enters
basis.pickle as KMODULI and KAPPA, and KAPPA feeds sigma_loss directly. Verified
in the pre-flight that the sampled POINTS and the stored WEIGHTS are kmoduli-independent
(the weights implement the kmoduli-independent |Omega|^2/CY measure) while KAPPA is not:
it moves 0.1172 -> 1.0905 between t=(1,1,1,1) and t=(4,1,3,4/3). Never reuse a cloud
across t3 without recomputing kappa.
"""
import argparse
import itertools
import os

import numpy as np

_ap = argparse.ArgumentParser()
_ap.add_argument("--t3", type=float, required=True, help="D-flat shape modulus, t3 > 2")
_ap.add_argument("--n-points", dest="n_points", type=int, default=500000)
_ap.add_argument("--cloud-dir", dest="cloud_dir", required=True)
args = _ap.parse_args()
if args.t3 <= 2.0:
    raise SystemExit(f"t3 must exceed 2 for t1 > 0 (Kahler cone); got {args.t3}")

from cymetric.pointgen.pointgen_cicy import CICYPointGenerator  # noqa: E402

# Complex structure is FIXED for model 2 (model2_train.ipynb cell t3).
psi0, psi1 = 2.0, 1.0
ambient = np.array([1, 1, 1, 1])
combinations = list(itertools.product([0, 1], repeat=4))

# D-flat locus, verified independently in moduli_scan/model2/test_dflat_locus.py.
t2 = 1.0
t3 = args.t3
t1 = (t3 - 2.0) * (t3 + 1.0)
t4 = t1 / t3
kmoduli = np.array([t1, t2, t3, t4], dtype=float)
assert np.all(kmoduli > 0), f"NOT in Kahler cone: {kmoduli}"

# Defining polynomial, model2_train.ipynb cell t4 (verbatim).
_mons, _cfs = [], []
for c in combinations:
    _mons.append([2 * (1 - c[0]), 2 * c[0], 2 * (1 - c[1]), 2 * c[1],
                  2 * (1 - c[2]), 2 * c[2], 2 * (1 - c[3]), 2 * c[3]])
    _cfs.append(1. if sum(c) % 2 == 0 else psi0)
_mons.append([1, 1, 1, 1, 1, 1, 1, 1])
_cfs.append(psi1)
monomials = np.array(_mons, dtype=np.int64)
coefficients = np.array(_cfs)

print(f"[SCAN] t3={t3}  kmoduli={kmoduli}  n={args.n_points}  dir={args.cloud_dir}")

basis_path = os.path.join(args.cloud_dir, "basis.pickle")
ds_path = os.path.join(args.cloud_dir, "dataset.npz")
pg = CICYPointGenerator([monomials], [coefficients], kmoduli, ambient)

# Size-aware idempotence guard (as moduli_scan/model1/gen_cloud.py): skip only if a big-enough cloud
# is already there. prepare_dataset uses a 90/10 train/val split.
if os.path.exists(basis_path) and os.path.exists(ds_path):
    _n_have = int(np.load(ds_path)["X_train"].shape[0])
    _n_want = int(0.9 * args.n_points)
    if _n_have >= 0.8 * _n_want:
        print(f"[SCAN] cloud already present ({_n_have} train pts >= 0.8*{_n_want}); skipping")
        raise SystemExit(0)
    print(f"[SCAN] existing cloud too small ({_n_have} < 0.8*{_n_want}); regenerating")

os.makedirs(args.cloud_dir, exist_ok=True)
kappa = pg.prepare_dataset(args.n_points, args.cloud_dir)
pg.prepare_basis(args.cloud_dir, kappa)
print(f"[SCAN] done. kappa={kappa}")
