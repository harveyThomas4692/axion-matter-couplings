#!/usr/bin/env python3
"""Model 2: rebuild the full canonical coupling matrices M = -i*xi from a saved
results.json, so the OFF-DIAGONAL (flavour-changing) structure can be studied offline --
no re-run, no trained networks, nothing that lives only on a cluster.

Mirrors moduli_scan/model1/fcnc_lib.py (model 1). Two things differ, both structural:

  * MODEL 2 ALREADY STORES THE FULL MATRICES. stage3 writes `coupling_full` for every
    axion direction (moduli_scan/model2/stage3_axion.py:199-202), not just the diagonals, so this
    module is not strictly needed to READ the off-diagonals. It exists anyway because a
    reconstruction from the stored G and Lambda is an INDEPENDENT path to the same
    number: reconstruct_and_check() compares the two and is the reason anything
    downstream may be trusted. (Model 1 had no `coupling_full`, so there the
    reconstruction was the only path.)

  * THE AXION BASIS IS t3-DEPENDENT. Model 1's directions come from the axion kinetic
    metric at the single SUSY locus t=(1,1,1,1) and are the same for every run. Model 2's
    D-flat point moves with t3, so g_axion, the eaten span and the volume/shape split are
    all per-run. They are rebuilt here from the stored `axion_kinetic_metric` and the two
    anomalous charge vectors, and cross-checked against the `directions` stage3 stored.

WHAT THE DIRECTIONS MEAN (moduli_scan/model2/stage3_axion.py:122-149)
  volume  t/|t|_g -- the overall breathing mode t -> lambda t.
  shape   the remaining light direction, g-orthogonal to volume; equivalently (Euler's
          identity for V homogeneous of degree 3 gives (g t)_i = V_i/(4V), so
          t.g.s = 0 <=> V_i s^i = 0) the VOLUME-PRESERVING direction.
  eaten0/1  the two combinations eaten by the anomalous U(1)s.
The volume direction is never eaten -- k_a^i g_ij t^j = mu(L_a)/(8V), which vanishes on
the D-flat locus (moduli_scan/model2/test_dflat_locus.py CHECK 6).

FLAVOUR STRUCTURE. Only Q, u^c, e^c (bundle L5) are genuine 3x3. d^c and L are
U(3)-universal, one field each (moduli_scan/model2/stage2_matter.py:241-247), so their coupling matrix
is a multiple of the identity BY SYMMETRY and has no off-diagonal to measure. Any FCNC
statement here is about the 10 only.

Usage (self-check over every stored run):
    python moduli_scan/model2/fcnc_lib.py [--runs-dir moduli_scan/model2/data/runs]
"""
import json
import math

import numpy as np

# --- normalisation, identical to model 1 (moduli_scan/model1/fcnc_lib.py) ---------------------------
# The bare coupling xi = e_hat^i G^{-1/2}(i Lambda_i) G^{-1/2} normalises the COMPLEXIFIED
# Kahler modulus, not the real axion; the canonical real axion needs 1/(2 sqrt 2).
# Fixed upstream in commit 1d9e7be of this repository. Cross-check: the analytic
# single-modulus value 2/sqrt(3) divided by 2 sqrt(2) is exactly 1/sqrt(6) = 0.4082483.
AXION_NORM = 2.0 * math.sqrt(2.0)
LAMBDA_VERTEX_FACTOR = 0.5              # folded into Lambda itself (GS/one-loop 1/(4 pi))
REAL_SCALAR_DIVISOR = math.sqrt(2.0)    # applied in stage3's canon()
assert abs(LAMBDA_VERTEX_FACTOR / REAL_SCALAR_DIVISOR - 1.0 / AXION_NORM) < 1e-15

NORMALISATION_KEY = "coupling_normalisation"
CANONICAL_MARKER = "canonical"

# The two independent anomalous-U(1) charge directions (moduli_scan/model2/stage3_axion.py:86).
K1 = np.array([-1.0, -1.0, 1.0, 1.0])
K4 = np.array([1.0, 2.0, -3.0, -1.0])


def is_canonical(run):
    """True iff this results.json was written in the canonical convention."""
    return run.get(NORMALISATION_KEY) == CANONICAL_MARKER


def residual_divisors(run):
    """Normalisation still MISSING from a stored run, applied by division.

    Returns (div_GLambda, div_full), where div_GLambda divides a reconstruction built
    from the stored G and Lambda, and div_full divides the stored coupling_diag /
    coupling_full. Same semantics and same single-source-of-truth role as
    scan/fcnc_lib.residual_divisors(); see that module's docstring for why these are
    divisors rather than multipliers.

    Every model-2 run is canonical -- stage3 has stamped the marker since the first
    scan2 run -- but the legacy branch is kept so a bare file cannot be silently
    misread if one is ever produced.
    """
    if is_canonical(run):
        return REAL_SCALAR_DIVISOR, 1.0
    return AXION_NORM, AXION_NORM


def _g_orth(vecs, metric, prev=None):
    """Gram-Schmidt in `metric` (verbatim from moduli_scan/model2/stage3_axion.py:77-83)."""
    out = list(prev) if prev else []
    for v in vecs:
        v = np.asarray(v, float).copy()
        for u in out:
            v = v - (v @ metric @ u) / (u @ metric @ u) * u
        if np.sqrt(abs(v @ metric @ v)) > 1e-9:
            out.append(v / np.sqrt(v @ metric @ v))
    return out


def axion_directions(run):
    """Rebuild {volume, shape, eaten0, eaten1} from the stored kinetic metric and t.

    Mirrors moduli_scan/model2/stage3_axion.py:86-92 and 122-143. Rebuilt rather than read back so
    that check_directions() has two independent objects to compare.
    """
    g = np.asarray(run["axion_kinetic_metric"], float)
    t = np.asarray(run["t_locus"], float)
    eaten = _g_orth([K1, K4], g)
    light = _g_orth([np.eye(4)[i] for i in range(4)], g, prev=eaten)[len(eaten):]
    vol = t / np.sqrt(t @ g @ t)
    shape = None
    for e in light:
        c = e - (e @ g @ vol) * vol
        n = float(np.sqrt(abs(c @ g @ c)))
        if n > 1e-8:
            shape = c / n
            break
    if shape is None:
        raise AssertionError("no shape direction g-orthogonal to volume")
    if shape[0] < 0:                       # stage3's sign convention (t3-monotone)
        shape = -shape
    dirs = {"volume": vol, "shape": shape}
    # light_raw* is the notebook's own Gram-Schmidt light basis -- an ARBITRARY basis of
    # the same 2-plane, kept because it is what model2_axion_coupling_all.ipynb
    # prints ("TWO DIAGONALISED LIGHT axions", cell 9), so it is the axis on which the
    # numbers can be compared against the notebook's embedded outputs. It is not used for figures: it
    # can rotate or swap between adjacent t3, which would make a coupling curve jump.
    dirs.update({f"light_raw{a}": np.asarray(e, float) for a, e in enumerate(light)})
    dirs.update({f"eaten{a}": np.asarray(e, float) for a, e in enumerate(eaten)})
    return dirs


def check_directions(run, tol=1e-10):
    """Worst |rebuilt - stored| over the directions both sides define."""
    rebuilt = axion_directions(run)
    worst = 0.0
    for nm, v in rebuilt.items():
        stored = np.asarray(run["directions"][nm], float)
        worst = max(worst, float(np.abs(v - stored).max()))
    if worst > tol:
        raise AssertionError(f"axion directions disagree with the stored ones: {worst:.2e}")
    return worst


def _inv_sqrt(Mx):
    ev, U = np.linalg.eigh(0.5 * (Mx + Mx.conj().T))
    return U @ np.diag(1.0 / np.sqrt(ev)) @ U.conj().T


def _canon(G, Lam, direction, divisor):
    """M = -i*xi for one axion direction (moduli_scan/model2/stage3_axion.py:94-98). Hermitian."""
    Gis = _inv_sqrt(G)
    bare = -1j * (Gis @ (1j * np.einsum('i,iIJ->IJ', direction, Lam)) @ Gis)
    return bare / divisor


def _cplx(d):
    return np.asarray(d["re"]) + 1j * np.asarray(d["im"])


def coupling_matrices(run, sectors=None):
    """(directions, M) with M[sector][axion] the full complex matrix, reconstructed.

    `sectors` lets a caller pass an already-relabelled sectors dict (moduli_scan/model2/labels.py) so
    the reconstruction and the naming cannot drift apart.
    """
    dirs = axion_directions(run)
    div, _ = residual_divisors(run)
    src = run["sectors"] if sectors is None else sectors
    out = {}
    for sec, sd in src.items():
        G = _cplx(sd["G"])
        Lam = _cplx(sd["Lambda"])                      # (4, n, n)
        out[sec] = {nm: _canon(G, Lam, d, div) for nm, d in dirs.items()}
    return dirs, out


def reconstruct_and_check(run, tol=1e-10):
    """Reconstruct M and check it against BOTH stored objects.

    Returns (M, worst_vs_full, worst_vs_diag, worst_directions). The `coupling_full`
    comparison is the strong one -- it covers the off-diagonals, which are the whole
    point of this module and which nothing else in the repo has ever checked.
    """
    wdir = check_directions(run)
    _, M = coupling_matrices(run)
    _, div_full = residual_divisors(run)
    worst_full = worst_diag = 0.0
    for sec, sd in run["sectors"].items():
        for nm, stored in sd["coupling_full"].items():
            if nm not in M[sec]:
                continue                               # light_raw*: not rebuilt here
            ref = _cplx(stored) / div_full
            worst_full = max(worst_full, float(np.abs(M[sec][nm] - ref).max()))
        for nm, stored in sd["coupling_diag"].items():
            if nm not in M[sec]:
                continue
            ref = np.asarray(stored, float) / div_full
            worst_diag = max(worst_diag,
                             float(np.abs(np.real(np.diag(M[sec][nm])) - ref).max()))
    if max(worst_full, worst_diag) > tol:
        raise AssertionError(
            f"reconstruction mismatch: full {worst_full:.2e}, diag {worst_diag:.2e} > {tol:.0e}")
    return M, worst_full, worst_diag, wdir


if __name__ == "__main__":
    import argparse
    import glob
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", dest="runs_dir", default="moduli_scan/model2/data/runs")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.runs_dir, "*", "results.json")))
    if not files:
        raise SystemExit(f"no results.json under {a.runs_dir}")
    wf = wd = wx = 0.0
    for f in files:
        with open(f) as fh:
            run = json.load(fh)
        _, a_full, a_diag, a_dir = reconstruct_and_check(run)
        wf, wd, wx = max(wf, a_full), max(wd, a_diag), max(wx, a_dir)
    print(f"checked {len(files)} runs")
    print(f"  worst |recon - stored coupling_full| = {wf:.2e}")
    print(f"  worst |recon - stored coupling_diag| = {wd:.2e}")
    print(f"  worst |rebuilt - stored direction|   = {wx:.2e}")
    print("  -> PASS" if max(wf, wd, wx) < 1e-10 else "  -> FAIL")
