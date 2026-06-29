#!/usr/bin/env python3
"""
Reconstruct the full canonical axion-matter coupling matrices  M = -i*xi  from a
saved scan results.json (the stored complex G and Lambda), so we can study the
OFF-DIAGONAL (FCNC) and imaginary (CP) structure offline -- no re-run needed.

The reconstruction mirrors stage3_axion.py's canon()/light/eaten canonicalization.
It is validated by reconstruct_and_check(): the reconstructed diagonals must equal
the stored coupling_diag to ~1e-9.

Axion directions (light + 3 eaten) depend only on the axion kinetic metric at the
SUSY locus t=(1,1,1,1), which is psi1-independent, so they are identical across the
whole scan.
"""
import json
import numpy as np

# --- axion direction setup (stage3_axion.py:91-114, t=(1,1,1,1) SUSY locus) ---
# k-charge matrix of the 4 model-dependent axions under the 5 anomalous-U(1) factors.
_K_MATRIX = np.array([[-1, -1, 0, 1, 1],
                      [0, -3, 1, 1, 1],
                      [0, 2, -1, -1, 0],
                      [1, 2, 0, -1, -2]], dtype=float)


def _g_orth(vecs, metric, against):
    """Gram-Schmidt orthogonalisation against `against`, in `metric` (stage3:106-112)."""
    out = []
    for v in vecs:
        v = v - (v @ metric @ against) / (against @ metric @ against) * against
        for u in out:
            v = v - (v @ metric @ u) / (u @ metric @ u) * u
        if np.sqrt(abs(v @ metric @ v)) > 1e-8:
            out.append(v / np.sqrt(abs(v @ metric @ v)))
    return out


def axion_directions(g_axion):
    """Return (light, [eaten0, eaten1, eaten2]) 4-vectors (stage3:104-113)."""
    g_axion = np.asarray(g_axion, float)
    light = np.array([1., 1., 1., 1.])
    light = light / np.sqrt(light @ g_axion @ light)
    eaten = _g_orth(list(_K_MATRIX.T), g_axion, light)
    return light, eaten


def _inv_sqrt(Mx):
    ev, U = np.linalg.eigh(0.5 * (Mx + Mx.conj().T))
    return U @ np.diag(1.0 / np.sqrt(ev)) @ U.conj().T


def _canon(G, Lam, direction):
    """Canonical coupling M = -i*xi for one axion direction (stage3:118-120). Hermitian."""
    Gis = _inv_sqrt(G)
    return -1j * (Gis @ (1j * np.einsum('i,iIJ->IJ', direction, Lam)) @ Gis)


def _cplx(d):
    return np.asarray(d["re"]) + 1j * np.asarray(d["im"])


def coupling_matrices(run):
    """
    Given a parsed results.json dict, return:
      directions: dict axion_name -> 4-vector
      M: dict sector -> dict axion_name -> full complex (n x n) M=-i*xi matrix
    """
    g_axion = np.asarray(run["axion_kinetic_metric"], float)
    light, eaten = axion_directions(g_axion)
    dirs = {"light": light, **{f"eaten{a}": e for a, e in enumerate(eaten)}}
    out = {}
    for sec, sd in run["sectors"].items():
        G = _cplx(sd["G"])
        Lam = _cplx(sd["Lambda"])              # (4, n, n)
        out[sec] = {nm: _canon(G, Lam, d) for nm, d in dirs.items()}
    return dirs, out


def reconstruct_and_check(run, tol=1e-7):
    """
    Reconstruct M for every sector/axion and verify the diagonals equal the stored
    coupling_diag (the anchored quantity). Returns (M_dict, worst_abs_diff).
    """
    _, M = coupling_matrices(run)
    worst = 0.0
    for sec, sd in run["sectors"].items():
        for nm, stored in sd["coupling_diag"].items():
            recon = np.real(np.diag(M[sec][nm]))
            worst = max(worst, float(np.abs(recon - np.asarray(stored)).max()))
    if worst > tol:
        raise AssertionError(f"reconstruction diag mismatch {worst:.2e} > {tol:.0e}")
    return M, worst


if __name__ == "__main__":
    import glob
    files = sorted(glob.glob("results/moduli_scan/runs/psi1=*_seed=*/results.json"))
    worst_all = 0.0
    for f in files:
        _, w = reconstruct_and_check(json.load(open(f)))
        worst_all = max(worst_all, w)
    print(f"checked {len(files)} runs; worst |recon_diag - stored_diag| = {worst_all:.2e}  (PASS if < 1e-7)")
