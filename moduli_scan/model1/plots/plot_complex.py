#!/usr/bin/env python3
"""
Plot the complex-psi1 scan. The CP-violating proxy is |Im M12|, the imaginary part of
the eaten-axion 1<->2 off-diagonal coupling. On the real axis it is provably exactly
zero, so the value measured there is the pure NN-seed noise FLOOR; the question is
whether Im(psi1)!=0 lifts |Im M12| above that floor (it does not).

Three panels:
  1. FCNC magnitude |M12| over (Re psi1, Im psi1)   -- the genuine, varying result.
  2. CP proxy |Im M12| over the same plane, colour scale ANCHORED AT 0 so the eye sees
     it is near-zero everywhere (NOT auto-scaled to the noise band, which would dress
     up noise as structure).
  3. Null-result histogram: distribution of |Im M12| for the real-axis points (Im=0,
     where it is provably zero -> the noise floor) vs the complex points (Im!=0),
     overlaid. Coincident distributions => no CP signal.

Also prints the symmetry-probe checks: M12(psibar) vs conj(M12(psi)) [CP],
and M12(-psi) vs M12(psi) [psi->-psi].

Usage: python moduli_scan/model1/plots/plot_complex.py [--combined moduli_scan/model1/data/combined_complex.json]
                                    [--runs-dir moduli_scan/model1/data/runs]
                                    [--outdir figures] [--no-latex]
"""
import argparse, glob, json, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--combined", default="moduli_scan/model1/data/combined_complex.json")
ap.add_argument("--runs-dir", default="moduli_scan/model1/data/runs")
ap.add_argument("--outdir", default="figures")
ap.add_argument("--no-latex", action="store_true")
args = ap.parse_args()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
if args.no_latex:
    matplotlib.rcParams["text.usetex"] = False

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)                                         # plots/ (scan_plot_style)
sys.path.insert(0, os.path.dirname(_here))  # moduli_scan/model1/ (fcnc_lib)
from fcnc_lib import coupling_matrices

data = json.load(open(args.combined))
pts = data["points"]
os.makedirs(args.outdir, exist_ok=True)

re = np.array([p["re"] for p in pts])
im = np.array([p["im"] for p in pts])
cp = np.array([p["cp_phase"] for p in pts])
fc = np.array([p["fcnc_mag"] for p in pts])

# --- per-seed |Im M12| (the noise floor needs the raw per-seed values, not the
#     seed-averaged combined.json) so the null histogram is honest. ---
AX = ["eaten0", "eaten1", "eaten2"]
floor_seed, complex_seed = [], []   # |Im M12| per individual run
for f in sorted(glob.glob(os.path.join(args.runs_dir, "*", "results.json"))):
    d = json.load(open(f))
    rev = float(d.get("psi1_re", d.get("psi1", 0.0)))
    imv = float(d.get("psi1_im", 0.0))
    _, M = coupling_matrices(d)
    cpv = 0.0
    for sec, sd in M.items():
        for ax in AX:
            mat = sd.get(ax)
            if mat is None or mat.shape[0] < 2:
                continue
            cpv = max(cpv, abs(complex(mat[0, 1]).imag))
    (floor_seed if imv == 0.0 else complex_seed).append(cpv)
floor_seed = np.array(floor_seed)
complex_seed = np.array(complex_seed)


# --- the two map panels are a clean RECTANGULAR heatmap over the complete
#     sub-grid that was sampled as a full grid: Re{0,1,2,4,6} x Im{0,0.5,1,2,4}
#     (25 cells, all filled). The real-axis baseline TAIL (Re 7-10, already in the
#     real-scan FCNC figure) and the 4 asymmetric symmetry probes are NOT a grid,
#     so they are excluded from the maps and used only for panel (c) + the probe
#     test below. Categorical (evenly-spaced) ticks render the uneven Re/Im
#     spacing as a proper rectangle.
GRID_RE = [0.0, 1.0, 2.0, 4.0, 6.0]
GRID_IM = [0.0, 0.5, 1.0, 2.0, 4.0]
val = {(p["re"], p["im"]): p for p in pts}


def grid_array(key):
    Z = np.full((len(GRID_IM), len(GRID_RE)), np.nan)
    for j, iv in enumerate(GRID_IM):
        for i, rv in enumerate(GRID_RE):
            p = val.get((rv, iv))
            if p is not None:
                Z[j, i] = p[key]
    return Z


def panel_map(ax, key, label, title, vmin=None, vmax=None):
    Z = grid_array(key)
    nx, ny = len(GRID_RE), len(GRID_IM)
    xedges, yedges = np.arange(nx + 1) - 0.5, np.arange(ny + 1) - 0.5
    im_h = ax.pcolormesh(xedges, yedges, np.ma.masked_invalid(Z),
                         cmap="Reds", vmin=vmin, vmax=vmax)
    ax.set_aspect("auto")
    ax.set_xticks(range(nx)); ax.set_xticklabels([f"{v:g}" for v in GRID_RE])
    ax.set_yticks(range(ny)); ax.set_yticklabels([f"{v:g}" for v in GRID_IM])
    ax.tick_params(top=False, right=False, which="both")  # heatmap: no mirrored ticks
    # annotate each cell with its value for readability (white on dark/high, black on light/low)
    lo = np.nanmin(Z) if vmin is None else vmin
    hi = np.nanmax(Z) if vmax is None else vmax
    for j in range(ny):
        for i in range(nx):
            if not np.isnan(Z[j, i]):
                rel = (Z[j, i] - lo) / (hi - lo + 1e-12)
                ax.text(i, j, f"{Z[j, i]:.3f}", ha="center", va="center",
                        fontsize=6.5, color=("w" if rel > 0.6 else "k"))
    ax.set_xlabel(r"$\mathrm{Re}\,\psi_1$")
    ax.set_ylabel(r"$\mathrm{Im}\,\psi_1$")
    ax.set_title(title, fontsize=10)
    cb = plt.colorbar(im_h, ax=ax); cb.set_label(label)


fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
panel_map(axes[0], "fcnc_mag", r"$|M_{12}|$ (FCNC)", r"(a)")
# CP panel: anchor colour scale at 0 so near-zero reads as near-zero.
panel_map(axes[1], "cp_phase", r"$|\mathrm{Im}\,M_{12}|$ ($CP$ proxy)",
          r"(b)", vmin=0.0)

# Null-result histogram.
hi = max(floor_seed.max(), complex_seed.max()) * 1.05
bins = np.linspace(0, hi, 14)
ax = axes[2]
ax.hist(floor_seed, bins=bins, density=True, histtype="step", lw=1.5, color="0.55",
        label=fr"real axis ($\mathrm{{Im}}\,\psi_1{{=}}0$): {floor_seed.mean():.4f}$\pm${floor_seed.std():.4f}")
ax.hist(complex_seed, bins=bins, density=True, histtype="step", lw=1.5, color="k",
        label=fr"complex ($\mathrm{{Im}}\,\psi_1{{\neq}}0$): {complex_seed.mean():.4f}$\pm${complex_seed.std():.4f}")
ax.axvline(floor_seed.mean(), color="0.55", lw=1.2, ls="--")
ax.axvline(complex_seed.mean(), color="k", lw=1.2, ls="--")
ax.set_xlabel(r"$|\mathrm{Im}\,M_{12}|$ (per run)")
ax.set_ylabel("density")
ax.set_title(r"(c)", fontsize=10)
ax.legend(fontsize=7, loc="upper right")

fig.tight_layout()
out = os.path.join(args.outdir, "complex_psi1_cp_map.pdf")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print("wrote:", out)
print(f"  real-axis |Im M12| floor : mean={floor_seed.mean():.4f} std={floor_seed.std():.4f} max={floor_seed.max():.4f}  (n={floor_seed.size})")
print(f"  complex   |Im M12|       : mean={complex_seed.mean():.4f} std={complex_seed.std():.4f} max={complex_seed.max():.4f}  (n={complex_seed.size})")


# --- symmetry-probe verification (printed) ---
def find(rev, imv, tol=1e-6):
    for p in pts:
        if abs(p["re"] - rev) < tol and abs(p["im"] - imv) < tol:
            return p
    return None

def m01_vec(p):
    out = {}
    for sec, axd in p["M01"].items():
        for ax, z in axd.items():
            out[(sec, ax)] = complex(*z)
    return out

print("\n=== symmetry probes (eaten 1<->2 coupling M12) ===")
checks = [("CP: M12(1-i) =? conj(M12(1+i))", (1, -1), (1, 1), True),
          ("psi->-psi: M12(-1+i) =? M12(1+i)", (-1, 1), (1, 1), False),
          ("CP+flip: M12(-1-i) =? conj(M12(1+i))", (-1, -1), (1, 1), True)]
for desc, a, b, conj in checks:
    pa, pb = find(*a), find(*b)
    if pa is None or pb is None:
        print(f"  {desc}: missing point(s) {a if pa is None else ''} {b if pb is None else ''}")
        continue
    va, vb = m01_vec(pa), m01_vec(pb)
    keys = set(va) & set(vb)
    if not keys:
        print(f"  {desc}: no common sectors"); continue
    d = max(abs(va[k] - (np.conj(vb[k]) if conj else vb[k])) for k in keys)
    print(f"  {desc}:  max|diff| = {d:.4f}  ({'HOLDS' if d < 0.05 else 'BROKEN'})")
