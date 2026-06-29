#!/usr/bin/env python3
"""
Complex-psi1 analogs of the real-scan diagonal-coupling figures
(moduli_scan/plots/plot_couplings.py).

The real scan plotted the diagonal canonical coupling M=-i*xi vs the 1-D axis psi1.
On the complex plane the abscissa becomes 2-D (Re psi1, Im psi1), so each line of the
real figures becomes a heatmap. We reconstruct the full M per run from the stored G,
Lambda via fcnc_lib.coupling_matrices (the same path plot_complex.py uses) and
seed-average the (real) diagonal at each grid point. NOTE: the diagonal of M is real
to machine precision even for Im psi1 != 0 (verified: max|Im(diag)| = 0 across runs),
so we plot the real diagonal directly.

Two figures, the direct analogs:
  1. complex_diag_light_map.pdf  -- 3x3 grid, one heatmap per sector, of the
     generation-averaged LIGHT-axion diagonal coupling over the plane.
     Should stay in the moduli-robust ~1.2 band everywhere (the headline check).
  2. complex_diag_<sec>.pdf  -- one file per sector. Rows = axions
     (light + 3 eaten), cols = generations; each cell a heatmap of that diagonal
     coupling over the plane. The eaten axions are strongly generation-dependent.

Grid = the complete rectangular sub-grid Re{0,1,2,4,6} x Im{0,0.5,1,2,4} (same as
plot_complex.py); the Re=7..10 real tail and the 4 off-grid symmetry probes are not a
grid and are omitted (they appear in the real-scan figures).

Usage: python moduli_scan/plots/plot_complex_diag.py [--runs-dir moduli_scan/data/runs]
                                         [--outdir figures]
                                         [--no-latex]
"""
import argparse, glob, json, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--runs-dir", default="moduli_scan/data/runs")
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
sys.path.insert(0, os.path.dirname(_here))  # moduli_scan/ (fcnc_lib)
from fcnc_lib import coupling_matrices
from scan_plot_style import sector_label, axion_label

GRID_RE = [0.0, 1.0, 2.0, 4.0, 6.0]
GRID_IM = [0.0, 0.5, 1.0, 2.0, 4.0]
AXIONS = ["light", "eaten0", "eaten1", "eaten2"]
SECTOR_ORDER = ["Q", "U", "E", "H", "Hd", "L_4_5", "D_4_5", "L_2_4", "D_2_4"]

# --- load every run, reconstruct M, keep the real diagonal per (sector, axion) ---
# accum[(re,im)][sec][axion] -> list over seeds of the real diagonal vector (len = ngen)
accum = {}
for f in sorted(glob.glob(os.path.join(args.runs_dir, "*", "results.json"))):
    d = json.load(open(f))
    rev = float(d.get("psi1_re", d.get("psi1", 0.0)))
    imv = float(d.get("psi1_im", 0.0))
    if rev not in GRID_RE or imv not in GRID_IM:
        continue  # tail / off-grid probe: not part of the rectangular map
    _, M = coupling_matrices(d)
    cell = accum.setdefault((rev, imv), {})
    for sec, sd in M.items():
        secd = cell.setdefault(sec, {})
        for ax in AXIONS:
            mat = sd.get(ax)
            if mat is None:
                continue
            secd.setdefault(ax, []).append(np.real(np.diag(mat)))

# seed-average: avg[(re,im)][sec][axion] -> mean real diagonal vector
avg = {}
for k, cell in accum.items():
    avg[k] = {sec: {ax: np.mean(vs, axis=0) for ax, vs in secd.items()}
              for sec, secd in cell.items()}

# number of generations/fields per sector (from any filled cell)
NGEN = {}
for cell in avg.values():
    for sec, secd in cell.items():
        if "light" in secd:
            NGEN[sec] = len(secd["light"])

os.makedirs(args.outdir, exist_ok=True)


def grid_array(sec, ax, reduce_gen=False, gen=None):
    """2-D array Z[j,i] over (Im=GRID_IM[j], Re=GRID_RE[i]) of the diagonal coupling.
    reduce_gen -> generation-average; else pick generation `gen`."""
    Z = np.full((len(GRID_IM), len(GRID_RE)), np.nan)
    for j, iv in enumerate(GRID_IM):
        for i, rv in enumerate(GRID_RE):
            vec = avg.get((rv, iv), {}).get(sec, {}).get(ax)
            if vec is None:
                continue
            Z[j, i] = float(np.mean(vec)) if reduce_gen else float(vec[gen])
    return Z


def draw_heatmap(ax, Z, cmap, vmin, vmax, annot=True):
    nx, ny = len(GRID_RE), len(GRID_IM)
    xedges, yedges = np.arange(nx + 1) - 0.5, np.arange(ny + 1) - 0.5
    h = ax.pcolormesh(xedges, yedges, np.ma.masked_invalid(Z), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("auto")
    ax.set_xticks(range(nx)); ax.set_xticklabels([f"{v:g}" for v in GRID_RE])
    ax.set_yticks(range(ny)); ax.set_yticklabels([f"{v:g}" for v in GRID_IM])
    ax.tick_params(top=False, right=False, which="both")
    if annot:
        for j in range(ny):
            for i in range(nx):
                if not np.isnan(Z[j, i]):
                    # contrast on sequential Reds: white text on dark/high, black on light/low
                    rel = (Z[j, i] - vmin) / (vmax - vmin + 1e-12)
                    col = "w" if rel > 0.6 else "k"
                    ax.text(i, j, f"{Z[j, i]:.2f}", ha="center", va="center",
                            fontsize=6, color=col)
    return h


# === Figure 1: gen-averaged light diagonal, per sector ===
secs = [s for s in SECTOR_ORDER if s in NGEN]
Zs = {s: grid_array(s, "light", reduce_gen=True) for s in secs}
allv = np.concatenate([Z[~np.isnan(Z)].ravel() for Z in Zs.values()])
vmin, vmax = float(allv.min()), float(allv.max())
fig, axes = plt.subplots(3, 3, figsize=(13, 10))
for k, sec in enumerate(secs):
    ax = axes.flat[k]
    h = draw_heatmap(ax, Zs[sec], "Reds", vmin, vmax)
    ax.set_title(sector_label(sec), fontsize=11)
    if k % 3 == 0:
        ax.set_ylabel(r"$\mathrm{Im}\,\psi_1$")
    if k // 3 == 2:
        ax.set_xlabel(r"$\mathrm{Re}\,\psi_1$")
for k in range(len(secs), 9):
    axes.flat[k].axis("off")
cb = fig.colorbar(h, ax=axes, shrink=0.85, pad=0.02)
cb.set_label(r"light-axion $M=-i\xi$ (gen-avg, diag.)")
out1 = os.path.join(args.outdir, "complex_diag_light_map.pdf")
fig.savefig(out1, bbox_inches="tight")
plt.close(fig)
print("wrote:", out1, f"  (light gen-avg range {vmin:.3f}..{vmax:.3f})")


# === Figure 2 (per sector): axions x generations heatmaps ===
def plot_sector(sec):
    ng = NGEN[sec]
    # one sequential Reds scale spanning the actual data range across all panels of this
    # sector (signed couplings; numeric annotations carry the sign)
    vals = np.concatenate([grid_array(sec, ax, gen=g)[~np.isnan(grid_array(sec, ax, gen=g))].ravel()
                           for ax in AXIONS for g in range(ng)])
    vlo, vhi = float(np.nanmin(vals)), float(np.nanmax(vals))
    fig, axes = plt.subplots(len(AXIONS), ng, figsize=(2.7 * ng + 1.5, 2.5 * len(AXIONS)),
                             squeeze=False)
    for r, ax_name in enumerate(AXIONS):
        for g in range(ng):
            a = axes[r][g]
            Z = grid_array(sec, ax_name, gen=g)
            h = draw_heatmap(a, Z, "Reds", vlo, vhi)
            if r == 0:
                # avoid nesting subscripts on already-subscripted labels (e.g. L_{4,5})
                a.set_title(rf"gen.\ {g+1}" if ng > 1 else sector_label(sec), fontsize=10)
            if g == 0:
                a.set_ylabel(axion_label(ax_name) + "\n" + r"$\mathrm{Im}\,\psi_1$", fontsize=8)
            if r == len(AXIONS) - 1:
                a.set_xlabel(r"$\mathrm{Re}\,\psi_1$")
    cb = fig.colorbar(h, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label(r"$M=-i\xi$ (diag.)")
    out = os.path.join(args.outdir, f"complex_diag_{sec}.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


for sec in secs:
    print("wrote:", plot_sector(sec))
