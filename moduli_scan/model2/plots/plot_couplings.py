#!/usr/bin/env python3
"""Model 2: plot axion-matter couplings vs the D-flat Kahler shape modulus t3.

This is the figure the model-2 study exists to produce: unlike model 1 -- whose SUSY locus
is the single point t=(1,1,1,1), leaving nothing to vary -- model 2 has a 2-parameter
D-flat cone, so the couplings genuinely depend on a ratio of Kahler parameters.

Figures written:
  volume_axion_benchmark.pdf  the volume-axion coupling vs t3 for every sector, against
                              the analytic 1/sqrt(6). This is a CORRECTNESS GATE, not a
                              result: the dilation + no-scale identities force it to be
                              flat and universal, so any t3-dependence or sector spread
                              is numerical error.
  shape_axion_vs_t3.pdf       the shape-axion coupling vs t3 per sector -- THE physics
                              result (the genuinely Kahler-dependent coupling).
  couplings_per_sector.pdf    per-sector panels, one line per axion.

Uses plotting_defaults.py (vendored in this folder).

Usage:
  python moduli_scan/model2/plot_couplings.py [--combined moduli_scan/model2/data/combined_parity=corrected.json]
                                 [--outdir figures]
                                 [--no-latex]
"""
import argparse
import json
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--combined", default="moduli_scan/model2/data/combined_parity=corrected.json")
ap.add_argument("--outdir", default="figures")
ap.add_argument("--no-latex", action="store_true")
ap.add_argument("--paper", action="store_true",
                help="restrict to PAPER_T3_WINDOW (collaborator guidance, see "
                     "scan_plot_style.py) and add PAPER_SUFFIX to every output name; "
                     "the full-range figures are kept alongside")
args = ap.parse_args()

# Shared plotting defaults (vendored in this folder).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plotting_defaults  # noqa: F401,E402
import matplotlib  # noqa: E402
if args.no_latex:
    matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan_plot_style import (SECTOR_ORDER, SECTOR_COLOR, SECTOR_MARKER, SECTOR_LABEL,
                             AXION_ORDER, AXION_COLOR, AXION_MARKER, AXION_LABEL,
                             VOLUME_AXION_CANONICAL, PAPER_T3_WINDOW, PAPER_SUFFIX,
                             apply_paper_fontscale, add_headroom)  # noqa: E402

if args.paper:
    apply_paper_fontscale()

YLABEL = r"canonical coupling $\xi$" if args.paper \
    else r"canonical coupling $\xi$ (diagonal)"

with open(args.combined) as f:
    C = json.load(f)
os.makedirs(args.outdir, exist_ok=True)
sectors = [s for s in SECTOR_ORDER if s in C["sectors"]]
# The locus parametrisation belongs in the figure captions, not the axis label.
XLABEL = r"Kähler shape modulus $t_3$"


def series(sec, axion):
    """(t3, mean, std) arrays for one sector/axion, averaged over the sector's fields."""
    d = C["sectors"][sec]["by_axion"].get(axion)
    if d is None:
        return None
    x = np.asarray(d["t3"], float)
    m = np.asarray(d["mean"], float)          # (n_t3, n_fields)
    s = np.asarray(d["std"], float)
    if args.paper:  # trusted window only (scan_plot_style.PAPER_T3_WINDOW)
        keep = (x >= PAPER_T3_WINDOW[0]) & (x <= PAPER_T3_WINDOW[1])
        x, m, s = x[keep], m[keep], s[keep]
    return x, m, s


def save(fig, name):
    if args.paper:
        name = name.replace(".pdf", PAPER_SUFFIX + ".pdf")
    p = os.path.join(args.outdir, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")


# ---------------------------------------------------------------- figure 1: benchmark
fig, ax = plt.subplots(figsize=(7.0, 4.4))
for sec in sectors:
    r = series(sec, "volume")
    if r is None:
        continue
    x, m, s = r
    for j in range(m.shape[1]):
        ax.errorbar(x, m[:, j], yerr=s[:, j], color=SECTOR_COLOR[sec],
                    marker=SECTOR_MARKER[sec], ms=4, lw=1.2, capsize=2, alpha=0.85,
                    label=SECTOR_LABEL[sec] if j == 0 else None)
ax.axhline(VOLUME_AXION_CANONICAL, color="k", ls="--", lw=1.4,
           label=r"analytic $1/\sqrt{6}$")
ax.set_xlabel(XLABEL)
ax.set_ylabel(YLABEL)
if args.paper:  # keep the legend clear of the curves and the analytic line
    add_headroom(ax, 0.38, "bottom")
    ax.legend(ncol=3, fontsize="small", frameon=False, loc="lower center")
else:
    ax.legend(ncol=3, fontsize="small", frameon=False)
save(fig, "volume_axion_benchmark.pdf")

# ------------------------------------------------------------- figure 2: the result
fig, ax = plt.subplots(figsize=(7.0, 4.4))
for sec in sectors:
    r = series(sec, "shape")
    if r is None:
        continue
    x, m, s = r
    for j in range(m.shape[1]):
        ax.errorbar(x, m[:, j], yerr=s[:, j], color=SECTOR_COLOR[sec],
                    marker=SECTOR_MARKER[sec], ms=4, lw=1.2, capsize=2, alpha=0.85,
                    label=SECTOR_LABEL[sec] if j == 0 else None)
ax.set_xlabel(XLABEL)
ax.set_ylabel(YLABEL)
if args.paper:
    add_headroom(ax, 0.28, "top")
    ax.legend(ncol=3, fontsize="small", frameon=False, loc="upper right")
else:
    ax.legend(ncol=3, fontsize="small", frameon=False)
save(fig, "shape_axion_vs_t3.pdf")

# --------------------------------------------------------- figure 3: per-sector panels
n = len(sectors)
if n:
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow),
                             sharex=True, squeeze=False)
    for k, sec in enumerate(sectors):
        a = axes[k // ncol][k % ncol]
        for axion in AXION_ORDER:
            r = series(sec, axion)
            if r is None:
                continue
            x, m, s = r
            for j in range(m.shape[1]):
                a.errorbar(x, m[:, j], yerr=s[:, j], color=AXION_COLOR[axion],
                           marker=AXION_MARKER[axion], ms=3.5, lw=1.0, capsize=2,
                           alpha=0.85, label=AXION_LABEL[axion] if j == 0 else None)
        a.axhline(VOLUME_AXION_CANONICAL, color="k", ls=":", lw=1.0)
        a.set_title(SECTOR_LABEL[sec])
        if k % ncol == 0:
            a.set_ylabel(r"$\xi$")
        if k == 0:
            a.legend(fontsize="x-small", frameon=False)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.supxlabel(XLABEL)
    save(fig, "couplings_per_sector.pdf")

# ------------------------------------- figure 4: per-GENERATION panels, line per axion
# Model-2 analogue of model 1's Fig. 2 (scan/plot_couplings.py's coupling_vs_psi1_<sec>).
# Model 1's headline there was that the eaten axions are strongly generation-dependent
# (-0.78 to +0.60 within Q) while the light axion is flat. Model 2 puts the same question
# to a Kahler direction. Only the 10 (Q, u^c, e^c) can answer it: d^c and L are
# U(3)-universal 1x1 (moduli_scan/model2/stage2_matter.py:241-247), so they have one "generation" by
# construction and are skipped rather than drawn as a degenerate panel.
for sec in sectors:
    fields = C["sectors"][sec]["fields"]
    if len(fields) < 2:
        continue
    fig, axes = plt.subplots(1, len(fields), figsize=(4.0 * len(fields), 3.6),
                             sharex=True, sharey=True, squeeze=False)
    for j, fld in enumerate(fields):
        a = axes[0][j]
        for axion in AXION_ORDER:
            r = series(sec, axion)
            if r is None:
                continue
            x, m, s = r
            a.errorbar(x, m[:, j], yerr=s[:, j], color=AXION_COLOR[axion],
                       marker=AXION_MARKER[axion], ms=4, lw=1.2, capsize=2,
                       label=AXION_LABEL[axion])
        a.axhline(VOLUME_AXION_CANONICAL, color="k", ls=":", lw=1.0)
        a.axhline(0.0, color="0.6", lw=0.8)
        a.set_title(f"${fld}$")
        # every subpanel carries its own short t3 label
        a.set_xlabel(r"$t_3$")
        if j == 0:
            a.set_ylabel(YLABEL)
            a.legend(fontsize="x-small", frameon=False,
                     loc="upper left" if args.paper else "best")
    if args.paper:  # sharey=True, so one call opens headroom in every panel
        add_headroom(axes[0][0], 0.50, "top")
    save(fig, f"couplings_per_generation_{sec}.pdf")

grid = [t for t in C["t3_grid"]
        if not args.paper or PAPER_T3_WINDOW[0] <= t <= PAPER_T3_WINDOW[1]]
print(f"done: {C['n_runs']} runs, t3 grid {grid}"
      + (f" (windowed to {PAPER_T3_WINDOW} for the paper)" if args.paper else ""))
