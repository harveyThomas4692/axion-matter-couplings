#!/usr/bin/env python3
"""Model 2: plot the OFF-DIAGONAL (flavour-changing) axion couplings vs t3.

Model-2 analogue of model 1's Fig. 3 (scan/plot_fcnc.py), from
moduli_scan/model2/aggregate_fcnc.py's combined_fcnc.json.

Figures written:
  fcnc_vs_t3_summary.pdf    max|M_IJ| over the off-diagonal entries, one line per axion,
                            averaged over the three 3x3 sectors (they agree closely --
                            same bundle L5). The eaten axions carry the flavour-changing
                            couplings; the light ones sit on the measured error floor.
  fcnc_vs_t3_<sector>.pdf   the same per sector, with seed error bars.
  fcnc_floor_vs_t3.pdf      THE CAVEAT FIGURE. The volume axion's off-diagonals are an
                            ANALYTIC ZERO (dilation identity => xi_vol proportional to the
                            identity), so what is measured there is the pipeline's error,
                            in the very same observable. Plotted as a shaded floor under
                            the eaten-axion signal, as a fraction of the diagonal.

WHY THE FLOOR FIGURE EXISTS. Reading "the light axion is FCNC-suppressed" off these curves
would be circular: it is suppressed by an exact identity, not by dynamics. The honest
question is whether the EATEN off-diagonals stand above that floor, so the floor is drawn
rather than described.

Only Q, u^c and e^c appear: d^c and L are U(3)-universal 1x1 (moduli_scan/model2/stage2_matter.py:241-247)
and have no off-diagonal.

Uses plotting_defaults.py (vendored in this folder).

Usage:
  python moduli_scan/model2/plot_fcnc.py [--combined moduli_scan/model2/data/combined_fcnc.json]
                            [--outdir figures] [--no-latex]
"""
import argparse
import json
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--combined", default="moduli_scan/model2/data/combined_fcnc.json")
ap.add_argument("--outdir", default="figures")
ap.add_argument("--no-latex", action="store_true")
ap.add_argument("--paper", action="store_true",
                help="restrict to PAPER_T3_WINDOW (collaborator guidance, see "
                     "scan_plot_style.py) and add PAPER_SUFFIX to every output name; "
                     "the full-range figures are kept alongside")
args = ap.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plotting_defaults  # noqa: F401,E402
import matplotlib  # noqa: E402
if args.no_latex:
    matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan_plot_style import (AXION_COLOR, AXION_LABEL, AXION_LS, AXION_MARKER,  # noqa: E402
                             SECTOR_COLOR, SECTOR_LABEL, SECTOR_MARKER, SECTOR_ORDER,
                             PAPER_T3_WINDOW, PAPER_SUFFIX,
                             apply_paper_fontscale, add_headroom)

if args.paper:
    apply_paper_fontscale()

with open(args.combined) as f:
    C = json.load(f)
os.makedirs(args.outdir, exist_ok=True)

AX = ["eaten0", "eaten1", "volume", "shape"]
# The locus parametrisation belongs in the figure captions, not the axis label.
XLABEL = r"Kähler shape modulus $t_3$"
sectors = [s for s in SECTOR_ORDER if s in C["sectors"]]
written = []


def save(fig, name):
    if args.paper:
        name = name.replace(".pdf", PAPER_SUFFIX + ".pdf")
    p = os.path.join(args.outdir, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    written.append(p)


def rec(sec, ax):
    r = C["sectors"][sec]["offdiag"][ax]
    if not args.paper:
        return r
    # trusted window only (scan_plot_style.PAPER_T3_WINDOW); every per-t3 array in the
    # record is aligned with r["t3"], so mask them together
    t3 = np.asarray(r["t3"], float)
    keep = (t3 >= PAPER_T3_WINDOW[0]) & (t3 <= PAPER_T3_WINDOW[1])
    return {k: (np.asarray(v, float)[keep] if isinstance(v, list) and len(v) == len(t3)
                else v) for k, v in r.items()}


# ------------------------------------------------- figure A: per-sector, line per axion
for sec in sectors:
    fig, a = plt.subplots(figsize=(7.0, 4.4))
    for axn in AX:
        r = rec(sec, axn)
        a.errorbar(r["t3"], r["max_abs_mean"], yerr=r["max_abs_std"],
                   color=AXION_COLOR[axn], marker=AXION_MARKER[axn], ls=AXION_LS[axn],
                   ms=4, lw=1.3, capsize=2, label=AXION_LABEL[axn])
    a.set_xlabel(XLABEL)
    a.set_ylabel(r"$\max_{I\neq J}|M_{IJ}|$")
    if args.paper:
        add_headroom(a, 0.30, "top")
        a.legend(ncol=2, fontsize="small", frameon=False, loc="upper left")
    else:
        a.legend(ncol=2, fontsize="small", frameon=False)
    save(fig, f"fcnc_vs_t3_{sec}.pdf")

# ------------------------------------------------------- figure B: summary over sectors
fig, a = plt.subplots(figsize=(7.0, 4.4))
for axn in AX:
    grid = rec(sectors[0], axn)["t3"]
    stack = np.array([rec(s, axn)["max_abs_mean"] for s in sectors])
    a.errorbar(grid, stack.mean(0), yerr=stack.std(0),
               color=AXION_COLOR[axn], marker=AXION_MARKER[axn], ls=AXION_LS[axn],
               ms=4, lw=1.4, capsize=2, label=AXION_LABEL[axn])
a.set_xlabel(XLABEL)
a.set_ylabel(r"$\max_{I\neq J}|M_{IJ}|$" if args.paper else
                 r"$\max_{I\neq J}|M_{IJ}|$  (mean over the $\mathbf{10}$ sectors)")
if args.paper:
    add_headroom(a, 0.30, "top")
    a.legend(ncol=2, fontsize="small", frameon=False, loc="upper left")
else:
    a.legend(ncol=2, fontsize="small", frameon=False)
save(fig, "fcnc_vs_t3_summary.pdf")

# --------------------------------------------- figure C: the signal against its own floor
# Everything relative to ONE common denominator -- the volume axion's mean |diagonal|
# (the canonical unit; analytically 1/sqrt(6) and sector-universal, scan_plot_style
# VOLUME_AXION_CANONICAL). Normalising each axion by ITS OWN diagonal (the first cut of
# this figure) is meaningless for the shape axion, whose diagonal crosses zero at
# t3 = 4.11: the ratio spiked to ~110% there as a 0/0 artefact -- an artefact of
# dividing by a vanishing coupling, not flavour physics. A common unit keeps every curve finite and comparable.
fig, a = plt.subplots(figsize=(7.0, 4.4))
grid = np.asarray(rec(sectors[0], "volume")["t3"], float)
vdiag = np.array([np.asarray(rec(s, "volume")["mean_abs_diag_mean"], float)
                  for s in sectors]).mean(0)
floor = np.array([np.asarray(rec(s, "volume")["max_abs_mean"], float)
                  for s in sectors]).mean(0) / vdiag
a.fill_between(grid, 0, 100 * floor, color="0.85", alpha=0.8, lw=0,
               label="error floor (volume axion: analytic zero)")
for axn in ["eaten0", "eaten1", "shape"]:
    rel = np.array([np.asarray(rec(s, axn)["max_abs_mean"], float)
                    for s in sectors]).mean(0) / vdiag
    a.plot(grid, 100 * rel, color=AXION_COLOR[axn], marker=AXION_MARKER[axn],
           ls=AXION_LS[axn], ms=4, lw=1.4, label=AXION_LABEL[axn])
a.set_xlabel(XLABEL)
if args.paper:
    # the full label overflows the panel height at paper font scale: drop the vol
    # superscript (the caption states the denominator is the volume-axion diagonal)
    # and shave this one label's font by 12% so the [%] unit fits
    import matplotlib as mpl
    a.set_ylabel(r"$\max_{I\neq J}|M_{IJ}| \,/\, \overline{|M_{II}|}$ "
                 + ("[%]" if args.no_latex else r"[\%]"),
                 fontsize=0.88 * float(mpl.rcParams["axes.labelsize"]))
else:
    a.set_ylabel(r"$\max_{I\neq J}|M_{IJ}| \;/\; \overline{|M^{\rm vol}_{II}|}$   "
                 + ("[%]" if args.no_latex else r"[\%]"))
if args.paper:
    add_headroom(a, 0.38, "top")
    a.legend(fontsize="small", frameon=False, loc="upper left")
else:
    a.legend(fontsize="small", frameon=False)
save(fig, "fcnc_floor_vs_t3.pdf")

print("wrote:")
for w in written:
    print("  " + w)
b = C["benchmark_offdiag"]
bt3 = np.asarray(b["t3"], float)
bkeep = ((bt3 >= PAPER_T3_WINDOW[0]) & (bt3 <= PAPER_T3_WINDOW[1])) if args.paper \
    else np.ones(len(bt3), bool)
brel = np.asarray(b["rel_mean"], float)[bkeep]
print(f"\nvolume-axion off-diagonal (analytic ZERO) runs "
      f"{brel[0]:.2%} -> {brel[-1]:.2%} of the diagonal across the "
      + ("paper window" if args.paper else "grid"))
