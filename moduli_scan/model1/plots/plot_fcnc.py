#!/usr/bin/env python3
"""
Plot the FCNC (off-diagonal) axion-matter couplings vs psi1 from combined_fcnc.json
(moduli_scan/model1/aggregate_fcnc.py output).

The eaten (heavy) axions carry non-zero 1<->2 off-diagonal couplings
M[0,1] = -i*xi that mediate flavour-changing effects, while the light (volume)
axion is flavour-aligned (suppressed off-diagonal). This shows |M[0,1]| vs psi1 per
multi-generation sector, line per axion, seed error bars.

Usage: python moduli_scan/model1/plots/plot_fcnc.py [--combined moduli_scan/model1/data/combined_fcnc.json]
                                 [--outdir figures] [--no-latex]
"""
import argparse, json, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--combined", default="moduli_scan/model1/data/combined_fcnc.json")
ap.add_argument("--outdir", default="figures")
ap.add_argument("--no-latex", action="store_true")
args = ap.parse_args()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
if args.no_latex:
    matplotlib.rcParams["text.usetex"] = False

data = json.load(open(args.combined))
os.makedirs(args.outdir, exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan_plot_style import (SECTOR_COLOR, SECTOR_MARKER, sector_label,
                             AXION_COLOR, AXION_MARKER, AXION_LS, axion_label)

AX = ["eaten0", "eaten1", "eaten2", "light"]


def plot_sector(sec, sd):
    fig, ax = plt.subplots(figsize=(6, 4))
    for a in AX:
        rec = sd["off01"][a]
        x = np.array(rec["psi1"])
        y = np.array(rec["abs_mean"]); e = np.array(rec["abs_std"])
        ax.errorbar(x, y, yerr=e, marker=AXION_MARKER.get(a, "o"), ms=3, capsize=2,
                    lw=1.3, color=AXION_COLOR.get(a), ls=AXION_LS.get(a, "-"),
                    label=axion_label(a))
    ax.set_xlabel(r"$\psi_1$")
    ax.set_ylabel(r"$|M_{12}| = |\xi_{12}|$  (gen $1\!\leftrightarrow\!2$ FCNC)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(args.outdir, f"fcnc_vs_psi1_{sec}.pdf")
    fig.savefig(out); plt.close(fig)
    return out


written = []
for sec, sd in data["sectors"].items():
    written.append(plot_sector(sec, sd))

# summary: max FCNC (over eaten axions) per sector vs psi1
fig, ax = plt.subplots(figsize=(6, 4))
for sec, sd in data["sectors"].items():
    grid = sd["off01"]["eaten0"]["psi1"]
    stack = np.array([sd["off01"][a]["abs_mean"] for a in ["eaten0", "eaten1", "eaten2"]])
    ax.plot(grid, stack.max(0), marker=SECTOR_MARKER.get(sec, "o"), ms=4, lw=1.3,
            color=SECTOR_COLOR.get(sec), label=sector_label(sec))
ax.set_xlabel(r"$\psi_1$")
ax.set_ylabel(r"$\max_{\rm eaten}\,|M_{12}|$")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
out = os.path.join(args.outdir, "fcnc_vs_psi1_summary.pdf")
fig.savefig(out); plt.close(fig)
written.append(out)

dec = data["decoupling"]
print("wrote:")
for w in written:
    print("  " + w)
print(f"\n3rd-gen decoupling: worst |cross-block| over scan = "
      f"{dec['worst_abs_cross_entry']:.2e} ({dec['worst_at']})")
