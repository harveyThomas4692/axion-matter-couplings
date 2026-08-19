#!/usr/bin/env python3
"""
Plot axion-matter couplings vs psi1 from combined.json (moduli_scan/model1/aggregate.py output).

One figure per SM field (sector), a line per axion showing how the canonical
coupling M=-i*xi (diagonal, per generation) varies with psi1, with seed error bars.

Layout: one figure per sector; one panel per generation/field in that sector;
within a panel, one line per axion (light + eaten_a), mean +/- std over seeds.

Usage:
  python moduli_scan/model1/plots/plot_couplings.py [--combined moduli_scan/model1/data/combined.json]
                                 [--outdir figures] [--no-latex]
"""
import argparse, json, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--combined", default="moduli_scan/model1/data/combined.json")
ap.add_argument("--outdir", default="figures")
ap.add_argument("--no-latex", action="store_true", help="disable usetex (fallback)")
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


def plot_sector(sec, sd):
    forms = sd["forms"]
    by_ax = sd["by_axion"]
    axions = [a for a in ["light", "eaten0", "eaten1", "eaten2"] if a in by_ax]
    axions += [a for a in by_ax if a not in axions]
    n_fields = len(forms)

    fig, axes = plt.subplots(1, n_fields, figsize=(3.4*n_fields, 3.2),
                             squeeze=False, sharex=True)
    axes = axes[0]
    for fi, fname in enumerate(forms):
        ax = axes[fi]
        for a in axions:
            rec = by_ax[a]
            x = np.array(rec["psi1"])
            mean = np.array(rec["mean"])[:, fi]
            std = np.array(rec["std"])[:, fi]
            ax.errorbar(x, mean, yerr=std, marker=AXION_MARKER.get(a, "o"), ms=3,
                        capsize=2, lw=1.3, color=AXION_COLOR.get(a),
                        ls=AXION_LS.get(a, "-"), label=axion_label(a))
        ax.axhline(0, color="0.6", lw=0.6, ls=":")
        ax.set_xlabel(r"$\psi_1$")
        ax.set_title(rf"${fname}$")
        if fi == 0:
            ax.set_ylabel(r"$M=-i\xi$  (diag.)")
    axes[-1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    out = os.path.join(args.outdir, f"coupling_vs_psi1_{sec}.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_light_summary(data):
    """Headline: light-axion coupling (gen-averaged) per sector vs psi1.

    Two panels (3-generation matter vs Higgs/split sectors). Colour+marker per
    sector come from the SHARED canonical map (scan_plot_style), so a given sector
    has the same colour here and in every other figure.
    """
    GROUPS = [("3-generation matter", ["Q", "U", "E"]),
              (r"Higgs $+$ split $L,D$ sectors",
               ["H", "Hd", "L_4_5", "D_4_5", "L_2_4", "D_2_4"])]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, (title, secs) in zip(axes, GROUPS):
        for sec in secs:
            sd = data["sectors"].get(sec)
            if not sd or "light" not in sd["by_axion"]:
                continue
            rec = sd["by_axion"]["light"]
            x = np.array(rec["psi1"])
            m = np.array(rec["mean"]).mean(1)        # avg over generations/fields
            s = np.array(rec["std"]).mean(1)
            ax.errorbar(x, m, yerr=s, marker=SECTOR_MARKER.get(sec, "o"), ms=4,
                        capsize=2, lw=1.3, color=SECTOR_COLOR.get(sec),
                        label=sector_label(sec))
        ax.set_xlabel(r"$\psi_1$")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"light-axion $M=-i\xi$ (gen-avg)")
    fig.tight_layout()
    out = os.path.join(args.outdir, "coupling_vs_psi1_light_summary.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


written = []
for sec, sd in data["sectors"].items():
    written.append(plot_sector(sec, sd))
written.append(plot_light_summary(data))
print("wrote:")
for w in written:
    print("  " + w)
