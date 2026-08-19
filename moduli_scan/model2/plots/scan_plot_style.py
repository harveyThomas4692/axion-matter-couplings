#!/usr/bin/env python3
"""Canonical colour / marker / label maps shared across all model-2 scan figures.

Mirrors model 1's plots/scan_plot_style.py so a given sector or axion keeps ONE identity across
every figure. Model-2 sectors and axions differ from model 1's: the spectrum is a clean
3-generation chiral GUT with no Higgs, and there are 2 light axions (volume + shape)
rather than 1.
"""
import matplotlib.pyplot as plt

# Default palette: grey, black, maroon -- cycled when a
# figure carries more than three lines, with markers and linestyles carrying the
# rest of the identity. The near-degenerate curve families (Q/u^c/e^c from L5, and
# the U(3)-degenerate d^c/L pair) overlap almost pointwise anyway, so repeated
# colours land on curves that were indistinguishable already.
DEFAULT_COLORS = ["0.5", "black", "maroon"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=DEFAULT_COLORS)
_MARK = ["o", "s", "^", "D", "v", "P", "X", "*", "<"]

# --- per-sector identity ---
# Q, U, E come from L5 (the 10); D, L from O(0,1,-2,0) (the 5-bar), which is
# U(3)-degenerate so those are scalar x 3.
SECTOR_ORDER = ["Q", "U", "E", "D", "L"]
SECTOR_COLOR = {s: DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
                for i, s in enumerate(SECTOR_ORDER)}
SECTOR_MARKER = {s: _MARK[i % len(_MARK)] for i, s in enumerate(SECTOR_ORDER)}
SECTOR_LABEL = {"Q": r"$Q$", "U": r"$u^c$", "E": r"$e^c$", "D": r"$d^c$", "L": r"$L$"}

# --- per-axion identity ---
# 'volume' and 'shape' are the canonical t3-stable light basis built in stage3's [SCAN]
# footer; 'light_raw*' are the notebook's Gram-Schmidt basis, kept for traceability.
AXION_ORDER = ["volume", "shape", "eaten0", "eaten1"]
# Light axions in the strong colours -- volume = black, shape = maroon (THE result) --
# and the eaten pair in two greys, told apart from each other
# by shade and marker, and from the light axions by solid-vs-dashed (AXION_LS below).
AXION_COLOR = {"volume": "black", "shape": "maroon", "eaten0": "0.45", "eaten1": "0.65"}
AXION_MARKER = {"volume": "o", "shape": "D", "eaten0": "s", "eaten1": "^"}
AXION_LABEL = {"volume": "volume axion", "shape": "shape axion",
               "eaten0": "eaten 0", "eaten1": "eaten 1"}
# Dashed for the two LIGHT axions, solid for the eaten ones: in the FCNC figures the light
# curves are a measured error floor (their off-diagonals are an analytic zero, see
# moduli_scan/model2/aggregate_fcnc.py) and the eaten ones are the physical signal, so the two must not
# read as like-for-like.
AXION_LS = {"volume": "--", "shape": "--", "eaten0": "-", "eaten1": "-"}

# The universal volume-axion benchmark.
#
# Derivation (no free parameters, and independent of the manifold and of the Kahler point):
#   the dilation identity  sum_i t^i Lambda_i  proportional to  G  makes
#   xi_vol = (t^i / |t|_g) G^{-1/2}(i Lambda_i) G^{-1/2}  proportional to the identity,
#   with |t|_g fixed by the no-scale identity  t^i t^j g_ij = 3/4  (V homogeneous of
#   degree 3), verified to hold at every model-2 D-flat point in
#   moduli_scan/model2/test_dflat_locus.py CHECK 5. That gives the BARE value 2/sqrt(3), and the
#   canonical real-axion normalisation divides by 2 sqrt(2)
#   (Lambda^can = xi_bare / (2 sqrt 2), the canonical real-axion normalisation):
#       2/sqrt(3) / (2 sqrt 2) = 1/sqrt(6) = 0.4082482904638631
VOLUME_AXION_CANONICAL = 6.0 ** -0.5

# Model 1 measured 0.4305 for this quantity, +5.4% above the analytic value, and the
# result was reproduced independently upstream -- so it is not a harness artefact.
# Model 2 tests the same number on a different
# quotient, different bundles and a different Kahler point, so it discriminates a
# systematic in the machinery from something model-specific.
MODEL1_VOLUME_AXION_MEASURED = 0.4305

# --- the paper's trusted t3 window ---
# Trusted-window truncation for the paper-bound figures:
# cut at t3 ~ 6 and keep off the singular end, then caption both edges.
#   low edge:  the Kahler-cone wall is t3 = 2 (t1 = (t3-2)(t3+1) -> 0, a collapsing
#              cycle; moduli_scan/model2/test_dflat_locus.py), so the t3 = 2.25 grid point sits
#              nearly on it and shows the grid's largest volume-benchmark sector
#              spread, 25%, in the measured volume-axion benchmark.
#   high edge: beyond t3 ~ 6 the locus is strongly anisotropic (t1/t2 = 28 at t3 = 6,
#              180 at t3 = 14, from t1 = (t3-2)(t3+1) with t2 = 1) and the volume-axion
#              benchmark deviation passes +18% (1.180 at t3 = 6, 1.258 at t3 = 14).
# Full-range figures (no suffix) are kept alongside for reference; pass --paper to the
# plotters to produce the windowed versions with the suffix below.
PAPER_T3_WINDOW = (2.5, 6.0)
PAPER_SUFFIX = "_paper"


def apply_paper_fontscale(scale=1.8):
    """Enlarge fonts for the --paper figures.

    The panels are drawn ~7 in wide but printed at 0.49\\textwidth (~3.1 in, a x0.45
    shrink; the per-generation strip lands at x0.52), so the 12pt/10pt label/tick
    fonts of plotting_defaults arrive at ~5-6pt next to the paper's 11pt text.
    Scaling by 1.8 lands labels at ~10-11pt effective. Numeric rcParams are scaled
    directly; named sizes ('small' etc., used by the legend calls) follow font.size.
    """
    import matplotlib as mpl
    for k in ("font.size", "axes.labelsize", "axes.titlesize",
              "xtick.labelsize", "ytick.labelsize", "legend.fontsize"):
        try:
            mpl.rcParams[k] = float(mpl.rcParams[k]) * scale
        except (TypeError, ValueError):
            pass


def add_headroom(ax, frac, where="top"):
    """Grow one end of the y-axis by frac of the current span, opening empty space
    to pin a legend into (legends must not cross curves)."""
    lo, hi = ax.get_ylim()
    span = hi - lo
    if where == "top":
        ax.set_ylim(lo, hi + frac * span)
    else:
        ax.set_ylim(lo - frac * span, hi)
