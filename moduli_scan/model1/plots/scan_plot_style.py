#!/usr/bin/env python3
"""
Canonical colour / marker / label maps shared across ALL scan figures
(plot_couplings.py, plot_fcnc.py) so that a given sector or axion has the SAME
colour in every plot. Import this instead of hard-coding styles per script.

  - SECTOR_*  : used where one line == one SM sector (the summary plots).
  - AXION_*   : used where one line == one axion (the per-sector plots).
"""
import matplotlib.pyplot as plt

_TAB = plt.get_cmap("tab10").colors        # 10 distinct qualitative colours
_MARK = ["o", "s", "^", "D", "v", "P", "X", "*", "<"]

# --- per-sector identity (fixed across every figure) ---
SECTOR_ORDER = ["H", "Hd", "Q", "U", "E", "L_4_5", "D_4_5", "L_2_4", "D_2_4"]
SECTOR_COLOR = {s: _TAB[i % len(_TAB)] for i, s in enumerate(SECTOR_ORDER)}
SECTOR_MARKER = {s: _MARK[i % len(_MARK)] for i, s in enumerate(SECTOR_ORDER)}
SECTOR_LABEL = {"H": r"$H$", "Hd": r"$H_d$", "Q": r"$Q$", "U": r"$U$", "E": r"$E$",
                "L_4_5": r"$L_{4,5}$", "D_4_5": r"$D_{4,5}$",
                "L_2_4": r"$L_{2,4}$", "D_2_4": r"$D_{2,4}$"}

# --- per-axion identity (fixed across every figure) ---
AXION_ORDER = ["light", "eaten0", "eaten1", "eaten2"]
AXION_COLOR = {"light": "0.45", "eaten0": _TAB[0], "eaten1": _TAB[1], "eaten2": _TAB[3]}
AXION_MARKER = {"light": "o", "eaten0": "s", "eaten1": "^", "eaten2": "D"}
AXION_LS = {"light": "--", "eaten0": "-", "eaten1": "-", "eaten2": "-"}
AXION_LABEL = {"light": r"light ($b_{\rm vol}$)", "eaten0": r"eaten$_0$",
               "eaten1": r"eaten$_1$", "eaten2": r"eaten$_2$"}


def sector_label(s):
    return SECTOR_LABEL.get(s, s)


def axion_label(a):
    return AXION_LABEL.get(a, a)
