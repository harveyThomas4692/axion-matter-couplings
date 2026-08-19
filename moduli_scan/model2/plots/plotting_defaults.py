import matplotlib as mpl
import math
import matplotlib.pyplot as plt
import numpy as np


###### # Setup Plotting Defaults #
###################################### ##
# For more options see https://matplotlib.org/users/customizing.html

# --- Optional LaTeX path shim (so matplotlib usetex works without
# requiring `module load texlive/...` in the calling shell). Tries a
# rocky-8 texlive/2024 location first, then a legacy sl-7 texlive/2016 one.
# Idempotent: only prepends if pdflatex isn't already on PATH.
import os as _os
import shutil as _shutil
if _shutil.which("pdflatex") is None:
    for _tex_bin in (
        "/global/software/rocky-8.x86_64/manual/modules/tools/texlive/2024/bin/x86_64-linux",
        "/global/software/sl-7.x86_64/modules/tools/texlive/2016/bin/x86_64-linux",
    ):
        if _os.path.isdir(_tex_bin):
            _os.environ["PATH"] = _tex_bin + ":" + _os.environ["PATH"]
            break


from cycler import cycler

# Commands for high detail plots (much larger in file size though)
#mpl.rcParams['agg.path.chunksize'] = 1000
#mpl.rcParams['savefig.epi'] = 1000

# Line styles
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['lines.antialiased'] = True
mpl.rcParams['lines.dashed_pattern'] = 2.8, 1.5
mpl.rcParams['lines.dashdot_pattern'] = 4.8, 1.5, 0.8, 1.5
mpl.rcParams['lines.dotted_pattern'] = 1.1, 1.1
mpl.rcParams['lines.scale_dashes'] = True

# Default colors
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler('color',['cornflowerblue','forestgreen','maroon','goldenrod','firebrick','mediumorchid'])


# Fonts
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'CMU Serif'
mpl.rcParams['font.sans-serif'] = 'CMU Sans Serif, DejaVu Sans, Bitstream Vera Sans, Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica, Avant Garde, sans-serif'

# Axes
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['axes.labelpad'] = 5.0

# Grid off globally — turn on per-figure only when explicitly needed.
mpl.rcParams['axes.grid'] = False

# Tick marks - the essence of life
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.major.size'] = 4
mpl.rcParams['xtick.minor.size'] = 2.0
mpl.rcParams['xtick.major.width'] = 0.8
mpl.rcParams['xtick.minor.width'] = 0.6
mpl.rcParams['xtick.major.pad'] = 4
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['ytick.major.size'] = 4
mpl.rcParams['ytick.minor.size'] = 2.0
mpl.rcParams['ytick.major.width'] = 0.8
mpl.rcParams['ytick.minor.width'] = 0.6
mpl.rcParams['ytick.major.pad'] = 4
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['ytick.minor.visible'] = True

# Legend
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['legend.frameon'] = True
mpl.rcParams['legend.framealpha'] = 1.
#mpl.rcParams['legend.edgecolor'] = 'black'
mpl.rcParams['legend.fancybox'] = True
mpl.rcParams['legend.borderpad'] = 0.4 # border whitespace
mpl.rcParams['legend.labelspacing'] = 0.4 # the vertical space between the legend entries
mpl.rcParams['legend.handlelength'] = 1.5 # the length of the legend lines
mpl.rcParams['legend.handleheight'] = 0.7 # the height of the legend handle
mpl.rcParams['legend.handletextpad'] = 0.5 # the space between the legend line and legend text
mpl.rcParams['legend.borderaxespad'] = 0.4 # the border between the axes and legend edge
mpl.rcParams['legend.columnspacing'] = 1.5 # column separation


# Figure size — PRL single-column native (~3.4 inches wide).
# Scripts emitting two-column or full-page figures override locally.
mpl.rcParams['figure.figsize'] = 5.0, 3.5
mpl.rcParams['figure.dpi'] = 120

# Save details
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.05
mpl.rcParams['savefig.dpi'] = 300

# Use LaTeX for all text rendering (PRL-ready math + fonts).
mpl.rcParams['text.usetex'] = True
