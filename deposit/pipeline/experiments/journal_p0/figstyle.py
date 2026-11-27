"""Shared figure style for the paper_canonical figures.

Figures are generated at their FINAL display width (single-column = COL_W in,
double-column = TEXT_W in for IEEEtran) so \\includegraphics[width=\\columnwidth]
(or \\textwidth) applies ~no scaling and printed fonts stay >= 8pt. Call
figstyle.apply() right after matplotlib.use("Agg")."""
import matplotlib as mpl

COL_W = 3.5      # IEEEtran \columnwidth (inches)
TEXT_W = 7.16    # IEEEtran \textwidth  (inches)


def apply():
    mpl.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.6,
        "lines.markersize": 6,
        "axes.linewidth": 0.8,
        "savefig.bbox": "tight",     # trim to content -> no edge clipping
        "savefig.pad_inches": 0.02,
    })
