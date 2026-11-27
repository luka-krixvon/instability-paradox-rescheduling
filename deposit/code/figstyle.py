"""Figure style for this paper.

The visual grammar imitates the style of DiscoLoop (arXiv:2607.00341v2), retargeted
from IEEEtran to this manuscript's sn-jnl text block.

Data plots: Times New Roman, matplotlib tab10 colors, all four spines, light grid on both axes, centered regular
panel titles, framed (rounded) legends with titles. Schematics (Fig. 1): Arial, rounded panels with semantic
red/amber/green borders, light-filled nodes and pills, colored bold takeaway lines.

Layout notes (colors and fonts superseded above):

The visual grammar, in one place so every paper's figures come out of the same box:

- DejaVu Sans, 9.5pt base; axes titles 10pt bold and LEFT-aligned, phrased as the
  figure's CLAIM ("Cost bounded by ~0.5%; only 14B clears zero"), never as a caption
  ("Accuracy cost").
- White ground; faint x-grid (alpha 0.18) drawn below the data; no top/right chartjunk.
- Palette: ink `dark` for text and edges, `mid_gray` for de-emphasis, `blue` for point
  estimates, `red` reserved for the one thing the figure exists to point at, `green` for
  good/safe, `orange` for warm emphasis. Red is a budget, not a color: if two things are
  red, neither is.
- Value labels sit next to their marks (above a point estimate, beside a bar end), bold
  only when they carry the figure's message.
- Overview figures (Fig. 1) are schematics on a hidden axis: rounded FancyBboxPatch
  stages, arrow flow, red injection/emphasis arrows, and an italic top-right kicker with
  the headline number.
- Type-42 fonts and stripped PDF metadata, so a rebuild is byte-comparable and carries no
  local provenance.

Hard-won rule this module enforces rather than documents: TEXT IS PLACED BY MEASURING,
NOT BY EYE. Eyeballing rasterized figures missed three frame overruns in one paper that a
bbox measurement found immediately. `save()` therefore refuses to write a figure whose
text overflows its axes or its containing box; `fit_fontsize()` sizes panel text from the
measured width of its longest line.

Usage:
    import figstyle as fs
    fs.apply()
    fig, ax = plt.subplots(figsize=fs.COLUMN)           # or fs.WIDE for figure*
    ...
    fs.save(fig, "out/figure.pdf")                       # measures, then writes
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---- palette ----
# ---- data-plot palette: matplotlib tab10, as in DiscoLoop's result figures ----
tab_blue, tab_orange, tab_green, tab_red = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"
tab_purple, tab_brown, tab_gray = "#9467bd", "#8c564b", "#7f7f7f"
ink = "#000000"                                  # axis text and frames in data plots
dark = "#222222"                                 # annotation text
mid_gray = "#7f7f7f"                             # de-emphasis (tab gray)
line_gray = "#8c8c8c"                            # reference lines
grid_gray = "#b0b0b0"                            # grid (drawn with alpha, see _RC)
band_gray = "#f5f5f5"                            # shaded regions (zoom windows, null bands)
frame = "#000000"
# legacy names used by the figure scripts, mapped onto the new palette
blue, orange, green, red = tab_blue, tab_orange, tab_green, tab_red
face_green, face_orange = "#e8f5e9", "#fff3e0"

# ---- schematic palette, sampled from DiscoLoop Fig. 1 ----
SANS = ["Arial", "Helvetica", "DejaVu Sans"]
sch_red, sch_red_text = "#D0452F", "#C62828"          # "fails" panel border / takeaway text
sch_amber, sch_amber_text = "#F9B900", "#F2A600"      # "partial" panel
sch_green, sch_green_text = "#2E7D32", "#2E8B3A"      # "solution" panel
sch_blue_fill, sch_blue_edge = "#6D9EEB", "#4A7BD1"   # small embedding squares
sch_layer_fill, sch_layer_edge = "#EEF3FB", "#3C6CB4" # layer / stage boxes
sch_yellow_fill, sch_yellow_edge = "#F8D775", "#E2B93B"
sch_teal_fill, sch_teal_edge = "#D4ECE6", "#2A7E6F"   # entity pills
sch_purple_fill, sch_purple_edge = "#EEE8FB", "#7C5CD6"
sch_light_green = "#C8EFC8"
sch_sub = "#555555"                                   # gray subtitles ("Input embeddings")

# ---- canonical sizes (inches) for this manuscript (sn-jnl) ----
# \textwidth measured from the build log: 455.24408 pt = 6.30 in.
COLUMN = (3.10, 2.30)      # half of the text block
COLUMN_SHORT = (3.10, 2.10)
WIDE = (6.30, 2.85)        # full text width (figure*, the overview)

PDF_METADATA = {"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None}

_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.titleweight": "normal",
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "axes.edgecolor": frame,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.grid": False,          # scripts call ax.grid(True) on data axes (label/schematic axes must stay clean)
    "axes.axisbelow": True,
    "grid.color": grid_gray,
    "grid.alpha": 0.4,
    "grid.linewidth": 0.6,
    "text.color": ink,
    "axes.labelcolor": ink,
    "xtick.color": ink,
    "ytick.color": ink,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "legend.title_fontsize": 8.5,
    "legend.frameon": True,
    "legend.fancybox": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "lines.linewidth": 1.8,
    "lines.markersize": 4.5,
    "pdf.fonttype": 42,
}


def apply() -> None:
    """Install the rcParams. Call once, before creating figures."""
    matplotlib.use("Agg")
    plt.rcParams.update(_RC)


def fit_fontsize(fig, lines, avail_inches, start=7.05, floor=6.0):
    """Largest font size at which the widest of `lines` fits in `avail_inches`.

    `lines` is an iterable of (text, bold) pairs. Measures rendered extents on `fig`'s
    canvas; never guesses. Panel text sized this way cannot overrun its frame, which is
    why panels in one figure may end up at slightly different sizes -- that is correct
    behaviour, not an inconsistency.
    """
    r = fig.canvas.get_renderer()
    avail = avail_inches * fig.dpi
    size = start
    while size > floor:
        widest = max(fig.text(0, 0, t, fontsize=size,
                              weight="bold" if b else "normal").get_window_extent(r).width
                     for t, b in lines)
        fig.texts[-len(list(lines)):]
        for t in list(fig.texts):
            t.remove()
        if widest <= avail:
            return round(size, 2)
        size -= 0.05
    return floor


def text_overflows(fig, pad_px=1.5, box_pad_px=2.0):
    """Return a list of (description, dx, dy) for text that escapes its container.

    Two containers are checked, because they fail differently:
      - every visible text against its axes' bbox (catches labels past a spine);
      - every text whose center lies inside a FancyBboxPatch against that patch
        (catches panel text past its frame, which the axes-level check cannot see).
    Tick labels and axis labels are checked against the figure bbox. An artist whose
    gid is set to "allow-overflow" is skipped, for the rare label meant to hang outside.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    bad = []
    for ax in fig.axes:
        ab = ax.get_window_extent(r)
        boxes = [(p, p.get_window_extent(r)) for p in ax.patches
                 if isinstance(p, FancyBboxPatch)]
        for t in ax.texts:
            if not t.get_visible() or not t.get_text().strip():
                continue
            if t.get_gid() == "allow-overflow":
                continue
            tb = t.get_window_extent(r)
            cx, cy = (tb.x0 + tb.x1) / 2, (tb.y0 + tb.y1) / 2
            owner = next((pb for _, pb in boxes
                          if pb.x0 <= cx <= pb.x1 and pb.y0 <= cy <= pb.y1), None)
            if owner is not None:
                d = max(owner.x0 + box_pad_px - tb.x0, tb.x1 - (owner.x1 - box_pad_px),
                        tb.y1 - owner.y1, owner.y0 - tb.y0)
                if d > 0.5:
                    bad.append((f"box text {t.get_text()!r}", round(d, 1), 0.0))
                continue
            dx = max(0, ab.x0 - tb.x0) + max(0, tb.x1 - ab.x1)
            dy = max(0, ab.y0 - tb.y0) + max(0, tb.y1 - ab.y1)
            if dx > pad_px or dy > pad_px:
                bad.append((f"axes text {t.get_text()!r}", round(dx, 1), round(dy, 1)))
        # Ticks outside the view interval exist as artists but are never rendered;
        # measuring them reports phantom overflows, so only in-view ticks are checked.
        tick_texts = []
        for axis, (lo, hi) in ((ax.xaxis, sorted(ax.get_xlim())),
                               (ax.yaxis, sorted(ax.get_ylim()))):
            for tick in axis.get_major_ticks():
                if lo <= tick.get_loc() <= hi:
                    tick_texts.append(tick.label1)
        for t in tick_texts + [ax.title, ax.xaxis.label, ax.yaxis.label]:
            if not t.get_visible() or not t.get_text().strip():
                continue
            b = t.get_window_extent(r)
            if b.width and (b.x0 < fb.x0 - 0.5 or b.x1 > fb.x1 + 0.5 or b.y1 > fb.y1 + 0.5):
                bad.append((f"fig-edge {t.get_text()!r}",
                            round(max(0, fb.x0 - b.x0) + max(0, b.x1 - fb.x1), 1),
                            round(max(0, b.y1 - fb.y1), 1)))
    return bad


def save(fig, path, check=True):
    """Measure, then write. Refuses to save a figure whose text overflows.

    The refusal is the point: an unchecked save is how three overruns shipped once.
    Pass check=False only for a figure whose overflow is deliberate and reviewed.
    """
    if check:
        bad = text_overflows(fig)
        if bad:
            lines = "\n".join(f"  {d}: dx={dx} dy={dy}" for d, dx, dy in bad)
            raise ValueError(f"text overflows its container in {path}:\n{lines}")
    fig.savefig(path, metadata=PDF_METADATA)
