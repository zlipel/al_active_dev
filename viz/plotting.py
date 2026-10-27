"""Named colors, figure tiers, and export settings for project plots."""
from __future__ import annotations

from pathlib import Path

from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


COLORS = {
    "Ill_Red": "#BE1E2D", "Ill_Blue": "#2E358F", "Ill_Green": "#016A38",
    "orange": "#FF8C00", "darkorchid": "#9932CC", "brown": "#A52A2A",
    "black": "#000000", "red": "#d65858", "blue": "#1E90FF",
    "darkgreen": "#006400", "light_blue": "#00B0E4", "dark_blue": "#2C6AB5",
    "blue_1": "#00B0E4", "blue_2": "#0297C5", "blue_3": "#0074CA", "blue_4": "#035390",
}
CATEGORICAL_NAMES = ("Ill_Blue", "orange", "black", "light_blue", "Ill_Red", "darkorchid", "Ill_Green")
CYCLE = tuple(COLORS[name] for name in CATEGORICAL_NAMES)
BLUE_RAMP = tuple(COLORS[f"blue_{index}"] for index in range(1, 5))
MARKERS = ("o", "s", "^", "D", "v", "P", "X")
LINESTYLES = ("-", "--", ":", "-.", "-", "--", ":")
TIERS = {
    "column": {"font": 9, "label": 9, "tick": 8, "legend": 9, "title": 9, "dpi": 450, "formats": ("pdf", "png"), "size": (3.3, 2.6)},
    "page": {"font": 11, "label": 10, "tick": 9, "legend": 18, "title": 10, "dpi": 450, "formats": ("pdf", "png"), "size": (6.5, 8.8)},
    "talk": {"font": 20, "label": 20, "tick": 18, "legend": 18, "title": 22, "dpi": 300, "formats": ("png",), "size": (10.0, 5.625)},
    "general": {"font": 14, "label": 14, "tick": 12, "legend": 18, "title": 16, "dpi": 450, "formats": ("png",), "size": (6.4, 4.8)},
}


def apply_tier(tier: str) -> None:
    """Apply the classic style with Arial and the selected tier's font sizes."""
    style = TIERS[tier]
    plt.style.use("classic")
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": style["font"],
        "axes.labelsize": style["label"], "axes.titlesize": style["title"],
        "xtick.labelsize": style["tick"], "ytick.labelsize": style["tick"],
        "legend.fontsize": style["legend"], "legend.frameon": False,
        "legend.numpoints": 1, "legend.handlelength": 1.5,
        "axes.prop_cycle": (cycler(color=CYCLE) + cycler(marker=MARKERS)
                            + cycler(linestyle=LINESTYLES)), "axes.linewidth": 0.8,
        "axes.grid": False, "axes.formatter.useoffset": False,
        "axes.unicode_minus": False, "axes.titlepad": 8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.size": 2, "ytick.minor.size": 2,
        "lines.linewidth": 1.2, "lines.markersize": 3.5,
        "errorbar.capsize": 2, "figure.figsize": style["size"],
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.transparent": True, "savefig.dpi": style["dpi"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "custom", "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "text.usetex": False,
    })


def set_size(w: float, h: float, ax=None) -> None:
    """Set the selected axes' exact width and height in inches."""
    ax = plt.gca() if ax is None else ax
    bounds = ax.get_position()
    ax.figure.set_size_inches(float(w) / bounds.width, float(h) / bounds.height)


def page_panels(rows: int, columns: int, *, height: float):
    """Create a 6.5-inch-wide multi-panel canvas within a 7 by 11-inch page."""
    if height > 11:
        raise ValueError("Page figures must fit within an 11-inch page")
    apply_tier("page")
    fig, axes = plt.subplots(rows, columns, figsize=(6.5, height), squeeze=False)
    fig.subplots_adjust(left=0.14, right=0.975, bottom=0.6 / height,
                        top=1 - 1.1 / height, wspace=0.65, hspace=0.68)
    return fig, axes


def model_legend(fig, names) -> None:
    """Place a shared model legend inside the page canvas."""
    handles = [Line2D([], [], color=CYCLE[i], marker=MARKERS[i], linestyle=LINESTYLES[i], label=name)
               for i, name in enumerate(names)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=len(handles), columnspacing=0.9, handletextpad=0.4)


def seed_trajectory(ax, x, mean, sd, *, series: int, label=None, log=False):
    """Draw a seed mean and sample-standard-deviation error bars."""
    import numpy as np

    lower = np.minimum(sd, mean * 0.999) if log else sd
    if log:
        ax.set_yscale("log")
    ax.errorbar(x, mean, yerr=np.vstack([lower, sd]), color=CYCLE[series],
                marker=MARKERS[series], linestyle=LINESTYLES[series], label=label,
                elinewidth=0.75, capthick=0.75)


def save_figure(fig, path, *, tier: str, transparent: bool = True):
    """Save the tier's formats without cropping or changing the canvas size."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in TIERS[tier]["formats"]:
        target = path.with_suffix(f".{extension}")
        fig.savefig(target, dpi=TIERS[tier]["dpi"], transparent=transparent,
                    bbox_inches=None, facecolor="none" if transparent else "white")
        outputs.append(target)
    return outputs


def apply_manuscript_page() -> None:
    """Keep the manuscript's 10-point minimum and thicker axes on page figures."""
    apply_tier("page")
    mpl.rcParams.update({
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "axes.linewidth": 1.5, "lines.linewidth": 1.5,
        "svg.fonttype": "none",
    })


def apply_compact_si() -> None:
    """Use page styling with 10-point labels and compact SI legends."""
    apply_manuscript_page()
    mpl.rcParams.update({"xtick.top": False, "ytick.right": False,
                         "xtick.direction": "out", "ytick.direction": "out"})
