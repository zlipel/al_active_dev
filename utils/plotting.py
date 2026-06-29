# plotting.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

Number = Union[int, float]


@dataclass
class PlotStyle:
    font_family: str = "Arial"
    base_font_size: int = 14
    label_font_size: int = 14
    title_font_size: int = 16
    tick_font_size: int = 12
    legend_font_size: int = 12

    spine_linewidth: float = 1.2
    tick_width: float = 1.2
    tick_length: float = 4.0
    tick_direction: str = "in"

    # tick density control
    max_xticks: int = 6
    max_yticks: int = 6

    # layout
    tight_layout: bool = True

    # optional: keep background clean for slides
    transparent: bool = False


def set_plot_style(style: PlotStyle = PlotStyle()) -> None:
    """
    Set global matplotlib rcParams for consistent formatting across scripts.
    Call once near the top of your plotting entrypoint.
    """
    mpl.rcParams.update({
        "font.family": style.font_family,
        "font.size": style.base_font_size,
        "axes.titlesize": style.title_font_size,
        "axes.labelsize": style.label_font_size,
        "xtick.labelsize": style.tick_font_size,
        "ytick.labelsize": style.tick_font_size,
        "legend.fontsize": style.legend_font_size,
        "axes.linewidth": style.spine_linewidth,
        "xtick.direction": style.tick_direction,
        "ytick.direction": style.tick_direction,
        "xtick.major.width": style.tick_width,
        "ytick.major.width": style.tick_width,
        "xtick.major.size": style.tick_length,
        "ytick.major.size": style.tick_length,
    })


def _as_axes_list(axes: Union[Axes, Sequence[Axes], None], fig: Figure) -> list[Axes]:
    if axes is None:
        return [ax for ax in fig.axes if isinstance(ax, Axes)]
    if isinstance(axes, Axes):
        return [axes]
    return list(axes)

def _thin(labels, max_keep):
    n = len(labels)
    if max_keep is None or max_keep <= 0 or n <= max_keep:
        return labels, 1
    step = int(np.ceil(n / max_keep))
    return labels[::step], step


def _format_axes(ax: Axes, style: PlotStyle, label: bool = False, min_n_ticks: int = 3, shap: bool = False) -> None:
    # Spines + ticks
    for side in ["top", "bottom", "left", "right"]:
        if side in ax.spines:
            ax.spines[side].set_linewidth(style.spine_linewidth)

    ax.tick_params(
        axis="both",
        which="both",
        direction=style.tick_direction,
        width=style.tick_width,
        length=style.tick_length,
        labelsize=style.tick_font_size,
        labelbottom=True,
        labelleft=True,
    )

    ax.minorticks_off()

    # Labels (you usually want blank for slide assembly)
    if label:
        ax.set_xlabel(ax.get_xlabel(), fontsize=style.label_font_size)
        ax.set_ylabel(ax.get_ylabel(), fontsize=style.label_font_size)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")

    # ---- HEATMAP SAFE PATH ----
    # seaborn heatmap creates a QuadMesh in ax.collections[0] and uses categorical ticks.
    # If we apply numeric locators, labels can disappear. So for heatmaps, we DO NOT set locators.
    is_heatmap = len(ax.collections) > 0  # works for seaborn heatmap
    if is_heatmap and not shap:
        # Ensure tick labels are actually shown
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=True)

        # Optionally thin tick labels deterministically (keeps mapping stable)
        # If you truly want ALL ticks for small grids, set style.max_*ticks >= len(labels)
        xt = ax.get_xticklabels()
        yt = ax.get_yticklabels()

        # If seaborn hasn't populated labels yet, bail out
        if len(xt) == 0 or len(yt) == 0:
            return

        # Thin without changing locator (preserves label mapping)
        # Keep every k-th label so total <= max_xticks/max_yticks


        import numpy as np
        xt_keep, xstep = _thin(list(xt), style.max_xticks)
        yt_keep, ystep = _thin(list(yt), style.max_yticks)

        # Apply thinning by hiding labels (NOT resetting tick locations)
        # This is the safest approach with heatmaps.
        for i, t in enumerate(xt):
            t.set_visible(i % xstep == 0)
        for i, t in enumerate(yt):
            t.set_visible(i % ystep == 0)

        return

    # # Ensure tick labels are actually shown
    # ax.tick_params(axis="x", labelbottom=True)
    # ax.tick_params(axis="y", labelleft=True)

    # xt = ax.get_xticklabels()
    # yt = ax.get_yticklabels()

    # print(xt, yt)

    # if len(xt) == 0 or len(yt) == 0:
    #         return


    # xt_keep, xstep = _thin(list(xt))#, style.max_xticks)
    # yt_keep, ystep = _thin(list(yt))#, style.max_yticks)

    # for i, t in enumerate(xt):
    #     t.set_visible(i % xstep == 0)
    # for i, t in enumerate(yt):
    #     t.set_visible(i % ystep == 0)

    # return

    # ---- NUMERIC PLOTS ----
    ax.xaxis.set_major_locator(MaxNLocator(nbins=style.max_xticks, min_n_ticks = min_n_ticks))
    ax.yaxis.set_major_locator(MaxNLocator(min_n_ticks=style.max_yticks))



def _find_colorbar_for_axes(ax: Axes):
    """
    Try to find an attached colorbar for common cases (seaborn heatmap, imshow, contourf).
    Returns (colorbar, mappable) or (None, None).
    """
    # seaborn heatmap stores QuadMesh in ax.collections
    if ax.collections:
        mappable = ax.collections[0]
        cbar = getattr(mappable, "colorbar", None)
        if cbar is not None:
            return cbar, mappable

    # imshow/pcolormesh can also be in ax.images
    if ax.images:
        mappable = ax.images[0]
        cbar = getattr(mappable, "colorbar", None)
        if cbar is not None:
            return cbar, mappable

    return None, None


def _save_colorbar(
    mappable,
    save_path: Union[str, Path],
    label: Optional[str],
    dpi: int,
    fmt: Optional[str],
    style: PlotStyle,
    size: Tuple[Number, Number] = (1.2, 3.5),
) -> None:
    """
    Create a standalone colorbar figure from a mappable (QuadMesh/Image/ContourSet).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig_cb, ax_cb = plt.subplots(figsize=(size[0], size[1]))
    ax_cb.set_axis_off()

    # Create colorbar using mappable's norm/cmap
    cb = fig_cb.colorbar(mappable, ax=ax_cb, orientation="vertical", fraction=1.0)
    if label:
        cb.set_label(label, fontsize=style.label_font_size)

    cb.ax.tick_params(
        direction=style.tick_direction,
        width=style.tick_width,
        length=style.tick_length,
        labelsize=style.tick_font_size,
    )

    if fmt is None:
        fmt = save_path.suffix.lstrip(".") if save_path.suffix else "png"

    fig_cb.savefig(save_path, dpi=dpi, format=fmt, transparent=style.transparent, bbox_inches='tight')
    plt.close(fig_cb)


def format_and_save_figure(
    fig: Figure,
    axes: Union[Axes, Sequence[Axes], None] = None,
    save_path: Union[str, Path] = "figure.png",
    *,
    dpi: int = 450,
    fmt: Optional[str] = None,
    dimensions: Optional[Tuple[Number, Number]] = None,
    style: PlotStyle = PlotStyle(),
    save_colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    colorbar_size: Tuple[Number, Number] = (1.2, 3.5),
    tight_layout: Optional[bool] = None,
    close: bool = False,
    label: bool = False,
    min_n_ticks: int = 3,
    shap: bool = False
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine fmt EARLY (before cb_path creation)
    if fmt is None:
        fmt = save_path.suffix.lstrip(".") if save_path.suffix else "png"

    if dimensions is not None:
        fig.set_size_inches(dimensions[0], dimensions[1], forward=True)

    ax_list = _as_axes_list(axes, fig)
    for ax in ax_list:
        if not shap:
            _format_axes(ax, style, label=label, min_n_ticks=min_n_ticks,shap=shap)

        else:

            ax.tick_params(
                axis='x',
                direction=style.tick_direction,
                width=style.tick_width,
                length=style.tick_length,
                labelsize=style.tick_font_size,
            )

            #ax.minorticks_off()

            # Labels (you usually want blank for slide assembly)
            if label:
                ax.set_xlabel(ax.get_xlabel(), fontsize=style.label_font_size)
                ax.set_ylabel(ax.get_ylabel(), fontsize=style.label_font_size)
            else:
                ax.set_xlabel("")
                ax.set_ylabel("")

    use_tl = style.tight_layout if tight_layout is None else tight_layout
    if use_tl:
        fig.tight_layout()

    # Save colorbars and remember them so we can remove them from main fig
    cbars_to_remove = []

    if save_colorbar:
        seen = set()
        for ax in ax_list:
            cbar, mappable = _find_colorbar_for_axes(ax)
            if cbar is None or mappable is None:
                continue
            key = id(cbar)
            if key in seen:
                continue
            seen.add(key)

            # Decide label to save on standalone CB
            if colorbar_label is True:
                try:
                    cbar_label = cbar.ax.get_ylabel()
                except Exception:
                    cbar_label = None
            elif isinstance(colorbar_label, str):
                cbar_label = colorbar_label
            else:
                cbar_label = None

            cb_path = save_path.with_name(f"{save_path.stem}_colorbar.{fmt}")
            _save_colorbar(
                mappable=mappable,
                save_path=cb_path,
                label=cbar_label,
                dpi=dpi,
                fmt=fmt,
                style=style,
                size=colorbar_size,
            )

            cbars_to_remove.append(cbar)

    # Remove colorbars from main fig (so main saves WITHOUT CB)
    for cbar in cbars_to_remove:
        try:
            cbar.remove()
        except Exception:
            pass

    # Save main fig
    fig.savefig(save_path, dpi=dpi, format=fmt, transparent=style.transparent, bbox_inches="tight")

    if close:
        plt.close(fig)
