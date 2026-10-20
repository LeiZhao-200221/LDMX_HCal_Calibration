"""Matplotlib helpers of every stage (Agg backend, pdf pages, detector maps)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import geometry as geo  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "axes.grid": False,
})

ARBITRARY_UNITS = "Energy [arbitrary units]"
MIP_UNITS = "Energy [MIP]"


# new figure with subplots
def new_figure(nrows=1, ncols=1, figsize=(8.0, 6.0), **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kw)
    return fig, axes


# add a figure to a pdf and / or save it to a file, then close it
def save(fig, pdf=None, path=None):
    if pdf is not None:
        pdf.savefig(fig)
    if path is not None:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# pdf page of text lines
def text_page(pdf, title: str, lines: list[str]):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.05, 0.95, title, fontsize=14, weight="bold", va="top")
    fig.text(0.05, 0.90, "\n".join(lines), fontsize=9, va="top", family="monospace")
    save(fig, pdf)


# detector map: (19 x 12) layer / position grid, empty cells blank
def layer_map(ax, grid: np.ndarray, title: str = "", cbar_label: str = "", cmap="viridis",
              log=False, vmin=None, vmax=None, annotate=False, fmt="{:.0f}"):
    from matplotlib.colors import LogNorm

    data = np.ma.masked_invalid(grid)
    norm = None
    if log:
        positive = data.compressed()
        positive = positive[positive > 0]
        if len(positive):
            norm = LogNorm(vmin=vmin or positive.min(), vmax=vmax or positive.max())
    im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap, norm=norm,
                   vmin=None if norm else vmin, vmax=None if norm else vmax,
                   extent=(-0.5, geo.N_POSITIONS - 0.5, 0.5, geo.N_LAYERS + 0.5))
    ax.set_xticks(range(geo.N_POSITIONS))
    ax.set_yticks(range(1, geo.N_LAYERS + 1))
    ax.set_xlabel("physical strip position (8-bar layers: strips 0-7 at positions 2-9)")
    ax.set_ylabel("layer")
    if title:
        ax.set_title(title)
    cb = plt.colorbar(im, ax=ax)
    if cbar_label:
        cb.set_label(cbar_label)
    if annotate:
        for (iy, ix), v in np.ndenumerate(grid):
            if np.isfinite(v):
                ax.text(ix, iy + 1, fmt.format(v), ha="center", va="center", fontsize=6, color="w")
    return im


# histogram drawn as steps
def hist_steps(ax, values, bins, range_=None, label=None, density=False, **kw):
    counts, edges = np.histogram(np.asarray(values, dtype=float), bins=bins, range=range_, density=density)
    ax.stairs(counts, edges, label=label, **kw)
    return counts, edges


# box with fit-result lines
def fit_box(ax, lines: list[str], loc="upper right"):
    ax.legend(handles=[], title="\n".join(lines), loc=loc, frameon=True, title_fontsize=8)
