"""Publication-quality plotting helpers for ripplepy.

A single, consistent matplotlib style for scientific figures:
serif font with STIX math, colour-blind-friendly (Okabe-Ito) palette,
light grid, and vector (PDF) + high-resolution (PNG) export.

Typical usage::

    from ripplepy.plotting import setup_publication_style, save_figure, PUB_COLORS
    import matplotlib.pyplot as plt

    setup_publication_style()            # headless-safe (Agg backend)
    fig, ax = plt.subplots()
    ax.plot(x, y, "o-", color=PUB_COLORS["blue"])
    ...
    save_figure(fig, "my_figure")        # writes my_figure.pdf and my_figure.png
"""

import matplotlib as _mpl

# Colour-blind-friendly / print-safe palette (Okabe-Ito)
PUB_COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "cyan": "#56B4E9",
    "magenta": "#CC79A7",
    "black": "#000000",
}

# Publication-quality matplotlib rcParams.
PUBLICATION_RC = {
    # figure
    "figure.figsize": (9.0, 3.4),
    # fonts: serif + STIX math (Times-like, standard in journals)
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    # lines / axes
    "lines.linewidth": 1.6,
    "lines.markersize": 6,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "0.85",
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    # output
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def setup_publication_style():
    """Apply the publication style and switch to the headless Agg backend.

    Must be called before ``matplotlib.pyplot`` is imported.
    """
    _mpl.use("Agg")
    _mpl.rcParams.update(PUBLICATION_RC)


def save_figure(fig, path_stem, dpi=300):
    """Save a figure as vector PDF and high-resolution PNG.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path_stem : str or os.PathLike
        Output path without extension; ``<stem>.pdf`` and ``<stem>.png``
        are written.
    dpi : int
        Resolution of the PNG output.
    """
    fig.savefig(f"{path_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{path_stem}.png", dpi=dpi, bbox_inches="tight")
