"""Lazy matplotlib import and the axis plumbing every plot here shares.

matplotlib is an optional dependency, exactly as ASE is for
:mod:`autopcet.ase_io`: it is imported only once a plotting function actually
needs it, so ``import autopcet`` keeps working on NumPy and SciPy alone.
Install it with the ``plotting`` extra.

Nothing in :mod:`autopcet.plotting` saves, shows, or clears a figure, and
nothing touches the pyplot global state beyond creating a figure when the
caller passed no axes to draw on. Every function returns the axes it drew on so
the caller can go on styling and saving them.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

MATPLOTLIB_HINT = (
    '{need} needs matplotlib. Install it with `pip install "autopcet[plotting]"`.'
)


def pyplot(need: str = "Plotting") -> ModuleType:
    """Import ``matplotlib.pyplot``, naming the extra when it is not installed."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(MATPLOTLIB_HINT.format(need=need)) from error

    return plt


def prepare_axes(
    ax: Axes | None = None, need: str = "Plotting", **figure_kwargs: Any
) -> Axes:
    """Return the axes to draw on, making a figure when none was given.

    ``figure_kwargs`` go to ``plt.subplots``; the layout is always constrained,
    which resizes the axes the way ``tight_layout`` does without warning about
    the axes it cannot handle.
    """
    if ax is not None:
        return ax

    plt = pyplot(need)
    _, created = plt.subplots(layout="constrained", **figure_kwargs)
    return cast("Axes", created)


def prepare_axes_grid(
    axes: Sequence[Axes] | None = None,
    need: str = "Plotting",
    nrows: int = 1,
    ncols: int = 1,
    **figure_kwargs: Any,
) -> list[Axes]:
    """Return ``nrows * ncols`` axes to draw on, row by row.

    A grid handed in has to hold exactly that many axes, in the same order.
    """
    if axes is not None:
        if len(axes) != nrows * ncols:
            raise ValueError(
                f"Expected {nrows * ncols} axes to draw on, got {len(axes)}."
            )
        return list(axes)

    plt = pyplot(need)
    _, created = plt.subplots(nrows, ncols, layout="constrained", **figure_kwargs)
    return [cast("Axes", item) for item in np.atleast_1d(created).ravel()]


def figure_of(ax: Axes) -> Figure:
    """The figure an axes belongs to, narrowed off ``Figure | SubFigure | None``.

    An axes always has one, but matplotlib types the accessor optionally and
    lets a sub-figure stand in for it; ``root=True`` asks for the real figure,
    which is the one that can be saved.
    """
    return cast("Figure", ax.get_figure(root=True))


def add_legend(ax: Axes, **kwargs: Any) -> None:
    """Add a legend, but only once some artist on the axes carries a label.

    matplotlib warns when asked for a legend it has nothing to put in, and the
    test suite turns warnings into errors.
    """
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(**kwargs)
