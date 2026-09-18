"""Thermally averaging a rate constant over the donor-acceptor distance.

A rate constant computed at one donor-acceptor distance ``R`` is only half the
answer: the mode sweeps through a range of ``R``, and the reaction is carried
by whichever distances make ``P(R) k(R)`` largest. That is usually well inside
the short-distance tail, where ``k`` is large but ``P`` is already falling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._types import FloatArray
from ._mpl import add_legend, prepare_axes
from .style import DISTANCE_LABEL, DISTRIBUTION_COLOR, RATE_COLOR

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _scaled(values: FloatArray, normalize: bool) -> FloatArray:
    """Scale a curve to a maximum of one, so three of them share an axis."""
    if not normalize:
        return values

    peak = float(np.max(np.abs(values)))
    if peak == 0.0:
        raise ValueError("Cannot normalize a curve that is zero everywhere.")
    return values / peak


def plot_thermal_average(
    distances: FloatArray,
    rates: FloatArray,
    distribution: FloatArray,
    ax: Axes | None = None,
    *,
    normalize: bool = True,
    mark_maxima: bool = True,
) -> Axes:
    """``k(R)``, ``P(R)``, and the product that decides the average rate.

    All three are scaled to a maximum of one by default, which is the only way
    they share an axis: ``k(R)`` spans orders of magnitude while ``P(R)`` is a
    normalized probability density. With ``mark_maxima`` a grey dashed line
    marks the equilibrium distance, where ``P(R)`` peaks, and a black one the
    dominant distance, where ``P(R) k(R)`` does -- the gap between them is how
    far the reaction is pulled in from equilibrium.
    """
    distance = np.asarray(distances, dtype=np.float64)
    rate = np.asarray(rates, dtype=np.float64)
    weight = np.asarray(distribution, dtype=np.float64)
    weighted = weight * rate

    ax = prepare_axes(ax, "Plotting a thermal average")
    ax.plot(distance, _scaled(rate, normalize), "-", color=RATE_COLOR, label=r"$k(R)$")
    ax.plot(
        distance,
        _scaled(weight, normalize),
        "-",
        color=DISTRIBUTION_COLOR,
        label=r"$P(R)$",
    )
    ax.plot(distance, _scaled(weighted, normalize), "-", color="k", label=r"$P(R)k(R)$")

    if mark_maxima:
        ax.axvline(
            distance[np.argmax(weight)],
            lw=1.5,
            color="darkgray",
            linestyle=(0, (3, 3)),
        )
        ax.axvline(
            distance[np.argmax(weighted)], lw=1.5, color="k", linestyle=(0, (3, 3))
        )

    ax.set_xlabel(DISTANCE_LABEL)
    ax.set_ylabel("normalized" if normalize else "")
    add_legend(ax)
    return ax
