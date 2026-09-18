"""Rate constants swept over temperature and over the driving force.

The sweeps themselves live in :mod:`autopcet.rates` -- :func:`~autopcet.
temperature_sweep` and :func:`~autopcet.driving_force_sweep` -- so they stay
usable without matplotlib. These functions take the arrays those return.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._types import FloatArray
from ..constants import BOLTZMANN
from ._mpl import add_legend, prepare_axes
from .style import RATE_COLOR, RATE_LABEL

if TYPE_CHECKING:
    from matplotlib.axes import Axes

INVERSE_TEMPERATURE_LABEL = r"$1000\ /\ T$ / K$^{-1}$"
"""Axis label for the reciprocal temperature an Arrhenius plot runs on."""


def _require_positive(rates: FloatArray, name: str) -> FloatArray:
    """Reject rate constants a logarithm cannot be taken of."""
    values = np.asarray(rates, dtype=np.float64)
    if np.any(values <= 0):
        raise ValueError(
            f"'{name}' has to be positive everywhere to be plotted on a "
            "logarithmic axis; some of it underflowed to zero or below."
        )
    return values


def plot_arrhenius(
    temperatures: FloatArray,
    rates: FloatArray,
    ax: Axes | None = None,
    *,
    label: str | None = None,
    color: str = RATE_COLOR,
    fit: bool = True,
) -> Axes:
    """``ln k`` against ``1000 / T``, the classic Arrhenius plot.

    A vibronically nonadiabatic rate constant is not an Arrhenius rate, so this
    curve is generally not straight; how far it bends, and what apparent
    activation energy the straight line through it reports, is the point.
    With ``fit`` the least-squares line is drawn and its ``E_a`` named in the
    legend.
    """
    temperature = np.asarray(temperatures, dtype=np.float64)
    log_rate = np.log(_require_positive(rates, "rates"))
    inverse = 1000.0 / temperature

    ax = prepare_axes(ax, "Plotting an Arrhenius plot")
    ax.plot(inverse, log_rate, "o", color=color, ms=5, label=label)

    if fit:
        slope, intercept = np.polyfit(inverse, log_rate, 1)
        # d(ln k)/d(1/T) = -E_a / k_B, and the axis carries a factor of 1000
        activation_energy = -slope * 1000.0 * BOLTZMANN
        ax.plot(
            inverse,
            slope * inverse + intercept,
            "-",
            color=color,
            lw=1.5,
            alpha=0.7,
            label=rf"$E_{{\rm a}} = {activation_energy:.3f}$ eV",
        )

    ax.set_xlabel(INVERSE_TEMPERATURE_LABEL)
    ax.set_ylabel(r"$\ln(k\ /\ {\rm s}^{-1})$")
    add_legend(ax)
    return ax


def plot_kie_vs_temperature(
    temperatures: FloatArray,
    rates_light: FloatArray,
    rates_heavy: FloatArray,
    ax: Axes | None = None,
    *,
    label: str | None = None,
    color: str = RATE_COLOR,
) -> Axes:
    """The kinetic isotope effect against temperature.

    A dotted line marks unity, where the two isotopes react at the same rate.
    A KIE that grows as the temperature falls is the signature of a reaction
    running through the tunnelling states rather than over the barrier.
    """
    temperature = np.asarray(temperatures, dtype=np.float64)
    light = _require_positive(rates_light, "rates_light")
    heavy = _require_positive(rates_heavy, "rates_heavy")

    ax = prepare_axes(ax, "Plotting a kinetic isotope effect")
    ax.axhline(1.0, color="k", lw=1, linestyle=(0, (3, 3)))
    ax.plot(temperature, light / heavy, "o-", color=color, ms=5, label=label)

    ax.set_xlabel(r"$T$ / K")
    ax.set_ylabel(r"KIE, $k_{\rm H}\ /\ k_{\rm D}$")
    add_legend(ax)
    return ax


def plot_rate_vs_driving_force(
    reaction_free_energies: FloatArray,
    rates: FloatArray,
    ax: Axes | None = None,
    *,
    reorganization_energy: float | None = None,
    label: str | None = None,
    color: str = RATE_COLOR,
) -> Axes:
    """The Marcus curve: rate constant against reaction free energy.

    Pass ``reorganization_energy`` to mark ``-lambda``, the driving force at
    which the classical activation energy vanishes. Everything beyond it is the
    inverted region, where a more exergonic reaction goes more slowly -- though
    the excited product states a vibronic treatment includes fill that dip in.
    """
    free_energy = np.asarray(reaction_free_energies, dtype=np.float64)

    ax = prepare_axes(ax, "Plotting a Marcus curve")
    ax.plot(
        free_energy,
        _require_positive(rates, "rates"),
        "-",
        color=color,
        lw=2,
        label=label,
    )
    ax.set_yscale("log")

    if reorganization_energy is not None:
        ax.axvline(
            -reorganization_energy,
            color="k",
            lw=1.5,
            linestyle=(0, (3, 3)),
            label=r"$-\lambda$",
        )

    ax.set_xlabel(r"$\Delta G^\circ$ / eV")
    ax.set_ylabel(RATE_LABEL)
    add_legend(ax)
    return ax
