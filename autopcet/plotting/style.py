"""The house style every figure in the package shares.

The examples used to repeat the same figure settings at a dozen sites each --
``dpi=300``, 14 pt ticks against 16 pt labels, blue for the reactant state and
red for the product, the rainbow ramp over donor-acceptor distances. They all
live here now, as rcParams and as named constants the plotting functions
default to.
"""

import colorsys
from contextlib import AbstractContextManager
from typing import Any

import numpy as np

from ._mpl import pyplot

REACTANT_COLOR = "tab:blue"
"""Colour of the reactant diabatic state, blue by convention."""

PRODUCT_COLOR = "tab:red"
"""Colour of the product diabatic state, red by convention."""

ADIABAT_COLOR = "k"
"""Colour of the adiabatic potentials and of anything overlaid on the diabats."""

RATE_COLOR = "#ff7700"
"""Colour of a rate constant plotted against the donor-acceptor distance."""

DISTRIBUTION_COLOR = (0.1, 0.6, 0.2)
"""Colour of the donor-acceptor distribution ``P(R)``."""

WAVEFUNCTION_SCALE = 0.06
"""Default scaling of a wave function drawn on an energy axis, in eV."""

WAVEFUNCTION_FADE = 0.12
"""How much fainter each successive vibrational state is drawn."""

MIN_STATE_ALPHA = 0.2
"""Floor on that fading. Without it a tenth state would ask for a negative
alpha, which matplotlib rejects outright."""

PROTON_LABEL = r"$r_{\rm p}\ /\ \rm\AA$"
"""Axis label for the proton coordinate."""

DISTANCE_LABEL = r"$R\ /\ \rm\AA$"
"""Axis label for the proton donor-acceptor distance."""

ENERGY_LABEL = r"$E$ / eV"
"""Axis label for an energy in electronvolts."""

RATE_LABEL = r"$k$ / s$^{-1}$"
"""Axis label for a rate constant."""

STYLE: dict[str, Any] = {
    "axes.labelsize": 16,
    "axes.linewidth": 1.2,
    "axes.titlesize": 18,
    "figure.figsize": (6.0, 4.5),
    "font.size": 14,
    "legend.fontsize": 14,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "xtick.direction": "in",
    "xtick.labelsize": 14,
    "ytick.direction": "in",
    "ytick.labelsize": 14,
}
"""The rcParams behind every figure here, matching what the examples set by hand."""


def use_style(**overrides: Any) -> None:
    """Apply the house rcParams globally, for every figure drawn from now on."""
    plt = pyplot("Styling a figure")
    plt.rcParams.update({**STYLE, **overrides})


def style_context(**overrides: Any) -> AbstractContextManager[None]:
    """The house rcParams as a context manager, leaving the globals alone.

    Prefer this over :func:`use_style` inside a library or a notebook shared
    with other figures::

        with style_context(**{"figure.figsize": (8, 4)}):
            plot_proton_states(system)
    """
    plt = pyplot("Styling a figure")
    context: AbstractContextManager[None] = plt.rc_context({**STYLE, **overrides})
    return context


def distance_colors(n: int) -> list[tuple[float, float, float]]:
    """``n`` colours ramping red through blue, one per donor-acceptor distance.

    The ramp the examples build by hand: it runs over four fifths of the hue
    circle, so the shortest and longest distances stay easy to tell apart.
    """
    if n < 1:
        raise ValueError(f"Need at least one colour, got {n}.")

    return [colorsys.hls_to_rgb(hue, 0.5, 0.85) for hue in np.linspace(0.0, 0.8, n)]
