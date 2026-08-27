"""Nonadiabaticity, the double layer, and the scans a calculation starts from.

The figures examples 4 to 6 draw by hand, or in example 6's case never got
around to drawing: the diabatic and adiabatic proton potentials behind a
:class:`~autopcet.KappaCoupling` analysis, the interfacial potential drop of an
EDL model, and the raw energies coming back off a donor-acceptor or proton
scan.
"""

from typing import TYPE_CHECKING, Any

import numpy as np

from .._types import FloatArray, ScalarOrArrayFunction
from ..ase_io import DistanceScan, ProtonScan
from ..kappa import KappaCoupling
from ._mpl import add_legend, prepare_axes
from .style import (
    ADIABAT_COLOR,
    DISTANCE_LABEL,
    ENERGY_LABEL,
    PRODUCT_COLOR,
    PROTON_LABEL,
    REACTANT_COLOR,
    WAVEFUNCTION_SCALE,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _require_calculated(system: KappaCoupling) -> None:
    """Fail clearly when ``calculate`` has not run on a coupling analysis yet."""
    if not hasattr(system, "ground_adiabat"):
        raise ValueError(
            "This KappaCoupling analysis has no results yet. Call "
            "'system.calculate(...)' before plotting it."
        )


def _draw_diabats(ax: Axes, system: KappaCoupling) -> None:
    """The two aligned diabatic potentials, blue reactant over red product."""
    ax.plot(system.rp, system.shifted_reactant_potential, color=REACTANT_COLOR, lw=2)
    ax.plot(system.rp, system.shifted_product_potential, color=PRODUCT_COLOR, lw=2)
    ax.set_xlabel(PROTON_LABEL)
    ax.set_ylabel(ENERGY_LABEL)


def _draw_ground_state(
    ax: Axes,
    grid: FloatArray,
    energy: float,
    wavefunction: FloatArray,
    color: str,
    scale: float,
) -> None:
    """One wave function sitting on its own level, flipped largest-lobe-up."""
    sign = 1 if np.abs(np.max(wavefunction)) > np.abs(np.min(wavefunction)) else -1
    curve = energy + scale * sign * wavefunction
    ax.plot(grid, curve, color=color, lw=1)
    ax.fill_between(grid, curve, energy, color=color, alpha=0.4)


def plot_crossing(
    system: KappaCoupling,
    ax: Axes | None = None,
    *,
    tangent_half_width: float = 0.1,
    show_states: bool = True,
    scale: float = WAVEFUNCTION_SCALE,
) -> Axes:
    """The aligned diabats, where they cross, and their slopes through it.

    The crossing point and the two slopes are what the Georgievskii-Stuchebrukhov
    analysis reads the proton tunnelling time off; the ground states drawn on
    their levels show how much amplitude is actually there to tunnel.
    """
    _require_calculated(system)

    ax = prepare_axes(ax, "Plotting a potential crossing")
    _draw_diabats(ax, system)

    ax.plot(
        system.crossing_rp,
        system.crossing_energy,
        "o",
        ms=5,
        mew=2,
        mfc=ADIABAT_COLOR,
        mec=ADIABAT_COLOR,
    )

    tangent_rp = np.linspace(
        system.crossing_rp - tangent_half_width,
        system.crossing_rp + tangent_half_width,
        100,
    )
    for slope in (system.reactant_slope, system.product_slope):
        tangent = slope * (tangent_rp - system.crossing_rp) + system.crossing_energy
        ax.plot(tangent_rp, tangent, "--", color=ADIABAT_COLOR, lw=1.5)

    if show_states:
        _draw_ground_state(
            ax,
            system.rp,
            float(system.shifted_reactant_energies[0]),
            system.reactant_wavefunctions[0],
            REACTANT_COLOR,
            scale,
        )
        _draw_ground_state(
            ax,
            system.rp,
            float(system.shifted_product_energies[0]),
            system.product_wavefunctions[0],
            PRODUCT_COLOR,
            scale,
        )

    return ax


def plot_diabats_and_adiabats(
    system: KappaCoupling,
    ax: Axes | None = None,
    *,
    annotate_splitting: bool = True,
) -> Axes:
    """The aligned diabats with the two adiabatic potentials over them.

    The gap between the ground and excited adiabats at the crossing is twice
    the adiabatic vibronic coupling, which ``annotate_splitting`` names in the
    legend.
    """
    _require_calculated(system)

    ax = prepare_axes(ax, "Plotting adiabatic potentials")
    _draw_diabats(ax, system)

    label = (
        rf"$V^{{\rm ad}} = {system.v_adiabatic * 1000:.2f}$ meV"
        if annotate_splitting
        else None
    )
    ax.plot(
        system.rp, system.ground_adiabat, "--", color=ADIABAT_COLOR, lw=1.5, label=label
    )
    ax.plot(system.rp, system.excited_adiabat, "--", color=ADIABAT_COLOR, lw=1.5)

    add_legend(ax)
    return ax


def plot_edl_profile(
    potential_drop: ScalarOrArrayFunction,
    distances: FloatArray,
    ax: Axes | None = None,
    *,
    d_ihl: float | None = None,
    d_ohl: float | None = None,
    label: str | None = None,
    color: Any = None,
) -> Axes:
    """The interfacial potential drop of an EDL model against distance.

    ``potential_drop`` is what :func:`~autopcet.make_edl_model` returns. Given
    ``d_ihl`` and ``d_ohl`` the inner and outer Helmholtz planes are marked, so
    it is clear which part of the drop the reaction actually sits in.

    Call it once per applied potential to build up the family of curves example
    4 draws.
    """
    ax = prepare_axes(ax, "Plotting an EDL model", figsize=(6, 3.5))
    ax.plot(
        distances,
        np.asarray(potential_drop(distances), dtype=np.float64),
        "-",
        lw=1.5,
        color=color,
        label=label,
    )

    if d_ihl is not None:
        ax.axvline(d_ihl, lw=1.5, color="k", linestyle=(0, (3, 3)))
    if d_ohl is not None:
        if d_ihl is None:
            raise ValueError("'d_ohl' is measured from 'd_ihl', so it needs one.")
        ax.axvline(d_ihl + d_ohl, lw=1.5, color="k", linestyle=(0, (3, 3)))

    ax.set_xlabel(DISTANCE_LABEL)
    ax.set_ylabel(r"$\phi(R, E)$ / V")
    add_legend(ax, frameon=True, framealpha=1.0)
    return ax


def plot_distance_scan(
    scan: DistanceScan,
    ax: Axes | None = None,
    *,
    relative: bool = True,
    label: str | None = None,
    color: str = REACTANT_COLOR,
) -> Axes:
    """Energy against the frozen donor-acceptor distance it was optimized at.

    The minimum of this curve is the equilibrium distance the harmonic
    ``P(R)`` of :func:`~autopcet.donor_acceptor_distribution` is built around.
    """
    energies = np.asarray(scan.energies, dtype=np.float64)
    if relative:
        energies = energies - energies.min()

    ax = prepare_axes(ax, "Plotting a distance scan")
    ax.plot(scan.distances, energies, "o-", color=color, ms=5, label=label)
    ax.set_xlabel(DISTANCE_LABEL)
    ax.set_ylabel(r"$\Delta E$ / eV" if relative else ENERGY_LABEL)
    add_legend(ax)
    return ax


def plot_proton_scan(
    scan: ProtonScan,
    ax: Axes | None = None,
    *,
    relative: bool = True,
    label: str | None = None,
    color: str = REACTANT_COLOR,
) -> Axes:
    """A proton potential as the scan came back off the calculator.

    Worth looking at before fitting it: a scan that never climbs the far wall,
    or one with a point that failed to converge, makes a fit that puts the
    vibrational states in the wrong place.
    """
    energies = scan.relative_energies if relative else scan.energies

    ax = prepare_axes(ax, "Plotting a proton scan")
    ax.plot(
        scan.offsets,
        np.asarray(energies, dtype=np.float64),
        "o-",
        color=color,
        ms=5,
        label=label,
    )
    ax.set_xlabel(PROTON_LABEL)
    ax.set_ylabel(r"$\Delta E$ / eV" if relative else ENERGY_LABEL)
    add_legend(ax)
    return ax
