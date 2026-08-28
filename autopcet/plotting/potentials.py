"""Proton potentials, the vibrational states in them, and how well they fit.

The three near-identical ``plot_states`` helpers the examples used to carry
collapse onto :func:`plot_state_ladder` here, and the two-panel figure examples
1 to 3 each built by hand onto :func:`plot_proton_states`.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .._types import FloatArray, PotentialFunction
from ..constants import MASS_DEUTERON, MASS_PROTON, ROOM_TEMPERATURE
from ..rates import PCET, _require_solved
from ._mpl import add_legend, prepare_axes, prepare_axes_grid
from .style import (
    ENERGY_LABEL,
    MIN_STATE_ALPHA,
    PRODUCT_COLOR,
    PROTON_LABEL,
    REACTANT_COLOR,
    WAVEFUNCTION_FADE,
    WAVEFUNCTION_SCALE,
    distance_colors,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _zero_point_shifts(system: PCET, align: bool) -> tuple[float, float]:
    """Shifts lining the two zero-point levels up, as the examples plot them."""
    if not align:
        return 0.0, 0.0

    gap = float(system.product_energies[0] - system.reactant_energies[0])
    return max(gap, 0.0), max(-gap, 0.0)


def _ladder_extent(
    energies: FloatArray,
    wavefunctions: FloatArray,
    n_states: int,
    shift: float,
    scale: float,
) -> tuple[float, float]:
    """Bottom and top of what :func:`plot_state_ladder` will actually draw."""
    levels = energies[:n_states] + shift
    amplitude = scale * float(np.max(np.abs(wavefunctions[:n_states])))
    return float(np.min(levels)) - amplitude, float(np.max(levels)) + amplitude


def _autoscale_energy(
    panels: Sequence[Axes],
    extents: Sequence[tuple[float, float]],
    margin: float = 0.1,
) -> None:
    """Fit a shared energy axis around the states rather than the diabats.

    A diabatic potential climbs to tens of eV at the edges of the proton grid,
    which leaves the states -- the part worth looking at -- a few pixels tall.
    """
    bottom = min(low for low, _ in extents)
    top = max(high for _, high in extents)
    padding = margin * max(top - bottom, 1e-12)

    for panel in panels:
        panel.set_ylim(bottom - padding, top + padding)


def plot_state_ladder(
    ax: Axes,
    grid: FloatArray,
    energies: FloatArray,
    wavefunctions: FloatArray,
    color: str,
    *,
    n_states: int = 6,
    shift: float = 0.0,
    scale: float = WAVEFUNCTION_SCALE,
    label: str | None = None,
    linestyle: str = "-",
) -> Axes:
    """Draw the lowest vibrational states as wave functions on their own levels.

    Each wave function is scaled onto the energy axis by ``scale`` and flipped
    so its largest lobe points up, then filled down to the level it sits on.
    Higher states fade out, which keeps a crowded ladder readable -- but only
    down to ``MIN_STATE_ALPHA``: fading without a floor asks for a negative
    alpha from the tenth state on, which matplotlib rejects.
    """
    if n_states < 1:
        raise ValueError(f"Need at least one state to draw, got {n_states}.")

    levels = energies[:n_states]
    states = wavefunctions[:n_states]

    for i, (energy, wavefunction) in enumerate(zip(levels, states, strict=True)):
        # flip the wave function so its largest amplitude points up
        sign = 1 if np.abs(np.max(wavefunction)) > np.abs(np.min(wavefunction)) else -1
        level = energy + shift
        curve = level + scale * sign * wavefunction
        ax.plot(
            grid,
            curve,
            linestyle,
            color=color,
            lw=1,
            alpha=max(1 - WAVEFUNCTION_FADE * i, MIN_STATE_ALPHA),
            label=label if i == 0 else None,
        )
        ax.fill_between(grid, curve, level, color=color, alpha=0.4)

    return ax


def plot_proton_states(
    system: PCET,
    axes: Sequence[Axes] | None = None,
    *,
    n_states: int = 6,
    align_zero_point: bool = True,
    scale: float = WAVEFUNCTION_SCALE,
    orientation: Literal["horizontal", "vertical"] = "horizontal",
    autoscale: bool = True,
) -> list[Axes]:
    """The two diabatic proton potentials with their vibrational states.

    Reactant on the first panel and product on the second, sharing both axes.
    With ``align_zero_point`` the two potentials are shifted so their zero-point
    levels line up, which is how the excitation energies driving the rate
    constant are read off. Returns ``[reactant_axes, product_axes]``.

    The diabats climb to tens of eV at the edges of the grid and would squash
    the states into a band a few pixels tall, so ``autoscale`` fits the energy
    axis around the ladder instead. Turn it off to set the limits yourself.
    """
    _require_solved(system)

    shape = (1, 2) if orientation == "horizontal" else (2, 1)
    panels = prepare_axes_grid(
        axes,
        "Plotting proton states",
        *shape,
        sharex=True,
        sharey=True,
        figsize=(9, 4.5) if orientation == "horizontal" else (5, 7),
    )
    reactant_axes, product_axes = panels
    reactant_shift, product_shift = _zero_point_shifts(system, align_zero_point)

    for panel, potential, energies, wavefunctions, colour, shift in (
        (
            reactant_axes,
            system.reactant_potential,
            system.reactant_energies,
            system.reactant_wavefunctions,
            REACTANT_COLOR,
            reactant_shift,
        ),
        (
            product_axes,
            system.product_potential,
            system.product_energies,
            system.product_wavefunctions,
            PRODUCT_COLOR,
            product_shift,
        ),
    ):
        panel.plot(system.rp, potential(system.rp) + shift, color=colour, lw=2)
        plot_state_ladder(
            panel,
            system.rp,
            energies,
            wavefunctions,
            colour,
            n_states=n_states,
            shift=shift,
            scale=scale,
        )
        panel.set_xlabel(PROTON_LABEL)

    reactant_axes.set_ylabel(ENERGY_LABEL)
    if orientation == "vertical":
        product_axes.set_ylabel(ENERGY_LABEL)

    if autoscale:
        _autoscale_energy(
            panels,
            [
                _ladder_extent(
                    system.reactant_energies,
                    system.reactant_wavefunctions,
                    n_states,
                    reactant_shift,
                    scale,
                ),
                _ladder_extent(
                    system.product_energies,
                    system.product_wavefunctions,
                    n_states,
                    product_shift,
                    scale,
                ),
            ],
        )

    return panels


def plot_isotope_states(
    system: PCET,
    axes: Sequence[Axes] | None = None,
    *,
    masses: Sequence[float] = (MASS_PROTON, MASS_DEUTERON),
    labels: Sequence[str] = ("H", "D"),
    temperature: float = ROOM_TEMPERATURE,
    n_states: int = 4,
    scale: float = WAVEFUNCTION_SCALE,
) -> list[Axes]:
    """The same two diabats with one isotope's states drawn over another's.

    The heavier isotope sits lower and is more localized, which is the whole
    story behind a kinetic isotope effect. Each mass is drawn in its own line
    style, dashed after the first.

    This re-solves the proton states once per mass, so afterwards ``system``
    holds the states of the last mass in ``masses``.
    """
    if len(masses) != len(labels):
        raise ValueError(
            f"Got {len(masses)} masses but {len(labels)} labels to name them."
        )

    panels = prepare_axes_grid(
        axes,
        "Plotting isotope states",
        1,
        2,
        sharex=True,
        sharey=True,
        figsize=(9, 4.5),
    )
    reactant_axes, product_axes = panels
    linestyles = ["-", "--", ":", "-."]
    extents: list[tuple[float, float]] = []

    for index, (mass, name) in enumerate(zip(masses, labels, strict=True)):
        system.calculate(mass, temperature=temperature)
        linestyle = linestyles[index % len(linestyles)]

        for panel, energies, wavefunctions, colour in (
            (
                reactant_axes,
                system.reactant_energies,
                system.reactant_wavefunctions,
                REACTANT_COLOR,
            ),
            (
                product_axes,
                system.product_energies,
                system.product_wavefunctions,
                PRODUCT_COLOR,
            ),
        ):
            plot_state_ladder(
                panel,
                system.rp,
                energies,
                wavefunctions,
                colour,
                n_states=n_states,
                scale=scale,
                label=name,
                linestyle=linestyle,
            )
            extents.append(
                _ladder_extent(energies, wavefunctions, n_states, 0.0, scale)
            )

    for panel, potential, colour in (
        (reactant_axes, system.reactant_potential, REACTANT_COLOR),
        (product_axes, system.product_potential, PRODUCT_COLOR),
    ):
        panel.plot(system.rp, potential(system.rp), color=colour, lw=2)
        panel.set_xlabel(PROTON_LABEL)
        add_legend(panel)

    reactant_axes.set_ylabel(ENERGY_LABEL)
    _autoscale_energy(panels, extents)
    return panels


def plot_potential_family(
    rp: FloatArray,
    potentials: Sequence[PotentialFunction],
    ax: Axes | None = None,
    *,
    colors: Sequence[Any] | None = None,
    labels: Sequence[str] | None = None,
    energy_scale: float = 1.0,
    ylabel: str = ENERGY_LABEL,
) -> Axes:
    """A family of proton potentials on one axes, one curve per member.

    The family is usually one potential per donor-acceptor distance, so the
    curves default to the :func:`~autopcet.plotting.style.distance_colors` ramp.
    ``energy_scale`` converts off eV -- pass ``EV_TO_KCAL`` with a matching
    ``ylabel`` to plot in kcal/mol.
    """
    ax = prepare_axes(ax, "Plotting proton potentials")
    ramp = distance_colors(len(potentials)) if colors is None else colors

    if len(ramp) != len(potentials):
        raise ValueError(
            f"Got {len(potentials)} potentials but {len(ramp)} colours for them."
        )

    for index, potential in enumerate(potentials):
        ax.plot(
            rp,
            np.asarray(potential(rp), dtype=np.float64) * energy_scale,
            color=ramp[index],
            lw=2,
            label=None if labels is None else labels[index],
        )

    ax.set_xlabel(PROTON_LABEL)
    ax.set_ylabel(ylabel)
    add_legend(ax)
    return ax


def plot_potential_fit(
    rp: FloatArray,
    energies: FloatArray,
    potential: PotentialFunction,
    axes: Sequence[Axes] | None = None,
    *,
    residuals: bool = True,
    color: str = REACTANT_COLOR,
    n_points: int = 400,
) -> list[Axes]:
    """How closely a fitted potential follows the data it was fitted to.

    Draws the tabulated points with the smooth :func:`~autopcet.fit_potential`
    curve over them and, unless ``residuals`` is off, the fit error underneath.
    A fit that misses the barrier top or the well bottom moves the vibrational
    states, and with them the rate constant, so it is worth looking at.
    """
    panels = prepare_axes_grid(
        axes,
        "Plotting a potential fit",
        2 if residuals else 1,
        1,
        sharex=True,
        figsize=(6, 6) if residuals else (6, 4.5),
        height_ratios=(3, 1) if residuals else None,
    )

    fine = np.linspace(float(np.min(rp)), float(np.max(rp)), n_points)
    fitted = np.asarray(potential(fine), dtype=np.float64)

    panels[0].plot(rp, energies, "o", color=color, ms=5, label="data")
    panels[0].plot(fine, fitted, "-", color=color, lw=2, label="fit")
    panels[0].set_ylabel(ENERGY_LABEL)
    add_legend(panels[0])

    if residuals:
        error = energies - np.asarray(potential(rp), dtype=np.float64)
        panels[1].axhline(0.0, color="k", lw=1, linestyle=(0, (3, 3)))
        panels[1].plot(rp, error, "o-", color=color, ms=4, lw=1)
        panels[1].set_ylabel(r"$\Delta E$ / eV")

    panels[-1].set_xlabel(PROTON_LABEL)
    return panels
