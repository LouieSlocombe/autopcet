"""The vibronic state-pair matrices, drawn as maps rather than ASCII tables.

A golden-rule rate constant is a sum over pairs of reactant and product proton
states, and which pairs carry it is the most informative thing a calculation
has to say. :func:`autopcet.reporting.write_contribution_table` writes those
matrices out as text; these functions put them on an axes.
"""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from .._types import FloatArray
from ..rates import PCET, _require_solved
from ..reporting import contribution_percentages
from ._mpl import prepare_axes, prepare_axes_grid, pyplot

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.image import AxesImage


class _Quantity(NamedTuple):
    """One plottable state-pair matrix, with how it should be drawn."""

    values: Callable[[PCET], FloatArray]
    label: str
    colormap: str
    fmt: str
    diverging: bool = False


QUANTITIES: dict[str, _Quantity] = {
    "contribution": _Quantity(
        contribution_percentages,
        r"contribution to $k_{\rm tot}$ / %",
        "viridis",
        ".1f",
    ),
    "overlap": _Quantity(
        lambda system: np.abs(system.overlaps),
        r"$|S_{uv}|$",
        "magma",
        ".1e",
    ),
    "free_energy": _Quantity(
        lambda system: system.pair_free_energies,
        r"$\Delta G_{uv}$ / eV",
        "coolwarm",
        "+.2f",
        diverging=True,
    ),
    "activation_energy": _Quantity(
        lambda system: system.pair_activation_energies,
        r"$\Delta G^{\ddag}_{uv}$ / eV",
        "cividis",
        ".2f",
    ),
}
"""The state-pair matrices :func:`plot_state_pair_map` knows how to draw."""


def _annotate(ax: Axes, matrix: FloatArray, image: AxesImage, fmt: str) -> None:
    """Write each cell's value into it, in whichever of black or white reads."""
    for u, row in enumerate(matrix):
        for v, value in enumerate(row):
            red, green, blue, _ = image.cmap(image.norm(value))
            luminance = 0.299 * red + 0.587 * green + 0.114 * blue
            ax.text(
                v,
                u,
                format(value, fmt),
                ha="center",
                va="center",
                fontsize=8,
                color="k" if luminance > 0.5 else "w",
            )


def plot_state_pair_map(
    system: PCET,
    quantity: str = "contribution",
    ax: Axes | None = None,
    *,
    n_states: int | None = None,
    annotate: bool = True,
    colorbar: bool = True,
) -> Axes:
    """One reactant/product state-pair matrix as an annotated heat map.

    ``quantity`` picks the matrix, and names the columns
    :func:`autopcet.reporting.write_contribution_table` prints:
    ``"contribution"`` (the percentage of the total rate constant each pair
    carries), ``"overlap"``, ``"free_energy"`` (``Delta G_uv``), or
    ``"activation_energy"`` (``Delta G^#_uv``). The one difference is that
    ``"overlap"`` is drawn here as ``|S_uv|``, which reads better on a linear
    colour scale, where the table prints ``|S_uv|^2``.

    Reactant state ``u`` runs up the vertical axis and product state ``v``
    along the horizontal one, so the bottom left cell is the ground-to-ground
    pair.
    """
    _require_solved(system)

    if quantity not in QUANTITIES:
        raise ValueError(
            f"Unknown quantity '{quantity}'. Pick one of "
            f"{', '.join(sorted(QUANTITIES))}."
        )

    spec = QUANTITIES[quantity]
    shown = system.n_states if n_states is None else n_states
    matrix = np.asarray(spec.values(system), dtype=np.float64)[:shown, :shown]

    limit = float(np.max(np.abs(matrix))) if spec.diverging else 0.0
    ax = prepare_axes(ax, "Plotting a state-pair map", figsize=(5.5, 4.5))
    image = ax.imshow(
        matrix,
        origin="lower",
        cmap=spec.colormap,
        vmin=-limit if spec.diverging else None,
        vmax=limit if spec.diverging else None,
    )

    ax.set_xlabel("product state $v$")
    ax.set_ylabel("reactant state $u$")
    ax.set_xticks(range(len(matrix)))
    ax.set_yticks(range(len(matrix)))
    ax.tick_params(length=0)

    if annotate:
        _annotate(ax, matrix, image, spec.fmt)

    if colorbar:
        pyplot().colorbar(image, ax=ax, label=spec.label)
    else:
        ax.set_title(spec.label)

    return ax


def plot_state_pair_grid(
    system: PCET,
    quantities: Sequence[str] = tuple(QUANTITIES),
    axes: Sequence[Axes] | None = None,
    *,
    n_states: int | None = None,
    annotate: bool = True,
) -> list[Axes]:
    """Every state-pair matrix side by side, two to a row.

    The overview to reach for first: it shows at a glance whether the rate is
    carried by the ground-to-ground pair or by an excited one, and which of the
    overlap and the activation energy is deciding that.
    """
    if not quantities:
        raise ValueError("Need at least one quantity to draw.")

    columns = min(2, len(quantities))
    rows = -(-len(quantities) // columns)
    panels = prepare_axes_grid(
        axes,
        "Plotting state-pair maps",
        rows,
        columns,
        figsize=(5.5 * columns, 4.5 * rows),
    )

    for panel, quantity in zip(panels, quantities, strict=False):
        plot_state_pair_map(
            system, quantity, panel, n_states=n_states, annotate=annotate
        )

    # a grid wider than the number of quantities leaves empty panels behind
    for panel in panels[len(quantities) :]:
        panel.set_axis_off()

    return panels


def plot_populations(
    system: PCET,
    ax: Axes | None = None,
    *,
    n_states: int | None = None,
    color: str = "tab:blue",
) -> Axes:
    """Boltzmann populations of the reactant proton states, on a log axis.

    How steeply these fall away from the ground state decides how much of the
    rate constant excited reactant states can carry.
    """
    _require_solved(system)

    shown = system.n_states if n_states is None else n_states
    populations = system.populations[:shown]

    ax = prepare_axes(ax, "Plotting state populations")
    ax.bar(range(len(populations)), populations, color=color)
    ax.set_yscale("log")
    ax.set_xlabel("reactant state $u$")
    ax.set_ylabel("$P_u$")
    ax.set_xticks(range(len(populations)))
    return ax
