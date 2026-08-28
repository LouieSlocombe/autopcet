"""The vibronic state-pair results, written out as text.

A golden-rule rate constant is a sum over pairs of reactant and product proton
states, and which pairs carry it is the most informative thing a calculation
has to say. :func:`format_contribution_table` lays that sum out as a table;
:func:`autopcet.plotting.plot_state_pair_map` draws the same numbers as a heat
map. Nothing here needs matplotlib.
"""

from typing import TextIO

import numpy as np

from ._types import FloatArray
from .rates import PCET, _require_solved

_RULE_WIDTH = 125
"""Width of the rules :func:`format_contribution_table` draws."""


def contribution_percentages(system: PCET) -> FloatArray:
    """Each state pair's share of the total rate constant, as a percentage."""
    if system.total_rate_constant <= 0.0:
        raise ValueError(
            "The total rate constant is zero, so there are no shares of it to "
            "report. Check that 'calculate' ran and did not underflow."
        )
    percentages: FloatArray = 100 * system.rate_contributions
    return percentages / system.total_rate_constant


def format_contribution_table(
    system: PCET,
    *,
    label: str | None = None,
    n_states: int | None = None,
) -> str:
    """The vibronic state-pair results as a table, one row per pair.

    Each row carries the reactant state's Boltzmann population, the squared
    overlap ``|S_uv|^2`` that multiplies into the pair's rate contribution, the
    pair free energy and Marcus activation energy, and the percentage of the
    total rate constant the pair carries.

    ``label`` titles the block, which is how a table for H is told from one for
    D. ``n_states`` truncates to the lowest states, defaulting to every state
    solved for; asking for more than were solved for shows all of them rather
    than failing, exactly as it does for
    :func:`autopcet.plotting.plot_state_pair_map`.

    That map draws the same four matrices, except that its ``"overlap"``
    quantity is ``|S_uv|`` rather than the ``|S_uv|^2`` tabulated here: a heat
    map reads better on the amplitude, while the column beside ``% Contrib.``
    is more useful as the factor that actually enters the rate.
    """
    _require_solved(system)

    shown = system.n_states if n_states is None else n_states
    overlaps = np.asarray(system.overlaps, dtype=np.float64)[:shown, :shown] ** 2
    free_energies = system.pair_free_energies[:shown, :shown]
    activation_energies = system.pair_activation_energies[:shown, :shown]
    percentages = contribution_percentages(system)[:shown, :shown]

    lines = ["\n"]
    if label is not None:
        lines.append(f"{label}\n")
    lines.append("=" * _RULE_WIDTH + "\n")
    lines.append(
        "(u, v)\t\tP_u\t\t\t|S_uv|^2\t\tDelta G_uv / eV\t\t"
        "Delta G^#_uv / eV\t% Contrib.\n"
    )
    lines.append("-" * _RULE_WIDTH + "\n")
    for u in range(len(overlaps)):
        for v in range(len(overlaps)):
            lines.append(
                f"({u:d}, {v:d})\t\t{system.populations[u]:.3e}\t\t"
                f"{overlaps[u, v]:.3e}\t\t"
                f"{free_energies[u, v]:+.3f}\t\t\t"
                f"{activation_energies[u, v]:.3f}\t\t\t"
                f"{percentages[u, v]:.1f}\n"
            )
    lines.append("=" * _RULE_WIDTH + "\n\n")
    return "".join(lines)


def write_contribution_table(
    stream: TextIO,
    system: PCET,
    *,
    label: str | None = None,
    n_states: int | None = None,
) -> None:
    """Write :func:`format_contribution_table` to ``stream``."""
    stream.write(format_contribution_table(system, label=label, n_states=n_states))
