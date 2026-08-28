"""Tests for the text tables in :mod:`autopcet.reporting`.

These assert on the structure of the formatted block -- how many data rows it
holds, which columns it names, where the rules and the label sit -- rather than
on an exact transcript, so a change in the third significant digit of a rate
constant does not fail them.
"""

import io

import numpy as np
import pytest

from autopcet import PCET, contribution_percentages, format_contribution_table
from autopcet.plotting import QUANTITIES
from autopcet.reporting import _RULE_WIDTH, write_contribution_table


def data_rows(table: str) -> list[str]:
    """The ``(u, v)`` rows of a formatted table, without its header or rules."""
    return [
        line
        for line in table.splitlines()
        if line.startswith("(") and not line.startswith("(u, v)")
    ]


def test_a_table_has_one_row_per_state_pair(calculated_system: PCET) -> None:
    """Every pair of the states asked for gets a row of its own."""
    assert len(data_rows(format_contribution_table(calculated_system, n_states=3))) == 9


def test_a_table_defaults_to_every_state_solved_for(calculated_system: PCET) -> None:
    """Left to itself the table reports the whole matrix."""
    table = format_contribution_table(calculated_system)

    assert len(data_rows(table)) == calculated_system.n_states**2


def test_a_table_truncates_rather_than_overrunning(calculated_system: PCET) -> None:
    """Asking for more states than were solved for shows all of them."""
    table = format_contribution_table(calculated_system, n_states=99)

    assert len(data_rows(table)) == calculated_system.n_states**2


def test_a_table_names_every_column_it_prints(calculated_system: PCET) -> None:
    """The header names the population and all four state-pair matrices."""
    header = next(
        line
        for line in format_contribution_table(calculated_system).splitlines()
        if line.startswith("(u, v)")
    )

    for column in ("P_u", "|S_uv|^2", "Delta G_uv", "Delta G^#_uv", "% Contrib."):
        assert column in header


def test_the_table_and_the_map_report_the_same_quantities(
    calculated_system: PCET,
) -> None:
    """The map draws what the table prints, so their columns must line up.

    The one deliberate difference is the overlap: ``|S_uv|`` on a heat map,
    ``|S_uv|^2`` in the column that explains ``% Contrib.``.
    """
    assert set(QUANTITIES) == {
        "contribution",
        "overlap",
        "free_energy",
        "activation_energy",
    }

    percentages = QUANTITIES["contribution"].values(calculated_system)
    assert percentages == pytest.approx(contribution_percentages(calculated_system))


def test_a_label_titles_the_block(calculated_system: PCET) -> None:
    """The isotope label goes above the top rule, on a line of its own."""
    lines = format_contribution_table(calculated_system, label="D").splitlines()

    assert lines[1] == "D"
    assert set(lines[2]) == {"="}


def test_an_unlabelled_block_opens_on_its_rule(calculated_system: PCET) -> None:
    """Without a label there is no title line to push the rule down."""
    lines = format_contribution_table(calculated_system).splitlines()

    assert set(lines[1]) == {"="}


def test_no_line_overruns_the_rules(calculated_system: PCET) -> None:
    """The rules are the width of the table, so nothing may stick out past them."""
    lines = format_contribution_table(calculated_system, label="H").splitlines()

    assert {len(line) for line in lines if set(line) in ({"="}, {"-"})} == {_RULE_WIDTH}
    assert max(len(line) for line in lines) <= _RULE_WIDTH


def test_the_contributions_account_for_the_whole_rate_constant(
    calculated_system: PCET,
) -> None:
    """Every pair's share of the rate constant, summed, is all of it."""
    assert np.sum(contribution_percentages(calculated_system)) == pytest.approx(100.0)


def test_writing_a_table_puts_it_on_the_stream(calculated_system: PCET) -> None:
    """The writer is the formatter plus a ``write``, and adds nothing else."""
    stream = io.StringIO()
    write_contribution_table(stream, calculated_system, label="H", n_states=2)

    assert stream.getvalue() == format_contribution_table(
        calculated_system, label="H", n_states=2
    )


def test_an_unsolved_system_says_so(system: PCET) -> None:
    """Reporting on a system before ``calculate`` ran is a clear error."""
    with pytest.raises(ValueError, match="no proton states yet"):
        format_contribution_table(system)


def test_a_rate_constant_that_underflowed_says_so(calculated_system: PCET) -> None:
    """There are no shares of a total rate constant of zero to report."""
    calculated_system.total_rate_constant = 0.0

    with pytest.raises(ValueError, match="total rate constant is zero"):
        contribution_percentages(calculated_system)
