"""Reading Gaussian frequency output and writing the inputs a scan needs.

The fixture log is a minimal synthetic ``freq=HPmodes`` job for a three-atom
donor-acceptor model: three modes in a single printed group, of which only the
first stretches the donor-acceptor axis.
"""

from pathlib import Path

import numpy as np
import pytest

from autopcet import (
    AU_TO_MDYNE_PER_ANGSTROM,
    effective_da_mode,
    read_frequencies,
    write_gaussian_input,
)
from autopcet.gaussian_io import (
    CONSTRAINT_TEMPLATE,
    DEFAULT_OPT_TEMPLATE,
    DEFAULT_SP_TEMPLATE,
)

DATA = Path(__file__).resolve().parent / "data"
LOG = DATA / "freq_hpmodes.log"


def test_read_frequencies_recovers_the_geometry() -> None:
    """The standard-orientation block gives the atoms the modes refer to."""
    frequencies = read_frequencies(LOG)

    assert frequencies.atomic_numbers.tolist() == [8, 8, 1]
    assert frequencies.positions == pytest.approx(
        np.array([[0.0, 0.0, 0.0], [2.7, 0.0, 0.0], [0.98, 0.0, 0.0]])
    )


def test_read_frequencies_recovers_the_modes() -> None:
    """A nonlinear molecule has 3N-6 modes, each with its own printed data."""
    frequencies = read_frequencies(LOG)

    assert frequencies.frequencies == pytest.approx([412.1260, 1637.8279, 3812.4145])
    assert frequencies.reduced_masses == pytest.approx([12.4471, 1.0836, 1.0451])
    assert frequencies.force_constants == pytest.approx([1.2456, 1.7126, 8.9497])

    # the first mode is a pure donor-acceptor stretch: the two heavy atoms move
    # apart along x and the proton stays put
    assert frequencies.normal_modes.shape == (3, 9)
    assert frequencies.normal_modes[0] == pytest.approx(
        [-0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    )


def test_read_frequencies_rejects_a_log_without_frequencies(tmp_path: Path) -> None:
    """A job run without freq=HPmodes has no high-precision modes to read."""
    path = tmp_path / "opt.log"
    path.write_text(
        "                          Standard orientation:\n"
        + "\n" * 4
        + "      1          8           0     0.0   0.0   0.0\n"
        " ---------------------------------------------------------\n"
    )

    with pytest.raises(ValueError, match="freq=HPmodes"):
        read_frequencies(path)


def test_read_frequencies_rejects_a_log_without_a_geometry(tmp_path: Path) -> None:
    """Without a standard orientation there is no axis to project onto."""
    path = tmp_path / "empty.log"
    path.write_text(" Harmonic frequencies (cm**-1)\n")

    with pytest.raises(ValueError, match="standard orientation"):
        read_frequencies(path)


def test_effective_da_mode_isolates_the_donor_acceptor_stretch() -> None:
    """Only modes that stretch the axis contribute, so a pure stretch stands alone."""
    frequencies = read_frequencies(LOG)

    mode = effective_da_mode(
        frequencies.positions,
        0,
        1,
        frequencies.reduced_masses,
        frequencies.force_constants,
        frequencies.normal_modes,
    )

    assert mode.force_constant == pytest.approx(
        frequencies.force_constants[0] / AU_TO_MDYNE_PER_ANGSTROM
    )
    assert mode.reduced_mass == pytest.approx(frequencies.reduced_masses[0])
    # the fixture's k and mu are consistent with its printed frequency
    assert mode.frequency == pytest.approx(frequencies.frequencies[0], rel=1e-4)


def test_effective_da_mode_adds_contributions_as_compliances() -> None:
    """Two equally projecting modes give half the force constant of either."""
    positions = np.array([[0.0, 0.0, 0.0], [2.7, 0.0, 0.0]])
    normal_modes = np.array(
        [[-0.5, 0.0, 0.0, 0.5, 0.0, 0.0], [-0.5, 0.0, 0.0, 0.5, 0.0, 0.0]]
    )
    force_constants = np.array([2.0, 2.0])
    reduced_masses = np.array([8.0, 8.0])

    mode = effective_da_mode(
        positions, 0, 1, reduced_masses, force_constants, normal_modes
    )

    assert mode.force_constant == pytest.approx(1.0 / AU_TO_MDYNE_PER_ANGSTROM)
    assert mode.reduced_mass == pytest.approx(4.0)


def test_effective_da_mode_is_symmetric_in_donor_and_acceptor() -> None:
    """Swapping the two atoms flips the axis but not the projection's magnitude."""
    frequencies = read_frequencies(LOG)
    arguments = (
        frequencies.reduced_masses,
        frequencies.force_constants,
        frequencies.normal_modes,
    )

    forward = effective_da_mode(frequencies.positions, 0, 1, *arguments)
    backward = effective_da_mode(frequencies.positions, 1, 0, *arguments)

    assert forward == pytest.approx(backward)


def test_write_gaussian_input_lays_out_a_single_point(tmp_path: Path) -> None:
    """The header, the geometry, and a terminating blank line."""
    path = tmp_path / "reactant_sp.gjf"
    header = DEFAULT_SP_TEMPLATE.format(state="reactant", charge=0, multiplicity=1)

    write_gaussian_input(
        path, header, ["O", "H"], np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    )

    lines = path.read_text().splitlines()
    assert lines[3] == "# Nosymm B3LYP/6-31+g(d,p)"
    assert lines[7] == "0 1"
    assert lines[8].split() == ["O", "0.00000000", "0.00000000", "0.00000000"]
    assert lines[9].split() == ["H", "1.00000000", "0.00000000", "0.00000000"]
    assert lines[10] == ""


def test_write_gaussian_input_appends_a_constraint_block(tmp_path: Path) -> None:
    """A ModRedundant block follows the geometry, after its blank line."""
    path = tmp_path / "reac_opt.gjf"
    header = DEFAULT_OPT_TEMPLATE.format(state="reactant", charge=0, multiplicity=1)
    trailer = CONSTRAINT_TEMPLATE.format(donor=1, acceptor=3, distance=2.4)

    write_gaussian_input(path, header, ["O"], np.array([[0.0, 0.0, 0.0]]), trailer)

    lines = path.read_text().splitlines()
    assert lines[-4] == ""
    assert lines[-3] == "1   3   =2.40   B"
    assert lines[-2] == "1   3   F"
    assert lines[-1] == ""
