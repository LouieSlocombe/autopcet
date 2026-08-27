"""Reading Gaussian output and writing the input files a proton scan needs.

Two things are read back from a Gaussian ``freq=HPmodes`` job: the optimized
geometry and the normal modes. Projecting those modes onto the proton
donor-acceptor axis gives the effective force constant of the donor-acceptor
mode, which sets the width of the ``P(R)`` distribution the rate constant is
averaged over (see :func:`autopcet.rates.donor_acceptor_distribution`).

The input writers generate the constrained optimizations that scan the
donor-acceptor distance and the single points that scan the proton along its
transfer axis.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from ._types import FloatArray, IntArray
from .constants import (
    ANGSTROM_TO_CM,
    AU_TIME_TO_SECONDS,
    DALTON_TO_ELECTRON_MASS,
    MDYNE_PER_ANGSTROM_TO_AU,
    SPEED_OF_LIGHT,
)

MODES_PER_GROUP = 5
"""With ``HPmodes`` set, Gaussian prints the high-precision modes five per group."""

DEFAULT_OPT_TEMPLATE = """%chk={state}.chk
%nprocshared=24
%mem=80GB
# B3LYP/6-31+g(d,p) opt=(ModRedundant)

{state}

{charge} {multiplicity}
"""
"""Header for the constrained optimizations that scan the donor-acceptor distance."""

DEFAULT_SP_TEMPLATE = """%chk={state}.chk
%nprocshared=24
%mem=60GB
# Nosymm B3LYP/6-31+g(d,p)

{state} proton potential

{charge} {multiplicity}
"""
"""Header for the single points that scan the proton along its transfer axis.

``Nosymm`` is essential: without it Gaussian reorients the molecule and the
proton no longer sits where the scan put it.
"""

CONSTRAINT_TEMPLATE = """{donor}   {acceptor}   ={distance:.2f}   B
{donor}   {acceptor}   F
"""
"""ModRedundant block freezing the donor-acceptor distance. Indices are 1-based."""


@dataclass(frozen=True)
class GaussianFrequencies:
    """Geometry and normal modes read from a Gaussian ``freq=HPmodes`` job."""

    atomic_numbers: IntArray
    positions: FloatArray
    """Standard-orientation geometry in angstrom, shape ``(n_atoms, 3)``."""

    frequencies: FloatArray
    """Harmonic frequencies in cm^-1."""

    reduced_masses: FloatArray
    """Reduced masses in amu."""

    force_constants: FloatArray
    """Force constants in mDyne/A, as Gaussian prints them."""

    normal_modes: FloatArray
    """Cartesian displacements, shape ``(n_modes, 3 * n_atoms)``."""


class EffectiveMode(NamedTuple):
    """The donor-acceptor mode projected out of a full set of normal modes."""

    force_constant: float
    """Effective force constant in atomic units (hartree/bohr^2)."""

    reduced_mass: float
    """Effective reduced mass in amu."""

    frequency: float
    """Effective frequency in cm^-1."""


def read_frequencies(log_path: str | Path) -> GaussianFrequencies:
    """Read the geometry and normal modes of a Gaussian frequency job.

    The job must have been run with ``#P`` and ``freq=HPmodes``; the parser
    looks for the high-precision normal mode block those settings produce. The
    molecule is assumed to be nonlinear, so it has ``3 * n_atoms - 6`` modes.
    """
    lines = Path(log_path).read_text().splitlines()

    geometry_start = None
    frequency_start = None
    for i, line in enumerate(lines):
        if "Standard orientation" in line:
            geometry_start = i
        if line.startswith(" Harmonic frequencies"):
            # with HPmodes the high precision normal modes are printed first, so
            # only the first such line is of interest
            frequency_start = i
            break

    if geometry_start is None:
        raise ValueError(f"No standard orientation geometry found in {log_path}.")
    if frequency_start is None:
        raise ValueError(
            f"No harmonic frequencies found in {log_path}. The job must be run "
            "with '#P' and 'freq=HPmodes'."
        )

    atomic_numbers = []
    positions = []
    for line in lines[geometry_start + 5 :]:
        if line.lstrip().startswith("---"):
            break
        fields = line.split()
        atomic_numbers.append(int(fields[1]))
        positions.append([float(value) for value in fields[3:6]])

    n_atoms = len(atomic_numbers)
    n_modes = 3 * n_atoms - 6
    n_groups = -(-n_modes // MODES_PER_GROUP)  # ceiling division

    frequencies = np.zeros(n_modes)
    reduced_masses = np.zeros(n_modes)
    force_constants = np.zeros(n_modes)
    normal_modes = np.zeros((n_modes, 3 * n_atoms))

    line_index = frequency_start + 4
    for _ in range(n_groups):
        mode_indices = [int(value) - 1 for value in lines[line_index].split()]
        frequencies[mode_indices] = [
            float(value) for value in lines[line_index + 2].split()[2:]
        ]
        reduced_masses[mode_indices] = [
            float(value) for value in lines[line_index + 3].split()[3:]
        ]
        force_constants[mode_indices] = [
            float(value) for value in lines[line_index + 4].split()[3:]
        ]

        for j in range(3 * n_atoms):
            normal_modes[mode_indices, j] = [
                float(value) for value in lines[line_index + 7 + j].split()[3:]
            ]

        line_index += 7 + n_atoms * 3

    return GaussianFrequencies(
        atomic_numbers=np.array(atomic_numbers),
        positions=np.array(positions),
        frequencies=frequencies,
        reduced_masses=reduced_masses,
        force_constants=force_constants,
        normal_modes=normal_modes,
    )


def effective_da_mode(
    positions: FloatArray,
    donor: int,
    acceptor: int,
    reduced_masses: FloatArray,
    force_constants: FloatArray,
    normal_modes: FloatArray,
) -> EffectiveMode:
    """Project the normal modes onto the proton donor-acceptor axis.

    Each mode contributes in proportion to how much it stretches that axis, and
    the effective force constant and reduced mass add as compliances. Takes
    ``force_constants`` in mDyne/A and ``reduced_masses`` in amu, as
    :func:`read_frequencies` returns them.
    """
    force_constants_au = force_constants * MDYNE_PER_ANGSTROM_TO_AU

    # unit vector connecting the proton donor and acceptor
    axis = positions[acceptor] - positions[donor]
    axis = axis / np.linalg.norm(axis)

    # how much each normal mode stretches the donor-acceptor axis
    donor_slice = slice(donor * 3, donor * 3 + 3)
    acceptor_slice = slice(acceptor * 3, acceptor * 3 + 3)
    weights = (normal_modes[:, acceptor_slice] - normal_modes[:, donor_slice]) @ axis

    force_constant = 1 / np.sum(weights * weights / force_constants_au)
    reduced_mass = 1 / np.sum(weights * weights / reduced_masses)

    # the package's speed of light is in A/s, so convert it to cm/s for a wavenumber
    speed_of_light_cm = SPEED_OF_LIGHT * ANGSTROM_TO_CM
    frequency = (
        math.sqrt(force_constant / reduced_mass / DALTON_TO_ELECTRON_MASS)
        / AU_TIME_TO_SECONDS
        / speed_of_light_cm
        / 2
        / math.pi
    )

    return EffectiveMode(float(force_constant), float(reduced_mass), frequency)


def write_gaussian_input(
    path: str | Path,
    header: str,
    symbols: Sequence[str],
    positions: FloatArray,
    trailer: str = "",
) -> None:
    """Write a Gaussian input file: ``header``, the geometry, then ``trailer``.

    ``header`` is the complete route/title/charge block, already formatted.
    Gaussian expects a blank line after the geometry and after any trailing
    block, so both are terminated here.
    """
    with open(path, "w") as gaussian_input:
        gaussian_input.write(header)
        for symbol, (x, y, z) in zip(symbols, positions, strict=True):
            gaussian_input.write(f"{symbol:<2s} {x:15.8f} {y:15.8f} {z:15.8f}\n")
        gaussian_input.write("\n")
        if trailer:
            gaussian_input.write(trailer)
            gaussian_input.write("\n")
