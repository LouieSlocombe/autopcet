"""Compute the effective proton donor-acceptor force constant, reduced mass, and
frequency from a Gaussian frequency calculation run with HPmodes."""

import argparse

import numpy as np
from ase.io import read, write

from autopcet import (
    ANGSTROM_TO_CM,
    AU_TIME_TO_SECONDS,
    DALTON_TO_ELECTRON_MASS,
    SPEED_OF_LIGHT,
)

# 1 a.u. = 8.2387235038 mDyne and 1 Bohr = 0.529177 A, so this converts the
# mDyne/A force constants Gaussian prints into atomic units
MDYNE_PER_ANGSTROM_TO_AU = 8.2387235038 / 0.529177

# with HPmodes set, Gaussian prints the high-precision normal modes 5 per group
MODES_PER_GROUP = 5


def read_gaussian_frequencies(xyz_path, log_path):
    """Read the optimized geometry and normal modes of a Gaussian frequency job.

    Returns the atoms, frequencies (cm^-1), reduced masses (amu), force
    constants (mDyne/A), and the Cartesian displacements of each normal mode.
    """
    atoms = read(xyz_path)
    n_atoms = len(atoms)
    # number of vibrational degrees of freedom, assuming a nonlinear molecule
    n_modes = 3 * n_atoms - 6
    n_groups = -(-n_modes // MODES_PER_GROUP)  # ceiling division

    with open(log_path) as log_file:
        lines = log_file.readlines()

    geometry_start = 0
    frequency_start = 0
    for i, line in enumerate(lines):
        if "Standard orientation" in line:
            geometry_start = i
        if line.startswith(" Harmonic frequencies"):
            # with HPmodes the high precision normal modes are printed first, so
            # only the first such line is of interest
            frequency_start = i
            break

    # Read the optimized structure and update the Atoms object with it, since
    # Gaussian may have rotated the geometry during the calculation.
    positions = np.zeros((n_atoms, 3))
    first_atom_line = geometry_start + 5
    for j in range(n_atoms):
        positions[j] = [
            float(value) for value in lines[first_atom_line + j].split()[3:]
        ]
    atoms.set_positions(positions)

    write("optimized_geometry.xyz", atoms)

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

    return atoms, frequencies, reduced_masses, force_constants, normal_modes


def calc_effective_mode(
    atoms, donor_index, acceptor_index, reduced_masses, force_constants, normal_modes
):
    """Project the normal modes onto the proton donor-acceptor axis.

    ``force_constants`` are in mDyne/A, as printed by Gaussian. Returns the
    effective force constant (a.u.), reduced mass (amu), and frequency (cm^-1).
    """
    force_constants_au = force_constants / MDYNE_PER_ANGSTROM_TO_AU

    positions = atoms.get_positions()

    # unit vector connecting the proton donor and acceptor
    axis = positions[acceptor_index] - positions[donor_index]
    axis /= np.linalg.norm(axis)

    # how much each normal mode stretches the donor-acceptor axis
    donor_slice = slice(donor_index * 3, donor_index * 3 + 3)
    acceptor_slice = slice(acceptor_index * 3, acceptor_index * 3 + 3)
    weights = (normal_modes[:, acceptor_slice] - normal_modes[:, donor_slice]) @ axis

    effective_force_constant = 1 / np.sum(weights * weights / force_constants_au)
    effective_reduced_mass = 1 / np.sum(weights * weights / reduced_masses)

    # autopcet's speed of light is in A/s, so convert it to cm/s for a wavenumber
    speed_of_light_cm = SPEED_OF_LIGHT * ANGSTROM_TO_CM
    effective_frequency = (
        np.sqrt(
            effective_force_constant / effective_reduced_mass / DALTON_TO_ELECTRON_MASS
        )
        / AU_TIME_TO_SECONDS
        / speed_of_light_cm
        / 2
        / np.pi
    )

    return effective_force_constant, effective_reduced_mass, effective_frequency


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xyz",
        dest="xyz_path",
        required=True,
        help="select an xyz file for the molecule",
    )
    parser.add_argument(
        "--log",
        dest="log_path",
        required=True,
        help="select the log file of a Gaussian frequency calculation",
    )
    parser.add_argument(
        "-D",
        type=int,
        dest="donor_index",
        required=True,
        help="atomic index (start from 0) of the proton donor",
    )
    parser.add_argument(
        "-A",
        type=int,
        dest="acceptor_index",
        required=True,
        help="atomic index (start from 0) of the proton acceptor",
    )
    return parser


def main():
    options = build_parser().parse_args()

    atoms, _, reduced_masses, force_constants, normal_modes = read_gaussian_frequencies(
        options.xyz_path, options.log_path
    )
    force_constant, reduced_mass, frequency = calc_effective_mode(
        atoms,
        options.donor_index,
        options.acceptor_index,
        reduced_masses,
        force_constants,
        normal_modes,
    )

    print(f"Effective force constant in a.u.: {force_constant:.4f}")
    print(f"Effective reduced mass in amu: {reduced_mass:.3f}")
    print(f"Effective frequency in cm-1: {frequency: .2f}")


if __name__ == "__main__":
    main()
