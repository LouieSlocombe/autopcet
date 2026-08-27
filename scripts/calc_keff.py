"""Compute the effective proton donor-acceptor force constant, reduced mass, and
frequency from a Gaussian frequency calculation run with HPmodes."""

import argparse

import numpy as np
from ase.io import read, write

from autopcet import A2cm, Da2me, au2s, c


def read_Gaussian_freq_job(xyzfile, logfile):
    atoms = read(xyzfile)
    natoms = len(atoms)
    # number of vibrational degrees of freedom, assume a nonlinear molecule
    nDOFvib = 3 * natoms - 6

    # when HPmodes is set in Gaussian input file, the normal modes are printed 5 modes per group
    ngroup = int(nDOFvib / 5)
    nresidue = nDOFvib % 5
    if nresidue != 0:
        ngroup += 1

    with open(logfile) as logfp:
        lines = logfp.readlines()

    index_optimized_struct = 0
    index_freq_output = 0
    for i, line in enumerate(lines):
        if "Standard orientation" in line:
            index_optimized_struct = i
        if line.startswith(" Harmonic frequencies"):
            # when HPmodes is set, the high precision normal modes are printed first
            # we only need the line index that corresponds to the high precision outputs
            index_freq_output = i
            break

    # read optimized structure
    # update the coordinate in the Atoms object if they are not the same as the optimized structure
    # this is because Gaussian can rotate the geometry during the calculation
    new_poses = np.zeros([natoms, 3])
    i = index_optimized_struct + 5
    for j in range(natoms):
        new_poses[j] = [float(dat) for dat in lines[i + j].split()[3:]]
    atoms.set_positions(new_poses)

    write("optimized_geometry.xyz", atoms)

    freqs = np.zeros(nDOFvib)
    reduced_masses = np.zeros(nDOFvib)
    force_constants = np.zeros(nDOFvib)
    normal_modes = np.zeros([nDOFvib, 3 * natoms])

    # start reading normal modes
    i = index_freq_output + 4
    for _ in range(ngroup):
        mode_indices = [int(dat) - 1 for dat in lines[i].split()]
        freqs[mode_indices] = [float(dat) for dat in lines[i + 2].split()[2:]]
        reduced_masses[mode_indices] = [float(dat) for dat in lines[i + 3].split()[3:]]
        force_constants[mode_indices] = [float(dat) for dat in lines[i + 4].split()[3:]]

        for j in range(3 * natoms):
            normal_modes[mode_indices, j] = [
                float(dat) for dat in lines[i + 7 + j].split()[3:]
            ]

        i += 7 + natoms * 3

    # In Gaussian output, the frequencies are in cm-1, reduced masses in amu
    # force constants in mDyne/A
    # The printed normal modes by Gaussian are the Cartesian displacements, not the mass-weighted Cartesian displacements

    return atoms, freqs, reduced_masses, force_constants, normal_modes


def calc_keff(
    atoms, donor_index, acceptor_index, reduced_masses, force_constants, normal_modes
):
    # input force constants in mDyne/A, which is unit used in Gassuain outputs

    # convert the unit of force constant from mDyne/A to au
    # 1 au = 8.2387235038 mDyne, 1 Bohr = 0.529177 A
    scale = 8.2387235038 / 0.529177
    force_constants_au = force_constants / scale

    poses = atoms.get_positions()

    # calculate the unit vector connect the proton donor and acceptor
    eDA = poses[acceptor_index] - poses[donor_index]
    eDA /= np.linalg.norm(eDA)

    nDOFvib = len(force_constants_au)
    weights = np.zeros(nDOFvib)

    for imode in range(nDOFvib):
        lAi = normal_modes[imode, acceptor_index * 3 : acceptor_index * 3 + 3]
        lDi = normal_modes[imode, donor_index * 3 : donor_index * 3 + 3]
        weights[imode] = np.inner(eDA, lAi - lDi)

    effective_force_constant = 1 / (np.sum(weights * weights / force_constants_au))
    effective_reduced_mass = 1 / (np.sum(weights * weights / reduced_masses))

    # calculate effective proton DA vibrational frequency in cm-1
    # autopcet.c is in A/s, so convert it to cm/s for a wavenumber
    c_cm = c * A2cm
    effective_frequency = (
        np.sqrt(effective_force_constant / effective_reduced_mass / Da2me)
        / au2s
        / c_cm
        / 2
        / np.pi
    )

    return effective_force_constant, effective_reduced_mass, effective_frequency


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xyz",
        dest="xyzfile",
        required=True,
        help="select an xyz file for the molecule",
    )
    parser.add_argument(
        "--log",
        dest="logfile",
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

    atoms, _freqs, reduced_masses, force_constants, normal_modes = (
        read_Gaussian_freq_job(options.xyzfile, options.logfile)
    )
    effective_force_constant, effective_reduced_mass, effective_frequency = calc_keff(
        atoms,
        options.donor_index,
        options.acceptor_index,
        reduced_masses,
        force_constants,
        normal_modes,
    )

    print(f"Effective force constant in a.u.: {effective_force_constant:.4f}")
    print(f"Effective reduced mass in amu: {effective_reduced_mass:.3f}")
    print(f"Effective frequency in cm-1: {effective_frequency: .2f}")


if __name__ == "__main__":
    main()
