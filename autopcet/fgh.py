"""Fourier grid Hamiltonian solver for one-dimensional proton vibrational states."""

import numpy as np
from numba import jit
from scipy.integrate import simpson

from ._types import FloatArray
from .constants import ANGSTROM_TO_BOHR, EV_TO_HARTREE, HARTREE_TO_EV


@jit(nopython=True)
def fgh_1d(
    n_grid: int, grid_length: float, potential: FloatArray, mass: float
) -> tuple[FloatArray, FloatArray]:
    """Solve a 1D Schroedinger equation with the Fourier grid Hamiltonian method.

    All quantities are in atomic units: ``grid_length`` is the length of the
    (evenly spaced) grid, ``potential`` the potential energy on it, and ``mass``
    the particle mass. Returns the eigenvalues and eigenvectors of the
    Hamiltonian, with the eigenvectors held in the columns.
    """
    spacing = grid_length / (n_grid - 1)
    k_max = np.pi / spacing

    inverse_two_mass = 1 / (2 * mass)
    diagonal_kinetic = (k_max**2) / 3
    kinetic_prefactor = (2 * k_max**2) / (np.pi**2)

    hamiltonian = np.zeros((n_grid, n_grid))
    for i in range(n_grid):
        for j in range(n_grid):
            if i == j:
                hamiltonian[i, j] = inverse_two_mass * diagonal_kinetic + potential[j]
            else:
                separation = j - i
                hamiltonian[i, j] = inverse_two_mass * (
                    kinetic_prefactor * (((-1) ** separation) / (separation**2))
                )

    return np.linalg.eigh(hamiltonian)


def _solve_proton_states(
    rp: FloatArray, potential: FloatArray, mass: float, n_states: int
) -> tuple[FloatArray, FloatArray]:
    """Solve for proton vibrational states on the grid ``rp`` (in angstrom).

    ``potential`` is the potential energy on that grid in eV. Returns the
    lowest ``n_states`` energy levels, in eV, and their normalized wave
    functions, one per row.
    """
    rp_bohr = rp * ANGSTROM_TO_BOHR
    eigenvalues, eigenvectors = fgh_1d(
        len(rp_bohr),
        rp_bohr[-1] - rp_bohr[0],
        potential * EV_TO_HARTREE,
        mass,
    )

    energies = eigenvalues[:n_states] * HARTREE_TO_EV
    wavefunctions = eigenvectors.T[:n_states]
    norms = np.sqrt(simpson(wavefunctions * wavefunctions, x=rp, axis=1))
    return energies, wavefunctions / norms[:, np.newaxis]
