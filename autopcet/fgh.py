"""Fourier grid Hamiltonian solver for one-dimensional proton vibrational states."""

import numpy as np
from numba import jit
from scipy.integrate import simpson

from ._types import FloatArray
from .constants import A2Bohr, Ha2eV, eV2Ha


@jit(nopython=True)
def fgh_1d(
    ngrid: int, sgrid: float, potential: FloatArray, mass: float
) -> tuple[FloatArray, FloatArray]:
    """Solve a 1D Schroedinger equation with the Fourier grid Hamiltonian method.

    All quantities are in atomic units: ``sgrid`` is the grid length,
    ``potential`` the potential energy on the grid, and ``mass`` the particle
    mass. Returns the eigenvalues and eigenvectors of the Hamiltonian.
    """
    nx = ngrid
    dx = sgrid / (ngrid - 1)
    k = np.pi / dx

    vmat = np.zeros((nx, nx))
    tmat = np.zeros((nx, nx))
    hmat = np.zeros((nx, nx))

    for i in range(nx):
        for j in range(nx):
            if i == j:
                vmat[i, j] = potential[j]
                tmat[i, j] = (k**2) / 3
            else:
                dji = j - i
                vmat[i, j] = 0
                tmat[i, j] = (2 * k**2) / (np.pi**2) * (((-1) ** dji) / (dji**2))

            hmat[i, j] = (1 / (2 * mass)) * tmat[i, j] + vmat[i, j]

    hmat_soln = np.linalg.eigh(hmat)
    return hmat_soln


def _solve_vibrational_states(
    rp: FloatArray, potential_eV: FloatArray, mass: float, nstates: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Solve for proton vibrational states on the grid ``rp`` (in Angstrom).

    Returns the lowest ``nstates`` energy levels in eV, the corresponding
    normalized wave functions, and the raw FGH eigenvalues in Hartree.
    """
    rp_in_Bohr = rp * A2Bohr
    ngrid = len(rp_in_Bohr)
    sgrid = rp_in_Bohr[-1] - rp_in_Bohr[0]

    eigvals, eigvecs = fgh_1d(ngrid, sgrid, potential_eV * eV2Ha, mass)
    energy_levels = eigvals[:nstates] * Ha2eV

    unnormalized_wfcs = np.transpose(eigvecs)[:nstates]
    wave_functions = np.array(
        [wfci / np.sqrt(simpson(wfci * wfci, x=rp)) for wfci in unnormalized_wfcs]
    )
    return energy_levels, wave_functions, eigvals
