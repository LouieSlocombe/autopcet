"""Tests for the Fourier grid Hamiltonian solver against exact results."""

import numpy as np
import pytest

from autopcet import fgh_1d

N_GRID = 128
GRID_LENGTH = 16.0
X = np.linspace(-GRID_LENGTH / 2, GRID_LENGTH / 2, N_GRID)


def test_harmonic_oscillator_eigenvalues() -> None:
    """FGH reproduces E_n = n + 1/2 for a unit harmonic oscillator (a.u.)."""
    eigenvalues, _ = fgh_1d(N_GRID, GRID_LENGTH, 0.5 * X**2, 1.0)

    np.testing.assert_allclose(eigenvalues[:6], np.arange(6) + 0.5, atol=1e-8)


def test_heavier_mass_lowers_the_spectrum() -> None:
    """Quadrupling the mass halves the harmonic frequency (E_n = (n + 1/2)/2)."""
    eigenvalues, _ = fgh_1d(N_GRID, GRID_LENGTH, 0.5 * X**2, 4.0)

    np.testing.assert_allclose(eigenvalues[:4], (np.arange(4) + 0.5) / 2, atol=1e-8)


def test_eigenvectors_are_orthonormal() -> None:
    """The returned eigenvectors form a discrete orthonormal basis."""
    _, eigenvectors = fgh_1d(N_GRID, GRID_LENGTH, 0.5 * X**2, 1.0)

    np.testing.assert_allclose(
        eigenvectors.T @ eigenvectors, np.eye(N_GRID), atol=1e-10
    )


def test_ground_state_has_no_node() -> None:
    """The harmonic ground-state wave function does not change sign."""
    _, eigenvectors = fgh_1d(N_GRID, GRID_LENGTH, 0.5 * X**2, 1.0)
    ground = eigenvectors[:, 0]

    assert np.all(ground >= 0) or np.all(ground <= 0)
    assert np.max(np.abs(ground)) == pytest.approx(
        np.abs(ground[np.argmin(np.abs(X))]), rel=1e-2
    )
