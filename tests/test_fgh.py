"""Tests for the Fourier grid Hamiltonian solver against exact results."""

import numpy as np
import pytest

from autopcet import fgh_1d

NGRID = 128
SGRID = 16.0
X = np.linspace(-SGRID / 2, SGRID / 2, NGRID)


def test_harmonic_oscillator_eigenvalues() -> None:
    """FGH reproduces E_n = n + 1/2 for a unit harmonic oscillator (a.u.)."""
    eigvals, _ = fgh_1d(NGRID, SGRID, 0.5 * X**2, 1.0)

    np.testing.assert_allclose(eigvals[:6], np.arange(6) + 0.5, atol=1e-8)


def test_heavier_mass_lowers_the_spectrum() -> None:
    """Quadrupling the mass halves the harmonic frequency (E_n = (n + 1/2)/2)."""
    eigvals, _ = fgh_1d(NGRID, SGRID, 0.5 * X**2, 4.0)

    np.testing.assert_allclose(eigvals[:4], (np.arange(4) + 0.5) / 2, atol=1e-8)


def test_eigenvectors_are_orthonormal() -> None:
    """The returned eigenvectors form a discrete orthonormal basis."""
    _, eigvecs = fgh_1d(NGRID, SGRID, 0.5 * X**2, 1.0)

    np.testing.assert_allclose(eigvecs.T @ eigvecs, np.eye(NGRID), atol=1e-10)


def test_ground_state_has_no_node() -> None:
    """The harmonic ground-state wave function does not change sign."""
    _, eigvecs = fgh_1d(NGRID, SGRID, 0.5 * X**2, 1.0)
    ground = eigvecs[:, 0]

    assert np.all(ground >= 0) or np.all(ground <= 0)
    assert np.max(np.abs(ground)) == pytest.approx(
        np.abs(ground[np.argmin(np.abs(X))]), rel=1e-2
    )
