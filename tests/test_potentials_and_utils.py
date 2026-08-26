"""Tests for the analytic potentials, fitting helpers, and small utilities."""

import numpy as np
import pytest
from example1_data import E_REAC_DATA, RP_DATA
from scipy.integrate import simpson

from autopcet import (
    A2Bohr,
    Bohr2A,
    Ha2eV,
    Ha2kcal,
    au2s,
    copy_function,
    eV2Ha,
    eV2kcal,
    eV2wn,
    find_roots,
    fit_bspline,
    fit_poly6,
    fit_poly8,
    gaussian,
    h,
    hbar,
    inverted_morse,
    is_array,
    is_number,
    kB,
    kcal2eV,
    kcal2Ha,
    make_inverted_morse,
    make_morse,
    massD,
    massH,
    morse,
    poly6,
    poly8,
    wn2eV,
)


def test_morse_minimum_and_dissociation_limit() -> None:
    """A Morse potential is zero at its minimum and tends to De."""
    r0, de, beta = 0.95, 4.0, 2.5

    assert morse(r0, r0, de, beta) == 0.0
    assert morse(r0 + 50.0 / beta, r0, de, beta) == pytest.approx(de)


def test_inverted_morse_mirrors_morse() -> None:
    """The inverted Morse potential is the mirror image of the Morse one."""
    r = np.linspace(-1.0, 1.0, 101)
    r0, de, beta = 0.35, 4.0, 2.5

    np.testing.assert_allclose(
        inverted_morse(r, r0, de, beta), morse(2 * r0 - r, r0, de, beta)
    )


def test_make_morse_matches_direct_evaluation() -> None:
    """The Morse factories return callables equal to the plain functions."""
    r = np.linspace(-1.0, 1.0, 101)
    r0, de, beta = -0.35, 3.5, 2.0

    np.testing.assert_allclose(make_morse(r0, de, beta)(r), morse(r, r0, de, beta))
    np.testing.assert_allclose(
        make_inverted_morse(r0, de, beta)(r), inverted_morse(r, r0, de, beta)
    )


def test_gaussian_is_normalized_and_peaks_at_center() -> None:
    """The Gaussian has unit area and its maximum value at x0."""
    x = np.linspace(-8.0, 8.0, 4001)
    x0, sigma2 = 0.3, 0.5
    g = gaussian(x, x0, sigma2)

    assert simpson(g, x=x) == pytest.approx(1.0, abs=1e-6)
    assert gaussian(x0, x0, sigma2) == pytest.approx(1 / np.sqrt(2 * np.pi * sigma2))
    assert gaussian(x0 + 0.2, x0, sigma2) == pytest.approx(
        gaussian(x0 - 0.2, x0, sigma2)
    )


def test_polynomials_match_numpy_polyval() -> None:
    """poly6 and poly8 evaluate their coefficients in descending order."""
    x = np.linspace(-2.0, 2.0, 51)
    coeffs6 = [0.2, -0.5, 1.0, 0.7, -1.2, 0.3, 2.0]
    coeffs8 = [0.1, 0.2, -0.4, 0.5, 1.0, -0.7, 1.2, -0.3, 0.9]

    np.testing.assert_allclose(poly6(x, *coeffs6), np.polyval(coeffs6, x))
    np.testing.assert_allclose(poly8(x, *coeffs8), np.polyval(coeffs8, x))


def test_fit_poly6_reproduces_smooth_data() -> None:
    """A 6th-order fit reproduces smooth data shifted to a zero minimum."""
    x = np.linspace(-1.0, 1.0, 81)
    y = np.cos(x)
    fitted = fit_poly6(x, y)

    np.testing.assert_allclose(fitted(x), y - np.min(y), atol=1e-4)


def test_fit_poly8_reproduces_example1_potential() -> None:
    """The example 1 workflow fits the raw double well to within 0.1 eV."""
    fitted = fit_poly8(RP_DATA, E_REAC_DATA)

    np.testing.assert_allclose(
        fitted(RP_DATA), E_REAC_DATA - np.min(E_REAC_DATA), atol=0.1
    )


def test_fit_bspline_reproduces_smooth_data() -> None:
    """The B-spline fit follows smooth data shifted to a zero minimum."""
    x = np.linspace(-1.0, 1.0, 81)
    y = np.cos(x)
    fitted = fit_bspline(x, y)

    np.testing.assert_allclose(fitted(x), y - np.min(y), atol=5e-3)


def test_find_roots_locates_sign_changes() -> None:
    """find_roots returns grid points adjacent to each sign change."""
    x = np.linspace(-2.0, 2.0, 401)
    roots = find_roots(x, x**2 - 0.9)

    assert len(roots) == 2
    np.testing.assert_allclose(roots, [-np.sqrt(0.9), np.sqrt(0.9)], atol=0.02)


def test_find_roots_without_sign_change_is_empty() -> None:
    """A strictly positive curve has no roots."""
    x = np.linspace(-2.0, 2.0, 101)

    assert find_roots(x, x**2 + 1.0) == []


def test_is_number_and_is_array() -> None:
    """The type helpers accept scalars/sequences and reject other types."""
    assert is_number(3)
    assert is_number(3.14)
    assert not is_number("3.14")
    assert not is_number([3.14])

    assert is_array([1.0, 2.0])
    assert is_array((1.0, 2.0))
    assert is_array(np.array([1.0, 2.0]))
    assert not is_array(3.14)
    assert not is_array("abc")


def test_copy_function_preserves_behavior_and_metadata() -> None:
    """copy_function returns an independent but equivalent function."""

    def greet(name: str = "World", *, punctuation: str = "!") -> str:
        return f"Hello, {name}{punctuation}"

    copied = copy_function(greet)

    assert copied is not greet
    assert copied.__name__ == greet.__name__
    assert copied() == "Hello, World!"
    assert copied("Ada", punctuation="?") == "Hello, Ada?"


def test_physical_constants_have_expected_values() -> None:
    """The exported constants match CODATA values in the documented units."""
    assert kB == pytest.approx(8.617333e-5, rel=1e-6)
    assert hbar == pytest.approx(h / (2 * np.pi))
    assert Ha2eV == pytest.approx(27.211386, rel=1e-6)
    assert Ha2kcal == pytest.approx(627.5095, rel=1e-4)
    assert eV2wn == pytest.approx(8065.54, rel=1e-4)
    assert au2s == pytest.approx(2.418884e-17, rel=1e-6)
    assert massH == pytest.approx(1836.15267, rel=1e-6)
    assert massD / massH == pytest.approx(2.0, rel=1e-2)


def test_unit_conversions_round_trip() -> None:
    """Each forward/backward conversion pair multiplies to one."""
    assert Ha2eV * eV2Ha == pytest.approx(1.0)
    assert Ha2kcal * kcal2Ha == pytest.approx(1.0)
    assert eV2kcal * kcal2eV == pytest.approx(1.0)
    assert A2Bohr * Bohr2A == pytest.approx(1.0)
    assert wn2eV * eV2wn == pytest.approx(1.0)
