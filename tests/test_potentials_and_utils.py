"""Tests for the analytic potentials, fitting helpers, and small utilities."""

from collections.abc import Callable

import numpy as np
import pytest
from example1_data import REACTANT_ENERGIES, RP_GRID
from scipy.integrate import simpson

from autopcet import (
    ANGSTROM_TO_BOHR,
    AU_TIME_TO_SECONDS,
    BOHR_TO_ANGSTROM,
    BOLTZMANN,
    DEBYE_TO_AU,
    EV_TO_HARTREE,
    EV_TO_KCAL,
    EV_TO_WAVENUMBER,
    HARTREE_TO_EV,
    HARTREE_TO_KCAL,
    HBAR,
    KCAL_TO_EV,
    KCAL_TO_HARTREE,
    MASS_DEUTERON,
    MASS_PROTON,
    PLANCK,
    WAVENUMBER_TO_EV,
    find_roots,
    fit_bspline,
    fit_poly6,
    fit_poly8,
    fit_potential,
    gaussian,
    inverted_morse,
    is_array,
    is_number,
    make_inverted_morse,
    make_morse,
    morse,
    poly6,
    poly8,
)
from autopcet._types import FitMethod, FloatArray, PotentialFunction


def test_morse_minimum_and_dissociation_limit() -> None:
    """A Morse potential is zero at its minimum and tends to its well depth."""
    r0, well_depth, width = 0.95, 4.0, 2.5

    assert morse(r0, r0, well_depth, width) == 0.0
    assert morse(r0 + 50.0 / width, r0, well_depth, width) == pytest.approx(well_depth)


def test_inverted_morse_mirrors_morse() -> None:
    """The inverted Morse potential is the mirror image of the Morse one."""
    r = np.linspace(-1.0, 1.0, 101)
    r0, well_depth, width = 0.35, 4.0, 2.5

    np.testing.assert_allclose(
        inverted_morse(r, r0, well_depth, width),
        morse(2 * r0 - r, r0, well_depth, width),
    )


def test_make_morse_matches_direct_evaluation() -> None:
    """The Morse factories return callables equal to the plain functions."""
    r = np.linspace(-1.0, 1.0, 101)
    r0, well_depth, width = -0.35, 3.5, 2.0

    np.testing.assert_allclose(
        make_morse(r0, well_depth, width)(r), morse(r, r0, well_depth, width)
    )
    np.testing.assert_allclose(
        make_inverted_morse(r0, well_depth, width)(r),
        inverted_morse(r, r0, well_depth, width),
    )


def test_gaussian_is_normalized_and_peaks_at_center() -> None:
    """The Gaussian has unit area and its maximum value at its center."""
    x = np.linspace(-8.0, 8.0, 4001)
    center, variance = 0.3, 0.5
    density = gaussian(x, center, variance)

    assert simpson(density, x=x) == pytest.approx(1.0, abs=1e-6)
    assert gaussian(center, center, variance) == pytest.approx(
        1 / np.sqrt(2 * np.pi * variance)
    )
    assert gaussian(center + 0.2, center, variance) == pytest.approx(
        gaussian(center - 0.2, center, variance)
    )


def test_polynomials_match_numpy_polyval() -> None:
    """poly6 and poly8 evaluate their coefficients in descending order."""
    x = np.linspace(-2.0, 2.0, 51)
    coefficients6 = [0.2, -0.5, 1.0, 0.7, -1.2, 0.3, 2.0]
    coefficients8 = [0.1, 0.2, -0.4, 0.5, 1.0, -0.7, 1.2, -0.3, 0.9]

    np.testing.assert_allclose(poly6(x, *coefficients6), np.polyval(coefficients6, x))
    np.testing.assert_allclose(poly8(x, *coefficients8), np.polyval(coefficients8, x))


def test_fit_poly6_reproduces_smooth_data() -> None:
    """A 6th-order fit reproduces smooth data shifted to a zero minimum."""
    x = np.linspace(-1.0, 1.0, 81)
    y = np.cos(x)
    fitted = fit_poly6(x, y)

    np.testing.assert_allclose(fitted(x), y - np.min(y), atol=1e-4)


def test_fit_poly8_reproduces_example1_potential() -> None:
    """The example 1 workflow fits the raw double well to within 0.1 eV."""
    fitted = fit_poly8(RP_GRID, REACTANT_ENERGIES)

    np.testing.assert_allclose(
        fitted(RP_GRID), REACTANT_ENERGIES - np.min(REACTANT_ENERGIES), atol=0.1
    )


def test_fit_bspline_reproduces_smooth_data() -> None:
    """The B-spline fit follows smooth data shifted to a zero minimum."""
    x = np.linspace(-1.0, 1.0, 81)
    y = np.cos(x)
    fitted = fit_bspline(x, y)

    np.testing.assert_allclose(fitted(x), y - np.min(y), atol=5e-3)


@pytest.mark.parametrize("fit_method", ["poly6", "poly8", "bspline"])
def test_fit_potential_dispatches_to_each_backend(fit_method: FitMethod) -> None:
    """fit_potential reproduces the fit its named backend would have made."""
    x = np.linspace(-1.0, 1.0, 81)
    y = np.cos(x)
    backends: dict[str, Callable[[FloatArray, FloatArray], PotentialFunction]] = {
        "poly6": fit_poly6,
        "poly8": fit_poly8,
        "bspline": fit_bspline,
    }

    np.testing.assert_allclose(
        fit_potential(x, y, fit_method)(x), backends[fit_method](x, y)(x)
    )


def test_fit_potential_rejects_an_unknown_method() -> None:
    """An unrecognized fit_method raises ValueError."""
    x = np.linspace(-1.0, 1.0, 21)

    with pytest.raises(ValueError, match="fit_method"):
        fit_potential(x, np.cos(x), "nope")  # type: ignore[arg-type]


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
    assert is_number(np.float32(3.14))
    assert is_number(np.int64(3))
    assert not is_number(True)
    assert not is_number("3.14")
    assert not is_number([3.14])

    assert is_array([1.0, 2.0])
    assert is_array((1.0, 2.0))
    assert is_array(np.array([1.0, 2.0]))
    assert not is_array(3.14)
    assert not is_array("abc")


@pytest.mark.parametrize(
    ("name", "value", "expected", "tolerance"),
    [
        ("BOLTZMANN", BOLTZMANN, 8.617333e-5, 1e-6),
        ("HBAR", HBAR, PLANCK / (2 * np.pi), 1e-12),
        ("HARTREE_TO_EV", HARTREE_TO_EV, 27.211386, 1e-6),
        ("HARTREE_TO_KCAL", HARTREE_TO_KCAL, 627.5095, 1e-4),
        ("EV_TO_WAVENUMBER", EV_TO_WAVENUMBER, 8065.54, 1e-4),
        ("AU_TIME_TO_SECONDS", AU_TIME_TO_SECONDS, 2.418884e-17, 1e-6),
        ("MASS_PROTON", MASS_PROTON, 1836.15267, 1e-6),
        ("MASS_DEUTERON", MASS_DEUTERON / MASS_PROTON, 2.0, 1e-2),
        ("DEBYE_TO_AU", DEBYE_TO_AU, 0.3934303, 1e-6),
    ],
)
def test_physical_constants_have_expected_values(
    name: str, value: float, expected: float, tolerance: float
) -> None:
    """The exported constants match CODATA values in the documented units."""
    assert value == pytest.approx(expected, rel=tolerance), name


@pytest.mark.parametrize(
    ("forward", "backward"),
    [
        (HARTREE_TO_EV, EV_TO_HARTREE),
        (HARTREE_TO_KCAL, KCAL_TO_HARTREE),
        (EV_TO_KCAL, KCAL_TO_EV),
        (ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM),
        (WAVENUMBER_TO_EV, EV_TO_WAVENUMBER),
    ],
)
def test_unit_conversions_round_trip(forward: float, backward: float) -> None:
    """Each forward/backward conversion pair multiplies to one."""
    assert forward * backward == pytest.approx(1.0)
