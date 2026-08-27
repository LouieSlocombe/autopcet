"""Analytic proton potentials and the fitting helpers that smooth tabulated data."""

from collections.abc import Callable
from typing import cast

import numpy as np
from scipy.interpolate import BSpline, splrep
from scipy.optimize import curve_fit

from ._types import FitMethod, FloatArray, PotentialFunction
from .utils import find_roots


def morse[T: (float, FloatArray)](
    r: T, r0: float, well_depth: float, width: float
) -> T:
    """Morse potential with its minimum at ``r0``, rising to ``well_depth``."""
    energy: T = well_depth * (1 - np.exp(-width * (r - r0))) ** 2
    return energy


def inverted_morse[T: (float, FloatArray)](
    r: T, r0: float, well_depth: float, width: float
) -> T:
    """Mirror image of :func:`morse` about its minimum at ``r0``."""
    energy: T = well_depth * (1 - np.exp(width * (r - r0))) ** 2
    return energy


def gaussian[T: (float, FloatArray)](x: T, center: float, variance: float) -> T:
    """Normalized Gaussian centered at ``center`` with the given variance."""
    density: T = (
        1
        / np.sqrt(2 * np.pi * variance)
        * np.exp(-((x - center) ** 2) / (2 * variance))
    )
    return density


def poly6[T: (float, FloatArray)](
    x: T,
    c6: float,
    c5: float,
    c4: float,
    c3: float,
    c2: float,
    c1: float,
    c0: float,
) -> T:
    """Sixth-order polynomial with coefficients in descending order."""
    value: T = c6 * x**6 + c5 * x**5 + c4 * x**4 + c3 * x**3 + c2 * x**2 + c1 * x + c0
    return value


def poly8[T: (float, FloatArray)](
    x: T,
    c8: float,
    c7: float,
    c6: float,
    c5: float,
    c4: float,
    c3: float,
    c2: float,
    c1: float,
    c0: float,
) -> T:
    """Eighth-order polynomial with coefficients in descending order."""
    value: T = (
        c8 * x**8
        + c7 * x**7
        + c6 * x**6
        + c5 * x**5
        + c4 * x**4
        + c3 * x**3
        + c2 * x**2
        + c1 * x
        + c0
    )
    return value


def make_morse(r0: float, well_depth: float, width: float) -> PotentialFunction:
    """Return a Morse potential as a function of the coordinate only."""

    def morse_potential(r: FloatArray) -> FloatArray:
        return morse(r, r0, well_depth, width)

    return morse_potential


def make_inverted_morse(
    r0: float, well_depth: float, width: float
) -> PotentialFunction:
    """Return an inverted Morse potential as a function of the coordinate only."""

    def inverted_morse_potential(r: FloatArray) -> FloatArray:
        return inverted_morse(r, r0, well_depth, width)

    return inverted_morse_potential


def _fit_polynomial(x: FloatArray, y: FloatArray, degree: int) -> PotentialFunction:
    """Fit a polynomial of the given degree, shifted to a minimum of zero.

    The fit is done twice: once against the data, then again against the fitted
    curve resampled on a uniform grid, which re-levels the minimum to zero.
    """
    polynomial: Callable[..., FloatArray] = poly6 if degree == 6 else poly8

    coefficients: FloatArray = curve_fit(polynomial, x, y - np.min(y))[0]
    resampled_x = np.linspace(np.min(x), np.max(x), 500)
    resampled_y = polynomial(resampled_x, *coefficients)
    coefficients = curve_fit(
        polynomial, resampled_x, resampled_y - np.min(resampled_y)
    )[0]

    def fitted(x_new: FloatArray) -> FloatArray:
        return polynomial(x_new, *coefficients)

    return fitted


def fit_poly6(x: FloatArray, y: FloatArray) -> PotentialFunction:
    """Fit a sixth-order polynomial to the data, shifted to a minimum of zero."""
    return _fit_polynomial(x, y, degree=6)


def fit_poly8(x: FloatArray, y: FloatArray) -> PotentialFunction:
    """Fit an eighth-order polynomial to the data, shifted to a minimum of zero."""
    return _fit_polynomial(x, y, degree=8)


def fit_bspline(
    x: FloatArray, y: FloatArray, smoothing: float = 1e-4
) -> PotentialFunction:
    """Fit a smoothing B-spline to the data, shifted to a minimum of zero."""
    # cast: scipy is untyped
    spline = cast("PotentialFunction", BSpline(*splrep(x, y, s=smoothing)))
    offset = np.min(spline(np.linspace(np.min(x), np.max(x), 500)))

    def fitted(x_new: FloatArray) -> FloatArray:
        return spline(x_new) - offset

    return fitted


def fit_potential(
    x: FloatArray,
    y: FloatArray,
    fit_method: FitMethod = "bspline",
    smoothing: float = 1e-4,
) -> PotentialFunction:
    """Smooth a tabulated potential with the named fitting method.

    ``smoothing`` is the B-spline smoothing factor and is ignored by the
    polynomial fits.
    """
    match fit_method:
        case "poly6":
            return fit_poly6(x, y)
        case "poly8":
            return fit_poly8(x, y)
        case "bspline":
            return fit_bspline(x, y, smoothing)
        case _:
            raise ValueError(
                "'fit_method' must be one of 'poly6', 'poly8', or 'bspline'."
            )


def make_double_well(
    donor_well_depth: float,
    acceptor_well_depth: float,
    donor_width: float,
    acceptor_width: float,
    separation: float,
    asymmetry: float,
    coupling: float,
    fit_method: FitMethod = "poly8",
    smoothing: float = 8.0,
    n_grid: int = 500,
) -> PotentialFunction:
    """Build a smoothed double-well potential from two coupled Morse potentials.

    The donor and acceptor wells sit ``separation`` apart, offset in energy by
    ``asymmetry``, and are mixed by a Gaussian coupling of strength
    ``coupling`` centered on their crossing point.
    """
    r = np.linspace(-1.2 * separation, 1.2 * separation, n_grid)
    donor = morse(r, -separation / 2, donor_well_depth, donor_width)
    acceptor = (
        inverted_morse(r, separation / 2, acceptor_well_depth, acceptor_width)
        + asymmetry
    )

    crossings = find_roots(r, donor - acceptor)
    if len(crossings) != 3:
        raise RuntimeError(
            "These input parameters lead to a single-well potential. "
            "Please use a different set of parameters."
        )

    mixing = coupling * gaussian(r, crossings[1], separation * separation)
    splitting = np.sqrt((acceptor - donor) ** 2 + 4 * mixing**2)
    lower = 0.5 * (acceptor + donor - splitting)
    upper = 0.5 * (acceptor + donor + splitting)

    # Follow the lower adiabat between the outer two crossings and the upper
    # one beyond them, so the double well keeps its repulsive outer walls.
    inside = (
        np.heaviside(r - crossings[0], 0.0) + np.heaviside(crossings[2] - r, 0.0) - 1
    )
    energy = inside * lower + (1 - inside) * upper

    fitted = fit_potential(r, energy, fit_method, smoothing)
    offset = np.min(fitted(r))

    def double_well_potential(r_new: FloatArray) -> FloatArray:
        return fitted(r_new) - offset

    return double_well_potential
