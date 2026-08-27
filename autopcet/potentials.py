"""Analytic proton potentials and the fitting helpers that smooth tabulated data."""

from typing import cast, overload

import numpy as np
from scipy.interpolate import BSpline, splrep
from scipy.optimize import curve_fit

from ._types import FloatArray, PotentialFunction
from .utils import find_roots


@overload
def morse(r: float, r0: float, De: float, beta: float) -> float: ...
@overload
def morse(r: FloatArray, r0: float, De: float, beta: float) -> FloatArray: ...
def morse(
    r: float | FloatArray, r0: float, De: float, beta: float
) -> float | FloatArray:
    """Morse potential with minimum at ``r0``, well depth ``De``, and width ``beta``."""
    # cast: numpy ufuncs are typed as returning Any for scalar inputs
    return cast("float | FloatArray", De * (1 - np.exp(-beta * (r - r0))) ** 2)


@overload
def inverted_morse(r: float, r0: float, De: float, beta: float) -> float: ...
@overload
def inverted_morse(r: FloatArray, r0: float, De: float, beta: float) -> FloatArray: ...
def inverted_morse(
    r: float | FloatArray, r0: float, De: float, beta: float
) -> float | FloatArray:
    """Mirror image of :func:`morse` about its minimum at ``r0``."""
    return cast("float | FloatArray", De * (1 - np.exp(beta * (r - r0))) ** 2)


@overload
def gaussian(x: float, x0: float, sigma2: float) -> float: ...
@overload
def gaussian(x: FloatArray, x0: float, sigma2: float) -> FloatArray: ...
def gaussian(x: float | FloatArray, x0: float, sigma2: float) -> float | FloatArray:
    """Normalized Gaussian centered at ``x0`` with variance ``sigma2``."""
    return cast(
        "float | FloatArray",
        1 / np.sqrt(2 * np.pi * sigma2) * np.exp(-((x - x0) ** 2) / (2 * sigma2)),
    )


@overload
def poly6(
    x: float,
    c6: float,
    c5: float,
    c4: float,
    c3: float,
    c2: float,
    c1: float,
    c0: float,
) -> float: ...
@overload
def poly6(
    x: FloatArray,
    c6: float,
    c5: float,
    c4: float,
    c3: float,
    c2: float,
    c1: float,
    c0: float,
) -> FloatArray: ...
def poly6(
    x: float | FloatArray,
    c6: float,
    c5: float,
    c4: float,
    c3: float,
    c2: float,
    c1: float,
    c0: float,
) -> float | FloatArray:
    """Sixth-order polynomial with coefficients in descending order."""
    return c6 * x**6 + c5 * x**5 + c4 * x**4 + c3 * x**3 + c2 * x**2 + c1 * x + c0


@overload
def poly8(
    x: float,
    c8: float,
    c7: float,
    c6: float,
    c5: float,
    c4: float,
    c3: float,
    c2: float,
    c1: float,
    c0: float,
) -> float: ...
@overload
def poly8(
    x: FloatArray,
    c8: float,
    c7: float,
    c6: float,
    c5: float,
    c4: float,
    c3: float,
    c2: float,
    c1: float,
    c0: float,
) -> FloatArray: ...
def poly8(
    x: float | FloatArray,
    c8: float,
    c7: float,
    c6: float,
    c5: float,
    c4: float,
    c3: float,
    c2: float,
    c1: float,
    c0: float,
) -> float | FloatArray:
    """Eighth-order polynomial with coefficients in descending order."""
    return (
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


def make_morse(r0: float, De: float, beta: float) -> PotentialFunction:
    """Return a Morse potential as a function of the coordinate only."""

    def morse_potential(r: FloatArray) -> FloatArray:
        return morse(r, r0, De, beta)

    return morse_potential


def make_inverted_morse(r0: float, De: float, beta: float) -> PotentialFunction:
    """Return an inverted Morse potential as a function of the coordinate only."""

    def inverted_morse_potential(r: FloatArray) -> FloatArray:
        return inverted_morse(r, r0, De, beta)

    return inverted_morse_potential


def make_double_well(
    De1: float,
    De2: float,
    beta1: float,
    beta2: float,
    R0: float,
    Delta: float,
    VPT0: float,
    smooth: str = "poly8",
    s: float = 8,
    NGrid: int = 500,
) -> PotentialFunction:
    """Build a smoothed double-well potential from two coupled Morse potentials."""
    r = np.linspace(-1.2 * R0, 1.2 * R0, NGrid)
    ED = morse(r, -R0 / 2, De1, beta1)
    EA = inverted_morse(r, R0 / 2, De2, beta2) + Delta

    roots = find_roots(r, ED - EA)
    if len(roots) != 3:
        raise RuntimeError(
            "These input parameters lead to a single-well potential."
            "Please use a different set of parameters."
        )
    VPT = VPT0 * gaussian(r, roots[1], R0 * R0)

    E1 = 0.5 * (EA + ED - np.sqrt((EA - ED) ** 2 + 4 * VPT**2))
    E2 = 0.5 * (EA + ED + np.sqrt((EA - ED) ** 2 + 4 * VPT**2))

    switch = np.heaviside(r - roots[0], 0 * r) + np.heaviside(roots[2] - r, 0 * r) - 1

    E = switch * E1 + (1 - switch) * E2

    fitted_E: PotentialFunction
    if smooth == "poly8":
        fitted_E = fit_poly8(r, E)
    elif smooth == "poly6":
        fitted_E = fit_poly6(r, E)
    elif smooth == "bspline":
        tck = splrep(r, E, s=s)

        def bspline_E(rr: FloatArray) -> FloatArray:
            # cast: scipy is untyped
            return cast("FloatArray", BSpline(*tck)(rr))

        fitted_E = bspline_E
    else:
        raise ValueError("'smooth' must be one of 'poly6', 'poly8', or 'bspline'.")

    def double_well_potential(rr: FloatArray) -> FloatArray:
        return fitted_E(rr) - np.min(fitted_E(r))

    return double_well_potential


def fit_poly6(xdata: FloatArray, ydata: FloatArray) -> PotentialFunction:
    """Fit a sixth-order polynomial to the data, shifted to a minimum of zero."""
    coeffs: FloatArray = curve_fit(poly6, xdata, ydata - np.min(ydata))[0]
    xdata_tmp = np.linspace(np.min(xdata), np.max(xdata), 500)
    ydata_tmp = poly6(xdata_tmp, *coeffs)
    coeffs = curve_fit(poly6, xdata_tmp, ydata_tmp - np.min(ydata_tmp))[0]

    def poly6_fit(x: FloatArray) -> FloatArray:
        return poly6(x, *coeffs)

    return poly6_fit


def fit_poly8(xdata: FloatArray, ydata: FloatArray) -> PotentialFunction:
    """Fit an eighth-order polynomial to the data, shifted to a minimum of zero."""
    coeffs: FloatArray = curve_fit(poly8, xdata, ydata - np.min(ydata))[0]
    xdata_tmp = np.linspace(np.min(xdata), np.max(xdata), 500)
    ydata_tmp = poly8(xdata_tmp, *coeffs)
    coeffs = curve_fit(poly8, xdata_tmp, ydata_tmp - np.min(ydata_tmp))[0]

    def poly8_fit(x: FloatArray) -> FloatArray:
        return poly8(x, *coeffs)

    return poly8_fit


def fit_bspline(
    xdata: FloatArray, ydata: FloatArray, s: float = 1e-4
) -> PotentialFunction:
    """Fit a smoothing B-spline to the data, shifted to a minimum of zero."""
    tck = splrep(xdata, ydata, s=s)
    xdata_tmp = np.linspace(np.min(xdata), np.max(xdata), 500)

    def bspline_fit(x: FloatArray) -> FloatArray:
        # cast: scipy is untyped
        return cast("FloatArray", BSpline(*tck)(x) - np.min(BSpline(*tck)(xdata_tmp)))

    return bspline_fit
