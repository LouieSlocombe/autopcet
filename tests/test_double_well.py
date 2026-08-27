"""Tests for the analytic double-well potential factory."""

import numpy as np
import pytest
from scipy.signal import find_peaks

from autopcet import make_double_well
from autopcet._types import FitMethod

# Two 4 eV Morse wells 0.7 angstrom apart with a modest diabatic coupling.
WELL_DEPTH = 4.0
WIDTH = 2.5
SEPARATION = 0.7
ASYMMETRY = 0.1
COUPLING = 0.5


@pytest.mark.parametrize("fit_method", ["poly8", "poly6", "bspline"])
def test_double_well_has_two_minima_and_zero_floor(fit_method: FitMethod) -> None:
    """Every smoothing backend yields a double well with its minimum at zero."""
    potential = make_double_well(
        WELL_DEPTH,
        WELL_DEPTH,
        WIDTH,
        WIDTH,
        SEPARATION,
        ASYMMETRY,
        COUPLING,
        fit_method=fit_method,
    )
    r = np.linspace(-1.1 * SEPARATION, 1.1 * SEPARATION, 501)
    energy = potential(r)

    assert np.all(np.isfinite(energy))
    assert np.min(energy) == pytest.approx(0.0, abs=0.02)

    minima, _ = find_peaks(-energy)
    assert len(minima) == 2
    barrier_top = np.max(energy[minima[0] : minima[1]])
    assert barrier_top > energy[minima[0]]
    assert barrier_top > energy[minima[1]]


def test_single_well_parameters_are_rejected() -> None:
    """Parameters that do not produce three diabat crossings raise an error."""
    with pytest.raises(RuntimeError, match="single-well"):
        make_double_well(
            WELL_DEPTH, WELL_DEPTH, WIDTH, WIDTH, SEPARATION, 10.0, COUPLING
        )


def test_unknown_fit_method_is_rejected() -> None:
    """An unrecognized fit_method keyword raises ValueError."""
    with pytest.raises(ValueError, match="fit_method"):
        make_double_well(
            WELL_DEPTH,
            WELL_DEPTH,
            WIDTH,
            WIDTH,
            SEPARATION,
            ASYMMETRY,
            COUPLING,
            fit_method="nope",  # type: ignore[arg-type]
        )
