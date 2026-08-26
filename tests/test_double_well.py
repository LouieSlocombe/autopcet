"""Tests for the analytic double-well potential factory."""

import numpy as np
import pytest
from scipy.signal import find_peaks

from autopcet import make_double_well

# Two 4 eV Morse wells 0.7 angstrom apart with a modest diabatic coupling.
DE = 4.0
BETA = 2.5
R0 = 0.7
DELTA = 0.1
VPT0 = 0.5


@pytest.mark.parametrize("smooth", ["poly8", "poly6", "BSpline"])
def test_double_well_has_two_minima_and_zero_floor(smooth: str) -> None:
    """Every smoothing backend yields a double well with its minimum at zero."""
    potential = make_double_well(DE, DE, BETA, BETA, R0, DELTA, VPT0, smooth=smooth)
    r = np.linspace(-1.1 * R0, 1.1 * R0, 501)
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
        make_double_well(DE, DE, BETA, BETA, R0, 10.0, VPT0)


def test_unknown_smoothing_option_is_rejected() -> None:
    """An unrecognized smooth keyword raises ValueError."""
    with pytest.raises(ValueError, match="smooth"):
        make_double_well(DE, DE, BETA, BETA, R0, DELTA, VPT0, smooth="nope")
