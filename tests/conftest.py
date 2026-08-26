"""Shared fixtures built from the bundled example calculations."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest
from example1_data import E_PROD_DATA, E_REAC_DATA, RP_DATA

from autopcet import fit_poly8

PotentialFunc = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


@pytest.fixture(scope="session")
def reac_proton_pot() -> PotentialFunc:
    """Reactant proton potential fitted exactly as in example 1."""
    return fit_poly8(RP_DATA, E_REAC_DATA)


@pytest.fixture(scope="session")
def prod_proton_pot() -> PotentialFunc:
    """Product proton potential fitted exactly as in example 1."""
    return fit_poly8(RP_DATA, E_PROD_DATA)
