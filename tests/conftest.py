"""Shared fixtures built from the bundled example calculations."""

import pytest
from example1_data import E_PROD_DATA, E_REAC_DATA, RP_DATA

from autopcet import fit_poly8
from autopcet._types import PotentialFunction


@pytest.fixture(scope="session")
def reac_proton_pot() -> PotentialFunction:
    """Reactant proton potential fitted exactly as in example 1."""
    return fit_poly8(RP_DATA, E_REAC_DATA)


@pytest.fixture(scope="session")
def prod_proton_pot() -> PotentialFunction:
    """Product proton potential fitted exactly as in example 1."""
    return fit_poly8(RP_DATA, E_PROD_DATA)
