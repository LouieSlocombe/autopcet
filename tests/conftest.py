"""Shared fixtures built from the bundled example calculations."""

import pytest
from example1_data import PRODUCT_ENERGIES, REACTANT_ENERGIES, RP_GRID

from autopcet import fit_poly8
from autopcet._types import PotentialFunction


@pytest.fixture(scope="session")
def reactant_potential() -> PotentialFunction:
    """Reactant proton potential fitted exactly as in example 1."""
    return fit_poly8(RP_GRID, REACTANT_ENERGIES)


@pytest.fixture(scope="session")
def product_potential() -> PotentialFunction:
    """Product proton potential fitted exactly as in example 1."""
    return fit_poly8(RP_GRID, PRODUCT_ENERGIES)
