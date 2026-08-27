"""Shared fixtures built from the bundled example calculations."""

from collections.abc import Iterator

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from example1_data import (
    ELECTRONIC_COUPLING,
    PRODUCT_ENERGIES,
    REACTANT_ENERGIES,
    REACTION_FREE_ENERGY,
    REORGANIZATION_ENERGY,
    RP_GRID,
    TEMPERATURE,
)

from autopcet import (
    MASS_PROTON,
    PCET,
    KappaCoupling,
    fit_poly8,
    inverted_morse,
    morse,
)
from autopcet._types import FloatArray, PotentialFunction

# Every figure the suite draws is drawn headless. Nothing has made a figure by
# the time this runs, so switching the backend here is safe.
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    """Close whatever figures a test opened.

    matplotlib warns once more than 20 are open at a time, and the suite turns
    warnings into errors, so leaking them would fail an unrelated test later.
    """
    yield
    plt.close("all")


@pytest.fixture(scope="session")
def reactant_potential() -> PotentialFunction:
    """Reactant proton potential fitted exactly as in example 1."""
    return fit_poly8(RP_GRID, REACTANT_ENERGIES)


@pytest.fixture(scope="session")
def product_potential() -> PotentialFunction:
    """Product proton potential fitted exactly as in example 1."""
    return fit_poly8(RP_GRID, PRODUCT_ENERGIES)


@pytest.fixture
def system(
    reactant_potential: PotentialFunction, product_potential: PotentialFunction
) -> PCET:
    """A fresh PCET system set up exactly as in example 1."""
    return PCET(
        reactant_potential,
        product_potential,
        REACTION_FREE_ENERGY,
        REORGANIZATION_ENERGY,
        electronic_coupling=ELECTRONIC_COUPLING,
    )


@pytest.fixture
def calculated_system(system: PCET) -> PCET:
    """The same system, with its proton states already solved for."""
    system.calculate(MASS_PROTON, temperature=TEMPERATURE)
    return system


# A symmetric model double well in the style of example 5: two Morse diabats
# crossing at rp = 0, on the power-of-two grid KappaCoupling insists on.
_KAPPA_GRID: FloatArray = np.linspace(-0.55, 0.55, 512)
_KAPPA_COUPLING = 0.0434


@pytest.fixture(scope="session")
def kappa_system() -> KappaCoupling:
    """A calculated nonadiabaticity analysis of a symmetric double well."""
    analysis = KappaCoupling(
        _KAPPA_GRID,
        morse(_KAPPA_GRID, -0.35, 4.0, 2.5),
        inverted_morse(_KAPPA_GRID, 0.35, 4.0, 2.5),
        _KAPPA_COUPLING,
    )
    analysis.calculate(MASS_PROTON)
    return analysis
