"""The inputs of example 1, shared by the fixtures and the tests that pin them.

The potentials are read from the same data file the example itself loads, so
there is one copy of the numbers rather than a transcription that can drift.
"""

from pathlib import Path

import numpy as np

from autopcet._types import FloatArray

_POTENTIALS = (
    Path(__file__).resolve().parent.parent
    / "examples"
    / "example1_basic_usage"
    / "double_well_potentials.dat"
)

RP_GRID: FloatArray
REACTANT_ENERGIES: FloatArray
PRODUCT_ENERGIES: FloatArray
RP_GRID, REACTANT_ENERGIES, PRODUCT_ENERGIES = np.loadtxt(_POTENTIALS, unpack=True)

# The thermodynamic parameters example 1 declares, in eV.
ELECTRONIC_COUPLING = 0.0434
REACTION_FREE_ENERGY = -0.50
REORGANIZATION_ENERGY = 1.00
TEMPERATURE = 298  # kelvin
