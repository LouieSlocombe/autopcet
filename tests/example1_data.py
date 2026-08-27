"""Reference input data taken from example 1 (examples/example1_basic_usage)."""

import numpy as np

# First-principles double-well proton potentials from example 1.
# Distances in angstrom, energies in eV.
RP_GRID = np.array(
    [
        0.614,
        0.550,
        0.485,
        0.420,
        0.356,
        0.291,
        0.226,
        0.162,
        0.097,
        0.032,
        -0.032,
        -0.097,
        -0.162,
        -0.226,
        -0.291,
        -0.356,
        -0.420,
        -0.485,
        -0.550,
        -0.614,
    ]
)
REACTANT_ENERGIES = np.array(
    [
        5.283,
        4.534,
        4.061,
        3.794,
        3.673,
        3.646,
        3.680,
        3.688,
        3.634,
        3.646,
        3.602,
        3.513,
        3.392,
        3.257,
        3.138,
        3.078,
        3.140,
        3.413,
        4.022,
        5.144,
    ]
)
PRODUCT_ENERGIES = np.array(
    [
        4.847,
        4.134,
        3.706,
        3.495,
        3.452,
        3.509,
        3.620,
        3.801,
        3.949,
        4.073,
        4.157,
        4.194,
        4.189,
        4.157,
        4.128,
        4.144,
        4.270,
        4.597,
        5.250,
        6.411,
    ]
)

# Thermodynamic parameters used in example 1, in eV.
REACTION_FREE_ENERGY = -0.50
REORGANIZATION_ENERGY = 1.00
ELECTRONIC_COUPLING = 0.0434
TEMPERATURE = 298  # kelvin
