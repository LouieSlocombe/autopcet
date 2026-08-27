"""Physical constants and unit conversions, derived from CODATA via SciPy.

Energies are in electronvolts, lengths in angstrom, and particle masses in
electron masses (atomic units) unless a name says otherwise.
"""

import numpy as np
from scipy.constants import (
    N_A,
    angstrom,
    calorie,
    centi,
    dyne,
    milli,
    nano,
    speed_of_light,
    value,
)

ROOM_TEMPERATURE: float = 298.15
"""Default temperature in kelvin."""

BOLTZMANN: float = value("Boltzmann constant in eV/K")
PLANCK: float = value("Planck constant in eV/Hz")
HBAR: float = PLANCK / 2 / np.pi
SPEED_OF_LIGHT: float = speed_of_light / angstrom
"""Speed of light in angstrom per second."""

MASS_PROTON: float = value("proton-electron mass ratio")
MASS_DEUTERON: float = value("deuteron-electron mass ratio") + 1
"""Deuteron mass plus the electron it carries, i.e. the mass of a D atom."""

HARTREE_TO_EV: float = value("Hartree energy in eV")
HARTREE_TO_KCAL: float = value("Hartree energy") * N_A / (1000 * calorie)
KCAL_TO_HARTREE: float = 1 / HARTREE_TO_KCAL
EV_TO_HARTREE: float = 1 / HARTREE_TO_EV
EV_TO_KCAL: float = EV_TO_HARTREE * HARTREE_TO_KCAL
KCAL_TO_EV: float = 1 / EV_TO_KCAL

ANGSTROM_TO_BOHR: float = angstrom / value("Bohr radius")
ANGSTROM_TO_NM: float = angstrom / nano
ANGSTROM_TO_CM: float = angstrom / centi
BOHR_TO_ANGSTROM: float = 1 / ANGSTROM_TO_BOHR
CM_TO_ANGSTROM: float = 1 / ANGSTROM_TO_CM
NM_TO_ANGSTROM: float = 1 / ANGSTROM_TO_NM

WAVENUMBER_TO_EV: float = PLANCK * SPEED_OF_LIGHT * ANGSTROM_TO_CM
EV_TO_WAVENUMBER: float = 1 / WAVENUMBER_TO_EV

DALTON_TO_ELECTRON_MASS: float = value("atomic mass constant") / value("electron mass")
ELECTRON_MASS_TO_DALTON: float = 1 / DALTON_TO_ELECTRON_MASS

AU_TIME_TO_SECONDS: float = value("atomic unit of time")

AU_TO_MDYNE_PER_ANGSTROM: float = (
    value("Hartree energy") / value("Bohr radius") ** 2 / (milli * dyne / angstrom)
)
"""Force constants: atomic units (hartree/bohr^2) to the mDyne/A Gaussian prints."""

MDYNE_PER_ANGSTROM_TO_AU: float = 1 / AU_TO_MDYNE_PER_ANGSTROM

DEBYE_TO_AU: float = (1e-21 / speed_of_light) / value(
    "atomic unit of electric dipole mom."
)
