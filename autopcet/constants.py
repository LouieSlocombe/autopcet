"""Physical constants and unit conversions, derived from CODATA via SciPy."""

import numpy as np
from scipy.constants import (
    N_A,
    angstrom,
    calorie,
    centi,
    nano,
    speed_of_light,
    value,
)

kB: float = value("Boltzmann constant in eV/K")
h: float = value("Planck constant in eV/Hz")
hbar: float = h / 2 / np.pi
c: float = speed_of_light / angstrom
massH: float = value("proton-electron mass ratio")
massD: float = value("deuteron-electron mass ratio") + 1

Ha2eV: float = value("Hartree energy in eV")
Ha2kcal: float = value("Hartree energy") * N_A / (1000 * calorie)
kcal2Ha: float = 1 / Ha2kcal
eV2Ha: float = 1 / Ha2eV
eV2kcal: float = eV2Ha * Ha2kcal
kcal2eV: float = 1 / eV2kcal

A2Bohr: float = angstrom / value("Bohr radius")
A2nm: float = angstrom / nano
A2cm: float = angstrom / centi
Bohr2A: float = 1 / A2Bohr
cm2A: float = 1 / A2cm
nm2A: float = 1 / A2nm

wn2eV: float = h * c * A2cm
eV2wn: float = 1 / wn2eV

Da2me: float = value("atomic mass constant") / value("electron mass")
me2Da: float = 1 / Da2me

au2s: float = value("atomic unit of time")

Debye2au: float = (1e-21 / speed_of_light) / value(
    "atomic unit of electric dipole mom."
)
