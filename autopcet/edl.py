"""Electrical double layer model of the interfacial potential drop."""

from typing import Literal, cast

import numpy as np
from scipy.constants import N_A, elementary_charge
from scipy.optimize import fsolve

from ._types import FloatArray, ScalarOrArrayFunction
from .constants import (
    ANGSTROM_TO_BOHR,
    BOHR_TO_ANGSTROM,
    BOLTZMANN,
    CM_TO_ANGSTROM,
    DEBYE_TO_AU,
    EV_TO_HARTREE,
    HARTREE_TO_EV,
    ROOM_TEMPERATURE,
)
from .utils import is_array, is_number


def fermi_distribution[T: (float, FloatArray)](
    energy: T, fermi_level: float = 0.0, temperature: float = ROOM_TEMPERATURE
) -> T:
    """Fermi-Dirac occupation of a state at ``energy`` (in eV)."""
    occupation: T = 1 / (np.exp((energy - fermi_level) / BOLTZMANN / temperature) + 1)
    return occupation


def _langevin[T: (float, FloatArray)](x: T) -> T:
    """Langevin function, the classical orientational average of a dipole."""
    value: T = 1 / np.tanh(x) - 1 / x
    return value


def make_edl_model(
    potential_vs_she: float,
    d_ihl: float,
    d_ohl: float,
    eps_ihl: float,
    eps_static: float,
    eps_optical: float,
    dipole: float | Literal["calculate"],
    solvent_density: float,
    solvent_molar_mass: float,
    ion_concentration: float,
    edl_capacitance: float,
    pzfc_vs_she: float,
    temperature: float = ROOM_TEMPERATURE,
    eta_kirkwood: float = 1.0,
    g_kirkwood: float = 2.4,
    verbose: bool = False,
) -> ScalarOrArrayFunction:
    """Build an electrical double layer (EDL) model of the potential drop.

    The electrode is described by its potential against the standard hydrogen
    electrode and its potential of zero free charge, the compact layers by
    their thicknesses (in angstrom) and permittivities, and the electrolyte by
    its solvent density (g/cm^3), solvent molar mass (g/mol), and ion
    concentration (mol/L). ``edl_capacitance`` is in microfarad/cm^2.

    ``dipole`` is a solvent dipole moment in debye, or ``"calculate"`` to
    derive one from the permittivities. Passing any ``eta_kirkwood`` other than
    1 replaces it with the Kirkwood correlation factor built from
    ``g_kirkwood``.

    Returns a function mapping the distance from the electrode (in angstrom)
    to the potential drop (in V).
    """
    thermal_energy_au = BOLTZMANN * temperature * EV_TO_HARTREE
    potential_vs_pzfc = potential_vs_she - pzfc_vs_she

    if verbose:
        print(f"E vs. SHE = {potential_vs_she:.2f} V")

    surface_charge = (edl_capacitance * potential_vs_pzfc) / (
        elementary_charge * 1e6 * (CM_TO_ANGSTROM * ANGSTROM_TO_BOHR) ** 2
    )
    if verbose:
        print(f"sigma_M = {surface_charge:.6e} a.u.")

    solvent_density_au = (
        solvent_density
        / solvent_molar_mass
        * N_A
        / (CM_TO_ANGSTROM * ANGSTROM_TO_BOHR) ** 3
    )

    if dipole == "calculate":
        if eta_kirkwood != 1:
            eta_kirkwood = (
                2 * (2 * eps_static + eps_optical) / (3 * g_kirkwood * eps_static)
            )
        dipole_au = (
            3
            / (2 + eps_optical)
            * np.sqrt(
                3
                * thermal_energy_au
                * (eps_static - eps_optical)
                * eta_kirkwood
                / (8 * np.pi * solvent_density_au)
            )
        )
        if verbose:
            print(f"dipole = {dipole_au:.6f} a.u.")
    elif is_number(dipole):
        dipole_au = float(dipole) * DEBYE_TO_AU
    else:
        raise ValueError("'dipole' must be a number (in debye) or 'calculate'.")

    ion_density_au = (ion_concentration * N_A * 1000) / (1e10 * ANGSTROM_TO_BOHR) ** 3
    d_ihl_bohr = d_ihl * ANGSTROM_TO_BOHR
    d_ohl_bohr = d_ohl * ANGSTROM_TO_BOHR

    phi_ohp_au = (
        2
        * thermal_energy_au
        * np.arcsinh(
            surface_charge
            / np.sqrt((2 * thermal_energy_au * eps_static * ion_density_au) / np.pi)
        )
    )
    phi_ohp = phi_ohp_au * HARTREE_TO_EV
    if verbose:
        print(f"phi_OHP = {phi_ohp:.6f} V")

    def permittivity_ohl[T: (float, FloatArray)](field_au: T) -> T:
        """Field-dependent permittivity of the outer Helmholtz layer."""
        value: T = eps_optical + (4 * np.pi * (2 + eps_optical)) / (
            3 * field_au
        ) * solvent_density_au * dipole_au * _langevin(
            ((2 + eps_optical) * dipole_au * field_au) / (2 * thermal_energy_au)
        )
        return value

    def residual(field_au: FloatArray) -> FloatArray:
        """Zero when ``field_au`` reproduces the applied potential drop."""
        mismatch: FloatArray = (
            field_au
            * ((d_ihl_bohr * permittivity_ohl(field_au) / eps_ihl) + d_ohl_bohr)
            - potential_vs_pzfc * EV_TO_HARTREE
            + phi_ohp_au
        )
        return mismatch

    field_ohl_au = float(fsolve(residual, x0=0.1)[0])
    field_ihl_au = float(field_ohl_au * permittivity_ohl(field_ohl_au) / eps_ihl)

    field_ohl = field_ohl_au * HARTREE_TO_EV / BOHR_TO_ANGSTROM
    field_ihl = field_ihl_au * HARTREE_TO_EV / BOHR_TO_ANGSTROM

    if verbose:
        print(f"eps_OHL = {permittivity_ohl(field_ohl_au):.6f}")
        print(f"E_OHL = {field_ohl:.6f} V/A")
        print(f"E_IHL = {field_ihl:.6f} V/A")
        print()

    inverse_debye_length = float(
        np.sqrt((8 * np.pi * ion_density_au) / (eps_static * thermal_energy_au))
        / BOHR_TO_ANGSTROM
    )
    thermal_potential = BOLTZMANN * temperature

    def drop_profile(distance: FloatArray) -> FloatArray:
        """Potential drop across the compact layers and the diffuse layer."""
        compact = distance <= d_ihl
        diffuse = distance > d_ihl + d_ohl
        outer_helmholtz = ~(compact | diffuse)

        drop = np.empty_like(distance)
        drop[compact] = potential_vs_pzfc - distance[compact] * field_ihl
        drop[outer_helmholtz] = (
            potential_vs_pzfc
            - d_ihl * field_ihl
            - (distance[outer_helmholtz] - d_ihl) * field_ohl
        )
        drop[diffuse] = (
            4
            * thermal_potential
            * np.arctanh(
                np.tanh(phi_ohp / (4 * thermal_potential))
                * np.exp(-inverse_debye_length * (distance[diffuse] - d_ihl - d_ohl))
            )
        )
        return drop

    def potential_drop(distance: float | FloatArray) -> float | FloatArray:
        if is_number(distance):
            return float(drop_profile(np.array([float(cast("float", distance))]))[0])
        if is_array(distance):
            return drop_profile(np.asarray(distance, dtype=np.float64))
        raise TypeError("'distance' must be a number or a 1D array.")

    # cast: the union signature above cannot express that a scalar in gives a
    # scalar out and an array in gives an array out, which the protocol does.
    return cast("ScalarOrArrayFunction", potential_drop)
