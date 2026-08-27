"""Electrical double layer model of the interfacial potential drop."""

from typing import cast, overload

import numpy as np
from scipy.constants import N_A, elementary_charge
from scipy.optimize import fsolve

from ._types import FloatArray, _ScalarArrayFunction
from .constants import A2Bohr, Bohr2A, Debye2au, Ha2eV, cm2A, eV2Ha, kB
from .utils import is_array, is_number


def make_edl_model(
    EvsSHE: float,
    dIHL: float,
    dOHL: float,
    eps_IHL: float,
    eps_st: float,
    eps_op: float,
    dipole: float | str,
    rho_solvent: float,
    m_solvent: float,
    c_ions: float,
    C_EDL: float,
    PZFCvsSHE: float,
    T: float = 298.15,
    eta_Kirkwood: float = 1,
    g_Kirkwood: float = 2.4,
    print_data: bool = False,
) -> _ScalarArrayFunction:
    """Build an electrical double layer (EDL) model of the potential drop.

    Returns a function mapping the distance from the electrode (in Angstrom)
    to the potential drop (in V).
    """
    EvsPZFC = EvsSHE - PZFCvsSHE
    if print_data:
        print(f"E vs. SHE = {EvsSHE:.2f} V")

    sigma_M = (C_EDL * EvsPZFC) / (elementary_charge * 1e6 * (cm2A * A2Bohr) ** 2)

    if print_data:
        print(f"sigma_M = {sigma_M:.6e} a.u.")

    n_solvent_au = rho_solvent / m_solvent * N_A / (cm2A * A2Bohr) ** 3

    if dipole == "calculate":
        if eta_Kirkwood != 1:
            eta_Kirkwood = 2 * (2 * eps_st + eps_op) / (3 * g_Kirkwood * eps_st)
        dipole_au = (
            3
            / (2 + eps_op)
            * np.sqrt(
                3
                * kB
                * T
                * eV2Ha
                * (eps_st - eps_op)
                * eta_Kirkwood
                / (8 * np.pi * n_solvent_au)
            )
        )
        if print_data:
            print(f"dipole = {dipole_au:.6f} a.u.")
    elif is_number(dipole):
        dipole_au = float(dipole) * Debye2au
    else:
        raise ValueError("'dipole' must be a number (in Debye) or 'calculate'.")

    n_ions_au = (c_ions * N_A * 1000) / (1e10 * A2Bohr) ** 3
    dIHL_Bohr = dIHL * A2Bohr
    dOHL_Bohr = dOHL * A2Bohr

    phi_OHP_au = (
        2
        * kB
        * T
        * eV2Ha
        * np.arcsinh(
            sigma_M / np.sqrt((2 * kB * T * eV2Ha * eps_st * n_ions_au) / np.pi)
        )
    )
    phi_OHP = phi_OHP_au * Ha2eV

    if print_data:
        print(f"phi_OHP = {phi_OHP:.6f} V")

    def langevin(u: float | FloatArray) -> float | FloatArray:
        # cast: numpy ufuncs are typed as returning Any for scalar inputs
        return cast("float | FloatArray", 1 / np.tanh(u) - 1 / u)

    def eps_ohl(E_OHL_au: float | FloatArray) -> float | FloatArray:
        return cast(
            "float | FloatArray",
            eps_op
            + (4 * np.pi * (2 + eps_op))
            / (3 * E_OHL_au)
            * n_solvent_au
            * dipole_au
            * langevin(((2 + eps_op) * dipole_au * E_OHL_au) / (2 * kB * T * eV2Ha)),
        )

    def ohl_field_equation(E_OHL_au: float | FloatArray) -> float | FloatArray:
        return cast(
            "float | FloatArray",
            E_OHL_au * ((dIHL_Bohr * eps_ohl(E_OHL_au) / eps_IHL) + dOHL_Bohr)
            - EvsPZFC * eV2Ha
            + phi_OHP_au,
        )

    E_solution_au = fsolve(ohl_field_equation, x0=0.1)

    E_OHL_au = float(E_solution_au[0])
    E_OHL = E_OHL_au * Ha2eV / Bohr2A

    E_IHL_au = float(E_OHL_au * eps_ohl(E_OHL_au) / eps_IHL)
    E_IHL = E_IHL_au * Ha2eV / Bohr2A

    if print_data:
        print(f"eps_OHL = {eps_ohl(E_OHL_au):.6f}")
        print(f"E_OHL = {E_OHL:.6f} V/A")
        print(f"E_IHL = {E_IHL:.6f} V/A")
        print()

    kappa = float(np.sqrt((8 * np.pi * n_ions_au) / (eps_st * kB * T * eV2Ha)) / Bohr2A)

    @overload
    def edl_potential_drop(R: float) -> float: ...
    @overload
    def edl_potential_drop(R: FloatArray) -> FloatArray: ...
    def edl_potential_drop(R: float | FloatArray) -> float | FloatArray:
        if is_number(R):
            R_num = float(R)
            if R_num <= dIHL:
                return EvsPZFC - R_num * E_IHL
            elif (dIHL < R_num) and (R_num <= dIHL + dOHL):
                return EvsPZFC - dIHL * E_IHL - (R_num - dIHL) * E_OHL
            else:
                return float(
                    4
                    * kB
                    * T
                    * np.arctanh(
                        np.tanh(phi_OHP / (4 * kB * T))
                        * np.exp(-kappa * (R_num - dIHL - dOHL))
                    )
                )
        elif is_array(R):
            R_arr = np.asarray(R, dtype=np.float64)
            result = np.zeros(len(R_arr))
            for i, Ri in enumerate(R_arr):
                if Ri <= dIHL:
                    result[i] = EvsPZFC - Ri * E_IHL
                elif (dIHL < Ri) and (Ri <= dIHL + dOHL):
                    result[i] = EvsPZFC - dIHL * E_IHL - (Ri - dIHL) * E_OHL
                else:
                    result[i] = (
                        4
                        * kB
                        * T
                        * np.arctanh(
                            np.tanh(phi_OHP / (4 * kB * T))
                            * np.exp(-kappa * (Ri - dIHL - dOHL))
                        )
                    )
            return result
        else:
            raise TypeError("'R' must be a number or a 1D array.")

    return edl_potential_drop


@overload
def fermi_distribution(E: float, E_Fermi: float = 0, T: float = 298.15) -> float: ...
@overload
def fermi_distribution(
    E: FloatArray, E_Fermi: float = 0, T: float = 298.15
) -> FloatArray: ...
def fermi_distribution(
    E: float | FloatArray, E_Fermi: float = 0, T: float = 298.15
) -> float | FloatArray:
    """Fermi-Dirac occupation of a state at energy ``E`` (in eV)."""
    return cast("float | FloatArray", 1 / (np.exp((E - E_Fermi) / kB / T) + 1))
