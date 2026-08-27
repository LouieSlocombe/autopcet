"""Vibronic couplings and nonadiabaticity analysis for PCET reactions."""

import warnings

import numpy as np
from scipy.integrate import simpson
from scipy.special import gamma

from ._types import FloatArray
from .constants import Bohr2A, Ha2eV, au2s, eV2Ha, hbar, massH
from .fgh import _solve_vibrational_states
from .utils import _find_first_crossing, is_array, is_number


class KappaCoupling:
    """Vibronic couplings and nonadiabaticity analysis for a PCET reaction.

    Takes the reactant and product proton potentials (in eV) tabulated on the
    proton coordinate grid ``rp`` (in Angstrom) and the electronic coupling
    ``Vel`` (in eV), and computes the semiclassical vibronic coupling and the
    nonadiabaticity parameters of Georgievskii and Stuchebrukhov.
    """

    rp: FloatArray
    ReacProtonPot: FloatArray
    ProdProtonPot: FloatArray
    Vel: FloatArray
    NStates: int
    mu: int
    nu: int
    ReacProtonEnergyLevels: FloatArray
    ReacProtonWaveFunctions: FloatArray
    ProdProtonEnergyLevels: FloatArray
    ProdProtonWaveFunctions: FloatArray
    ShiftedReacProtonPot: FloatArray
    ShiftedProdProtonPot: FloatArray
    ShiftedReacProtonEnergyLevels: FloatArray
    ShiftedProdProtonEnergyLevels: FloatArray
    rp_crossing: float
    E_crossing: float
    Vel_crossing: float
    slope_reac: float
    slope_prod: float
    tau_p: float
    tau_e: float
    p: float
    kappa: float
    AdiabaticProtonPotGS: FloatArray
    AdiabaticProtonPotES: FloatArray
    AdiabaticGSProtonEnergyLevels: FloatArray
    AdiabaticGSProtonWaveFunctions: FloatArray
    V_ad: float
    V_nad: float
    V_sc: float

    def __init__(
        self,
        rp: FloatArray,
        ReacProtonPot: FloatArray,
        ProdProtonPot: FloatArray,
        Vel: float | FloatArray,
        NStates: int = 10,
        mu: int = 0,
        nu: int = 0,
    ) -> None:
        if not (is_array(rp) and is_array(ReacProtonPot) and is_array(ProdProtonPot)):
            raise TypeError(
                "'rp', 'ReacProtonPot', and 'ProdProtonPot' must be 1D arrays."
            )
        if is_array(Vel):
            Vel_arr = np.asarray(Vel, dtype=np.float64)
        elif is_number(Vel):
            Vel_arr = np.ones(len(rp)) * Vel
        else:
            raise TypeError("'Vel' must be a number or a 1D array.")

        if (
            len(ReacProtonPot) != len(rp)
            or len(ProdProtonPot) != len(rp)
            or len(Vel_arr) != len(rp)
        ):
            raise ValueError(
                "'rp', 'ReacProtonPot', 'ProdProtonPot', and 'Vel' must have the same dimension."
            )

        if np.abs(np.log2(len(rp)) - int(np.log2(len(rp)))) > 1e-4:
            raise ValueError(
                "The number of grid points should be some integer power of 2."
            )

        if not (0 <= mu < NStates and 0 <= nu < NStates):
            raise ValueError("'mu' and 'nu' must satisfy 0 <= mu, nu < NStates.")

        self.rp = np.asarray(rp, dtype=np.float64)
        self.ReacProtonPot = np.asarray(ReacProtonPot, dtype=np.float64)
        self.ProdProtonPot = np.asarray(ProdProtonPot, dtype=np.float64)
        self.Vel = Vel_arr
        self.NStates = NStates
        self.mu = mu
        self.nu = nu

    def calc_proton_vibrational_states(self, mass: float = massH) -> None:
        """Solve for the proton vibrational states in both diabatic potentials."""
        (
            self.ReacProtonEnergyLevels,
            self.ReacProtonWaveFunctions,
            _,
        ) = _solve_vibrational_states(self.rp, self.ReacProtonPot, mass, self.NStates)
        (
            self.ProdProtonEnergyLevels,
            self.ProdProtonWaveFunctions,
            _,
        ) = _solve_vibrational_states(self.rp, self.ProdProtonPot, mass, self.NStates)

    def analyze_proton_potentials(self, mass: float = massH) -> None:
        """Shift the potentials to degeneracy and locate their crossing point."""
        self.calc_proton_vibrational_states(mass)
        E_reac = self.ReacProtonEnergyLevels[self.mu]
        E_prod = self.ProdProtonEnergyLevels[self.nu]

        if E_prod < E_reac:
            dEr = 0
            dEp = -E_prod + E_reac
        else:
            dEr = E_prod - E_reac
            dEp = 0

        self.ShiftedReacProtonPot = self.ReacProtonPot + dEr
        self.ShiftedProdProtonPot = self.ProdProtonPot + dEp
        self.ShiftedReacProtonEnergyLevels = self.ReacProtonEnergyLevels + dEr
        self.ShiftedProdProtonEnergyLevels = self.ProdProtonEnergyLevels + dEp

        deltaE = self.ShiftedReacProtonPot - self.ShiftedProdProtonPot

        rp_crossing_index, self.rp_crossing = _find_first_crossing(deltaE, self.rp)
        self.E_crossing = (
            self.ShiftedReacProtonPot[rp_crossing_index]
            + self.ShiftedReacProtonPot[rp_crossing_index - 1]
            + self.ShiftedProdProtonPot[rp_crossing_index]
            + self.ShiftedProdProtonPot[rp_crossing_index - 1]
        ) / 4
        self.Vel_crossing = (
            self.Vel[rp_crossing_index] + self.Vel[rp_crossing_index - 1]
        ) / 2

        self.slope_reac = (
            self.ShiftedReacProtonPot[rp_crossing_index]
            - self.ShiftedReacProtonPot[rp_crossing_index - 1]
        ) / (self.rp[rp_crossing_index] - self.rp[rp_crossing_index - 1])
        self.slope_prod = (
            self.ShiftedProdProtonPot[rp_crossing_index]
            - self.ShiftedProdProtonPot[rp_crossing_index - 1]
        ) / (self.rp[rp_crossing_index] - self.rp[rp_crossing_index - 1])

    def calculate(self, mass: float = massH, overlap_thresh: float = 0.8) -> None:
        """Compute the nonadiabaticity parameters and vibronic couplings."""
        self.analyze_proton_potentials(mass)

        E0 = self.ShiftedReacProtonEnergyLevels[self.mu]

        if self.E_crossing < E0:
            raise RuntimeError(
                "The tunneling energy is higher than the energy at the crossing point."
            )

        vt = np.sqrt(2 * (self.E_crossing - E0) * eV2Ha / mass) * Bohr2A / au2s

        self.tau_p = self.Vel_crossing / (
            np.abs(self.slope_reac - self.slope_prod) * vt
        )
        self.tau_e = hbar / self.Vel_crossing
        self.p = self.tau_p / self.tau_e
        self.kappa = (
            np.sqrt(2 * np.pi * self.p)
            * np.exp(self.p * np.log(self.p) - self.p)
            / gamma(self.p + 1)
        )

        self.AdiabaticProtonPotGS = 0.5 * (
            self.ShiftedReacProtonPot
            + self.ShiftedProdProtonPot
            - np.sqrt(
                (self.ShiftedProdProtonPot - self.ShiftedReacProtonPot) ** 2
                + 4 * self.Vel_crossing**2
            )
        )
        self.AdiabaticProtonPotES = 0.5 * (
            self.ShiftedReacProtonPot
            + self.ShiftedProdProtonPot
            + np.sqrt(
                (self.ShiftedProdProtonPot - self.ShiftedReacProtonPot) ** 2
                + 4 * self.Vel_crossing**2
            )
        )

        (
            self.AdiabaticGSProtonEnergyLevels,
            self.AdiabaticGSProtonWaveFunctions,
            eigvals,
        ) = _solve_vibrational_states(
            self.rp, self.AdiabaticProtonPotGS, mass, 2 * self.NStates
        )

        Smunu = simpson(
            self.ReacProtonWaveFunctions[self.mu]
            * self.ProdProtonWaveFunctions[self.nu],
            x=self.rp,
        )
        sign = 1 if Smunu > 0 else -1
        wfc_symm = (
            self.ReacProtonWaveFunctions[self.mu]
            + sign * self.ProdProtonWaveFunctions[self.nu]
        ) / np.sqrt(2)
        wfc_anti = (
            self.ReacProtonWaveFunctions[self.mu]
            - sign * self.ProdProtonWaveFunctions[self.nu]
        ) / np.sqrt(2)
        wfc_symm /= np.sqrt(simpson(wfc_symm**2, x=self.rp))
        wfc_anti /= np.sqrt(simpson(wfc_anti**2, x=self.rp))

        overlap_w_symm = np.array(
            [
                np.abs(simpson(wfci * wfc_symm, x=self.rp))
                for wfci in self.AdiabaticGSProtonWaveFunctions
            ]
        )
        overlap_w_anti = np.array(
            [
                np.abs(simpson(wfci * wfc_anti, x=self.rp))
                for wfci in self.AdiabaticGSProtonWaveFunctions
            ]
        )

        index_max_overlap_symm = 0
        for i in range(2 * self.NStates):
            if overlap_w_symm[i] > overlap_w_symm[index_max_overlap_symm]:
                index_max_overlap_symm = i

        index_max_overlap_anti = 0
        for i in range(2 * self.NStates):
            if (
                overlap_w_anti[i] > overlap_w_anti[index_max_overlap_anti]
                and i != index_max_overlap_symm
            ):
                index_max_overlap_anti = i

        if (
            overlap_w_symm[index_max_overlap_symm] < overlap_thresh
            or overlap_w_anti[index_max_overlap_anti] < overlap_thresh
        ):
            warnings.warn(
                "The maximum overlap between the proton vibrational wave functions in the "
                "adiabatic potential and the symmetric/antisymmetric combinations of the wave "
                f"functions in diabatic potentials is less than {overlap_thresh:.1f}.",
                stacklevel=2,
            )
        if index_max_overlap_anti < index_max_overlap_symm:
            warnings.warn(
                "The identified antisymmetric state is lower in energy than the identified "
                f"symmetric state. The symmetric state is state {index_max_overlap_symm:d} and "
                f"the antisymmetric state is state {index_max_overlap_anti:d}.",
                stacklevel=2,
            )
        if np.abs(index_max_overlap_anti - index_max_overlap_symm) > 1:
            warnings.warn(
                "There are multiple states lying in between the identified symmetric and "
                f"antisymmetric states. The symmetric state is state {index_max_overlap_symm:d} "
                f"and the antisymmetric state is state {index_max_overlap_anti:d}.",
                stacklevel=2,
            )

        tunneling_splitting = (
            eigvals[index_max_overlap_anti] - eigvals[index_max_overlap_symm]
        ) * Ha2eV
        self.V_ad = 0.5 * tunneling_splitting

        self.V_nad = self.Vel_crossing * Smunu
        self.V_sc = self.kappa * self.V_ad

    def get_reactant_proton_states(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return the shifted reactant potential, energy levels, and wave functions."""
        return (
            self.ShiftedReacProtonPot,
            self.ShiftedReacProtonEnergyLevels,
            self.ReacProtonWaveFunctions,
        )

    def get_product_proton_states(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return the shifted product potential, energy levels, and wave functions."""
        return (
            self.ShiftedProdProtonPot,
            self.ShiftedProdProtonEnergyLevels,
            self.ProdProtonWaveFunctions,
        )

    def get_adiabatic_proton_potentials(self) -> tuple[FloatArray, FloatArray]:
        """Return the ground- and excited-state adiabatic proton potentials."""
        return self.AdiabaticProtonPotGS, self.AdiabaticProtonPotES

    def get_ground_adiabatic_proton_states(self) -> tuple[FloatArray, FloatArray]:
        """Return the energy levels and wave functions in the ground adiabatic potential."""
        return self.AdiabaticGSProtonEnergyLevels, self.AdiabaticGSProtonWaveFunctions

    def get_nonadiabaticity_parameters(self) -> tuple[float, float, float, float]:
        """Return ``tau_e``, ``tau_p``, the adiabaticity parameter ``p``, and ``kappa``."""
        return self.tau_e, self.tau_p, self.p, self.kappa

    def get_vibronic_couplings(self) -> tuple[float, float, float]:
        """Return the semiclassical, nonadiabatic, and adiabatic vibronic couplings."""
        return self.V_sc, self.V_nad, self.V_ad
