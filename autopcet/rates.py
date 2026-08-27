"""Golden-rule PCET rate constants from diabatic proton potentials."""

import numpy as np
from scipy.integrate import simpson

from ._types import FloatArray, PotentialFunction
from .constants import hbar, kB, massH
from .fgh import _solve_vibrational_states
from .potentials import fit_bspline, fit_poly6, fit_poly8
from .utils import is_array


def _smoothed_potential(
    pot: PotentialFunction | tuple[FloatArray, FloatArray] | list[FloatArray],
    smooth: str,
    name: str,
) -> tuple[PotentialFunction, float | None, float | None]:
    """Turn a tabulated ``(rp, V)`` potential into a callable via smoothing.

    Callables pass through unchanged. Returns the potential function and the
    bounds of the tabulated data (``None`` for callables).
    """
    if callable(pot):
        return pot, None, None
    if is_array(pot) and len(pot) == 2:
        r = np.asarray(pot[0], dtype=np.float64)
        v = np.asarray(pot[1], dtype=np.float64)
        if smooth == "poly6":
            fitted = fit_poly6(r, v)
        elif smooth == "poly8":
            fitted = fit_poly8(r, v)
        elif smooth == "bspline":
            fitted = fit_bspline(r, v)
        else:
            raise ValueError("'smooth' must be one of 'poly6', 'poly8', or 'bspline'.")
        return fitted, float(np.min(r)), float(np.max(r))
    raise TypeError(f"'{name}' must be a callable or a pair (rp, V) of 1D arrays.")


class PCET:
    """Golden-rule PCET rate constants from diabatic proton potentials.

    Takes the reactant and product proton potentials (callables in eV, or
    tabulated ``(rp, V)`` pairs to be smoothed), the reaction free energy
    ``DeltaG`` (eV), the reorganization energy ``Lambda`` (eV), and the
    electronic coupling ``Vel`` (eV), and computes the vibronically
    nonadiabatic PCET rate constant.
    """

    rp: FloatArray
    ReacProtonPot: PotentialFunction
    ProdProtonPot: PotentialFunction
    DeltaG: float
    Lambda: float
    Vel: float
    NStates: int
    Pu: FloatArray
    Suv: FloatArray
    dGuv: FloatArray
    kuv: FloatArray
    Iuv: FloatArray
    k_tot: float
    MassUsedPreviously: float | None
    ReacProtonEnergyLevels: FloatArray
    ReacProtonWaveFunctions: FloatArray
    ProdProtonEnergyLevels: FloatArray
    ProdProtonWaveFunctions: FloatArray

    def __init__(
        self,
        ReacProtonPot: PotentialFunction
        | tuple[FloatArray, FloatArray]
        | list[FloatArray],
        ProdProtonPot: PotentialFunction
        | tuple[FloatArray, FloatArray]
        | list[FloatArray],
        DeltaG: float,
        Lambda: float,
        Vel: float = 0.0434,
        NStates: int = 10,
        NGridPot: int = 256,
        smooth: str = "bspline",
        rmin: float | None = None,
        rmax: float | None = None,
    ) -> None:
        self.ReacProtonPot, rmin1, rmax1 = _smoothed_potential(
            ReacProtonPot, smooth, "ReacProtonPot"
        )
        self.ProdProtonPot, rmin2, rmax2 = _smoothed_potential(
            ProdProtonPot, smooth, "ProdProtonPot"
        )

        if rmin is None:
            rmin = (
                min(rmin1, rmin2) if rmin1 is not None and rmin2 is not None else -0.8
            )
        if rmax is None:
            rmax = max(rmax1, rmax2) if rmax1 is not None and rmax2 is not None else 0.8
        self.rp = np.linspace(rmin, rmax, NGridPot)

        self.DeltaG = DeltaG
        self.Lambda = Lambda
        self.Vel = Vel
        self.NStates = NStates

        self.Pu = np.zeros(NStates)
        self.Suv = np.zeros((NStates, NStates))
        self.dGuv = np.zeros((NStates, NStates))
        self.kuv = np.zeros((NStates, NStates))
        self.Iuv = np.zeros((NStates, NStates))
        self.k_tot = 0.0
        self.MassUsedPreviously = None

    def calc_proton_vibrational_states(self, mass: float = massH) -> None:
        """Solve for the proton vibrational states in both diabatic potentials."""
        self.MassUsedPreviously = mass
        E_reac = np.asarray(self.ReacProtonPot(self.rp), dtype=np.float64)
        E_prod = np.asarray(self.ProdProtonPot(self.rp), dtype=np.float64)

        (
            self.ReacProtonEnergyLevels,
            self.ReacProtonWaveFunctions,
            _,
        ) = _solve_vibrational_states(self.rp, E_reac, mass, self.NStates)
        (
            self.ProdProtonEnergyLevels,
            self.ProdProtonWaveFunctions,
            _,
        ) = _solve_vibrational_states(self.rp, E_prod, mass, self.NStates)

    def calc_reactant_state_distribution(self, T: float = 298.15) -> FloatArray:
        """Compute the Boltzmann populations of the reactant proton states."""
        Boltzmann_factors = np.exp(-self.ReacProtonEnergyLevels / kB / T)
        partition_func = np.sum(Boltzmann_factors)
        self.Pu = Boltzmann_factors / partition_func
        return self.Pu

    def calc_proton_overlap_matrix(self) -> FloatArray:
        """Compute the overlap matrix of reactant and product proton states."""
        for u in range(self.NStates):
            for v in range(self.NStates):
                self.Suv[u, v] = simpson(
                    self.ReacProtonWaveFunctions[u] * self.ProdProtonWaveFunctions[v],
                    x=self.rp,
                )
        return self.Suv

    def calc_reaction_free_energy_matrix(self) -> FloatArray:
        """Compute the reaction free energy for each pair of proton states."""
        for u in range(self.NStates):
            for v in range(self.NStates):
                self.dGuv[u, v] = (
                    self.DeltaG
                    + (self.ProdProtonEnergyLevels[v] - self.ProdProtonEnergyLevels[0])
                    - (self.ReacProtonEnergyLevels[u] - self.ReacProtonEnergyLevels[0])
                )
        return self.dGuv

    def calc_rate_contribution_matrix(self, T: float = 298.15) -> FloatArray:
        """Compute the rate contribution of each pair of proton states."""
        k0 = 2 * np.pi / hbar * self.Vel * self.Vel
        self.Iuv = (
            1
            / np.sqrt(4 * np.pi * self.Lambda * kB * T)
            * np.exp(-((self.dGuv + self.Lambda) ** 2) / (4 * self.Lambda * kB * T))
        )
        self.kuv = k0 * np.matmul(np.diag(self.Pu), self.Suv * self.Suv * self.Iuv)
        return self.kuv

    def calculate(
        self,
        mass: float = massH,
        T: float = 298.15,
        reuse_saved_proton_states: bool = False,
    ) -> float:
        """Compute the total PCET rate constant at temperature ``T``."""
        if self.MassUsedPreviously != mass:
            reuse_saved_proton_states = False

        if not reuse_saved_proton_states:
            self.calc_proton_vibrational_states(mass)
            self.calc_proton_overlap_matrix()

        self.calc_reactant_state_distribution(T)
        self.calc_reaction_free_energy_matrix()
        self.calc_rate_contribution_matrix(T)

        self.k_tot = np.sum(self.kuv)
        return self.k_tot

    def set_parameters(
        self,
        DeltaG: float | None = None,
        Lambda: float | None = None,
        Vel: float | None = None,
    ) -> None:
        """Update the reaction free energy, reorganization energy, or coupling."""
        if DeltaG is not None:
            self.DeltaG = DeltaG
        if Lambda is not None:
            self.Lambda = Lambda
        if Vel is not None:
            self.Vel = Vel

    def get_reactant_proton_states(self) -> tuple[FloatArray, FloatArray]:
        """Return the reactant proton energy levels and wave functions."""
        return self.ReacProtonEnergyLevels, self.ReacProtonWaveFunctions

    def get_product_proton_states(self) -> tuple[FloatArray, FloatArray]:
        """Return the product proton energy levels and wave functions."""
        return self.ProdProtonEnergyLevels, self.ProdProtonWaveFunctions

    def get_reactant_state_distribution(self) -> FloatArray:
        """Return the Boltzmann populations of the reactant proton states."""
        return self.Pu

    def get_proton_overlap_matrix(self) -> FloatArray:
        """Return the overlap matrix of reactant and product proton states."""
        return self.Suv

    def get_reaction_free_energy_matrix(self) -> FloatArray:
        """Return the reaction free energy matrix."""
        return self.dGuv

    def get_activation_free_energy_matrix(self) -> FloatArray:
        """Return the Marcus activation free energy matrix."""
        return (self.dGuv + self.Lambda) ** 2 / (4 * self.Lambda)

    def get_rate_contribution_matrix(self) -> FloatArray:
        """Return the rate contribution matrix."""
        return self.kuv

    def get_total_rate_constant(self) -> float:
        """Return the total PCET rate constant."""
        return self.k_tot
