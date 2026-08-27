"""Vibronic couplings and nonadiabaticity analysis for PCET reactions."""

import warnings

import numpy as np
from scipy.integrate import simpson
from scipy.special import gamma

from ._types import FloatArray
from .constants import (
    AU_TIME_TO_SECONDS,
    BOHR_TO_ANGSTROM,
    EV_TO_HARTREE,
    HBAR,
    MASS_PROTON,
)
from .fgh import _solve_proton_states
from .utils import _find_first_crossing, is_array, is_number


def _best_overlap_index(overlaps: FloatArray, exclude: int) -> int:
    """Index of the largest overlap, skipping ``exclude``.

    The search starts from index 0, so when ``exclude`` is 0 its own overlap
    still sets the bar the other states have to clear, and is returned if none
    of them do. The caller warns about that case rather than hiding it.
    """
    best = 0
    for i in range(1, len(overlaps)):
        if i != exclude and overlaps[i] > overlaps[best]:
            best = i
    return best


class KappaCoupling:
    """Vibronic couplings and nonadiabaticity analysis for a PCET reaction.

    Takes the reactant and product proton potentials (in eV) tabulated on the
    proton coordinate grid ``rp`` (in angstrom) and the electronic coupling
    (in eV, constant or tabulated on the same grid), and computes the
    semiclassical vibronic coupling and the nonadiabaticity parameters of
    Georgievskii and Stuchebrukhov.

    ``mu`` and ``nu`` select which reactant and product vibrational states the
    analysis is carried out for.
    """

    rp: FloatArray
    reactant_potential: FloatArray
    product_potential: FloatArray
    electronic_coupling: FloatArray
    n_states: int
    mu: int
    nu: int

    reactant_energies: FloatArray
    reactant_wavefunctions: FloatArray
    product_energies: FloatArray
    product_wavefunctions: FloatArray

    shifted_reactant_potential: FloatArray
    """Reactant potential shifted so that states ``mu`` and ``nu`` are degenerate."""

    shifted_product_potential: FloatArray
    shifted_reactant_energies: FloatArray
    shifted_product_energies: FloatArray

    crossing_rp: float
    """Proton coordinate where the shifted diabatic potentials cross."""

    crossing_energy: float
    crossing_coupling: float
    reactant_slope: float
    product_slope: float

    tau_proton: float
    """Proton tunneling time in seconds."""

    tau_electron: float
    """Electronic transition time in seconds."""

    adiabaticity: float
    """Ratio ``tau_proton / tau_electron``; small values are nonadiabatic."""

    kappa: float
    """Prefactor bridging the nonadiabatic and adiabatic coupling limits."""

    ground_adiabat: FloatArray
    excited_adiabat: FloatArray
    adiabatic_energies: FloatArray
    adiabatic_wavefunctions: FloatArray

    v_adiabatic: float
    """Adiabatic vibronic coupling, half the tunneling splitting."""

    v_nonadiabatic: float
    """Nonadiabatic vibronic coupling, the coupling times the state overlap."""

    v_semiclassical: float
    """Semiclassical vibronic coupling, ``kappa * v_adiabatic``."""

    def __init__(
        self,
        rp: FloatArray,
        reactant_potential: FloatArray,
        product_potential: FloatArray,
        electronic_coupling: float | FloatArray,
        n_states: int = 10,
        mu: int = 0,
        nu: int = 0,
    ) -> None:
        if not (
            is_array(rp)
            and is_array(reactant_potential)
            and is_array(product_potential)
        ):
            raise TypeError(
                "'rp', 'reactant_potential', and 'product_potential' must be 1D arrays."
            )

        if is_array(electronic_coupling):
            coupling = np.asarray(electronic_coupling, dtype=np.float64)
        elif is_number(electronic_coupling):
            coupling = np.full(len(rp), float(electronic_coupling))
        else:
            raise TypeError("'electronic_coupling' must be a number or a 1D array.")

        if not (
            len(reactant_potential)
            == len(product_potential)
            == len(coupling)
            == len(rp)
        ):
            raise ValueError(
                "'rp', 'reactant_potential', 'product_potential', and "
                "'electronic_coupling' must have the same dimension."
            )

        if np.abs(np.log2(len(rp)) - int(np.log2(len(rp)))) > 1e-4:
            raise ValueError(
                "The number of grid points should be some integer power of 2."
            )

        if not (0 <= mu < n_states and 0 <= nu < n_states):
            raise ValueError("'mu' and 'nu' must satisfy 0 <= mu, nu < n_states.")

        self.rp = np.asarray(rp, dtype=np.float64)
        self.reactant_potential = np.asarray(reactant_potential, dtype=np.float64)
        self.product_potential = np.asarray(product_potential, dtype=np.float64)
        self.electronic_coupling = coupling
        self.n_states = n_states
        self.mu = mu
        self.nu = nu

    def calculate(
        self, mass: float = MASS_PROTON, overlap_threshold: float = 0.8
    ) -> None:
        """Compute the nonadiabaticity parameters and the vibronic couplings.

        ``overlap_threshold`` sets how well the proton states of the ground
        adiabatic potential must match the symmetric and antisymmetric
        combinations of the diabatic states before a warning is issued.
        """
        self._solve_states(mass)
        self._align_potentials()
        self._locate_crossing()
        self._compute_nonadiabaticity(mass)
        self._build_adiabats()
        self._compute_couplings(mass, overlap_threshold)

    def _solve_states(self, mass: float) -> None:
        """Solve for the proton vibrational states in both diabatic potentials."""
        self.reactant_energies, self.reactant_wavefunctions = _solve_proton_states(
            self.rp, self.reactant_potential, mass, self.n_states
        )
        self.product_energies, self.product_wavefunctions = _solve_proton_states(
            self.rp, self.product_potential, mass, self.n_states
        )

    def _align_potentials(self) -> None:
        """Shift the potentials so that states ``mu`` and ``nu`` are degenerate."""
        gap = self.product_energies[self.nu] - self.reactant_energies[self.mu]
        reactant_shift, product_shift = max(gap, 0.0), max(-gap, 0.0)

        self.shifted_reactant_potential = self.reactant_potential + reactant_shift
        self.shifted_product_potential = self.product_potential + product_shift
        self.shifted_reactant_energies = self.reactant_energies + reactant_shift
        self.shifted_product_energies = self.product_energies + product_shift

    def _locate_crossing(self) -> None:
        """Find where the shifted diabatic potentials cross, and how steeply."""
        index, self.crossing_rp = _find_first_crossing(
            self.rp, self.shifted_reactant_potential - self.shifted_product_potential
        )
        bracket = slice(index - 1, index + 1)

        self.crossing_energy = float(
            np.mean(
                [
                    self.shifted_reactant_potential[bracket],
                    self.shifted_product_potential[bracket],
                ]
            )
        )
        self.crossing_coupling = float(np.mean(self.electronic_coupling[bracket]))

        rp_step = self.rp[index] - self.rp[index - 1]
        self.reactant_slope = float(
            np.diff(self.shifted_reactant_potential[bracket])[0] / rp_step
        )
        self.product_slope = float(
            np.diff(self.shifted_product_potential[bracket])[0] / rp_step
        )

    def _compute_nonadiabaticity(self, mass: float) -> None:
        """Compare the proton tunneling time with the electronic transition time."""
        tunneling_energy = self.shifted_reactant_energies[self.mu]
        if self.crossing_energy < tunneling_energy:
            raise RuntimeError(
                "The tunneling energy is higher than the energy at the crossing point."
            )

        tunneling_velocity = (
            np.sqrt(
                2 * (self.crossing_energy - tunneling_energy) * EV_TO_HARTREE / mass
            )
            * BOHR_TO_ANGSTROM
            / AU_TIME_TO_SECONDS
        )

        self.tau_proton = self.crossing_coupling / (
            np.abs(self.reactant_slope - self.product_slope) * tunneling_velocity
        )
        self.tau_electron = HBAR / self.crossing_coupling
        self.adiabaticity = self.tau_proton / self.tau_electron

        p = self.adiabaticity
        self.kappa = np.sqrt(2 * np.pi * p) * np.exp(p * np.log(p) - p) / gamma(p + 1)

    def _build_adiabats(self) -> None:
        """Mix the shifted diabats with the coupling at their crossing point."""
        mean = 0.5 * (self.shifted_reactant_potential + self.shifted_product_potential)
        half_splitting = 0.5 * np.sqrt(
            (self.shifted_product_potential - self.shifted_reactant_potential) ** 2
            + 4 * self.crossing_coupling**2
        )
        self.ground_adiabat = mean - half_splitting
        self.excited_adiabat = mean + half_splitting

    def _compute_couplings(self, mass: float, overlap_threshold: float) -> None:
        """Evaluate the adiabatic, nonadiabatic, and semiclassical couplings."""
        self.adiabatic_energies, self.adiabatic_wavefunctions = _solve_proton_states(
            self.rp, self.ground_adiabat, mass, 2 * self.n_states
        )

        reactant = self.reactant_wavefunctions[self.mu]
        product = self.product_wavefunctions[self.nu]
        overlap = simpson(reactant * product, x=self.rp)

        # Pick the relative phase that makes the in-phase combination the
        # symmetric, nodeless one.
        phase = 1.0 if overlap > 0 else -1.0
        symmetric = self._normalized(reactant + phase * product)
        antisymmetric = self._normalized(reactant - phase * product)

        overlap_symmetric = self._overlaps_with(symmetric)
        overlap_antisymmetric = self._overlaps_with(antisymmetric)

        symmetric_index = int(np.argmax(overlap_symmetric))
        antisymmetric_index = _best_overlap_index(
            overlap_antisymmetric, exclude=symmetric_index
        )
        self._warn_on_poor_assignment(
            overlap_symmetric[symmetric_index],
            overlap_antisymmetric[antisymmetric_index],
            symmetric_index,
            antisymmetric_index,
            overlap_threshold,
        )

        tunneling_splitting = (
            self.adiabatic_energies[antisymmetric_index]
            - self.adiabatic_energies[symmetric_index]
        )
        self.v_adiabatic = 0.5 * tunneling_splitting
        self.v_nonadiabatic = self.crossing_coupling * overlap
        self.v_semiclassical = self.kappa * self.v_adiabatic

    def _normalized(self, wavefunction: FloatArray) -> FloatArray:
        """Scale a wave function to unit norm on the proton grid."""
        normalized: FloatArray = wavefunction / np.sqrt(
            simpson(wavefunction**2, x=self.rp)
        )
        return normalized

    def _overlaps_with(self, wavefunction: FloatArray) -> FloatArray:
        """Absolute overlap of every adiabatic state with ``wavefunction``."""
        overlaps: FloatArray = np.abs(
            simpson(self.adiabatic_wavefunctions * wavefunction, x=self.rp, axis=1)
        )
        return overlaps

    @staticmethod
    def _warn_on_poor_assignment(
        symmetric_overlap: float,
        antisymmetric_overlap: float,
        symmetric_index: int,
        antisymmetric_index: int,
        overlap_threshold: float,
    ) -> None:
        """Flag adiabatic states that do not cleanly match the diabatic pair."""
        states = (
            f"The symmetric state is state {symmetric_index:d} and the "
            f"antisymmetric state is state {antisymmetric_index:d}."
        )
        if min(symmetric_overlap, antisymmetric_overlap) < overlap_threshold:
            warnings.warn(
                "The maximum overlap between the proton vibrational wave functions "
                "in the adiabatic potential and the symmetric/antisymmetric "
                "combinations of the wave functions in diabatic potentials is less "
                f"than {overlap_threshold:.1f}.",
                stacklevel=2,
            )
        if antisymmetric_index < symmetric_index:
            warnings.warn(
                "The identified antisymmetric state is lower in energy than the "
                f"identified symmetric state. {states}",
                stacklevel=2,
            )
        if np.abs(antisymmetric_index - symmetric_index) > 1:
            warnings.warn(
                "There are multiple states lying in between the identified "
                f"symmetric and antisymmetric states. {states}",
                stacklevel=2,
            )
