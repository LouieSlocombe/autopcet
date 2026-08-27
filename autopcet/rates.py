"""Golden-rule PCET rate constants from diabatic proton potentials."""

import numpy as np
from scipy.integrate import simpson

from ._types import FitMethod, FloatArray, PotentialFunction, TabulatedPotential
from .constants import (
    ANGSTROM_TO_BOHR,
    BOLTZMANN,
    HARTREE_TO_EV,
    HBAR,
    MASS_PROTON,
    ROOM_TEMPERATURE,
)
from .fgh import _solve_proton_states
from .potentials import fit_potential
from .utils import is_array

_DEFAULT_GRID_HALF_WIDTH = 0.8
"""Half-width in angstrom of the proton grid used for callable potentials."""


def _as_potential_function(
    potential: PotentialFunction | TabulatedPotential,
    fit_method: FitMethod,
    name: str,
) -> tuple[PotentialFunction, tuple[float, float] | None]:
    """Turn a tabulated ``(rp, V)`` potential into a callable by smoothing it.

    Callables pass through unchanged. Returns the potential function and the
    range spanned by the tabulated data (``None`` for callables).
    """
    if callable(potential):
        return potential, None
    if is_array(potential) and len(potential) == 2:
        rp = np.asarray(potential[0], dtype=np.float64)
        energy = np.asarray(potential[1], dtype=np.float64)
        return fit_potential(rp, energy, fit_method), (
            float(np.min(rp)),
            float(np.max(rp)),
        )
    raise TypeError(f"'{name}' must be a callable or a pair (rp, V) of 1D arrays.")


class PCET:
    """Golden-rule PCET rate constants from diabatic proton potentials.

    Takes the reactant and product proton potentials -- callables returning eV
    given the proton coordinate in angstrom, or tabulated ``(rp, V)`` pairs to
    be smoothed -- along with the reaction free energy, the reorganization
    energy, and the electronic coupling, all in eV.

    Calling :meth:`calculate` fills in the per-state results: ``populations``,
    ``overlaps``, ``pair_free_energies``, ``pair_activation_energies``,
    ``rate_contributions``, and ``total_rate_constant``.
    """

    rp: FloatArray
    """Proton coordinate grid in angstrom."""

    reactant_potential: PotentialFunction
    product_potential: PotentialFunction
    reaction_free_energy: float
    reorganization_energy: float
    electronic_coupling: float
    n_states: int

    reactant_energies: FloatArray
    reactant_wavefunctions: FloatArray
    product_energies: FloatArray
    product_wavefunctions: FloatArray

    populations: FloatArray
    """Boltzmann population of each reactant proton state."""

    overlaps: FloatArray
    """Overlap of each reactant state with each product state."""

    pair_free_energies: FloatArray
    """Reaction free energy of each reactant/product state pair."""

    rate_contributions: FloatArray
    """Rate constant contributed by each reactant/product state pair."""

    total_rate_constant: float
    """Sum of ``rate_contributions``, in inverse seconds."""

    _mass: float | None
    """Mass the currently stored proton states were solved for."""

    def __init__(
        self,
        reactant_potential: PotentialFunction | TabulatedPotential,
        product_potential: PotentialFunction | TabulatedPotential,
        reaction_free_energy: float,
        reorganization_energy: float,
        electronic_coupling: float = 0.0434,
        n_states: int = 10,
        n_grid: int = 256,
        fit_method: FitMethod = "bspline",
        r_min: float | None = None,
        r_max: float | None = None,
    ) -> None:
        self.reactant_potential, reactant_range = _as_potential_function(
            reactant_potential, fit_method, "reactant_potential"
        )
        self.product_potential, product_range = _as_potential_function(
            product_potential, fit_method, "product_potential"
        )

        # Tabulated potentials set the grid limits from their own data; a pair
        # of callables has no data to read them off, so fall back to defaults.
        if reactant_range is not None and product_range is not None:
            data_min = min(reactant_range[0], product_range[0])
            data_max = max(reactant_range[1], product_range[1])
        else:
            data_min = -_DEFAULT_GRID_HALF_WIDTH
            data_max = _DEFAULT_GRID_HALF_WIDTH

        self.rp = np.linspace(
            data_min if r_min is None else r_min,
            data_max if r_max is None else r_max,
            n_grid,
        )

        self.reaction_free_energy = reaction_free_energy
        self.reorganization_energy = reorganization_energy
        self.electronic_coupling = electronic_coupling
        self.n_states = n_states

        self.populations = np.zeros(n_states)
        self.overlaps = np.zeros((n_states, n_states))
        self.pair_free_energies = np.zeros((n_states, n_states))
        self.rate_contributions = np.zeros((n_states, n_states))
        self.total_rate_constant = 0.0
        self._mass = None

    @property
    def pair_activation_energies(self) -> FloatArray:
        """Marcus activation free energy of each reactant/product state pair."""
        return (self.pair_free_energies + self.reorganization_energy) ** 2 / (
            4 * self.reorganization_energy
        )

    def calculate(
        self,
        mass: float = MASS_PROTON,
        temperature: float = ROOM_TEMPERATURE,
        reuse_states: bool = False,
    ) -> float:
        """Compute the total PCET rate constant at the given temperature.

        Set ``reuse_states`` to skip re-solving the proton vibrational states,
        which is worth doing when only the thermodynamic parameters changed.
        Stored states for a different mass are always re-solved.
        """
        if not reuse_states or self._mass != mass:
            self._solve_states(mass)
            self._compute_overlaps()

        self._compute_populations(temperature)
        self._compute_pair_free_energies()
        self._compute_rate_contributions(temperature)

        self.total_rate_constant = float(np.sum(self.rate_contributions))
        return self.total_rate_constant

    def _solve_states(self, mass: float) -> None:
        """Solve for the proton vibrational states in both diabatic potentials."""
        self._mass = mass
        reactant_energy = np.asarray(self.reactant_potential(self.rp), dtype=np.float64)
        product_energy = np.asarray(self.product_potential(self.rp), dtype=np.float64)

        self.reactant_energies, self.reactant_wavefunctions = _solve_proton_states(
            self.rp, reactant_energy, mass, self.n_states
        )
        self.product_energies, self.product_wavefunctions = _solve_proton_states(
            self.rp, product_energy, mass, self.n_states
        )

    def _compute_overlaps(self) -> None:
        """Overlap integrals between every reactant and product proton state."""
        products = (
            self.reactant_wavefunctions[:, np.newaxis, :]
            * self.product_wavefunctions[np.newaxis, :, :]
        )
        self.overlaps = simpson(products, x=self.rp, axis=2)

    def _compute_populations(self, temperature: float) -> None:
        """Boltzmann populations of the reactant proton states."""
        boltzmann_factors = np.exp(-self.reactant_energies / BOLTZMANN / temperature)
        self.populations = boltzmann_factors / np.sum(boltzmann_factors)

    def _compute_pair_free_energies(self) -> None:
        """Reaction free energy for each pair of proton states."""
        product_excitation = self.product_energies - self.product_energies[0]
        reactant_excitation = self.reactant_energies - self.reactant_energies[0]
        self.pair_free_energies = (
            self.reaction_free_energy + product_excitation[np.newaxis, :]
        ) - reactant_excitation[:, np.newaxis]

    def _compute_rate_contributions(self, temperature: float) -> None:
        """Rate constant contributed by each pair of proton states."""
        thermal_energy = BOLTZMANN * temperature
        reorganization = self.reorganization_energy
        prefactor = 2 * np.pi / HBAR * self.electronic_coupling**2

        # Marcus nuclear factor for each state pair
        marcus_factors = (
            1
            / np.sqrt(4 * np.pi * reorganization * thermal_energy)
            * np.exp(
                -((self.pair_free_energies + reorganization) ** 2)
                / (4 * reorganization * thermal_energy)
            )
        )

        self.rate_contributions = (
            prefactor
            * self.populations[:, np.newaxis]
            * self.overlaps**2
            * marcus_factors
        )


def donor_acceptor_distribution(
    distance: FloatArray,
    equilibrium: float,
    force_constant: float,
    temperature: float = ROOM_TEMPERATURE,
) -> FloatArray:
    """Harmonic ``P(R)`` for the proton donor-acceptor mode, in angstrom.

    The donor-acceptor coordinate is treated as a classical harmonic oscillator
    about ``equilibrium``, so its distribution is the Boltzmann weight of
    ``k (R - R_eq)^2 / 2``. ``force_constant`` is in atomic units, as
    :func:`autopcet.gaussian_io.effective_da_mode` returns it.

    The weights are returned unnormalized: a rate constant thermally averaged
    over R needs both ``P(R)`` and ``k(R)`` integrated on the same grid, so
    normalization is left to the caller.
    """
    energy = (
        0.5
        * force_constant
        * (distance - equilibrium) ** 2
        * ANGSTROM_TO_BOHR**2
        * HARTREE_TO_EV
    )
    weights: FloatArray = np.exp(-energy / (BOLTZMANN * temperature))
    return weights
