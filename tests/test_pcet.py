"""Tests for the PCET rate-constant workflow used in example 1."""

import numpy as np
import pytest
from example1_data import (
    ELECTRONIC_COUPLING,
    PRODUCT_ENERGIES,
    REACTANT_ENERGIES,
    REACTION_FREE_ENERGY,
    REORGANIZATION_ENERGY,
    RP_GRID,
    TEMPERATURE,
)
from scipy.integrate import simpson

from autopcet import (
    ANGSTROM_TO_BOHR,
    BOLTZMANN,
    HARTREE_TO_EV,
    MASS_DEUTERON,
    MASS_PROTON,
    PCET,
    ROOM_TEMPERATURE,
    donor_acceptor_distribution,
)
from autopcet._types import FitMethod, PotentialFunction


@pytest.fixture
def system(
    reactant_potential: PotentialFunction, product_potential: PotentialFunction
) -> PCET:
    """A fresh PCET system set up exactly as in example 1."""
    return PCET(
        reactant_potential,
        product_potential,
        REACTION_FREE_ENERGY,
        REORGANIZATION_ENERGY,
        electronic_coupling=ELECTRONIC_COUPLING,
    )


def test_rate_constant_is_positive_and_kie_is_normal(system: PCET) -> None:
    """The example 1 system reproduces its rate constant and H/D KIE."""
    rate_h = system.calculate(MASS_PROTON, temperature=TEMPERATURE)
    assert rate_h == system.total_rate_constant

    rate_d = system.calculate(MASS_DEUTERON, temperature=TEMPERATURE)
    assert rate_d == system.total_rate_constant

    # regression values from running the example 1 workflow
    assert rate_h == pytest.approx(2.337e8, rel=0.05)
    assert rate_h / rate_d == pytest.approx(2.72, rel=0.05)


def test_reactant_state_distribution_is_a_boltzmann_distribution(
    system: PCET,
) -> None:
    """State populations are normalized and decrease with energy."""
    system.calculate(MASS_PROTON, temperature=TEMPERATURE)
    populations = system.populations

    assert populations.shape == (system.n_states,)
    assert np.sum(populations) == pytest.approx(1.0)
    assert np.all(np.diff(populations) <= 0)


def test_proton_states_are_normalized(system: PCET) -> None:
    """Vibrational energies are sorted and wave functions are normalized."""
    system.calculate(MASS_PROTON, temperature=TEMPERATURE)

    for energies, wavefunctions in (
        (system.reactant_energies, system.reactant_wavefunctions),
        (system.product_energies, system.product_wavefunctions),
    ):
        assert energies.shape == (system.n_states,)
        assert wavefunctions.shape == (system.n_states, len(system.rp))
        assert np.all(np.diff(energies) >= 0)
        norms = simpson(wavefunctions**2, x=system.rp, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-8)


def test_overlap_matrix_is_bounded(system: PCET) -> None:
    """Overlaps between normalized states cannot exceed one."""
    system.calculate(MASS_PROTON, temperature=TEMPERATURE)

    assert system.overlaps.shape == (system.n_states, system.n_states)
    assert np.all(np.abs(system.overlaps) <= 1.0 + 1e-6)


def test_free_energy_matrices_follow_their_definitions(system: PCET) -> None:
    """The pair free energies and activation energies match their definitions."""
    system.calculate(MASS_PROTON, temperature=TEMPERATURE)
    pair_free_energies = system.pair_free_energies

    assert pair_free_energies[0, 0] == pytest.approx(REACTION_FREE_ENERGY)
    expected = (
        REACTION_FREE_ENERGY
        + (system.product_energies[np.newaxis, :] - system.product_energies[0])
        - (system.reactant_energies[:, np.newaxis] - system.reactant_energies[0])
    )
    np.testing.assert_allclose(pair_free_energies, expected)
    np.testing.assert_allclose(
        system.pair_activation_energies,
        (pair_free_energies + REORGANIZATION_ENERGY) ** 2 / (4 * REORGANIZATION_ENERGY),
    )


def test_total_rate_is_the_sum_of_state_contributions(system: PCET) -> None:
    """The total rate accumulates the non-negative contribution matrix."""
    total = system.calculate(MASS_PROTON, temperature=TEMPERATURE)

    assert np.all(system.rate_contributions >= 0)
    assert total == pytest.approx(np.sum(system.rate_contributions))


def test_reusing_saved_proton_states_reproduces_the_rate(system: PCET) -> None:
    """Reusing cached vibrational states must not change the answer."""
    first = system.calculate(MASS_PROTON, temperature=TEMPERATURE)
    reused = system.calculate(MASS_PROTON, temperature=TEMPERATURE, reuse_states=True)
    deuterium = system.calculate(
        MASS_DEUTERON, temperature=TEMPERATURE, reuse_states=True
    )

    assert reused == first
    assert deuterium != first


def test_updating_parameters_changes_the_rate(system: PCET) -> None:
    """The thermodynamic parameters are plain attributes and can be reassigned."""
    before = system.calculate(MASS_PROTON, temperature=TEMPERATURE)

    system.reaction_free_energy = -0.3
    system.reorganization_energy = 0.9
    system.electronic_coupling = 0.05
    after = system.calculate(MASS_PROTON, temperature=TEMPERATURE, reuse_states=True)

    assert system.reaction_free_energy == -0.3
    assert system.reorganization_energy == 0.9
    assert system.electronic_coupling == 0.05
    assert after != before


def test_array_input_defines_the_proton_grid() -> None:
    """Tabulated potentials set the rp grid limits from the data."""
    ascending = RP_GRID[::-1]
    system = PCET(
        (ascending, REACTANT_ENERGIES[::-1]),
        (ascending, PRODUCT_ENERGIES[::-1]),
        REACTION_FREE_ENERGY,
        REORGANIZATION_ENERGY,
        electronic_coupling=ELECTRONIC_COUPLING,
    )

    assert system.rp[0] == pytest.approx(np.min(RP_GRID))
    assert system.rp[-1] == pytest.approx(np.max(RP_GRID))
    total = system.calculate(MASS_PROTON, temperature=TEMPERATURE)
    assert np.isfinite(total)
    assert total > 0


@pytest.mark.parametrize("fit_method", ["poly6", "poly8", "bspline"])
def test_array_input_supports_all_fit_methods(fit_method: FitMethod) -> None:
    """Each smoothing backend yields callable proton potentials."""
    ascending = RP_GRID[::-1]
    system = PCET(
        (ascending, REACTANT_ENERGIES[::-1]),
        (ascending, PRODUCT_ENERGIES[::-1]),
        REACTION_FREE_ENERGY,
        REORGANIZATION_ENERGY,
        fit_method=fit_method,
    )

    assert np.all(np.isfinite(system.reactant_potential(np.array([0.0]))))
    assert np.all(np.isfinite(system.product_potential(np.array([0.0]))))


def test_grid_limit_keywords_override_defaults(
    reactant_potential: PotentialFunction, product_potential: PotentialFunction
) -> None:
    """r_min/r_max keywords take precedence; callables default to +/-0.8."""
    default = PCET(
        reactant_potential,
        product_potential,
        REACTION_FREE_ENERGY,
        REORGANIZATION_ENERGY,
    )
    custom = PCET(
        reactant_potential,
        product_potential,
        REACTION_FREE_ENERGY,
        REORGANIZATION_ENERGY,
        r_min=-0.5,
        r_max=0.6,
    )

    assert default.rp[0] == pytest.approx(-0.8)
    assert default.rp[-1] == pytest.approx(0.8)
    assert custom.rp[0] == pytest.approx(-0.5)
    assert custom.rp[-1] == pytest.approx(0.6)


def test_invalid_potential_inputs_are_rejected(
    reactant_potential: PotentialFunction,
) -> None:
    """Non-callable, non-tabulated potentials raise TypeError."""
    with pytest.raises(TypeError, match="reactant_potential"):
        PCET(5.0, reactant_potential, REACTION_FREE_ENERGY, REORGANIZATION_ENERGY)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="product_potential"):
        PCET(reactant_potential, 5.0, REACTION_FREE_ENERGY, REORGANIZATION_ENERGY)  # type: ignore[arg-type]


def test_invalid_fit_method_is_rejected(
    reactant_potential: PotentialFunction,
) -> None:
    """An unknown fit_method raises ValueError for either potential."""
    tabulated = (RP_GRID[::-1], REACTANT_ENERGIES[::-1])

    with pytest.raises(ValueError, match="fit_method"):
        PCET(
            tabulated,
            reactant_potential,
            REACTION_FREE_ENERGY,
            REORGANIZATION_ENERGY,
            fit_method="nope",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="fit_method"):
        PCET(
            reactant_potential,
            tabulated,
            REACTION_FREE_ENERGY,
            REORGANIZATION_ENERGY,
            fit_method="nope",  # type: ignore[arg-type]
        )


# the effective donor-acceptor force constant and equilibrium distance of the
# benzimidazole-phenol system in example 3, in atomic units and angstrom
FORCE_CONSTANT = 0.0443
EQUILIBRIUM_DISTANCE = 2.58


def test_donor_acceptor_distribution_peaks_at_the_equilibrium_distance() -> None:
    """P(R) is the Boltzmann weight of a harmonic well centred on R_eq."""
    # a grid centred on the equilibrium distance, so the weights mirror about it
    distances = np.linspace(EQUILIBRIUM_DISTANCE - 0.3, EQUILIBRIUM_DISTANCE + 0.3, 61)

    distribution = donor_acceptor_distribution(
        distances, EQUILIBRIUM_DISTANCE, FORCE_CONSTANT, TEMPERATURE
    )

    assert distances[np.argmax(distribution)] == pytest.approx(EQUILIBRIUM_DISTANCE)
    assert np.max(distribution) == pytest.approx(1.0)
    assert distribution == pytest.approx(distribution[::-1])


def test_donor_acceptor_distribution_matches_the_boltzmann_factor() -> None:
    """A displaced point carries exp(-k dR^2 / 2kT), with k converted from a.u."""
    displacement = 0.1
    energy = (
        0.5 * FORCE_CONSTANT * displacement**2 * ANGSTROM_TO_BOHR**2 * HARTREE_TO_EV
    )

    weight = donor_acceptor_distribution(
        np.array([EQUILIBRIUM_DISTANCE + displacement]),
        EQUILIBRIUM_DISTANCE,
        FORCE_CONSTANT,
        TEMPERATURE,
    )

    assert weight[0] == pytest.approx(np.exp(-energy / (BOLTZMANN * TEMPERATURE)))


def test_donor_acceptor_distribution_broadens_with_temperature() -> None:
    """A hotter donor-acceptor mode samples short and long R more readily."""
    distances = np.linspace(2.3, 2.9, 61)

    cold = donor_acceptor_distribution(
        distances, EQUILIBRIUM_DISTANCE, FORCE_CONSTANT, 200.0
    )
    hot = donor_acceptor_distribution(
        distances, EQUILIBRIUM_DISTANCE, FORCE_CONSTANT, 400.0
    )

    cold /= simpson(cold, x=distances)
    hot /= simpson(hot, x=distances)

    def width(distribution: np.ndarray) -> float:
        mean = simpson(distribution * distances, x=distances)
        return float(simpson(distribution * (distances - mean) ** 2, x=distances))

    assert width(hot) > width(cold)


def test_donor_acceptor_distribution_defaults_to_room_temperature() -> None:
    """Leaving the temperature out matches passing ROOM_TEMPERATURE explicitly."""
    distances = np.linspace(2.4, 2.8, 21)

    assert donor_acceptor_distribution(
        distances, EQUILIBRIUM_DISTANCE, FORCE_CONSTANT
    ) == pytest.approx(
        donor_acceptor_distribution(
            distances, EQUILIBRIUM_DISTANCE, FORCE_CONSTANT, ROOM_TEMPERATURE
        )
    )
