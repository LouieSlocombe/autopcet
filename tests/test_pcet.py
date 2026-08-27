"""Tests for the PCET rate-constant workflow used in example 1."""

import numpy as np
import pytest
from example1_data import (
    DELTA_G,
    E_PROD_DATA,
    E_REAC_DATA,
    LAMBDA,
    RP_DATA,
    TEMPERATURE,
    VEL,
)
from scipy.integrate import simpson

from autopcet import PCET, massD, massH
from autopcet._types import PotentialFunction


@pytest.fixture
def system(
    reac_proton_pot: PotentialFunction, prod_proton_pot: PotentialFunction
) -> PCET:
    """A fresh PCET system set up exactly as in example 1."""
    return PCET(reac_proton_pot, prod_proton_pot, DELTA_G, LAMBDA, Vel=VEL)


def test_rate_constant_is_positive_and_kie_is_normal(system: PCET) -> None:
    """The example 1 system reproduces its rate constant and H/D KIE."""
    k_tot_h = system.calculate(massH, T=TEMPERATURE)
    assert k_tot_h == system.get_total_rate_constant()

    k_tot_d = system.calculate(massD, T=TEMPERATURE)
    assert k_tot_d == system.get_total_rate_constant()

    # regression values from running the example 1 workflow
    assert k_tot_h == pytest.approx(2.337e8, rel=0.05)
    assert k_tot_h / k_tot_d == pytest.approx(2.72, rel=0.05)


def test_reactant_state_distribution_is_a_boltzmann_distribution(
    system: PCET,
) -> None:
    """State populations are normalized and decrease with energy."""
    system.calculate(massH, T=TEMPERATURE)
    pu = system.get_reactant_state_distribution()

    assert pu.shape == (system.NStates,)
    assert np.sum(pu) == pytest.approx(1.0)
    assert np.all(np.diff(pu) <= 0)


def test_proton_states_are_normalized(system: PCET) -> None:
    """Vibrational energies are sorted and wave functions are normalized."""
    system.calculate(massH, T=TEMPERATURE)

    for energies, wfcs in (
        system.get_reactant_proton_states(),
        system.get_product_proton_states(),
    ):
        assert energies.shape == (system.NStates,)
        assert wfcs.shape == (system.NStates, len(system.rp))
        assert np.all(np.diff(energies) >= 0)
        norms = simpson(wfcs**2, x=system.rp, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-8)


def test_overlap_matrix_is_bounded(system: PCET) -> None:
    """Overlaps between normalized states cannot exceed one."""
    system.calculate(massH, T=TEMPERATURE)
    suv = system.get_proton_overlap_matrix()

    assert suv.shape == (system.NStates, system.NStates)
    assert np.all(np.abs(suv) <= 1.0 + 1e-6)


def test_free_energy_matrices_follow_their_definitions(system: PCET) -> None:
    """dG_uv and the activation matrix match their analytic expressions."""
    system.calculate(massH, T=TEMPERATURE)
    dguv = system.get_reaction_free_energy_matrix()
    e_reac, _ = system.get_reactant_proton_states()
    e_prod, _ = system.get_product_proton_states()

    assert dguv[0, 0] == pytest.approx(DELTA_G)
    expected = (
        DELTA_G
        + (e_prod[np.newaxis, :] - e_prod[0])
        - (e_reac[:, np.newaxis] - e_reac[0])
    )
    np.testing.assert_allclose(dguv, expected)
    np.testing.assert_allclose(
        system.get_activation_free_energy_matrix(),
        (dguv + LAMBDA) ** 2 / (4 * LAMBDA),
    )


def test_total_rate_is_the_sum_of_state_contributions(system: PCET) -> None:
    """k_tot accumulates the non-negative contribution matrix."""
    k_tot = system.calculate(massH, T=TEMPERATURE)
    kuv = system.get_rate_contribution_matrix()

    assert np.all(kuv >= 0)
    assert k_tot == pytest.approx(np.sum(kuv))


def test_reusing_saved_proton_states_reproduces_the_rate(system: PCET) -> None:
    """Reusing cached vibrational states must not change the answer."""
    k_first = system.calculate(massH, T=TEMPERATURE)
    k_reused = system.calculate(massH, T=TEMPERATURE, reuse_saved_proton_states=True)
    k_deuterium = system.calculate(massD, T=TEMPERATURE, reuse_saved_proton_states=True)

    assert k_reused == k_first
    assert k_deuterium != k_first


def test_set_parameters_updates_the_model(system: PCET) -> None:
    """set_parameters overrides DeltaG, Lambda, and Vel."""
    system.set_parameters(DeltaG=-0.3, Lambda=0.9, Vel=0.05)

    assert system.DeltaG == -0.3
    assert system.Lambda == 0.9
    assert system.Vel == 0.05


def test_array_input_defines_the_proton_grid() -> None:
    """Tabulated potentials set the rp grid limits from the data."""
    rp_ascending = RP_DATA[::-1]
    system = PCET(
        (rp_ascending, E_REAC_DATA[::-1]),
        (rp_ascending, E_PROD_DATA[::-1]),
        DELTA_G,
        LAMBDA,
        Vel=VEL,
    )

    assert system.rp[0] == pytest.approx(np.min(RP_DATA))
    assert system.rp[-1] == pytest.approx(np.max(RP_DATA))
    k_tot = system.calculate(massH, T=TEMPERATURE)
    assert np.isfinite(k_tot)
    assert k_tot > 0


@pytest.mark.parametrize("smooth", ["poly6", "poly8", "bspline"])
def test_array_input_supports_all_smoothing_options(smooth: str) -> None:
    """Each smoothing backend yields callable proton potentials."""
    rp_ascending = RP_DATA[::-1]
    system = PCET(
        (rp_ascending, E_REAC_DATA[::-1]),
        (rp_ascending, E_PROD_DATA[::-1]),
        DELTA_G,
        LAMBDA,
        smooth=smooth,
    )

    assert np.all(np.isfinite(system.ReacProtonPot(np.array([0.0]))))
    assert np.all(np.isfinite(system.ProdProtonPot(np.array([0.0]))))


def test_grid_limit_keywords_override_defaults(
    reac_proton_pot: PotentialFunction, prod_proton_pot: PotentialFunction
) -> None:
    """rmin/rmax keywords take precedence; callables default to +/-0.8."""
    default = PCET(reac_proton_pot, prod_proton_pot, DELTA_G, LAMBDA)
    custom = PCET(
        reac_proton_pot, prod_proton_pot, DELTA_G, LAMBDA, rmin=-0.5, rmax=0.6
    )

    assert default.rp[0] == pytest.approx(-0.8)
    assert default.rp[-1] == pytest.approx(0.8)
    assert custom.rp[0] == pytest.approx(-0.5)
    assert custom.rp[-1] == pytest.approx(0.6)


def test_invalid_potential_inputs_are_rejected(
    reac_proton_pot: PotentialFunction,
) -> None:
    """Non-callable, non-tabulated potentials raise TypeError."""
    with pytest.raises(TypeError, match="ReacProtonPot"):
        PCET(5.0, reac_proton_pot, DELTA_G, LAMBDA)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ProdProtonPot"):
        PCET(reac_proton_pot, 5.0, DELTA_G, LAMBDA)  # type: ignore[arg-type]


def test_invalid_smoothing_option_is_rejected(
    reac_proton_pot: PotentialFunction,
) -> None:
    """An unknown smooth choice raises ValueError for either potential."""
    rp_ascending = RP_DATA[::-1]
    tabulated = (rp_ascending, E_REAC_DATA[::-1])

    with pytest.raises(ValueError, match="smooth"):
        PCET(tabulated, reac_proton_pot, DELTA_G, LAMBDA, smooth="nope")
    with pytest.raises(ValueError, match="smooth"):
        PCET(reac_proton_pot, tabulated, DELTA_G, LAMBDA, smooth="nope")
