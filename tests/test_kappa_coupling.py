"""Tests for the nonadiabaticity analysis used in example 5."""

import numpy as np
import numpy.typing as npt
import pytest
from scipy.integrate import simpson

from autopcet import (
    HBAR,
    MASS_DEUTERON,
    MASS_PROTON,
    KappaCoupling,
    inverted_morse,
    morse,
)
from autopcet.utils import _find_first_crossing

# A symmetric model double well in the style of the RNR Y356-Y731 system of
# example 5: two Morse diabats crossing at rp = 0, on a 512-point grid.
# The grid stops at +/-0.55 A so the only diabat crossing inside it is the
# central one (the repulsive walls cross the opposing Morse tails further out).
SEPARATION = 0.35
WELL_DEPTH = 4.0
WIDTH = 2.5
COUPLING = 0.0434

RP_GRID = np.linspace(-0.55, 0.55, 512)
REACTANT_POTENTIAL = morse(RP_GRID, -SEPARATION, WELL_DEPTH, WIDTH)
PRODUCT_POTENTIAL = inverted_morse(RP_GRID, SEPARATION, WELL_DEPTH, WIDTH)


def make_system(
    coupling: float | npt.NDArray[np.float64] = COUPLING,
) -> KappaCoupling:
    return KappaCoupling(RP_GRID, REACTANT_POTENTIAL, PRODUCT_POTENTIAL, coupling)


def test_crossing_point_of_a_symmetric_double_well() -> None:
    """The diabats of a symmetric well cross at rp = 0 with mirrored slopes."""
    system = make_system()
    system.calculate(MASS_PROTON)

    assert system.crossing_rp == pytest.approx(0.0, abs=0.01)
    assert system.crossing_energy == pytest.approx(
        morse(0.0, -SEPARATION, WELL_DEPTH, WIDTH), abs=0.05
    )
    assert system.crossing_coupling == pytest.approx(COUPLING)
    assert system.reactant_slope == pytest.approx(-system.product_slope, rel=0.05)


def test_nonadiabaticity_parameters_are_consistent() -> None:
    """The tunneling times, adiabaticity, and kappa obey their relations."""
    system = make_system()
    system.calculate(MASS_PROTON)

    assert system.tau_electron == pytest.approx(HBAR / COUPLING)
    assert system.tau_proton > 0
    assert system.adiabaticity == pytest.approx(system.tau_proton / system.tau_electron)
    assert 0 < system.kappa <= 1


def test_vibronic_couplings_are_positive_and_ordered() -> None:
    """v_semiclassical = kappa * v_adiabatic; v_nonadiabatic is bounded by Vel."""
    system = make_system()
    system.calculate(MASS_PROTON)

    assert system.v_adiabatic > 0
    # the sign of v_nonadiabatic follows the arbitrary phase of the states
    assert 0 < abs(system.v_nonadiabatic) <= COUPLING
    assert system.v_semiclassical == pytest.approx(system.kappa * system.v_adiabatic)
    assert system.v_semiclassical <= system.v_adiabatic


def test_state_alignment_and_normalization() -> None:
    """Selected reactant/product levels are aligned; states are normalized."""
    system = make_system()
    system.calculate(MASS_PROTON)

    assert system.shifted_reactant_energies[system.mu] == pytest.approx(
        system.shifted_product_energies[system.nu]
    )
    assert simpson(system.reactant_wavefunctions[0] ** 2, x=RP_GRID) == pytest.approx(
        1.0
    )
    assert simpson(system.product_wavefunctions[0] ** 2, x=RP_GRID) == pytest.approx(
        1.0
    )


def test_adiabatic_potentials_bracket_the_diabats() -> None:
    """The ground (excited) adiabat lies below (above) both diabats."""
    system = make_system()
    system.calculate(MASS_PROTON)
    reactant = system.shifted_reactant_potential
    product = system.shifted_product_potential

    assert np.all(system.ground_adiabat <= np.minimum(reactant, product) + 1e-12)
    assert np.all(system.excited_adiabat >= np.maximum(reactant, product) - 1e-12)

    assert system.adiabatic_energies.shape == (2 * system.n_states,)
    assert system.adiabatic_wavefunctions.shape == (
        2 * system.n_states,
        len(RP_GRID),
    )


def test_deuterium_tunnels_less_than_protium() -> None:
    """Deuterium has a lower ZPE and a smaller tunneling splitting."""
    protium = make_system()
    protium.calculate(MASS_PROTON)
    deuterium = make_system()
    deuterium.calculate(MASS_DEUTERON)

    assert deuterium.reactant_energies[0] < protium.reactant_energies[0]
    assert deuterium.v_adiabatic < protium.v_adiabatic


def test_constant_electronic_coupling_array_matches_scalar() -> None:
    """A constant coupling array gives the same couplings as the scalar input."""
    scalar = make_system()
    scalar.calculate(MASS_PROTON)
    array = make_system(np.full(len(RP_GRID), COUPLING))
    array.calculate(MASS_PROTON)

    assert array.v_semiclassical == pytest.approx(scalar.v_semiclassical)
    assert array.kappa == pytest.approx(scalar.kappa)


def test_low_overlap_threshold_emits_a_warning() -> None:
    """An unattainable overlap threshold triggers the diagnostic warning."""
    system = make_system()

    with pytest.warns(UserWarning, match="maximum overlap"):
        system.calculate(MASS_PROTON, overlap_threshold=1.01)


def test_tunneling_above_the_barrier_is_rejected() -> None:
    """A barrier below the proton ZPE raises RuntimeError."""
    shallow_reactant = morse(RP_GRID, -0.2, WELL_DEPTH, 0.5)
    shallow_product = inverted_morse(RP_GRID, 0.2, WELL_DEPTH, 0.5)
    system = KappaCoupling(RP_GRID, shallow_reactant, shallow_product, COUPLING)

    with pytest.raises(RuntimeError, match="tunneling energy"):
        system.calculate(MASS_PROTON)


def test_invalid_inputs_are_rejected() -> None:
    """Constructor type and shape validation raises informative errors."""
    with pytest.raises(TypeError, match="1D arrays"):
        KappaCoupling(0.5, REACTANT_POTENTIAL, PRODUCT_POTENTIAL, COUPLING)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="electronic_coupling"):
        KappaCoupling(RP_GRID, REACTANT_POTENTIAL, PRODUCT_POTENTIAL, "strong")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="same dimension"):
        KappaCoupling(RP_GRID, REACTANT_POTENTIAL[:-1], PRODUCT_POTENTIAL, COUPLING)
    with pytest.raises(ValueError, match="power of 2"):
        KappaCoupling(
            RP_GRID[:500],
            REACTANT_POTENTIAL[:500],
            PRODUCT_POTENTIAL[:500],
            COUPLING,
        )


def test_out_of_range_state_indices_are_rejected() -> None:
    """State indices outside 0 <= mu, nu < n_states raise ValueError."""
    with pytest.raises(ValueError, match="mu"):
        KappaCoupling(
            RP_GRID,
            REACTANT_POTENTIAL,
            PRODUCT_POTENTIAL,
            COUPLING,
            n_states=5,
            mu=5,
        )
    with pytest.raises(ValueError, match="nu"):
        KappaCoupling(
            RP_GRID,
            REACTANT_POTENTIAL,
            PRODUCT_POTENTIAL,
            COUPLING,
            n_states=5,
            nu=-1,
        )


def test_non_crossing_potentials_are_rejected() -> None:
    """A curve without a sign change has no crossing to locate."""
    with pytest.raises(RuntimeError, match="do not cross"):
        _find_first_crossing(np.linspace(0.0, 1.0, 8), np.ones(8))
