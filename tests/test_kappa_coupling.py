"""Tests for the nonadiabaticity analysis used in example 5."""

import numpy as np
import numpy.typing as npt
import pytest
from scipy.integrate import simpson

from autopcet import (
    KappaCoupling,
    hbar,
    inverted_morse,
    massD,
    massH,
    morse,
)
from autopcet.utils import _find_first_crossing

# A symmetric model double well in the style of the RNR Y356-Y731 system of
# example 5: two Morse diabats crossing at rp = 0, on a 512-point grid.
# The grid stops at +/-0.55 A so the only diabat crossing inside it is the
# central one (the repulsive walls cross the opposing Morse tails further out).
R0 = 0.35
DE = 4.0
BETA = 2.5
VEL = 0.0434

RP_GRID = np.linspace(-0.55, 0.55, 512)
E_REAC = morse(RP_GRID, -R0, DE, BETA)
E_PROD = inverted_morse(RP_GRID, R0, DE, BETA)


def make_system(vel: float | npt.NDArray[np.float64] = VEL) -> KappaCoupling:
    return KappaCoupling(RP_GRID, E_REAC, E_PROD, vel)


def test_crossing_point_of_a_symmetric_double_well() -> None:
    """The diabats of a symmetric well cross at rp = 0 with mirrored slopes."""
    system = make_system()
    system.calculate(massH)

    assert system.rp_crossing == pytest.approx(0.0, abs=0.01)
    assert system.E_crossing == pytest.approx(morse(0.0, -R0, DE, BETA), abs=0.05)
    assert system.Vel_crossing == pytest.approx(VEL)
    assert system.slope_reac == pytest.approx(-system.slope_prod, rel=0.05)


def test_nonadiabaticity_parameters_are_consistent() -> None:
    """tau_e, tau_p, p, and kappa obey their defining relations."""
    system = make_system()
    system.calculate(massH)
    tau_e, tau_p, p, kappa = system.get_nonadiabaticity_parameters()

    assert tau_e == pytest.approx(hbar / VEL)
    assert tau_p > 0
    assert p == pytest.approx(tau_p / tau_e)
    assert 0 < kappa <= 1


def test_vibronic_couplings_are_positive_and_ordered() -> None:
    """V_sc = kappa * V_ad; V_nad is nonzero but bounded by Vel."""
    system = make_system()
    system.calculate(massH)
    v_sc, v_nad, v_ad = system.get_vibronic_couplings()

    assert v_ad > 0
    # the sign of V_nad follows the arbitrary phase of the vibrational states
    assert 0 < abs(v_nad) <= VEL
    assert v_sc == pytest.approx(system.kappa * v_ad)
    assert v_sc <= v_ad


def test_state_alignment_and_normalization() -> None:
    """Selected reactant/product levels are aligned; states are normalized."""
    system = make_system()
    system.calculate(massH)
    _, e_reac, wfc_reac = system.get_reactant_proton_states()
    _, e_prod, wfc_prod = system.get_product_proton_states()

    assert e_reac[system.mu] == pytest.approx(e_prod[system.nu])
    assert simpson(wfc_reac[0] ** 2, x=RP_GRID) == pytest.approx(1.0)
    assert simpson(wfc_prod[0] ** 2, x=RP_GRID) == pytest.approx(1.0)


def test_adiabatic_potentials_bracket_the_diabats() -> None:
    """The ground (excited) adiabat lies below (above) both diabats."""
    system = make_system()
    system.calculate(massH)
    e_gs, e_es = system.get_adiabatic_proton_potentials()
    reac = system.ShiftedReacProtonPot
    prod = system.ShiftedProdProtonPot

    assert np.all(e_gs <= np.minimum(reac, prod) + 1e-12)
    assert np.all(e_es >= np.maximum(reac, prod) - 1e-12)

    energies, wfcs = system.get_ground_adiabatic_proton_states()
    assert energies.shape == (2 * system.NStates,)
    assert wfcs.shape == (2 * system.NStates, len(RP_GRID))


def test_deuterium_tunnels_less_than_protium() -> None:
    """Deuterium has a lower ZPE and a smaller tunneling splitting."""
    system_h = make_system()
    system_h.calculate(massH)
    system_d = make_system()
    system_d.calculate(massD)

    assert system_d.ReacProtonEnergyLevels[0] < system_h.ReacProtonEnergyLevels[0]
    assert system_d.V_ad < system_h.V_ad


def test_constant_electronic_coupling_array_matches_scalar() -> None:
    """A constant Vel array gives the same couplings as the scalar input."""
    system_scalar = make_system()
    system_scalar.calculate(massH)
    system_array = make_system(np.full(len(RP_GRID), VEL))
    system_array.calculate(massH)

    assert system_array.V_sc == pytest.approx(system_scalar.V_sc)
    assert system_array.kappa == pytest.approx(system_scalar.kappa)


def test_low_overlap_threshold_emits_a_warning() -> None:
    """An unattainable overlap threshold triggers the diagnostic warning."""
    system = make_system()

    with pytest.warns(UserWarning, match="maximum overlap"):
        system.calculate(massH, overlap_thresh=1.01)


def test_tunneling_above_the_barrier_is_rejected() -> None:
    """A barrier below the proton ZPE raises RuntimeError."""
    shallow_reac = morse(RP_GRID, -0.2, DE, 0.5)
    shallow_prod = inverted_morse(RP_GRID, 0.2, DE, 0.5)
    system = KappaCoupling(RP_GRID, shallow_reac, shallow_prod, VEL)

    with pytest.raises(RuntimeError, match="tunneling energy"):
        system.calculate(massH)


def test_invalid_inputs_are_rejected() -> None:
    """Constructor type and shape validation raises informative errors."""
    with pytest.raises(TypeError, match="1D arrays"):
        KappaCoupling(0.5, E_REAC, E_PROD, VEL)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Vel"):
        KappaCoupling(RP_GRID, E_REAC, E_PROD, "strong")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="same dimension"):
        KappaCoupling(RP_GRID, E_REAC[:-1], E_PROD, VEL)
    with pytest.raises(ValueError, match="power of 2"):
        KappaCoupling(RP_GRID[:500], E_REAC[:500], E_PROD[:500], VEL)


def test_out_of_range_state_indices_are_rejected() -> None:
    """State indices outside 0 <= mu, nu < NStates raise ValueError."""
    with pytest.raises(ValueError, match="mu"):
        KappaCoupling(RP_GRID, E_REAC, E_PROD, VEL, NStates=5, mu=5)
    with pytest.raises(ValueError, match="nu"):
        KappaCoupling(RP_GRID, E_REAC, E_PROD, VEL, NStates=5, nu=-1)


def test_non_crossing_potentials_are_rejected() -> None:
    """A curve without a sign change has no crossing to locate."""
    with pytest.raises(RuntimeError, match="do not cross"):
        _find_first_crossing(np.ones(8), np.linspace(0.0, 1.0, 8))
