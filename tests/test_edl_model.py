"""Tests for the electric double layer model and Fermi distribution (example 4)."""

from typing import Any

import numpy as np
import pytest

from autopcet import fermi_distribution, make_edl_model
from autopcet._types import _ScalarArrayFunction

# Electrode/electrolyte parameters for CoTPP on graphene from example 4.
E_VS_SHE = -0.6
D_IHL = 3.6  # angstrom
D_OHL = 3.5  # angstrom
EPS_IHL = 2.7
EPS_ST = 78.0
EPS_OP = 1.78
RHO_WATER = 0.9970470  # g/cm^3
M_WATER = 18.01528  # g/mol
C_IONS = 0.5  # mol/L
C_EDL = 15  # microFarad/cm^2
PZFC_VS_SHE = 0.04  # V


def make_model(**overrides: Any) -> _ScalarArrayFunction:
    kwargs: dict[str, Any] = {
        "EvsSHE": E_VS_SHE,
        "dIHL": D_IHL,
        "dOHL": D_OHL,
        "eps_IHL": EPS_IHL,
        "eps_st": EPS_ST,
        "eps_op": EPS_OP,
        "dipole": "calculate",
        "rho_solvent": RHO_WATER,
        "m_solvent": M_WATER,
        "c_ions": C_IONS,
        "C_EDL": C_EDL,
        "PZFCvsSHE": PZFC_VS_SHE,
        "print_data": False,
    }
    kwargs.update(overrides)
    return make_edl_model(**kwargs)


def test_potential_drop_at_the_electrode_surface() -> None:
    """At R = 0 the drop equals the potential relative to the PZFC."""
    drop = make_model()

    assert drop(0.0) == pytest.approx(E_VS_SHE - PZFC_VS_SHE)


def test_scalar_and_array_evaluation_agree_in_all_regions() -> None:
    """Scalar and array inputs agree in the IHL, OHL, and diffuse layers."""
    drop = make_model()
    r_values = np.array([1.0, 5.0, 9.0])
    array_result = drop(r_values)

    for i, r in enumerate(r_values):
        assert drop(float(r)) == pytest.approx(array_result[i])


def test_potential_drop_is_continuous_across_layer_boundaries() -> None:
    """The piecewise profile is continuous at the IHL and OHL boundaries."""
    drop = make_model()

    for boundary in (D_IHL, D_IHL + D_OHL):
        assert drop(boundary - 1e-6) == pytest.approx(drop(boundary + 1e-6), abs=1e-3)


def test_potential_drop_decays_into_the_bulk() -> None:
    """The magnitude decays with distance and vanishes in the bulk."""
    drop = make_model()

    assert drop(9.0) < 0
    assert abs(drop(20.0)) < abs(drop(9.0))
    assert drop(60.0) == pytest.approx(0.0, abs=1e-4)


def test_unsupported_position_type_is_rejected() -> None:
    """Inputs that are neither numbers nor arrays raise TypeError."""
    drop = make_model()

    with pytest.raises(TypeError, match="R"):
        drop("nowhere")  # type: ignore[call-overload]


def test_numeric_dipole_and_kirkwood_options() -> None:
    """A dipole in Debye and a non-unit Kirkwood eta are both accepted."""
    drop_numeric = make_model(dipole=1.85)
    drop_kirkwood = make_model(eta_Kirkwood=0.9)

    assert np.isfinite(drop_numeric(1.0))
    assert np.isfinite(drop_kirkwood(1.0))


def test_invalid_dipole_is_rejected() -> None:
    """A non-numeric, non-'calculate' dipole raises ValueError."""
    with pytest.raises(ValueError, match="dipole"):
        make_model(dipole="oops")


def test_print_data_reports_the_model_setup(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With print_data=True the model prints its derived quantities."""
    make_model(print_data=True)
    out = capsys.readouterr().out

    assert "E vs. SHE" in out
    assert "phi_OHP" in out


def test_fermi_distribution_basic_properties() -> None:
    """The Fermi function is 1/2 at the Fermi level and steps from 1 to 0."""
    assert fermi_distribution(0.0) == pytest.approx(0.5)
    assert fermi_distribution(0.3, E_Fermi=0.3) == pytest.approx(0.5)
    assert fermi_distribution(-1.0) == pytest.approx(1.0, abs=1e-10)
    assert fermi_distribution(1.0) == pytest.approx(0.0, abs=1e-10)
    # electron-hole symmetry about the Fermi level
    assert fermi_distribution(0.2) + fermi_distribution(-0.2) == pytest.approx(1.0)


def test_fermi_distribution_sharpens_at_low_temperature() -> None:
    """Lowering T sharpens the step edge."""
    assert fermi_distribution(0.05, T=100) < fermi_distribution(0.05, T=1000)

    energies = np.linspace(-0.5, 0.5, 11)
    occupations = fermi_distribution(energies)
    assert np.all(np.diff(occupations) < 0)
