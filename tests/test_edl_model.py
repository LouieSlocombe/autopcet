"""Tests for the electric double layer model and Fermi distribution (example 4)."""

from typing import Any

import numpy as np
import pytest

from autopcet import fermi_distribution, make_edl_model
from autopcet._types import ScalarOrArrayFunction

# Electrode/electrolyte parameters for CoTPP on graphene from example 4.
POTENTIAL_VS_SHE = -0.6
D_IHL = 3.6  # angstrom
D_OHL = 3.5  # angstrom
EPS_IHL = 2.7
EPS_STATIC = 78.0
EPS_OPTICAL = 1.78
WATER_DENSITY = 0.9970470  # g/cm^3
WATER_MOLAR_MASS = 18.01528  # g/mol
ION_CONCENTRATION = 0.5  # mol/L
EDL_CAPACITANCE = 15  # microfarad/cm^2
PZFC_VS_SHE = 0.04  # V


def make_model(**overrides: Any) -> ScalarOrArrayFunction:
    arguments: dict[str, Any] = {
        "potential_vs_she": POTENTIAL_VS_SHE,
        "d_ihl": D_IHL,
        "d_ohl": D_OHL,
        "eps_ihl": EPS_IHL,
        "eps_static": EPS_STATIC,
        "eps_optical": EPS_OPTICAL,
        "dipole": "calculate",
        "solvent_density": WATER_DENSITY,
        "solvent_molar_mass": WATER_MOLAR_MASS,
        "ion_concentration": ION_CONCENTRATION,
        "edl_capacitance": EDL_CAPACITANCE,
        "pzfc_vs_she": PZFC_VS_SHE,
        "verbose": False,
    }
    arguments.update(overrides)
    return make_edl_model(**arguments)


def test_potential_drop_at_the_electrode_surface() -> None:
    """At the surface the drop equals the potential relative to the PZFC."""
    drop = make_model()

    assert drop(0.0) == pytest.approx(POTENTIAL_VS_SHE - PZFC_VS_SHE)


def test_scalar_and_array_evaluation_agree_in_all_regions() -> None:
    """Scalar and array inputs agree in the IHL, OHL, and diffuse layers."""
    drop = make_model()
    distances = np.array([1.0, 5.0, 9.0])
    from_array = drop(distances)

    for i, distance in enumerate(distances):
        assert drop(float(distance)) == pytest.approx(from_array[i])


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

    with pytest.raises(TypeError, match="distance"):
        drop("nowhere")  # type: ignore[type-var]


def test_numeric_dipole_and_kirkwood_options() -> None:
    """A dipole in debye and a non-unit Kirkwood eta are both accepted."""
    numeric_dipole = make_model(dipole=1.85)
    kirkwood = make_model(eta_kirkwood=0.9)

    assert np.isfinite(numeric_dipole(1.0))
    assert np.isfinite(kirkwood(1.0))


def test_invalid_dipole_is_rejected() -> None:
    """A non-numeric, non-'calculate' dipole raises ValueError."""
    with pytest.raises(ValueError, match="dipole"):
        make_model(dipole="oops")


def test_verbose_reports_the_model_setup(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With verbose=True the model prints its derived quantities."""
    make_model(verbose=True)
    printed = capsys.readouterr().out

    assert "E vs. SHE" in printed
    assert "phi_OHP" in printed


def test_fermi_distribution_basic_properties() -> None:
    """The Fermi function is 1/2 at the Fermi level and steps from 1 to 0."""
    assert fermi_distribution(0.0) == pytest.approx(0.5)
    assert fermi_distribution(0.3, fermi_level=0.3) == pytest.approx(0.5)
    assert fermi_distribution(-1.0) == pytest.approx(1.0, abs=1e-10)
    assert fermi_distribution(1.0) == pytest.approx(0.0, abs=1e-10)
    # electron-hole symmetry about the Fermi level
    assert fermi_distribution(0.2) + fermi_distribution(-0.2) == pytest.approx(1.0)


def test_fermi_distribution_sharpens_at_low_temperature() -> None:
    """Lowering the temperature sharpens the step edge."""
    assert fermi_distribution(0.05, temperature=100) < fermi_distribution(
        0.05, temperature=1000
    )

    occupations = fermi_distribution(np.linspace(-0.5, 0.5, 11))
    assert np.all(np.diff(occupations) < 0)
