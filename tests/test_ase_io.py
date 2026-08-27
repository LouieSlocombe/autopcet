"""Running the PCET scans in-process through ASE calculators.

Everything here runs against ASE's pure-Python Morse calculator or hand-built
Hessians, so no quantum chemistry program is needed. EMT would work too, but
its H/O parameters are explicitly flagged as toys, while a Morse potential is
analytic in every quantity these tests check.
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from autopcet import (
    ANGSTROM_TO_BOHR,
    EV_TO_HARTREE,
    effective_mode_from_vibrations,
    optimize_proton,
    proton_scan_grid,
    read_scan_energies,
    run_da_scan,
    run_proton_scan,
    run_vibrations,
)
from autopcet._types import FloatArray

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.morse import MorsePotential
    from ase.vibrations import VibrationsData

SENTINEL: Any = object()

MORSE = {"epsilon": 1.0, "rho0": 6.0, "r0": 1.0, "rcut1": 10.0, "rcut2": 12.0}
"""Morse parameters with the cutoffs pushed far out, so scans see no kinks."""


def make_morse(**overrides: float) -> MorsePotential:
    from ase.calculators.morse import MorsePotential

    return MorsePotential(**{**MORSE, **overrides})


def make_oho(proton_x: float) -> Atoms:
    """An O-H-O chain along x with the donor at 0 and the acceptor at 2.4."""
    from ase import Atoms

    return Atoms("OHO", positions=[[0, 0, 0], [proton_x, 0, 0], [2.4, 0, 0]])


def spring_hessian(force_constant: float, axis: int = 0) -> FloatArray:
    """Two atoms coupled by a spring along one axis, as a ``(6, 6)`` Hessian."""
    hessian = np.zeros((6, 6))
    first, second = axis, 3 + axis
    hessian[first, first] = hessian[second, second] = force_constant
    hessian[first, second] = hessian[second, first] = -force_constant
    return hessian


def spring_vibrations(hessian: FloatArray) -> VibrationsData:
    from ase import Atoms
    from ase.vibrations import VibrationsData

    atoms = Atoms("H2", positions=[[0, 0, 0], [0.75, 0, 0]])
    return VibrationsData.from_2d(atoms, hessian)


def test_run_proton_scan_walks_only_the_proton() -> None:
    """The scan moves the proton on the shared grid and nothing else."""
    pytest.importorskip("ase")

    reactant, product = make_oho(0.9), make_oho(1.5)
    scan = run_proton_scan(reactant, product, 1, make_morse(), points=9)

    offsets, geometries = proton_scan_grid(
        reactant.get_positions(), product.get_positions(), 1, 9
    )
    assert scan.offsets == pytest.approx(offsets, abs=0.0)
    assert scan.positions == pytest.approx(geometries, abs=0.0)

    # the frame is mirror symmetric about the midpoint, so the potential is
    # symmetric in the offset
    assert scan.energies == pytest.approx(scan.energies[::-1])


def test_run_proton_scan_energies_come_from_the_calculator() -> None:
    """Each grid point holds the calculator's energy for that geometry."""
    pytest.importorskip("ase")
    from ase import Atoms

    reactant, product = make_oho(0.9), make_oho(1.5)
    scan = run_proton_scan(reactant, product, 1, make_morse(), points=5)

    probe = Atoms("OHO", positions=scan.positions[3])
    probe.calc = make_morse()
    assert scan.energies[3] == pytest.approx(probe.get_potential_energy())

    assert scan.relative_energies.min() == 0.0
    assert scan.relative_energies == pytest.approx(scan.energies - scan.energies.min())


def test_run_proton_scan_validates_its_indices() -> None:
    """Mismatched structures and out-of-range protons are rejected up front."""
    pytest.importorskip("ase")
    from ase import Atoms

    reactant, product = make_oho(0.9), make_oho(1.5)
    diatomic = Atoms("OH", positions=[[0, 0, 0], [1, 0, 0]])
    with pytest.raises(ValueError, match="same atoms"):
        run_proton_scan(reactant, diatomic, 1, SENTINEL)
    with pytest.raises(ValueError, match="out of range"):
        run_proton_scan(reactant, product, 3, SENTINEL)


@pytest.mark.parametrize("fix", [0, 1])
def test_run_da_scan_holds_the_frozen_distance(fix: int) -> None:
    """Each point relaxes with the donor-acceptor distance pinned at its R."""
    pytest.importorskip("ase")

    distances = [2.3, 2.5]
    scan = run_da_scan(make_oho(0.9), 0, 2, make_morse(), distances, fix=fix)

    assert scan.distances == pytest.approx(distances)
    for distance, structure, energy in zip(
        scan.distances, scan.structures, scan.energies, strict=True
    ):
        assert structure.get_distance(0, 2) == pytest.approx(distance)
        assert structure.calc is None
        assert structure.constraints == []

        # the recorded energy belongs to the returned geometry
        probe = structure.copy()
        probe.calc = make_morse()
        assert energy == pytest.approx(probe.get_potential_energy())


def test_run_da_scan_raises_when_the_optimizer_gives_up() -> None:
    """An unconverged constrained optimization must not pass silently."""
    pytest.importorskip("ase")

    with pytest.raises(RuntimeError, match="did not converge"):
        run_da_scan(make_oho(0.5), 0, 2, make_morse(), [2.4], fmax=1e-12, steps=1)


def test_optimize_proton_freezes_the_frame() -> None:
    """Only the proton relaxes, onto the analytic Morse minimum."""
    pytest.importorskip("ase")
    from ase import Atoms

    # the second oxygen sits beyond the Morse cutoff, so the proton's only
    # neighbor is the origin and its minimum is exactly r0 away
    structure = Atoms("OOH", positions=[[0, 0, 0], [50, 0, 0], [0.8, 0, 0]])
    relaxed = optimize_proton(structure, 2, make_morse(), fmax=1e-4)

    assert relaxed.get_positions()[:2] == pytest.approx(
        structure.get_positions()[:2], abs=0.0
    )
    assert relaxed.get_positions()[2] == pytest.approx([MORSE["r0"], 0, 0], abs=1e-3)
    assert relaxed.calc is None
    assert relaxed.constraints == []

    # the input is untouched
    assert structure.get_positions()[2] == pytest.approx([0.8, 0, 0], abs=0.0)

    with pytest.raises(ValueError, match="out of range"):
        optimize_proton(structure, -1, SENTINEL)


def test_optimize_proton_raises_when_the_optimizer_gives_up() -> None:
    """An unconverged proton optimization must not pass silently."""
    pytest.importorskip("ase")

    with pytest.raises(RuntimeError, match="did not converge"):
        optimize_proton(make_oho(0.5), 1, make_morse(), fmax=1e-12, steps=1)


def test_effective_mode_matches_an_analytic_spring() -> None:
    """A pure two-atom spring comes back with its physical mode quantities.

    This pins down the whole conversion from ASE's mass-weighted modes to the
    Gaussian-convention reduced masses and force constants
    :func:`autopcet.gaussian_io.effective_da_mode` expects.
    """
    pytest.importorskip("ase")

    force_constant = 5.0  # eV/A^2
    vibrations = spring_vibrations(spring_hessian(force_constant))
    mode = effective_mode_from_vibrations(vibrations, 0, 1)

    mass = float(vibrations.get_atoms().get_masses()[0])
    assert mode.reduced_mass == pytest.approx(mass / 2, rel=1e-9)
    assert mode.force_constant == pytest.approx(
        force_constant * EV_TO_HARTREE / ANGSTROM_TO_BOHR**2, rel=1e-5
    )
    # the round trip through force constant and reduced mass recovers ASE's
    # wavenumber for the stretch
    stretch = float(np.max(vibrations.get_frequencies().real))
    assert mode.frequency == pytest.approx(stretch, rel=1e-9)


def test_effective_mode_drops_rigid_body_modes() -> None:
    """The five zero modes of a free spring never reach the compliance sum."""
    pytest.importorskip("ase")

    vibrations = spring_vibrations(spring_hessian(5.0))
    assert len(vibrations.get_frequencies()) == 6

    mode = effective_mode_from_vibrations(vibrations, 0, 1)
    assert np.isfinite(mode.force_constant)

    with pytest.raises(ValueError, match="No vibrational modes"):
        effective_mode_from_vibrations(vibrations, 0, 1, frequency_floor=1e6)


def test_effective_mode_drops_imaginary_modes() -> None:
    """An unstable mode is excluded rather than poisoning the projection."""
    pytest.importorskip("ase")

    stable = spring_vibrations(spring_hessian(5.0))
    unstable = spring_vibrations(spring_hessian(5.0) + spring_hessian(-3.0, axis=1))
    assert np.any(unstable.get_frequencies().imag != 0)

    assert effective_mode_from_vibrations(unstable, 0, 1) == pytest.approx(
        effective_mode_from_vibrations(stable, 0, 1)
    )


def test_effective_mode_needs_the_axis_atoms_in_the_hessian() -> None:
    """A frozen axis atom would silently zero the projection, so it is an error."""
    ase = pytest.importorskip("ase")
    from ase.vibrations import VibrationsData

    atoms = ase.Atoms("H2", positions=[[0, 0, 0], [0.75, 0, 0]])
    vibrations = VibrationsData.from_2d(atoms, np.zeros((3, 3)), indices=[0])

    with pytest.raises(ValueError, match=r"acceptor .* was not displaced"):
        effective_mode_from_vibrations(vibrations, 0, 1)


def test_effective_mode_agrees_with_finite_differences(tmp_path: Path) -> None:
    """Finite difference vibrations recover the analytic Morse curvature."""
    pytest.importorskip("ase")
    from ase import Atoms

    structure = Atoms("H2", positions=[[0, 0, 0], [MORSE["r0"], 0, 0]])

    vibrations = run_vibrations(
        structure, make_morse(), delta=0.005, directory=tmp_path / "vib"
    )
    mode = effective_mode_from_vibrations(vibrations, 0, 1)

    # the Morse curvature at its minimum is 2 epsilon (rho0 / r0)^2; finite
    # differences pick up a little of the well's anharmonicity on top
    curvature = 2 * MORSE["epsilon"] * (MORSE["rho0"] / MORSE["r0"]) ** 2
    assert mode.force_constant == pytest.approx(
        curvature * EV_TO_HARTREE / ANGSTROM_TO_BOHR**2, rel=1e-2
    )
    assert mode.reduced_mass == pytest.approx(
        float(structure.get_masses()[0]) / 2, rel=1e-3
    )


def test_read_scan_energies_collects_numbered_directories(tmp_path: Path) -> None:
    """Energies come back in grid order, from the numbered directories only."""
    pytest.importorskip("ase")
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write

    energies = [-2.0, -3.5, -3.0, -1.5, 0.5]
    for i, energy in enumerate(energies):
        directory = tmp_path / f"{i:02d}"
        directory.mkdir()
        atoms = Atoms("H", positions=[[0, 0, 0]])
        atoms.calc = SinglePointCalculator(atoms, energy=energy)
        write(directory / "reactant_sp.xyz", atoms)

    # neither a stray directory nor a digit-named file is a grid point
    (tmp_path / "vib").mkdir()
    (tmp_path / "99").touch()

    assert read_scan_energies(tmp_path) == pytest.approx(energies)


def test_read_scan_energies_requires_an_output_per_point(tmp_path: Path) -> None:
    """A grid point without a finished job is reported, not skipped."""
    pytest.importorskip("ase")

    with pytest.raises(ValueError, match="No numbered scan directories"):
        read_scan_energies(tmp_path)

    (tmp_path / "00").mkdir()
    with pytest.raises(FileNotFoundError, match="product_sp"):
        read_scan_energies(tmp_path, state="product")


@pytest.mark.parametrize(
    ("blocked", "call", "match"),
    [
        (
            "ase.constraints",
            lambda tmp_path: run_da_scan(SENTINEL, 0, 2, SENTINEL, [2.4]),
            "donor-acceptor distance",
        ),
        (
            "ase.constraints",
            lambda tmp_path: optimize_proton(SENTINEL, 1, SENTINEL),
            "Optimizing the proton",
        ),
        (
            "ase.vibrations",
            lambda tmp_path: run_vibrations(SENTINEL, SENTINEL),
            "frequency calculation",
        ),
        (
            "ase.io",
            lambda tmp_path: read_scan_energies(tmp_path),
            "Reading scan outputs",
        ),
    ],
)
def test_runners_say_so_when_ase_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    blocked: str,
    call: Any,
    match: str,
) -> None:
    """ASE only ships in the `ase` extra, so its absence must be explained."""
    monkeypatch.setitem(sys.modules, blocked, None)

    with pytest.raises(ImportError, match=match) as error:
        call(tmp_path)

    assert "autopcet[ase]" in str(error.value)
