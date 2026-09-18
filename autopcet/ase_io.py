"""Running the PCET scans through ASE, with any of its calculators.

Everything :mod:`autopcet.gaussian_io` prepares Gaussian input files for can
also run in-process against an ASE calculator -- ORCA, xTB, or anything else
implementing ASE's ``BaseCalculator`` interface: the constrained donor-acceptor
distance scan, the proton endpoint optimizations, the proton potential scan,
and the frequency calculation behind the effective donor-acceptor mode. For
jobs run elsewhere, :func:`read_scan_energies` collects the energies from the
output files instead.

The diabatic electronic states are selected through the calculator: configure
one calculator per state, carrying that state's charge and multiplicity, and
run each scan once per state. Energies are in eV and positions in angstrom
throughout, as ASE returns them.

Everything that optimizes or displaces atoms -- :func:`run_da_scan`,
:func:`optimize_proton`, :func:`run_vibrations` -- needs forces from the
calculator, so a calculator running an external program has to be configured
to compute the gradient: ORCA needs ``EnGrad`` in its ``orcasimpleinput``, for
one. :func:`run_proton_scan` and :func:`read_scan_energies` get by on energies
alone.

ASE itself is only imported when a function here needs it; install it with the
``ase`` extra.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from ._types import FloatArray, IntArray
from .constants import (
    ANGSTROM_TO_CM,
    AU_TIME_TO_SECONDS,
    AU_TO_MDYNE_PER_ANGSTROM,
    DALTON_TO_ELECTRON_MASS,
    SPEED_OF_LIGHT,
)
from .gaussian_io import EffectiveMode, effective_da_mode
from .structure import proton_scan_grid

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import BaseCalculator
    from ase.vibrations import VibrationsData

ASE_HINT = (
    "{need} needs the Atomic Simulation Environment. Install it with "
    '`pip install "autopcet[ase]"`.'
)

SCAN_OUTPUT_SUFFIXES = (".log", ".out", ".xyz", ".traj")
"""Suffixes :func:`read_scan_energies` looks for, in order of preference."""

FORCES_HINT = (
    "{need} needs forces, but the calculator did not return any ({error}). A "
    "calculator that runs an external program has to be told to compute the "
    "gradient: ORCA needs `EnGrad` in its `orcasimpleinput`, for instance. "
    "Only run_proton_scan and read_scan_energies get by on energies alone."
)


def _require_forces(atoms: Atoms, need: str) -> None:
    """Fail before a long scan when the calculator cannot return forces.

    ASE caches a calculator's results against the state of the atoms it was
    given, so the optimizer's own first force evaluation on the same object
    reuses this one: the check costs a wasted job only when it fails.
    """
    try:
        atoms.get_forces()
    except NotImplementedError as error:
        # ASE raises PropertyNotImplementedError, a NotImplementedError, both
        # when the calculator declares no forces and when a run produced none
        raise RuntimeError(FORCES_HINT.format(need=need, error=error)) from error


def _finite_energy(atoms: Atoms, where: str) -> float:
    """Read a potential energy, refusing the NaN of a failed calculation.

    ASE hands back ``nan`` rather than raising when a calculation converged
    badly enough to say so -- an ORCA output whose final energy is flagged
    ``Wavefunction not fully converged``, for one. Left alone that ``nan``
    travels all the way to the rate constant.
    """
    energy = float(atoms.get_potential_energy())
    if not np.isfinite(energy):
        directory = getattr(atoms.calc, "directory", None)
        location = f" Its files are in {directory}." if directory else ""
        raise RuntimeError(
            f"The calculator returned a non-finite energy ({energy}) {where}, "
            f"which usually means the calculation did not converge.{location}"
        )
    return energy


@dataclass(frozen=True)
class ProtonScan:
    """A proton potential computed along the proton transfer axis."""

    offsets: FloatArray
    """Proton displacement from the endpoint midpoint in angstrom, ``(points,)``."""

    energies: FloatArray
    """Energy at each grid point in eV, as the calculator returned them."""

    positions: FloatArray
    """Geometry at each grid point in angstrom, ``(points, n_atoms, 3)``."""

    @property
    def relative_energies(self) -> FloatArray:
        """Energies shifted so the lowest grid point sits at zero.

        ``(scan.offsets, scan.relative_energies)`` is a tabulated potential
        ready for :class:`autopcet.rates.PCET`.
        """
        return self.energies - self.energies.min()


@dataclass(frozen=True)
class DistanceScan:
    """One diabatic state reoptimized at each frozen donor-acceptor distance."""

    distances: FloatArray
    """Donor-acceptor distances in angstrom, ``(points,)``."""

    energies: FloatArray
    """Energy at each constrained minimum in eV."""

    structures: list[Atoms]
    """The optimized geometry at each distance, with no calculator attached."""


def run_proton_scan(
    reactant: Atoms,
    product: Atoms,
    proton: int,
    calculator: BaseCalculator,
    points: int = 20,
) -> ProtonScan:
    """Compute a proton potential along the transfer axis with ``calculator``.

    Takes the two averaged structures with the proton optimized on the donor
    (``reactant``) and on the acceptor (``product``), builds the same grid the
    ``autopcet-scan-proton`` helper writes Gaussian inputs for -- see
    :func:`autopcet.structure.proton_scan_grid` -- and computes a single point
    energy at each geometry. Run it once per diabatic state, with a calculator
    carrying that state's charge and multiplicity.

    Calculators that run an external program reuse their scratch directory
    from point to point; to keep every job's files, write inputs with the
    command-line helper instead and collect the energies afterwards with
    :func:`read_scan_energies`.
    """
    if len(reactant) != len(product):
        raise ValueError(
            "The reactant and product structures must hold the same atoms, "
            f"got {len(reactant)} and {len(product)}."
        )
    if list(reactant.symbols) != list(product.symbols):
        raise ValueError(
            "The reactant and product structures must list the same elements "
            "in the same order."
        )
    if not 0 <= proton < len(reactant):
        raise ValueError(
            f"Proton index {proton} is out of range for {len(reactant)} atoms."
        )

    offsets, geometries = proton_scan_grid(
        reactant.get_positions(), product.get_positions(), proton, points
    )

    energies = np.zeros(points)
    for i, geometry in enumerate(geometries):
        frame = reactant.copy()
        frame.set_positions(geometry)
        frame.calc = calculator
        energies[i] = _finite_energy(frame, f"at grid point {i} of {points}")

    return ProtonScan(offsets=offsets, energies=energies, positions=geometries)


def run_da_scan(
    structure: Atoms,
    donor: int,
    acceptor: int,
    calculator: BaseCalculator,
    distances: Sequence[float] | FloatArray,
    fix: int = 0,
    fmax: float = 0.05,
    steps: int = 200,
) -> DistanceScan:
    """Reoptimize one state at each frozen donor-acceptor distance.

    The in-process counterpart of the ``autopcet-scan-da`` helper: for each
    distance the structure is scaled to it, the donor-acceptor distance is
    frozen, and everything else is relaxed with BFGS until the largest force
    drops below ``fmax`` (eV/angstrom). ``fix`` names which of the two atoms
    stays put while the distance is set: 0 for the donor (the reactant's
    anchor), 1 for the acceptor (the product's). Only the starting guess for
    the constrained optimization depends on it.

    Raises ``RuntimeError`` if the calculator returns no forces, if an
    optimization has not converged after ``steps`` steps, or if a converged
    point comes back with a non-finite energy.
    """
    try:
        from ase.constraints import FixBondLength
        from ase.optimize import BFGS
    except ImportError as error:
        raise ImportError(
            ASE_HINT.format(need="Scanning the donor-acceptor distance")
        ) from error

    distance_grid = np.asarray(distances, dtype=float)
    energies = np.zeros(distance_grid.size)
    structures: list[Atoms] = []
    for i, distance in enumerate(distance_grid):
        scaled = structure.copy()
        scaled.set_distance(donor, acceptor, float(distance), fix=fix)
        scaled.set_constraint([*scaled.constraints, FixBondLength(donor, acceptor)])
        scaled.calc = calculator
        _require_forces(scaled, "Scanning the donor-acceptor distance")

        if not BFGS(scaled, logfile=None).run(fmax=fmax, steps=steps):
            raise RuntimeError(
                f"The constrained optimization at R = {distance:.2f} A did not "
                f"converge to fmax = {fmax} within {steps} steps."
            )

        energies[i] = _finite_energy(scaled, f"at R = {distance:.2f} A")
        scaled.calc = None
        scaled.set_constraint(structure.constraints)
        structures.append(scaled)

    return DistanceScan(
        distances=distance_grid, energies=energies, structures=structures
    )


def optimize_proton(
    structure: Atoms,
    proton: int,
    calculator: BaseCalculator,
    fmax: float = 0.01,
    steps: int = 200,
) -> Atoms:
    """Relax the proton with every other atom frozen.

    Prepares the endpoint structures :func:`run_proton_scan` needs: starting
    from the averaged geometry, relax the proton onto the donor with the
    reactant state's calculator, and onto the acceptor with the product's.
    Returns the relaxed structure, leaving the input untouched; raises
    ``RuntimeError`` if the calculator returns no forces, or if the
    optimization has not converged after ``steps`` steps.
    """
    try:
        from ase.constraints import FixAtoms
        from ase.optimize import BFGS
    except ImportError as error:
        raise ImportError(ASE_HINT.format(need="Optimizing the proton")) from error

    if not 0 <= proton < len(structure):
        raise ValueError(
            f"Proton index {proton} is out of range for {len(structure)} atoms."
        )

    relaxed: Atoms = structure.copy()
    relaxed.set_constraint(FixAtoms(mask=np.arange(len(relaxed)) != proton))
    relaxed.calc = calculator
    _require_forces(relaxed, "Optimizing the proton")

    if not BFGS(relaxed, logfile=None).run(fmax=fmax, steps=steps):
        raise RuntimeError(
            f"The proton optimization did not converge to fmax = {fmax} "
            f"within {steps} steps."
        )

    relaxed.calc = None
    relaxed.set_constraint(structure.constraints)
    return relaxed


def run_vibrations(
    structure: Atoms,
    calculator: BaseCalculator,
    indices: Sequence[int] | None = None,
    delta: float = 0.01,
    directory: str | Path = "vib",
) -> VibrationsData:
    """Compute normal modes by finite differences with ``calculator``.

    The result feeds :func:`effective_mode_from_vibrations`. ``indices``
    restricts the displaced atoms; the default displaces all of them, by
    ``delta`` angstrom each way. The displacement forces are cached in
    ``directory``, so an interrupted calculation resumes where it stopped.
    ASE would reuse that cache for any structure, so what filled it is
    recorded alongside; a fresh structure needs a fresh directory, and
    reusing one raises ``RuntimeError`` rather than returning the previous
    structure's Hessian.
    """
    try:
        from ase.vibrations import Vibrations
    except ImportError as error:
        raise ImportError(
            ASE_HINT.format(need="A finite difference frequency calculation")
        ) from error

    work = structure.copy()
    work.calc = calculator
    vibrations = Vibrations(
        work,
        indices=None if indices is None else list(indices),
        name=str(directory),
        delta=delta,
    )
    _check_vibration_cache(Path(directory), work, vibrations.indices, delta)
    _require_forces(work, "A finite difference frequency calculation")
    vibrations.run()
    return cast("VibrationsData", vibrations.get_vibrations())


def _check_vibration_cache(
    directory: Path,
    structure: Atoms,
    indices: IntArray,
    delta: float,
) -> None:
    """Refuse a cache that was filled by a different calculation.

    ASE keys its displacement cache on the name of the displacement alone --
    ``0x+``, ``0y-`` and so on -- and never on the geometry, so a second run
    in the same directory silently reuses the first structure's forces. Record
    what filled the cache and compare on the way back in. The file sits beside
    ASE's own entries without disturbing them: ASE globs for ``cache.*.json``.
    """
    fingerprint = {
        "symbols": list(structure.symbols),
        "positions": structure.get_positions().round(8).tolist(),
        "indices": [int(index) for index in indices],
        "delta": delta,
    }
    record = directory / "autopcet-structure.json"
    if record.is_file():
        if json.loads(record.read_text()) != fingerprint:
            raise RuntimeError(
                f"The displacement cache in {directory} was filled by a "
                "different structure, and ASE would silently reuse its forces. "
                "Delete the directory, or point this calculation at a fresh one."
            )
        return

    directory.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(fingerprint))


def effective_mode_from_vibrations(
    vibrations: VibrationsData,
    donor: int,
    acceptor: int,
    frequency_floor: float = 10.0,
) -> EffectiveMode:
    """Project ASE normal modes onto the proton donor-acceptor axis.

    The ASE counterpart of feeding a Gaussian ``freq=HPmodes`` log through
    :func:`autopcet.gaussian_io.read_frequencies` and
    :func:`autopcet.gaussian_io.effective_da_mode`: the modes are converted to
    the reduced masses, force constants, and unit-normalized displacements of
    Gaussian's convention and projected the same way.

    A finite difference Hessian keeps the six rigid-body modes, whose near-zero
    force constants the projection cannot divide by, so modes with imaginary
    frequencies or frequencies below ``frequency_floor`` (cm^-1) are dropped
    first. Dropping an imaginary mode above the floor warns: at a geometry that
    is not a true minimum -- one optimized under a frozen donor-acceptor
    distance, say -- the mode along that axis is the one most likely to come
    back imaginary, and discarding it overestimates the effective force
    constant. Both axis atoms must have been displaced in the underlying
    frequency calculation.
    """
    indices = vibrations.get_indices()
    for atom, name in ((donor, "donor"), (acceptor, "acceptor")):
        if atom not in indices:
            raise ValueError(
                f"The {name} (atom {atom}) was not displaced in the frequency "
                "calculation, so its motion is missing from the normal modes."
            )

    positions = vibrations.get_atoms().get_positions()
    frequencies = np.asarray(vibrations.get_frequencies())
    modes = np.asarray(vibrations.get_modes(all_atoms=True))
    displacements = modes.reshape(modes.shape[0], -1)

    # drop rigid-body and unstable modes before the projection divides by
    # their force constants; a rigid-body mode lands on either side of zero, so
    # only an imaginary frequency above the floor says anything about the
    # structure
    unstable = frequencies.imag > frequency_floor
    if unstable.any():
        warnings.warn(
            f"Dropping {int(unstable.sum())} imaginary mode(s), the largest at "
            f"{frequencies.imag[unstable].max():.1f}i cm^-1, before projecting "
            "onto the donor-acceptor axis. The structure is not a minimum, so "
            "the effective mode is missing whatever compliance they carried.",
            stacklevel=2,
        )

    vibrational = (frequencies.imag == 0) & (frequencies.real > frequency_floor)
    if not vibrational.any():
        raise ValueError(
            f"No vibrational modes above {frequency_floor} cm^-1; the structure "
            "is not at a minimum, or the floor is set too high."
        )
    wavenumbers = frequencies.real[vibrational]
    displacements = displacements[vibrational]

    # ASE modes are mass-weighted eigenvectors over sqrt(mass): their squared
    # norm is 1/mu, and Gaussian prints them renormalized to unit length
    reduced_masses = 1 / np.sum(displacements**2, axis=1)
    normal_modes = displacements * np.sqrt(reduced_masses)[:, np.newaxis]

    # invert the wavenumber formula of gaussian_io.effective_da_mode
    angular = 2 * pi * wavenumbers * SPEED_OF_LIGHT * ANGSTROM_TO_CM
    force_constants_au = (
        reduced_masses * DALTON_TO_ELECTRON_MASS * (angular * AU_TIME_TO_SECONDS) ** 2
    )
    force_constants = force_constants_au * AU_TO_MDYNE_PER_ANGSTROM

    return effective_da_mode(
        positions, donor, acceptor, reduced_masses, force_constants, normal_modes
    )


def read_scan_energies(
    directory: str | Path,
    state: str = "reactant",
) -> FloatArray:
    """Collect the energies of an externally run proton scan.

    Walks the numbered subdirectories ``autopcet-scan-proton`` created, reads
    the completed ``<state>_sp`` output in each -- any format ASE recognizes,
    trying the suffixes in :data:`SCAN_OUTPUT_SUFFIXES` -- and returns the
    energies in eV, in grid order. :func:`autopcet.structure.proton_scan_grid`
    with the same inputs recovers the matching proton coordinates.
    """
    try:
        from ase.io import read
    except ImportError as error:
        raise ImportError(ASE_HINT.format(need="Reading scan outputs")) from error

    directory = Path(directory)
    grid_points = sorted(
        (
            child
            for child in directory.iterdir()
            if child.is_dir() and child.name.isdigit()
        ),
        key=lambda child: int(child.name),
    )
    if not grid_points:
        raise ValueError(f"No numbered scan directories found in {directory}.")

    energies = np.zeros(len(grid_points))
    for i, grid_point in enumerate(grid_points):
        for suffix in SCAN_OUTPUT_SUFFIXES:
            output = grid_point / f"{state}_sp{suffix}"
            if output.exists():
                # the default index selects the last frame, always a single Atoms
                energies[i] = cast("Atoms", read(output)).get_potential_energy()
                break
        else:
            raise FileNotFoundError(
                f"No {state}_sp output found in {grid_point} with any of the "
                f"suffixes {', '.join(SCAN_OUTPUT_SUFFIXES)}."
            )
    return energies
