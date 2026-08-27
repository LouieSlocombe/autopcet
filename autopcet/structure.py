"""Reading, aligning, and averaging the molecular structures a PCET scan needs.

A vibronically nonadiabatic rate calculation starts from proton potentials
computed on a single averaged geometry, so the reactant and product structures
first have to be brought into a common frame: the proton donor-acceptor axis
along Z with its midpoint at the origin, and the remaining nuclei as close to
each other as a rotation about that axis allows.

Positions are ``(n_atoms, 3)`` arrays in angstrom.
"""

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from ._types import FloatArray

_XYZ_PRECISION = 8
"""Decimal places written per coordinate, and the precision alignment works at."""


def read_xyz(path: str | Path) -> tuple[list[str], FloatArray]:
    """Read an xyz file into its element symbols and ``(n_atoms, 3)`` positions."""
    lines = Path(path).read_text().splitlines()

    n_atoms = int(lines[0])
    symbols = []
    positions = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        symbol, x, y, z = lines[i + 2].split()[:4]
        symbols.append(symbol)
        positions[i] = (float(x), float(y), float(z))
    return symbols, positions


def write_xyz(
    path: str | Path,
    symbols: Sequence[str],
    positions: FloatArray,
    comment: str = "",
) -> None:
    """Write ``(n_atoms, 3)`` positions to an xyz file."""
    with open(path, "w") as xyz_file:
        xyz_file.write(f"{len(symbols)}\n")
        xyz_file.write(comment + "\n")
        for symbol, (x, y, z) in zip(symbols, positions, strict=True):
            xyz_file.write(
                f"{symbol:<2s} {x:15.{_XYZ_PRECISION}f} "
                f"{y:15.{_XYZ_PRECISION}f} {z:15.{_XYZ_PRECISION}f}\n"
            )


def round_to_xyz_precision(positions: FloatArray) -> FloatArray:
    """Round positions to the precision :func:`write_xyz` records.

    Alignment is done on coordinates rounded this way so that its result matches
    the structures written to disk rather than the full double precision behind
    them.
    """
    rounded = [float(f"{value:.{_XYZ_PRECISION}f}") for value in positions.ravel()]
    return np.reshape(np.array(rounded), positions.shape)


def _rotation_about_z(cos: float, sin: float) -> FloatArray:
    """Rotation about Z, from the angle's cosine and sine."""
    return np.array([[cos, sin, 0.0], [-sin, cos, 0.0], [0.0, 0.0, 1.0]])


def _rotation_about_y(cos: float, sin: float) -> FloatArray:
    """Rotation about Y, from the angle's cosine and sine."""
    return np.array([[cos, 0.0, -sin], [0.0, 1.0, 0.0], [sin, 0.0, cos]])


def rotation_about_z(angle: float) -> FloatArray:
    """Rotation matrix about the Z axis by ``angle`` radians.

    Apply it to ``(n_atoms, 3)`` positions as ``positions @ rotation.T``.
    """
    return _rotation_about_z(math.cos(angle), math.sin(angle))


def rotation_about_y(angle: float) -> FloatArray:
    """Rotation matrix about the Y axis by ``angle`` radians."""
    return _rotation_about_y(math.cos(angle), math.sin(angle))


def align_da_to_z(
    positions: FloatArray, donor: int, acceptor: int
) -> tuple[FloatArray, float]:
    """Put the donor at the origin and the donor-acceptor axis along Z.

    The axis is taken from the acceptor towards the donor, so the donor ends up
    on the positive Z side. Returns the rotated positions and the
    donor-acceptor distance.
    """
    positions = positions - positions[donor]

    separation = positions[donor] - positions[acceptor]
    distance = math.sqrt(float(separation @ separation))
    axis = separation / distance

    # Angle of the XY projection with the x-axis. If the donor-acceptor axis
    # already lies along Z there is no XY projection to take an angle from, and
    # the rotation about Z is the identity.
    xy_norm = math.hypot(axis[0], axis[1])
    if xy_norm < 1e-12:
        cos_xy, sin_xy = 1.0, 0.0
    else:
        cos_xy, sin_xy = axis[0] / xy_norm, axis[1] / xy_norm

    # angle with the z axis
    cos_z = float(axis[2])
    sin_z = math.sin(math.acos(cos_z))

    spin_into_xz_plane = _rotation_about_z(cos_xy, sin_xy)
    tilt_onto_z = _rotation_about_y(cos_z, sin_z)

    aligned = (positions @ spin_into_xz_plane.T) @ tilt_onto_z.T
    return aligned, distance


def centre_da_midpoint(positions: FloatArray, acceptor: int) -> FloatArray:
    """Shift along Z so the donor-acceptor midpoint sits at the origin.

    Assumes :func:`align_da_to_z` has already put the donor at the origin.
    """
    shifted = positions.copy()
    shifted[:, 2] -= positions[acceptor, 2] / 2
    return shifted


def minimize_rmsd_rotation(
    reference: FloatArray, positions: FloatArray
) -> tuple[float, float]:
    """Rotate about Z to best overlay ``positions`` on ``reference``.

    Returns the minimum RMSD and the rotation angle in radians that achieves it.
    """

    def rmsd(angle: float) -> float:
        rotated = positions @ rotation_about_z(angle).T
        return math.sqrt(float(np.mean(np.sum((reference - rotated) ** 2, axis=1))))

    minimization = minimize_scalar(rmsd, bounds=(0, 2 * math.pi), method="bounded")
    return float(minimization.fun), float(minimization.x)


def average_structures(first: FloatArray, second: FloatArray) -> FloatArray:
    """Average two sets of aligned positions atom by atom."""
    return (first + second) / 2


PROTON_SCAN_MINIMUM_HALF_WIDTH = 0.5
"""Smallest half-width in angstrom a proton scan spans."""

PROTON_SCAN_SPAN_FACTOR = 1.7
"""The scan spans this multiple of the distance between the proton endpoints."""


def proton_scan_grid(
    reactant_positions: FloatArray,
    product_positions: FloatArray,
    proton: int,
    points: int,
) -> tuple[FloatArray, FloatArray]:
    """Grid of geometries scanning the proton along its transfer axis.

    The axis passes through the proton's two endpoint positions: optimized on
    the donor in ``reactant_positions`` and on the acceptor in
    ``product_positions``. To build a proton potential the proton has to come
    very close to both the donor and the acceptor, so the scan spans
    :data:`PROTON_SCAN_SPAN_FACTOR` times the distance between the endpoints,
    and never less than twice :data:`PROTON_SCAN_MINIMUM_HALF_WIDTH`.

    Returns ``(offsets, geometries)``: the signed displacement of each grid
    point from the midpoint of the two endpoint positions, shape ``(points,)``,
    and the geometry at each point, shape ``(points, n_atoms, 3)``. Every atom
    but the proton sits at its reactant position.
    """
    proton_shift = product_positions[proton] - reactant_positions[proton]
    transfer_distance = float(np.linalg.norm(proton_shift))
    axis = proton_shift / transfer_distance
    midpoint = 0.5 * (reactant_positions[proton] + product_positions[proton])

    half_width = max(
        PROTON_SCAN_MINIMUM_HALF_WIDTH,
        PROTON_SCAN_SPAN_FACTOR * transfer_distance / 2,
    )
    offsets = np.linspace(-half_width, half_width, points)

    geometries = np.repeat(reactant_positions[np.newaxis], points, axis=0)
    geometries[:, proton] = offsets[:, np.newaxis] * axis + midpoint
    return offsets, geometries
