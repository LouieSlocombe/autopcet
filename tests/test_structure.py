"""Alignment and averaging of the structures a proton potential scan starts from."""

import math
from pathlib import Path

import numpy as np
import pytest

from autopcet import (
    align_da_to_z,
    average_structures,
    centre_da_midpoint,
    minimize_rmsd_rotation,
    proton_scan_grid,
    read_xyz,
    rotation_about_y,
    rotation_about_z,
    round_to_xyz_precision,
    write_xyz,
)
from autopcet._types import FloatArray

DATA = Path(__file__).resolve().parent / "data"


def test_xyz_round_trips_through_a_file(tmp_path: Path) -> None:
    """Writing then reading an xyz recovers the symbols and positions."""
    symbols = ["O", "H", "N"]
    positions = np.array([[0.0, 0.0, 0.0], [0.98, 0.12, 0.05], [2.65, 0.31, 0.14]])

    path = tmp_path / "structure.xyz"
    write_xyz(path, symbols, positions, "a comment")

    read_symbols, read_positions = read_xyz(path)
    assert read_symbols == symbols
    assert read_positions == pytest.approx(positions)
    assert path.read_text().splitlines()[1] == "a comment"


def test_write_xyz_rejects_mismatched_symbols(tmp_path: Path) -> None:
    """Symbols and positions have to describe the same number of atoms."""
    with pytest.raises(ValueError):
        write_xyz(tmp_path / "bad.xyz", ["O", "H"], np.zeros((3, 3)))


def test_round_to_xyz_precision_matches_what_is_written(tmp_path: Path) -> None:
    """Rounding in memory agrees with a round trip through a written file."""
    rng = np.random.default_rng(0)
    positions = rng.normal(size=(8, 3)) * 3

    path = tmp_path / "structure.xyz"
    write_xyz(path, ["C"] * 8, positions)
    _, from_file = read_xyz(path)

    assert round_to_xyz_precision(positions) == pytest.approx(from_file, abs=0.0)


@pytest.mark.parametrize("rotation", [rotation_about_y, rotation_about_z])
def test_rotations_are_orthonormal(rotation: object) -> None:
    """Rotation matrices preserve lengths and angles."""
    matrix = rotation(0.7)  # type: ignore[operator]
    assert matrix @ matrix.T == pytest.approx(np.eye(3))
    assert np.linalg.det(matrix) == pytest.approx(1.0)


def test_align_da_to_z_puts_the_axis_on_z() -> None:
    """The donor lands on the origin and the acceptor on the Z axis below it."""
    positions = np.array(
        [[1.0, 2.0, 3.0], [1.5, 2.4, 3.9], [2.1, 3.6, 4.4], [0.2, 1.1, 2.0]]
    )
    donor, acceptor = 0, 2

    aligned, distance = align_da_to_z(positions, donor, acceptor)

    expected = np.linalg.norm(positions[acceptor] - positions[donor])
    assert distance == pytest.approx(expected)
    assert aligned[donor] == pytest.approx(np.zeros(3))
    # the axis runs from the acceptor towards the donor, so the acceptor is at -Z
    assert aligned[acceptor] == pytest.approx([0.0, 0.0, -distance])


def test_align_da_to_z_preserves_all_interatomic_distances() -> None:
    """Alignment is a rigid motion, so the internal geometry is untouched."""
    rng = np.random.default_rng(3)
    positions = rng.normal(size=(6, 3)) * 2

    aligned, _ = align_da_to_z(positions, 0, 3)

    def pair_distances(coordinates: FloatArray) -> FloatArray:
        difference = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
        distances: FloatArray = np.linalg.norm(difference, axis=-1)
        return distances

    assert pair_distances(aligned) == pytest.approx(pair_distances(positions))


def test_align_da_to_z_handles_an_axis_already_along_z() -> None:
    """A donor-acceptor axis on Z has no XY projection to take an angle from."""
    positions = np.array([[0.0, 0.0, 2.7], [0.4, 0.0, 1.5], [0.0, 0.0, 0.0]])

    aligned, distance = align_da_to_z(positions, 0, 2)

    assert distance == pytest.approx(2.7)
    assert aligned[2] == pytest.approx([0.0, 0.0, -2.7])


def test_centre_da_midpoint_splits_the_axis_about_the_origin() -> None:
    """After centring, the donor and acceptor sit at plus and minus half of R."""
    aligned, distance = align_da_to_z(
        np.array([[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [2.6, 0.3, 0.1]]), 0, 2
    )

    centred = centre_da_midpoint(aligned, 2)

    assert centred[0, 2] == pytest.approx(distance / 2)
    assert centred[2, 2] == pytest.approx(-distance / 2)


def test_minimize_rmsd_rotation_recovers_a_known_rotation() -> None:
    """A structure rotated about Z is rotated back onto its reference."""
    rng = np.random.default_rng(11)
    reference = rng.normal(size=(7, 3))
    angle = 1.3
    rotated = reference @ rotation_about_z(-angle).T

    rmsd, recovered = minimize_rmsd_rotation(reference, rotated)

    # scipy's bounded minimizer defaults to xatol=1e-5, so the angle -- and with
    # it the residual RMSD -- is only recovered to about that precision
    assert rmsd == pytest.approx(0.0, abs=1e-5)
    assert recovered == pytest.approx(angle, abs=1e-4)


def test_minimize_rmsd_rotation_searches_the_whole_circle() -> None:
    """The optimum is found even when it lies near the top of the range."""
    rng = np.random.default_rng(12)
    reference = rng.normal(size=(7, 3))
    angle = 2 * math.pi - 0.4
    rotated = reference @ rotation_about_z(-angle).T

    rmsd, recovered = minimize_rmsd_rotation(reference, rotated)

    assert rmsd == pytest.approx(0.0, abs=1e-5)
    assert recovered == pytest.approx(angle, abs=1e-4)


def test_average_structures_is_the_midpoint() -> None:
    """Averaging two structures takes each atom to the midpoint of its pair."""
    first = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    second = np.array([[2.0, 4.0, 6.0], [3.0, 2.0, 1.0]])

    assert average_structures(first, second) == pytest.approx(
        np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])
    )


def test_proton_scan_grid_floors_the_half_width() -> None:
    """A short proton transfer still spans at least half an angstrom each way."""
    reactant = np.array([[0.0, 0.0, 0.0], [2.4, 0.0, 0.0], [1.0, 0.0, 0.0]])
    product = np.array([[0.0, 0.0, 0.0], [2.4, 0.0, 0.0], [1.4, 0.0, 0.0]])

    offsets, geometries = proton_scan_grid(reactant, product, 2, 11)

    # 1.7 * 0.4 / 2 = 0.34 is below the 0.5 angstrom floor
    assert offsets[0] == pytest.approx(-0.5)
    assert offsets[-1] == pytest.approx(0.5)
    assert offsets == pytest.approx(-offsets[::-1])
    assert geometries.shape == (11, 3, 3)


def test_proton_scan_grid_spans_the_transfer_distance() -> None:
    """A long transfer spans 1.7 times the endpoint separation, centred on it."""
    rng = np.random.default_rng(1)
    reactant = rng.normal(size=(4, 3))
    product = reactant.copy()
    product[1] = reactant[1] + np.array([0.6, 0.8, 0.0])  # transfer distance 1

    offsets, geometries = proton_scan_grid(reactant, product, 1, 5)

    assert offsets[-1] == pytest.approx(1.7 / 2)
    midpoint = (reactant[1] + product[1]) / 2
    axis = product[1] - reactant[1]
    for offset, geometry in zip(offsets, geometries, strict=True):
        # every atom but the proton sits exactly at its reactant position
        frame = geometry.copy()
        frame[1] = reactant[1]
        assert frame == pytest.approx(reactant, abs=0.0)
        assert geometry[1] == pytest.approx(midpoint + offset * axis)
    assert geometries[2, 1] == pytest.approx(midpoint)


def test_proton_scan_grid_rejects_coincident_endpoints() -> None:
    """Endpoints in the same place have no axis, and used to yield all nan."""
    reactant = np.array([[0.0, 0, 0], [1.2, 0, 0], [2.45, 0, 0]])

    with pytest.raises(ValueError, match="same place"):
        proton_scan_grid(reactant, reactant.copy(), 1, 5)
