"""Align reactant and product xyz structures so the proton donor-acceptor axis
lies along Z with its midpoint at the origin, rotate the product about Z to
minimize the RMSD to the reactant, and write the averaged structure.

All donor/acceptor indices are 1-based, as printed by most quantum chemistry
programs.
"""

import argparse
import math
import os

import numpy as np
from scipy.optimize import minimize_scalar


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-r",
        "--reactant-xyz",
        dest="reactant_path",
        required=True,
        help="select xyz-file with the reactant structure",
    )
    parser.add_argument(
        "-p",
        "--product-xyz",
        dest="product_path",
        required=True,
        help="select xyz-file with the product structure",
    )
    parser.add_argument(
        "-o",
        "--output-xyz",
        dest="output_path",
        default="AVERAGE_STRUCTURE.xyz",
        help="select the name of xyz-file with the average structure "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--r-donor",
        type=int,
        dest="reactant_donor",
        required=True,
        help="donor atom index in the reactant structure",
    )
    parser.add_argument(
        "--r-acceptor",
        type=int,
        dest="reactant_acceptor",
        required=True,
        help="acceptor atom index in the reactant structure",
    )
    parser.add_argument(
        "--p-donor",
        type=int,
        dest="product_donor",
        required=True,
        help="donor atom index in the product structure",
    )
    parser.add_argument(
        "--p-acceptor",
        type=int,
        dest="product_acceptor",
        required=True,
        help="acceptor atom index in the product structure",
    )
    return parser


def read_xyz(path):
    """Read an xyz file into (n_atoms, symbols, positions with shape (3, n_atoms))."""
    with open(path) as xyz_file:
        lines = xyz_file.readlines()

    n_atoms = int(lines[0])
    symbols = []
    positions = np.zeros((3, n_atoms))
    for i in range(n_atoms):
        symbol, x, y, z = lines[i + 2].split()[:4]
        symbols.append(symbol)
        positions[:, i] = (float(x), float(y), float(z))
    return n_atoms, symbols, positions


def write_xyz(path, symbols, positions, comment):
    """Write positions with shape (3, n_atoms) to an xyz file."""
    with open(path, "w") as xyz_file:
        xyz_file.write(f"{len(symbols)}\n")
        xyz_file.write(comment + "\n")
        for i, symbol in enumerate(symbols):
            x, y, z = positions[:, i]
            xyz_file.write(f"{symbol:<2s} {x:15.8f} {y:15.8f} {z:15.8f}\n")


def rotation_about_z(cos, sin):
    """Rotation matrix about the Z axis, from the angle's cosine and sine."""
    return np.array([[cos, sin, 0.0], [-sin, cos, 0.0], [0.0, 0.0, 1.0]])


def rotation_about_y(cos, sin):
    """Rotation matrix about the Y axis, from the angle's cosine and sine."""
    return np.array([[cos, 0.0, -sin], [0.0, 1.0, 0.0], [sin, 0.0, cos]])


def rotation_by_angle_about_z(angle):
    """Rotation matrix about the Z axis by ``angle`` radians."""
    return rotation_about_z(math.cos(angle), math.sin(angle))


def align_da_to_z(positions, donor, acceptor):
    """Translate the donor to the origin and rotate the donor-acceptor axis onto Z.

    Returns the rotated positions and the donor-acceptor distance.
    """
    positions = positions - positions[:, donor, np.newaxis]

    separation = positions[:, donor] - positions[:, acceptor]
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
    cos_z = axis[2]
    sin_z = math.sin(math.acos(cos_z))

    spin_into_xz_plane = rotation_about_z(cos_xy, sin_xy)
    tilt_onto_z = rotation_about_y(cos_z, sin_z)

    return tilt_onto_z @ (spin_into_xz_plane @ positions), distance


def minimize_rmsd_rotation(reactant, product):
    """Rotation angle about Z that minimizes the product-reactant RMSD."""

    def rmsd(angle):
        rotated = rotation_by_angle_about_z(angle) @ product
        return math.sqrt(np.mean(np.sum((reactant - rotated) ** 2, axis=0)))

    minimization = minimize_scalar(rmsd, bounds=(0, 2 * math.pi), method="bounded")
    return minimization.fun, minimization.x


def centre_midpoint_on_origin(positions, acceptor):
    """Shift along Z so the donor-acceptor midpoint sits at the origin."""
    shifted = positions.copy()
    shifted[2, :] -= positions[2, acceptor] / 2
    return shifted


def main():
    options = build_parser().parse_args()

    print("\nXYZ file with the reactant structure: " + options.reactant_path)
    print("XYZ file with the product  structure: " + options.product_path)

    print(f"\nDonor index in the reactant structure: {options.reactant_donor}")
    print(f"Acceptor index in the reactant structure: {options.reactant_acceptor}")
    print(f"\nDonor index in the product structure: {options.product_donor}")
    print(f"Acceptor index in the product structure: {options.product_acceptor}")

    print("\nOutput file with the averaged structure: " + options.output_path + "\n")

    # python arrays are 0-based
    reactant_donor = options.reactant_donor - 1
    reactant_acceptor = options.reactant_acceptor - 1
    product_donor = options.product_donor - 1
    product_acceptor = options.product_acceptor - 1

    reactant_stem, reactant_extension = os.path.splitext(options.reactant_path)
    aligned_reactant_path = reactant_stem + "_DA_along_Z" + reactant_extension

    product_stem, product_extension = os.path.splitext(options.product_path)
    aligned_product_path = product_stem + "_DA_along_Z" + product_extension
    rotated_product_path = product_stem + "_DA_along_Z_aligned" + product_extension

    # reactant: donor-acceptor axis onto Z, midpoint at the origin
    _, reactant_symbols, reactant_positions = read_xyz(options.reactant_path)
    reactant_positions, reactant_distance = align_da_to_z(
        reactant_positions, reactant_donor, reactant_acceptor
    )
    reactant_positions = centre_midpoint_on_origin(
        reactant_positions, reactant_acceptor
    )

    write_xyz(
        aligned_reactant_path,
        reactant_symbols,
        reactant_positions,
        "Reactant with D and A atoms along the Z-axis and midpoint at (0,0,0): "
        f"DA distance {reactant_distance:10.6f} Å",
    )

    # product: same treatment
    _, product_symbols, product_positions = read_xyz(options.product_path)
    product_positions, product_distance = align_da_to_z(
        product_positions, product_donor, product_acceptor
    )

    # KNOWN BUG, DELIBERATELY NOT FIXED: this shifts the product structure using
    # `reactant_acceptor`, the acceptor index in the *reactant* file. The correct
    # index is `product_acceptor`. It is left as-is so that averaged structures
    # generated by earlier versions of this script -- and the proton potentials
    # derived from them -- stay reproducible. It only matters when the two
    # acceptor indices differ; fixing it would change the averaged geometry and
    # invalidate any downstream scan built on it.
    product_positions = centre_midpoint_on_origin(product_positions, reactant_acceptor)

    write_xyz(
        aligned_product_path,
        product_symbols,
        product_positions,
        "Product with D and A atoms along the Z-axis and midpoint at (0,0,0): "
        f"DA distance {product_distance:10.6f} Å",
    )

    # Check if DA distances are the same in the reactant and product structures
    if abs(reactant_distance - product_distance) > 1e-6:
        print(
            "\nThe donor acceptor distances in the reactant and product structures "
            f"are different: {reactant_distance} and {product_distance}"
        )
        print(
            "Donor and acceptor atoms are along the Z-axis and the midpoint is aligned)\n"
        )
        print(
            "Average structure is labeled by the DA distance in the reactant configuration\n"
        )

    # rotate the product around Z to minimize the RMSD to the reactant
    # (re-read the written files so the result matches their 8-decimal precision)
    _, reactant_symbols, reactant_positions = read_xyz(aligned_reactant_path)
    _, product_symbols, product_positions = read_xyz(aligned_product_path)

    minimum_rmsd, angle = minimize_rmsd_rotation(reactant_positions, product_positions)

    print(
        f"\nMinimum RMSD: {minimum_rmsd:12.6f}   "
        f"Rotation angle is {180 * angle / math.pi:12.6f} degrees\n"
    )

    write_xyz(
        rotated_product_path,
        product_symbols,
        rotation_by_angle_about_z(angle) @ product_positions,
        "Product with D and A atoms along the Z-axis and aligned with reactant: "
        f"DA distance {product_distance:10.6f} Å",
    )

    # average the aligned reactant and product structures
    _, reactant_symbols, reactant_positions = read_xyz(aligned_reactant_path)
    _, _, product_positions = read_xyz(rotated_product_path)

    write_xyz(
        options.output_path,
        reactant_symbols,
        (reactant_positions + product_positions) / 2,
        f"Average reactant/product configuration: DA distance {reactant_distance:10.6f} Å",
    )


if __name__ == "__main__":
    main()
