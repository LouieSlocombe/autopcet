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
        dest="rea",
        required=True,
        help="select xyz-file with the reactant structure",
    )
    parser.add_argument(
        "-p",
        "--product-xyz",
        dest="pro",
        required=True,
        help="select xyz-file with the product structure",
    )
    parser.add_argument(
        "-o",
        "--output-xyz",
        dest="output_name",
        default="AVERAGE_STRUCTURE.xyz",
        help="select the name of xyz-file with the average structure "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--r-donor",
        type=int,
        dest="rdo",
        required=True,
        help="donor atom index in the reactant structure",
    )
    parser.add_argument(
        "--r-acceptor",
        type=int,
        dest="rac",
        required=True,
        help="acceptor atom index in the reactant structure",
    )
    parser.add_argument(
        "--p-donor",
        type=int,
        dest="pdo",
        required=True,
        help="donor atom index in the product structure",
    )
    parser.add_argument(
        "--p-acceptor",
        type=int,
        dest="pac",
        required=True,
        help="acceptor atom index in the product structure",
    )
    return parser


def read_xyz(path):
    """Read an xyz file into (natom, symbols, positions with shape (3, natom))."""
    with open(path) as fp:
        lines = fp.readlines()
    natom = int(lines[0])
    symbols = []
    pos = np.zeros((3, natom))
    for i in range(natom):
        xyz = lines[i + 2].split()
        symbols.append(xyz[0])
        pos[0][i] = float(xyz[1])
        pos[1][i] = float(xyz[2])
        pos[2][i] = float(xyz[3])
    return natom, symbols, pos


def write_xyz(path, symbols, pos, comment):
    """Write positions with shape (3, natom) to an xyz file."""
    natom = len(symbols)
    with open(path, "w") as fp:
        fp.write(str(natom) + "\n")
        fp.write(comment + "\n")
        for i in range(natom):
            fp.write(
                f"{symbols[i]:<2s} {pos[0][i]:15.8f} {pos[1][i]:15.8f} {pos[2][i]:15.8f}\n"
            )


def align_da_to_z(pos, donor, acceptor):
    """Translate the donor to the origin and rotate the donor-acceptor axis onto Z.

    Returns the rotated positions and the donor-acceptor distance.
    """
    natom = pos.shape[1]

    hr = [0.0, 0.0, 0.0]
    for i in range(3):
        hr[i] = pos[i][donor]

    for j in range(natom):
        for i in range(3):
            pos[i][j] = pos[i][j] - hr[i]

    rmag = math.sqrt(
        (pos[0][donor] - pos[0][acceptor]) ** 2
        + (pos[1][donor] - pos[1][acceptor]) ** 2
        + (pos[2][donor] - pos[2][acceptor]) ** 2
    )

    vec = [0.0, 0.0, 0.0]
    for i in range(3):
        vec[i] = (pos[i][donor] - pos[i][acceptor]) / rmag

    # angle of projection with x-axis
    costhx = vec[0] / math.sqrt(vec[0] ** 2 + vec[1] ** 2)
    sinthx = vec[1] / math.sqrt(vec[0] ** 2 + vec[1] ** 2)

    # angle with z axis
    costhz = vec[2]
    sinthz = math.sin(math.acos(costhz))

    rrx = np.zeros((3, 3))
    rrz = np.zeros((3, 3))

    rrx[0][0] = costhx
    rrx[1][1] = costhx
    rrx[0][1] = sinthx
    rrx[1][0] = -sinthx
    rrx[2][2] = 1.0e0

    rrz[0][0] = costhz
    rrz[2][2] = costhz
    rrz[0][2] = -sinthz
    rrz[2][0] = sinthz
    rrz[1][1] = 1.0e0

    return np.dot(np.dot(rrz, rrx), pos), rmag


def rotation_z(deg_rad):
    """Rotation matrix about the Z axis by ``deg_rad`` radians."""
    rrz = np.zeros((3, 3))
    rrz[0][0] = math.cos(deg_rad)
    rrz[0][1] = math.sin(deg_rad)
    rrz[1][0] = -math.sin(deg_rad)
    rrz[1][1] = math.cos(deg_rad)
    rrz[2][2] = 1.0e0
    return rrz


def minimize_rmsd_rotation(rea, pro, natom):
    """Rotation angle about Z that minimizes the product-reactant RMSD."""

    def rmsdfun(deg_rad):
        pro_new = np.dot(rotation_z(deg_rad), pro)
        e = 0
        for k in range(natom):
            a = (rea[0][k] - pro_new[0][k]) * (rea[0][k] - pro_new[0][k])
            b = (rea[1][k] - pro_new[1][k]) * (rea[1][k] - pro_new[1][k])
            c = (rea[2][k] - pro_new[2][k]) * (rea[2][k] - pro_new[2][k])
            d = a + b + c
            e = d + e
        return math.sqrt((1.0 / float(natom)) * e)

    minimization = minimize_scalar(rmsdfun, bounds=(0, 2 * math.pi), method="bounded")
    return minimization.fun, minimization.x


def main():
    options = build_parser().parse_args()

    print("\nXYZ file with the reactant structure: " + options.rea)
    print("XYZ file with the product  structure: " + options.pro)

    print(f"\nDonor index in the reactant structure: {options.rdo}")
    print(f"Acceptor index in the reactant structure: {options.rac}")
    print(f"\nDonor index in the product structure: {options.pdo}")
    print(f"Acceptor index in the product structure: {options.pac}")

    print("\nOutput file with the averaged structure: " + options.output_name + "\n")

    # (index starts from 0 in python arrays)
    rdo = options.rdo - 1
    rac = options.rac - 1
    pdo = options.pdo - 1
    pac = options.pac - 1

    reabasename, reaext = os.path.splitext(options.rea)
    reanew = reabasename + "_DA_along_Z" + reaext

    probasename, proext = os.path.splitext(options.pro)
    proali = probasename + "_DA_along_Z" + proext
    prorot = probasename + "_DA_along_Z_aligned" + proext

    # reactant: donor-acceptor axis onto Z, midpoint at the origin
    natom, rea_symbols, posr = read_xyz(options.rea)
    posr_new, rda_distance = align_da_to_z(posr, rdo, rac)

    shift = posr_new[2][rac] / 2
    for i in range(natom):
        posr_new[2][i] = posr_new[2][i] - shift

    write_xyz(
        reanew,
        rea_symbols,
        posr_new,
        "Reactant with D and A atoms along the Z-axis and midpoint at (0,0,0): "
        f"DA distance {rda_distance:10.6f} Å",
    )

    # product: same treatment
    natom, pro_symbols, posp = read_xyz(options.pro)
    posr_new_pro, pda_distance = align_da_to_z(posp, pdo, pac)

    # NOTE: the shift historically uses the reactant acceptor index, not the
    # product one; preserved for backward compatibility with earlier outputs
    shift = posr_new_pro[2][rac] / 2
    for i in range(natom):
        posr_new_pro[2][i] = posr_new_pro[2][i] - shift

    write_xyz(
        proali,
        pro_symbols,
        posr_new_pro,
        "Product with D and A atoms along the Z-axis and midpoint at (0,0,0): "
        f"DA distance {pda_distance:10.6f} Å",
    )

    # Check if DA distances are the same in the reactant and product structures
    if abs(rda_distance - pda_distance) > 10 ** (-6):
        print(
            "\nThe donor acceptor distances in the reactant and product structures "
            f"are different: {rda_distance} and {pda_distance}"
        )
        print(
            "Donor and acceptor atoms are along the Z-axis and the midpoint is aligned)\n"
        )
        print(
            "Average structure is labeled by the DA distance in the reactant configuration\n"
        )

    da_distance = rda_distance

    # rotate the product around Z to minimize the RMSD to the reactant
    # (re-read the written files so the result matches their 8-decimal precision)
    natom, rea_symbols, rea = read_xyz(reanew)
    natom, pro_symbols, pro = read_xyz(proali)

    rmsd_min, deg_rad = minimize_rmsd_rotation(rea, pro, natom)

    print(
        f"\nMinimum RMSD: {rmsd_min:12.6f}   Rotation angle is {180 * deg_rad / math.pi:12.6f} degrees\n"
    )

    pro_new = np.dot(rotation_z(deg_rad), pro)

    write_xyz(
        prorot,
        pro_symbols,
        pro_new,
        "Product with D and A atoms along the Z-axis and aligned with reactant: "
        f"DA distance {pda_distance:10.6f} Å",
    )

    # average the aligned reactant and product structures
    natom, rea_symbols, rea = read_xyz(reanew)
    natom, pro_symbols, pro = read_xyz(prorot)

    write_xyz(
        options.output_name,
        rea_symbols,
        (rea + pro) / 2,
        f"Average reactant/product configuration: DA distance {da_distance:10.6f} Å",
    )


if __name__ == "__main__":
    main()
