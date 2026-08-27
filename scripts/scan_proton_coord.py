"""Generate Gaussian single-point inputs that scan the proton along the axis
connecting its equilibrium positions in the reactant and product structures.

Edit the constants below for your system, then run the script from the
directory that holds the two xyz files.
"""

import os

import numpy as np
from ase.io import read

# xyz files with the proton optimized in the reactant and product states
REACTANT_XYZ = "averaged_optH_reac.xyz"
PRODUCT_XYZ = "averaged_optH_prod.xyz"

# atomic index (starting from 0) of the proton in these xyz files
PROTON_INDEX = 1

# the state (reactant or product), charge, and multiplicity
STATE = "reactant"
CHARGE = 0
MULTIPLICITY = 1

# number of grid points
N_POINTS = 20

# change the level of theory or add empirical dispersion/implicit solvent as needed
HEADER_TEMPLATE = """%chk={state}.chk
%nprocshared=24
%mem=60GB
# Nosymm B3LYP/6-31+g(d,p)

{state} proton potential

{charge} {multiplicity}
"""


def main():
    reactant = read(REACTANT_XYZ)
    product = read(PRODUCT_XYZ)

    reactant_positions = reactant.get_positions()
    product_positions = product.get_positions()

    # the proton axis passes through the two optimized proton positions
    proton_shift = product_positions[PROTON_INDEX] - reactant_positions[PROTON_INDEX]
    transfer_distance = np.linalg.norm(proton_shift)
    axis = proton_shift / transfer_distance
    midpoint = 0.5 * (
        reactant_positions[PROTON_INDEX] + product_positions[PROTON_INDEX]
    )

    # To generate the proton potential the proton has to come very close to the
    # donor and the acceptor, so the default scan range is 1.7 times the distance
    # between its equilibrium positions. That is not enough for very small R, so
    # the range is never shorter than 1 A in total.
    offsets = np.linspace(
        np.min([-0.5, -1.7 * transfer_distance / 2]),
        np.max([0.5, 1.7 * transfer_distance / 2]),
        N_POINTS,
    )

    for i, offset in enumerate(offsets):
        os.makedirs(f"{i:02d}", exist_ok=True)
        with open(f"{i:02d}/{STATE}_sp.gjf", "w") as gaussian_input:
            gaussian_input.write(
                HEADER_TEMPLATE.format(
                    state=STATE, charge=CHARGE, multiplicity=MULTIPLICITY
                )
            )

            structure = reactant.copy()
            positions = structure.get_positions()
            positions[PROTON_INDEX] = offset * axis + midpoint
            structure.set_positions(positions)

            for atom in structure:
                x, y, z = positions[atom.index]
                gaussian_input.write(
                    f"{atom.symbol}      {x:.8f}   {y:.8f}   {z:.8f}\n"
                )
            gaussian_input.write("\n")


if __name__ == "__main__":
    main()
