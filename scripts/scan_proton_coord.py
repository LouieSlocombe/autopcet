"""Generate Gaussian single-point inputs that scan the proton along the axis
connecting its equilibrium positions in the reactant and product structures.

Edit the constants below for your system, then run the script from the
directory that holds the two xyz files.
"""

import os

import numpy as np
from ase.io import read

# xyz files with the proton optimized in the reactant and product states
reac_xyz = "averaged_optH_reac.xyz"
prod_xyz = "averaged_optH_prod.xyz"

# atomic index (start with 0) of the proton in these xyz files
proton_index = 1

# define the state (reactant or product), charge and multiplicity
state = "reactant"
charge = 0
multiplicity = 1

# number of grid points
N = 20

# change the level of theory or add empirical dispersion/implicit solvent as needed,
heading = """%chk={state}.chk
%nprocshared=24
%mem=60GB
# Nosymm B3LYP/6-31+g(d,p)

{state} proton potential

{charge} {multiplicity}
"""


def main():
    struct1 = read(reac_xyz)
    struct2 = read(prod_xyz)

    pos1 = struct1.get_positions()
    pos2 = struct2.get_positions()

    # calculating the proton axis, which passes through the two optimized proton positions in the xyz files
    rPT = pos2[proton_index] - pos1[proton_index]
    dPT = np.linalg.norm(rPT)
    nPT = rPT / dPT
    center = 0.5 * (pos1[proton_index] + pos2[proton_index])

    # To effectively generate the proton potential, we need to move the proton very close to the donor or the acceptor
    # the default scan range is 1.7 times the distance between the equilibrium proton positions on its donor and acceptor
    # However, for very small R values, this is not sufficient. The scan range for this case is set to be 1 A
    dp_center = np.linspace(
        np.min([-0.5, -1.7 * dPT / 2]), np.max([0.5, 1.7 * dPT / 2]), N
    )

    for i in range(N):
        os.makedirs(f"{i:02d}", exist_ok=True)
        with open(f"{i:02d}/{state}_sp.gjf", "w") as outfp:
            outfp.write(
                heading.format(state=state, charge=charge, multiplicity=multiplicity)
            )

            rp = dp_center[i] * nPT + center

            struct_tmp = struct1.copy()
            pos_tmp = struct_tmp.get_positions()
            pos_tmp[proton_index] = rp
            struct_tmp.set_positions(pos_tmp)

            for atom in struct_tmp:
                x, y, z = pos_tmp[atom.index]
                outfp.write(f"{atom.symbol}      {x:.8f}   {y:.8f}   {z:.8f}\n")
            outfp.write("\n")


if __name__ == "__main__":
    main()
