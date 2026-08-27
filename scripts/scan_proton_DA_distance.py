"""Generate Gaussian constrained-optimization inputs that scan the proton
donor-acceptor distance for the reactant and product states.

Edit the constants below for your system, then run the script from the
directory that holds the two xyz files.
"""

import os

import numpy as np
from ase.io import read

# xyz files for the reactant and the product
REACTANT_XYZ = "reac.xyz"
PRODUCT_XYZ = "prod.xyz"

# atomic indices (starting from 0) of the proton donor and acceptor
DONOR_INDEX = 0
ACCEPTOR_INDEX = 2

# charge and multiplicity of the reactant and product; here the reactant is a
# neutral singlet and the product a doublet with charge +1
REACTANT_CHARGE = 0
REACTANT_MULTIPLICITY = 1
PRODUCT_CHARGE = 1
PRODUCT_MULTIPLICITY = 2

# range of proton donor-acceptor distances to scan, in angstrom
DISTANCES = np.linspace(2.4, 2.8, 9)

# change the level of theory or add empirical dispersion/implicit solvent as needed
HEADER_TEMPLATE = """%chk={state}.chk
%nprocshared=24
%mem=80GB
# B3LYP/6-31+g(d,p) opt=(ModRedundant)

{state}

{charge} {multiplicity}
"""

CONSTRAINT_TEMPLATE = """{donor}   {acceptor}   ={distance:.2f}   B
{donor}   {acceptor}   F
"""


def main():
    # read the fully optimized reactant and product structures; the atoms must
    # appear in the same order in both files
    reactant = read(REACTANT_XYZ)
    product = read(PRODUCT_XYZ)

    for distance in DISTANCES:
        directory = f"R{distance:.2f}A"
        os.makedirs(f"{directory}/reac_opt/", exist_ok=True)
        os.makedirs(f"{directory}/prod_opt/", exist_ok=True)

        # copy the structures and set the donor-acceptor distance on each copy
        scaled_reactant = reactant.copy()
        scaled_product = product.copy()
        scaled_reactant.set_distance(DONOR_INDEX, ACCEPTOR_INDEX, distance, fix=0)
        scaled_product.set_distance(DONOR_INDEX, ACCEPTOR_INDEX, distance, fix=1)

        symbols = scaled_reactant.symbols
        states = (
            (
                f"{directory}/reac_opt/reac_opt.gjf",
                "reactant",
                REACTANT_CHARGE,
                REACTANT_MULTIPLICITY,
                scaled_reactant.get_positions(),
            ),
            (
                f"{directory}/prod_opt/prod_opt.gjf",
                "product",
                PRODUCT_CHARGE,
                PRODUCT_MULTIPLICITY,
                scaled_product.get_positions(),
            ),
        )

        for path, state, charge, multiplicity, positions in states:
            with open(path, "w") as gaussian_input:
                gaussian_input.write(
                    HEADER_TEMPLATE.format(
                        state=state, charge=charge, multiplicity=multiplicity
                    )
                )
                for symbol, (x, y, z) in zip(symbols, positions, strict=True):
                    gaussian_input.write(
                        f"{symbol:2s}       {x: 3.6f}    {y: 3.6f}    {z: 3.6f}\n"
                    )
                gaussian_input.write("\n")
                # Gaussian indices start from 1
                gaussian_input.write(
                    CONSTRAINT_TEMPLATE.format(
                        donor=DONOR_INDEX + 1,
                        acceptor=ACCEPTOR_INDEX + 1,
                        distance=distance,
                    )
                )
                gaussian_input.write("\n")


if __name__ == "__main__":
    main()
