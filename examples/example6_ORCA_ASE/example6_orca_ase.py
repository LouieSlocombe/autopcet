"""The full PCET workflow run in-process through ASE, here with ORCA.

Replicates what the four ``autopcet-*`` command-line helpers prepare Gaussian
inputs for -- the donor-acceptor distance scan, the alignment and averaging,
the proton endpoint optimizations, the proton potentials, and the effective
donor-acceptor mode -- but drives a live calculator through
:mod:`autopcet.ase_io` instead of writing input files. Swapping ORCA for any
other ASE calculator only changes the two calculator constructors.

Unlike the other examples this needs a working ORCA installation and your own
fully optimized reactant and product structures, with the atoms in the same
order in both:

    python example6_orca_ase.py --orca /opt/orca/orca \\
        -r reac.xyz -p prod.xyz -D 0 -A 2 -H 1

Point ``--orca`` at the full path of the ORCA binary: on desktop Linux a bare
``orca`` on the PATH is usually the GNOME screen reader, not ORCA. The
reaction free energy and reorganization energy are inputs of the golden-rule
rate expression, not something this script computes; supply your system's
values through ``--delta-g`` and ``--reorganization``.
"""

import argparse

import numpy as np
from ase.calculators.orca import ORCA, OrcaProfile
from ase.io import read

from autopcet import (
    MASS_DEUTERON,
    MASS_PROTON,
    PCET,
    ROOM_TEMPERATURE,
    align_da_to_z,
    average_structures,
    centre_da_midpoint,
    donor_acceptor_distribution,
    effective_mode_from_vibrations,
    minimize_rmsd_rotation,
    optimize_proton,
    rotation_about_z,
    run_da_scan,
    run_proton_scan,
    run_vibrations,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--orca", required=True, help="full path to the ORCA binary")
parser.add_argument("-r", "--reactant", default="reac.xyz")
parser.add_argument("-p", "--product", default="prod.xyz")
parser.add_argument("-D", "--donor", type=int, required=True, help="0-based index")
parser.add_argument("-A", "--acceptor", type=int, required=True, help="0-based index")
parser.add_argument("-H", "--proton", type=int, required=True, help="0-based index")
parser.add_argument("--reactant-charge", type=int, default=0)
parser.add_argument("--reactant-multiplicity", type=int, default=1)
parser.add_argument("--product-charge", type=int, default=1)
parser.add_argument("--product-multiplicity", type=int, default=2)
parser.add_argument("--start", type=float, default=2.4, help="shortest R in angstrom")
parser.add_argument("--stop", type=float, default=2.8, help="longest R in angstrom")
parser.add_argument("--points", type=int, default=9, help="distances to scan")
parser.add_argument("--scan-points", type=int, default=20, help="proton grid points")
parser.add_argument("--method", default="B3LYP def2-SVP TightSCF")
parser.add_argument("--nprocs", type=int, default=1)
parser.add_argument("--delta-g", type=float, default=0.0, help="in eV")
parser.add_argument("--reorganization", type=float, default=1.0, help="in eV")
args = parser.parse_args()

# one calculator per diabatic state: the state is selected by its charge and
# multiplicity, exactly as --state/--charge/--multiplicity do for the helpers
profile = OrcaProfile(command=args.orca)
blocks = f"%pal nprocs {args.nprocs} end"
reactant_calc = ORCA(
    profile=profile,
    directory="reac",
    charge=args.reactant_charge,
    mult=args.reactant_multiplicity,
    orcasimpleinput=args.method,
    orcablocks=blocks,
)
product_calc = ORCA(
    profile=profile,
    directory="prod",
    charge=args.product_charge,
    mult=args.product_multiplicity,
    orcasimpleinput=args.method,
    orcablocks=blocks,
)

# 1. reoptimize both states with the donor-acceptor distance frozen at each R
#    (what autopcet-scan-da writes Gaussian inputs for)
distances = np.linspace(args.start, args.stop, args.points)
reactant_scan = run_da_scan(
    read(args.reactant), args.donor, args.acceptor, reactant_calc, distances, fix=0
)
product_scan = run_da_scan(
    read(args.product), args.donor, args.acceptor, product_calc, distances, fix=1
)

equilibrium = int(np.argmin(reactant_scan.energies))
distance = float(reactant_scan.distances[equilibrium])
print(f"Reactant equilibrium donor-acceptor distance: {distance:.2f} A")

# the rest of this script builds the rate at that single R; averaging over R
# repeats it per distance and weights with P(R), as example 2 does
reactant_structure = reactant_scan.structures[equilibrium]
product_structure = product_scan.structures[equilibrium]

# 2. bring both states into the common frame and average them
#    (what autopcet-align-average does)
reactant_positions, _ = align_da_to_z(
    reactant_structure.get_positions(), args.donor, args.acceptor
)
reactant_positions = centre_da_midpoint(reactant_positions, args.acceptor)
product_positions, _ = align_da_to_z(
    product_structure.get_positions(), args.donor, args.acceptor
)
product_positions = centre_da_midpoint(product_positions, args.acceptor)

_, angle = minimize_rmsd_rotation(reactant_positions, product_positions)
product_positions = product_positions @ rotation_about_z(angle).T

averaged = reactant_structure.copy()
averaged.set_positions(average_structures(reactant_positions, product_positions))

# 3. optimize the proton onto the donor and onto the acceptor with the rest of
#    the frame frozen (the step the command-line workflow leaves to the user)
reactant_endpoint = optimize_proton(averaged, args.proton, reactant_calc)
product_endpoint = optimize_proton(averaged, args.proton, product_calc)

# 4. diabatic proton potentials along the transfer axis
#    (what autopcet-scan-proton writes Gaussian single points for)
reactant_potential = run_proton_scan(
    reactant_endpoint, product_endpoint, args.proton, reactant_calc, args.scan_points
)
product_potential = run_proton_scan(
    reactant_endpoint, product_endpoint, args.proton, product_calc, args.scan_points
)

# 5. effective donor-acceptor mode from finite difference vibrations at the
#    reactant minimum (what autopcet-keff reads from a Gaussian HPmodes log)
mode = effective_mode_from_vibrations(
    run_vibrations(reactant_structure, reactant_calc, directory="vib"),
    args.donor,
    args.acceptor,
)
print(f"Effective force constant in a.u.: {mode.force_constant:.4f}")
print(f"Effective frequency in cm-1: {mode.frequency:.2f}")

distribution = donor_acceptor_distribution(
    reactant_scan.distances, distance, mode.force_constant
)
print(f"Unnormalized P(R) on the scanned grid: {np.array2string(distribution)}")

# 6. golden-rule rate constants and the H/D kinetic isotope effect at this R
system = PCET(
    (reactant_potential.offsets, reactant_potential.relative_energies),
    (product_potential.offsets, product_potential.relative_energies),
    args.delta_g,
    args.reorganization,
)
rate_h = system.calculate(MASS_PROTON, ROOM_TEMPERATURE)
rate_d = system.calculate(MASS_DEUTERON, ROOM_TEMPERATURE)

print(f"k(H) = {rate_h:.3e} s^-1")
print(f"k(D) = {rate_d:.3e} s^-1")
print(f"KIE = {rate_h / rate_d:.2f}")
