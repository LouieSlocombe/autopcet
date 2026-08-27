"""Gaussian constrained optimizations scanning the proton donor-acceptor distance.

Takes the fully optimized reactant and product structures and writes, for each
donor-acceptor distance R in the requested range, a pair of Gaussian inputs that
reoptimize the two states with R frozen. The atoms must appear in the same order
in both structures. Donor and acceptor indices are 0-based.

The default route line is a B3LYP/6-31+G(d,p) ModRedundant optimization; point
``--template`` at a file to use your own level of theory, dispersion correction,
or implicit solvent. The template is formatted with ``{state}``, ``{charge}``,
and ``{multiplicity}``.

This assumes the reactant and product diabatic electronic states differ only in
the charge of the whole system, as they do in homogeneous electrochemical PCET
or in photoexcited PCET with an external photoreceptor.
"""

import argparse
from pathlib import Path

import numpy as np

from ..gaussian_io import (
    CONSTRAINT_TEMPLATE,
    DEFAULT_OPT_TEMPLATE,
    write_gaussian_input,
)
from ._io import load_template, read_atoms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autopcet-scan-da", description=__doc__)
    parser.add_argument(
        "-r",
        "--reactant",
        dest="reactant_path",
        required=True,
        help="structure file with the optimized reactant",
    )
    parser.add_argument(
        "-p",
        "--product",
        dest="product_path",
        required=True,
        help="structure file with the optimized product",
    )
    parser.add_argument(
        "-D",
        "--donor",
        type=int,
        dest="donor",
        required=True,
        help="atomic index (starting from 0) of the proton donor",
    )
    parser.add_argument(
        "-A",
        "--acceptor",
        type=int,
        dest="acceptor",
        required=True,
        help="atomic index (starting from 0) of the proton acceptor",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=2.4,
        help="shortest donor-acceptor distance in angstrom (default: %(default)s)",
    )
    parser.add_argument(
        "--stop",
        type=float,
        default=2.8,
        help="longest donor-acceptor distance in angstrom (default: %(default)s)",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=9,
        help="number of distances to scan (default: %(default)s)",
    )
    parser.add_argument(
        "--reactant-charge",
        type=int,
        default=0,
        help="charge of the reactant state (default: %(default)s)",
    )
    parser.add_argument(
        "--reactant-multiplicity",
        type=int,
        default=1,
        help="spin multiplicity of the reactant state (default: %(default)s)",
    )
    parser.add_argument(
        "--product-charge",
        type=int,
        default=1,
        help="charge of the product state (default: %(default)s)",
    )
    parser.add_argument(
        "--product-multiplicity",
        type=int,
        default=2,
        help="spin multiplicity of the product state (default: %(default)s)",
    )
    parser.add_argument(
        "--template",
        dest="template_path",
        default=None,
        help="file holding the Gaussian header, in place of the built-in default",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        default=".",
        help="directory to write the R<distance>A trees into (default: %(default)s)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    options = build_parser().parse_args(argv)

    header_template = load_template(options.template_path, DEFAULT_OPT_TEMPLATE)
    output_dir = Path(options.output_dir)

    need = "Setting the donor-acceptor distance"
    reactant = read_atoms(options.reactant_path, need)
    product = read_atoms(options.product_path, need)

    # `fix` names which of the two atoms stays put while the distance is set: the
    # donor for the reactant, the acceptor for the product. Only the starting
    # guess for the constrained optimization depends on it.
    states = (
        (
            "reac_opt",
            "reactant",
            options.reactant_charge,
            options.reactant_multiplicity,
            reactant,
            0,
        ),
        (
            "prod_opt",
            "product",
            options.product_charge,
            options.product_multiplicity,
            product,
            1,
        ),
    )

    for distance in np.linspace(options.start, options.stop, options.points):
        directory = output_dir / f"R{distance:.2f}A"

        for name, state, charge, multiplicity, structure, fix in states:
            scaled = structure.copy()
            scaled.set_distance(options.donor, options.acceptor, distance, fix=fix)

            (directory / name).mkdir(parents=True, exist_ok=True)
            write_gaussian_input(
                directory / name / f"{name}.gjf",
                header_template.format(
                    state=state, charge=charge, multiplicity=multiplicity
                ),
                list(scaled.symbols),
                scaled.get_positions(),
                # Gaussian indices start from 1
                CONSTRAINT_TEMPLATE.format(
                    donor=options.donor + 1,
                    acceptor=options.acceptor + 1,
                    distance=distance,
                ),
            )


if __name__ == "__main__":
    main()
