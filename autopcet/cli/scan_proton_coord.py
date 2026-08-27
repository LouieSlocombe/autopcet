"""Gaussian single points scanning the proton along its transfer axis.

Takes the two averaged structures with the proton optimized on the donor and on
the acceptor, builds the axis through those two proton positions, and writes a
Gaussian single point for each point along it. Run it once for the reactant
state and once for the product state, changing ``--state``, ``--charge``, and
``--multiplicity``.

To build a proton potential the proton has to come very close to both the donor
and the acceptor, so the scan spans 1.7 times the distance between its two
equilibrium positions, and never less than 1 angstrom in total.

The default route line sets ``Nosymm``, which is essential: without it Gaussian
reorients the molecule and the proton no longer sits where the scan put it.
Point ``--template`` at a file to use your own level of theory; it is formatted
with ``{state}``, ``{charge}``, and ``{multiplicity}``.
"""

import argparse
from pathlib import Path

import numpy as np

from ..gaussian_io import DEFAULT_SP_TEMPLATE, write_gaussian_input
from ._io import load_template, read_structure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autopcet-scan-proton", description=__doc__)
    parser.add_argument(
        "-r",
        "--reactant",
        dest="reactant_path",
        required=True,
        help="averaged structure with the proton optimized on the donor",
    )
    parser.add_argument(
        "-p",
        "--product",
        dest="product_path",
        required=True,
        help="averaged structure with the proton optimized on the acceptor",
    )
    parser.add_argument(
        "-H",
        "--proton",
        type=int,
        dest="proton",
        required=True,
        help="atomic index (starting from 0) of the transferring proton",
    )
    parser.add_argument(
        "--state",
        default="reactant",
        help="name of the state being scanned (default: %(default)s)",
    )
    parser.add_argument(
        "--charge",
        type=int,
        default=0,
        help="charge of the state being scanned (default: %(default)s)",
    )
    parser.add_argument(
        "--multiplicity",
        type=int,
        default=1,
        help="spin multiplicity of the state being scanned (default: %(default)s)",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=20,
        help="number of grid points along the proton axis (default: %(default)s)",
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
        help="directory to write the numbered grid points into (default: %(default)s)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    options = build_parser().parse_args(argv)

    header = load_template(options.template_path, DEFAULT_SP_TEMPLATE).format(
        state=options.state,
        charge=options.charge,
        multiplicity=options.multiplicity,
    )
    output_dir = Path(options.output_dir)

    symbols, reactant_positions = read_structure(options.reactant_path)
    _, product_positions = read_structure(options.product_path)

    # the proton axis passes through the two optimized proton positions
    proton_shift = (
        product_positions[options.proton] - reactant_positions[options.proton]
    )
    transfer_distance = float(np.linalg.norm(proton_shift))
    axis = proton_shift / transfer_distance
    midpoint = 0.5 * (
        reactant_positions[options.proton] + product_positions[options.proton]
    )

    half_width = max(0.5, 1.7 * transfer_distance / 2)
    offsets = np.linspace(-half_width, half_width, options.points)

    for i, offset in enumerate(offsets):
        positions = reactant_positions.copy()
        positions[options.proton] = offset * axis + midpoint

        directory = output_dir / f"{i:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        write_gaussian_input(
            directory / f"{options.state}_sp.gjf", header, symbols, positions
        )


if __name__ == "__main__":
    main()
