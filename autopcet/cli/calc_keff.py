"""Effective proton donor-acceptor force constant from a Gaussian frequency job.

Projecting the normal modes onto the donor-acceptor axis gives the effective
force constant, reduced mass, and frequency of the donor-acceptor mode. The
force constant is what ``P(R)`` needs -- see
:func:`autopcet.rates.donor_acceptor_distribution`.

The frequency job must be run with ``#P`` and ``freq=HPmodes``: the parser reads
the high-precision normal modes those settings print, and will not work without
them. Donor and acceptor indices are 0-based.
"""

import argparse

from ..gaussian_io import effective_da_mode, read_frequencies
from ..structure import write_xyz
from ._io import symbols_from_atomic_numbers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autopcet-keff", description=__doc__)
    parser.add_argument(
        "--log",
        dest="log_path",
        required=True,
        help="log file of a Gaussian frequency calculation run with freq=HPmodes",
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
        "--write-geometry",
        dest="geometry_path",
        default=None,
        help="also write the standard-orientation geometry to this xyz file",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    options = build_parser().parse_args(argv)

    frequencies = read_frequencies(options.log_path)
    mode = effective_da_mode(
        frequencies.positions,
        options.donor,
        options.acceptor,
        frequencies.reduced_masses,
        frequencies.force_constants,
        frequencies.normal_modes,
    )

    if options.geometry_path is not None:
        # Gaussian may have reoriented the molecule, so this is the geometry the
        # normal modes above actually refer to.
        write_xyz(
            options.geometry_path,
            symbols_from_atomic_numbers([int(z) for z in frequencies.atomic_numbers]),
            frequencies.positions,
            "Standard orientation geometry from " + options.log_path,
        )

    print(f"Effective force constant in a.u.: {mode.force_constant:.4f}")
    print(f"Effective reduced mass in amu: {mode.reduced_mass:.3f}")
    print(f"Effective frequency in cm-1: {mode.frequency: .2f}")


if __name__ == "__main__":
    main()
