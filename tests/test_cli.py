"""The console scripts that prepare the inputs of a PCET calculation.

Each test drives ``main`` directly with an explicit argument list, so the tests
exercise the same code path the installed ``autopcet-*`` commands take.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from autopcet.cli import align_average, calc_keff, scan_da_distance, scan_proton_coord
from autopcet.structure import read_xyz

DATA = Path(__file__).resolve().parent / "data"


def read_gaussian_geometry(path: Path) -> tuple[list[str], np.ndarray]:
    """Pull the symbols and coordinates back out of a Gaussian input file."""
    symbols = []
    positions = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) == 4 and fields[0].isalpha():
            symbols.append(fields[0])
            positions.append([float(value) for value in fields[1:]])
    return symbols, np.array(positions)


@pytest.fixture
def structures(tmp_path: Path) -> Path:
    """A scratch directory holding the reactant and product fixtures."""
    for name in ("reactant.xyz", "product.xyz"):
        shutil.copy(DATA / name, tmp_path / name)
    return tmp_path


def assert_matches_reference(written: Path, reference: Path) -> None:
    """Compare two xyz files on their symbols and coordinates."""
    written_symbols, written_positions = read_xyz(written)
    reference_symbols, reference_positions = read_xyz(reference)

    assert written_symbols == reference_symbols
    assert written_positions == pytest.approx(reference_positions, abs=1e-6)


def test_align_average_reproduces_its_reference(structures: Path) -> None:
    """The averaged structure matches the one the helper has always produced."""
    align_average.main(
        [
            "-r",
            str(structures / "reactant.xyz"),
            "-p",
            str(structures / "product.xyz"),
            "--r-donor",
            "1",
            "--r-acceptor",
            "3",
            "--p-donor",
            "1",
            "--p-acceptor",
            "3",
            "-o",
            str(structures / "AVERAGE_STRUCTURE.xyz"),
        ]
    )

    assert_matches_reference(
        structures / "AVERAGE_STRUCTURE.xyz", DATA / "reference_average.xyz"
    )
    assert_matches_reference(
        structures / "reactant_DA_along_Z.xyz", DATA / "reference_reactant_aligned.xyz"
    )
    assert_matches_reference(
        structures / "product_DA_along_Z_aligned.xyz",
        DATA / "reference_product_aligned.xyz",
    )


def test_align_average_centres_the_product_on_the_reactant_acceptor(
    structures: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A long-standing quirk, pinned so a refactor cannot quietly change it.

    When the acceptor has a different index in the two files, the product is
    centred using the *reactant's* acceptor index. Averaged structures and the
    proton potentials derived from them depend on this, so it is preserved
    rather than fixed.
    """
    align_average.main(
        [
            "-r",
            str(structures / "reactant.xyz"),
            "-p",
            str(structures / "product.xyz"),
            "--r-donor",
            "1",
            "--r-acceptor",
            "3",
            "--p-donor",
            "1",
            "--p-acceptor",
            "4",
            "-o",
            str(structures / "AVERAGE_STRUCTURE.xyz"),
        ]
    )

    assert_matches_reference(
        structures / "AVERAGE_STRUCTURE.xyz",
        DATA / "reference_average_mismatched_acceptor.xyz",
    )
    # the two structures now have different donor-acceptor distances, which the
    # helper warns about
    assert "are different" in capsys.readouterr().out


def test_calc_keff_reports_the_donor_acceptor_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The effective force constant, reduced mass, and frequency are printed."""
    calc_keff.main(["--log", str(DATA / "freq_hpmodes.log"), "-D", "0", "-A", "1"])

    output = capsys.readouterr().out
    assert "Effective force constant in a.u.: 0.0800" in output
    assert "Effective reduced mass in amu: 12.447" in output
    assert "Effective frequency in cm-1:  412.13" in output


def test_calc_keff_can_write_the_standard_orientation(tmp_path: Path) -> None:
    """Gaussian may reorient the molecule, so the geometry is worth keeping."""
    pytest.importorskip("ase")
    geometry = tmp_path / "optimized_geometry.xyz"

    calc_keff.main(
        [
            "--log",
            str(DATA / "freq_hpmodes.log"),
            "-D",
            "0",
            "-A",
            "1",
            "--write-geometry",
            str(geometry),
        ]
    )

    symbols, positions = read_xyz(geometry)
    assert symbols == ["O", "O", "H"]
    assert positions[1] == pytest.approx([2.7, 0.0, 0.0])


def test_scan_da_distance_writes_a_pair_of_inputs_per_distance(
    structures: Path,
) -> None:
    """Each scanned R gets a constrained optimization for both states."""
    pytest.importorskip("ase")

    scan_da_distance.main(
        [
            "-r",
            str(structures / "reactant.xyz"),
            "-p",
            str(structures / "product.xyz"),
            "-D",
            "0",
            "-A",
            "2",
            "--start",
            "2.4",
            "--stop",
            "2.6",
            "--points",
            "3",
            "-o",
            str(structures / "scan"),
        ]
    )

    scan = structures / "scan"
    assert sorted(p.name for p in scan.iterdir()) == ["R2.40A", "R2.50A", "R2.60A"]

    text = (scan / "R2.50A" / "reac_opt" / "reac_opt.gjf").read_text()
    assert "opt=(ModRedundant)" in text
    # Gaussian indices are 1-based, unlike the -D/-A arguments
    assert "1   3   =2.50   B" in text

    _, positions = read_xyz(structures / "reactant.xyz")
    lines = [line.split() for line in text.splitlines() if line.startswith("N ")]
    scaled = np.array([float(value) for value in lines[0][1:]])
    assert np.linalg.norm(scaled - positions[0]) == pytest.approx(2.5)


def test_scan_da_distance_accepts_a_custom_template(structures: Path) -> None:
    """The route line and resources can be replaced without editing the package."""
    pytest.importorskip("ase")
    template = structures / "header.txt"
    template.write_text(
        "%mem=8GB\n# PBE1PBE/def2SVP opt=(ModRedundant)\n\n{state}\n\n{charge} {multiplicity}\n"
    )

    scan_da_distance.main(
        [
            "-r",
            str(structures / "reactant.xyz"),
            "-p",
            str(structures / "product.xyz"),
            "-D",
            "0",
            "-A",
            "2",
            "--start",
            "2.4",
            "--stop",
            "2.4",
            "--points",
            "1",
            "--product-charge",
            "2",
            "--product-multiplicity",
            "3",
            "--template",
            str(template),
            "-o",
            str(structures / "scan"),
        ]
    )

    text = (structures / "scan" / "R2.40A" / "prod_opt" / "prod_opt.gjf").read_text()
    assert "# PBE1PBE/def2SVP opt=(ModRedundant)" in text
    assert "2 3" in text


def test_scan_proton_coord_walks_the_proton_along_its_axis(structures: Path) -> None:
    """Every grid point moves only the proton, along the transfer axis."""
    scan_proton_coord.main(
        [
            "-r",
            str(structures / "reactant.xyz"),
            "-p",
            str(structures / "product.xyz"),
            "-H",
            "1",
            "--points",
            "5",
            "-o",
            str(structures / "scan"),
        ]
    )

    scan = structures / "scan"
    assert sorted(p.name for p in scan.iterdir()) == ["00", "01", "02", "03", "04"]

    _, reactant = read_xyz(structures / "reactant.xyz")
    protons = []
    for i in range(5):
        symbols, positions = read_gaussian_geometry(
            scan / f"{i:02d}" / "reactant_sp.gjf"
        )
        assert symbols == ["O", "H", "N", "C", "C", "H"]
        # every atom but the proton is frozen at its reactant position
        assert np.delete(positions, 1, axis=0) == pytest.approx(
            np.delete(reactant, 1, axis=0)
        )
        protons.append(positions[1])

    # the proton positions are evenly spaced and collinear, to the eight
    # decimals the input files record
    steps = np.diff(np.array(protons), axis=0)
    assert steps == pytest.approx(np.broadcast_to(steps[0], steps.shape), abs=1e-7)


def test_scan_proton_coord_uses_the_requested_state(structures: Path) -> None:
    """The state names the files and appears in the route and title blocks."""
    scan_proton_coord.main(
        [
            "-r",
            str(structures / "reactant.xyz"),
            "-p",
            str(structures / "product.xyz"),
            "-H",
            "1",
            "--state",
            "product",
            "--charge",
            "1",
            "--multiplicity",
            "2",
            "--points",
            "2",
            "-o",
            str(structures / "scan"),
        ]
    )

    text = (structures / "scan" / "00" / "product_sp.gjf").read_text()
    assert "%chk=product.chk" in text
    assert "product proton potential" in text
    assert "1 2" in text


@pytest.mark.parametrize(
    ("module", "arguments", "match"),
    [
        (
            scan_da_distance,
            ["-r", "reactant.mol", "-p", "product.mol", "-D", "0", "-A", "2"],
            "Setting the donor-acceptor distance",
        ),
        (
            scan_proton_coord,
            ["-r", "reactant.mol", "-p", "product.mol", "-H", "1"],
            "Reading '.mol' files",
        ),
    ],
)
def test_helpers_needing_ase_say_so_when_it_is_missing(
    structures: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    arguments: list[str],
    match: str,
) -> None:
    """ASE only ships in the `ase` extra, so its absence must be explained."""
    monkeypatch.setitem(sys.modules, "ase.io", None)

    paths = [
        str(structures / argument) if "." in argument else argument
        for argument in arguments
    ]
    with pytest.raises(ImportError, match=match) as error:
        module.main(paths)  # type: ignore[attr-defined]

    assert "autopcet[ase]" in str(error.value)


def test_writing_the_geometry_needs_ase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Element symbols come from ASE's periodic table."""
    monkeypatch.setitem(sys.modules, "ase.data", None)

    with pytest.raises(ImportError, match="Writing the optimized geometry"):
        calc_keff.main(
            [
                "--log",
                str(DATA / "freq_hpmodes.log"),
                "-D",
                "0",
                "-A",
                "1",
                "--write-geometry",
                str(tmp_path / "geometry.xyz"),
            ]
        )


def test_helpers_read_any_format_ase_understands(structures: Path) -> None:
    """Structures only have to be xyz if ASE is not installed."""
    ase_io = pytest.importorskip("ase.io")

    symbols, positions = read_xyz(structures / "reactant.xyz")
    for name in ("reactant", "product"):
        ase_io.write(
            structures / f"{name}.traj", ase_io.read(structures / f"{name}.xyz")
        )

    scan_proton_coord.main(
        [
            "-r",
            str(structures / "reactant.traj"),
            "-p",
            str(structures / "product.traj"),
            "-H",
            "1",
            "--points",
            "2",
            "-o",
            str(structures / "scan"),
        ]
    )

    written, written_positions = read_gaussian_geometry(
        structures / "scan" / "00" / "reactant_sp.gjf"
    )
    assert written == symbols
    assert written_positions[2] == pytest.approx(positions[2])
