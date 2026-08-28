"""Smoke tests that the bundled examples still reproduce their reference output.

Only the two cheap examples run here. Examples 2-4 sweep thousands of rate
evaluations each (example 4 alone is of order 10^5) and are far too slow for
CI, so they stay out.

The committed reference outputs were generated with older NumPy/SciPy releases
and no longer reproduce to the last printed digit, so the comparison is on the
labelled quantities at a relative tolerance rather than on the raw text.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import example1_data
import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

# Reported values drift in the third significant digit across library versions;
# a real regression moves them by orders of magnitude.
RTOL = 0.05

# "k_tot(H) = 2.34e+08", "tau_p = 1.897e-16", "V_ad = 1.68e-03 eV = 3.88e-02 ..."
_LABELLED_VALUE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*(?:\([A-Za-z]\))?)\s*=\s*"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)


def labelled_values(text: str) -> list[tuple[str, float]]:
    """Extract every ``label = number`` pair, in the order they are printed."""
    return [(m[1], float(m[2])) for m in _LABELLED_VALUE.finditer(text)]


def run_example(tmp_path: Path, directory: str, script: str, *args: str) -> str:
    """Run an example in a scratch copy of its directory and return its stdout."""
    workdir = tmp_path / directory
    shutil.copytree(EXAMPLES / directory, workdir)

    completed = subprocess.run(
        [sys.executable, script, *args],
        cwd=workdir,
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def assert_matches_reference(output: str, reference: Path) -> None:
    """Compare the labelled quantities of an example run against its reference."""
    actual = labelled_values(output)
    expected = labelled_values(reference.read_text())

    assert expected, f"no labelled values found in {reference}"
    assert [label for label, _ in actual] == [label for label, _ in expected]

    for (label, got), (_, want) in zip(actual, expected, strict=True):
        assert got == pytest.approx(want, rel=RTOL), label


def test_example1_reproduces_its_reference_output(tmp_path: Path) -> None:
    """Example 1 reproduces its rate constants and H/D KIE."""
    directory = "example1_basic_usage"
    output = run_example(tmp_path, directory, "example1_basic_usage.py")

    assert_matches_reference(output, EXAMPLES / directory / "reference_output.txt")


@pytest.mark.parametrize("config", ["gas", "env"])
def test_example5_reproduces_its_reference_output(tmp_path: Path, config: str) -> None:
    """Example 5 reproduces its couplings in both the gas and protein cases."""
    directory = "example5_RNR_nonadiabaticity"
    output = run_example(
        tmp_path, directory, "example5_nonadiabaticity_Y356-Y731.py", "--config", config
    )

    assert_matches_reference(
        output, EXAMPLES / directory / f"reference_output_{config}.txt"
    )


def test_example5_configurations_write_separate_figures(tmp_path: Path) -> None:
    """The two configurations must not overwrite each other's figures."""
    directory = "example5_RNR_nonadiabaticity"
    workdir = tmp_path / directory
    shutil.copytree(EXAMPLES / directory, workdir)
    script = "example5_nonadiabaticity_Y356-Y731.py"

    for config in ("gas", "env"):
        completed = subprocess.run(
            [sys.executable, script, "--config", config],
            cwd=workdir,
            env={**os.environ, "MPLBACKEND": "Agg"},
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr

    assert sorted(p.name for p in workdir.glob("*.png")) == [
        "Proton_pot_adiabatic_Y356_Y731_env.png",
        "Proton_pot_adiabatic_Y356_Y731_gas.png",
        "Proton_pot_w_slope_Y356_Y731_env.png",
        "Proton_pot_w_slope_Y356_Y731_gas.png",
    ]


# "kPCET_data_{OUTPUT_TAG}.log", "Proton_states_H_R{distance:.2f}_{OUTPUT_TAG}.png"
_OUTPUT_NAME = re.compile(r'"([^"\n]*\.(?:log|png))"')
_OUTPUT_TAG = re.compile(r'^OUTPUT_TAG = "(\w+)"$', re.MULTILINE)


def outputs_written_by(script: Path) -> set[str]:
    """The file names a script writes, with its ``OUTPUT_TAG`` resolved."""
    source = script.read_text()

    tag = _OUTPUT_TAG.search(source)
    assert tag, f"{script.name} declares no OUTPUT_TAG"

    names = _OUTPUT_NAME.findall(source)
    assert names, f"no output file names found in {script.name}"
    assert all("{OUTPUT_TAG}" in name for name in names), (
        f"{script.name} writes an untagged file: {sorted(names)}"
    )
    return {name.replace("{OUTPUT_TAG}", tag[1]) for name in names}


def test_the_example3_scripts_write_separate_files() -> None:
    """The two example 3 runs share a directory, so their outputs must differ.

    Both are far too slow to run here -- the electrochemical one sweeps 11
    distances by 101 electrode energies by two isotopes -- so this reads the
    file names out of the sources rather than running them.
    """
    directory = EXAMPLES / "example3_BIP_KIE"
    electrochemical = outputs_written_by(
        directory / "example3_BIP_KIE_electrochemical.py"
    )
    photochemical = outputs_written_by(directory / "example3_BIP_KIE_photochemical.py")

    assert not electrochemical & photochemical


def test_the_shared_fixture_parameters_are_still_example1s() -> None:
    """The fixtures claim example 1's thermodynamics, so check they still are.

    Its potentials come from the same data file the example reads, but these
    four scalars are stated on both sides and could drift apart.
    """
    source = (EXAMPLES / "example1_basic_usage" / "example1_basic_usage.py").read_text()
    declared = dict(labelled_values(source))

    for name, value in (
        ("TEMPERATURE", example1_data.TEMPERATURE),
        ("ELECTRONIC_COUPLING", example1_data.ELECTRONIC_COUPLING),
        ("REACTION_FREE_ENERGY", example1_data.REACTION_FREE_ENERGY),
        ("REORGANIZATION_ENERGY", example1_data.REORGANIZATION_ENERGY),
    ):
        assert declared[name] == value, name
