# autopcet

Automated proton-coupled electron transfer (PCET) rate calculations in Python.

The library computes vibronically nonadiabatic PCET rate constants from
diabatic proton potentials: it solves for the proton vibrational states with a
Fourier grid Hamiltonian (FGH) method, builds the golden-rule rate matrices
over pairs of reactant/product vibronic states (`PCET`), and evaluates
kinetic isotope effects (KIEs). It also provides a nonadiabaticity analysis
following Georgievskii and Stuchebrukhov (`KappaCoupling`) and an electrical
double layer (EDL) model of the interfacial potential drop for
electrochemical PCET (`make_edl_model`).

## Requirements

- Python 3.14 or newer
- pip 25.1 or newer (for dependency groups)

NumPy, SciPy, and Numba are installed automatically as dependencies.

## Installation

Create an isolated environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1` instead.

The examples additionally use matplotlib and pandas, and the helper scripts
use ASE; install them through the optional extras:

```bash
python -m pip install -e ".[examples]"
python -m pip install -e ".[scripts]"
```

For development, install the dev dependency group as well:

```bash
python -m pip install --group dev -e .
```

If you prefer Conda, `build_tools/environment.yml` creates the base environment:

```bash
conda env create -f build_tools/environment.yml
conda activate autopcet
python -m pip install --group dev -e .
```

## Quick start

Fit tabulated diabatic proton potentials (proton coordinate in Å, energies in
eV), set up the golden-rule rate model, and compute rate constants and the KIE:

```python
from autopcet import PCET, fit_poly8, massD, massH

# rp, E_reac, E_prod: 1D arrays with the tabulated diabatic proton potentials
ReacProtonPot = fit_poly8(rp, E_reac)
ProdProtonPot = fit_poly8(rp, E_prod)

# reaction free energy, reorganization energy, and electronic coupling in eV
system = PCET(ReacProtonPot, ProdProtonPot, DeltaG=-0.50, Lambda=1.00, Vel=0.0434)

k_H = system.calculate(massH, T=298)
k_D = system.calculate(massD, T=298)
print(f"k(H) = {k_H:.2e} s^-1, KIE = {k_H / k_D:.2f}")
```

The per-state populations, overlaps, free energies, and rate contributions are
available through the `get_*` methods after `calculate` has run.

To quantify the (non)adiabaticity of a reaction from the same inputs:

```python
from autopcet import KappaCoupling, massH

# potentials tabulated on a 2^n grid, electronic coupling Vel in eV
system = KappaCoupling(rp, E_reac, E_prod, Vel)
system.calculate(massH)
tau_e, tau_p, p, kappa = system.get_nonadiabaticity_parameters()
V_sc, V_nad, V_ad = system.get_vibronic_couplings()
```

## Examples

The `examples/` directory contains worked calculations, each with its input
data and a reference output to compare against:

1. `example1_basic_usage` — rate constant, vibronic-state analysis, and H/D
   KIE for a first-principles double-well potential.
2. `example2_Y356-Y731` — rate constants as a function of the proton
   donor-acceptor distance for the RNR Y356-Y731 interface, thermally averaged
   over an umbrella-sampled P(R) distribution.
3. `example3_BIP_KIE` — electrochemical and photochemical KIEs for a
   benzimidazole-phenol (BIP) system.
4. `example4_CoTPP` — heterogeneous electrochemical PCET for CoTPP on
   graphene, combining the EDL model with a density-of-states average.
5. `example5_RNR_nonadiabaticity` — vibronic couplings and nonadiabaticity
   analysis for the RNR Y356-Y731 interface, in the gas phase
   (`--config gas`) or in the protein environment (`--config env`).

Run an example from inside its own directory, e.g.:

```bash
cd examples/example1_basic_usage
python example1_basic_usage.py
```

Example 5 takes the configuration to run as an argument, and writes a separate
set of figures for each:

```bash
cd examples/example5_RNR_nonadiabaticity
python example5_nonadiabaticity_Y356-Y731.py --config gas
python example5_nonadiabaticity_Y356-Y731.py --config env
```

The reference outputs were generated with the NumPy and SciPy versions current
at the time; results can drift in the third significant digit across releases.
`tests/test_examples.py` re-runs examples 1 and 5 and compares them at a
relative tolerance.

The `scripts/` directory holds standalone Gaussian/ASE helpers for preparing
the inputs of such calculations (potential scans, structure alignment,
effective force constants); see [scripts/README.md](scripts/README.md).

## Development

Run the complete local checks:

```bash
ruff check .
ruff format --check .
mypy
pytest
python -m build
python -m twine check dist/*
```

Ruff can apply safe lint and formatting changes with:

```bash
ruff check --fix .
ruff format .
```

Install the Git hooks once, then pre-commit will run the fast checks before
each commit:

```bash
pre-commit install
pre-commit run --all-files
```

## Project layout

```text
.
├── .github/workflows/ci.yml   # automated quality and packaging checks
├── autopcet/                  # the installable, typed package
├── build_tools/               # optional Conda setup
├── examples/                  # worked examples with reference outputs
├── scripts/                   # Gaussian/ASE helper scripts
├── tests/                     # behavior-focused tests
└── pyproject.toml             # project metadata and tool configuration
```

## Next steps

1. Broaden the supported Python versions beyond 3.14 and add them to the CI
   matrix.
2. Configure trusted publishing and cut a first PyPI release.

## License

Released under the [MIT License](LICENSE).
