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

The examples additionally use matplotlib and pandas, and ASE powers both the
in-process scan runners in `autopcet.ase_io` and the structure-file reading of
the `autopcet-*` command-line helpers; install them through the optional
extras:

```bash
python -m pip install -e ".[examples]"
python -m pip install -e ".[ase]"
```

(`scripts` is kept as a legacy alias for the `ase` extra.)

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

Fit tabulated diabatic proton potentials, set up the golden-rule rate model,
and compute rate constants and the KIE:

```python
from autopcet import PCET, MASS_DEUTERON, MASS_PROTON, fit_poly8

# rp, reactant_energies, product_energies: 1D arrays holding the tabulated
# diabatic proton potentials
reactant_potential = fit_poly8(rp, reactant_energies)
product_potential = fit_poly8(rp, product_energies)

system = PCET(
    reactant_potential,
    product_potential,
    reaction_free_energy=-0.50,
    reorganization_energy=1.00,
    electronic_coupling=0.0434,
)

rate_h = system.calculate(MASS_PROTON, temperature=298)
rate_d = system.calculate(MASS_DEUTERON, temperature=298)
print(f"k(H) = {rate_h:.2e} s^-1, KIE = {rate_h / rate_d:.2f}")
```

Once `calculate` has run, the per-state results are plain attributes:
`populations`, `overlaps`, `pair_free_energies`, `pair_activation_energies`,
`rate_contributions`, and `total_rate_constant`. The thermodynamic parameters
are attributes too, so a sweep just reassigns one and calls `calculate` again
with `reuse_states=True` to keep the proton states it already solved for.

A rate constant computed at one proton donor-acceptor distance is usually
averaged over the distribution of distances the mode samples.
`donor_acceptor_distribution` builds the harmonic `P(R)` from the effective
force constant that `autopcet-keff` reports:

```python
from scipy.integrate import simpson

from autopcet import donor_acceptor_distribution

# distances: the grid k(R) was evaluated on; force_constant in atomic units
distribution = donor_acceptor_distribution(distances, 2.58, 0.0443, temperature=298)
distribution /= simpson(distribution, x=distances)
average_rate = simpson(distribution * rates, x=distances)
```

To quantify the (non)adiabaticity of a reaction from the same inputs:

```python
from autopcet import KappaCoupling, MASS_PROTON

# potentials tabulated on a 2^n grid, electronic coupling in eV
system = KappaCoupling(rp, reactant_energies, product_energies, electronic_coupling)
system.calculate(MASS_PROTON)

print(system.tau_electron, system.tau_proton, system.adiabaticity, system.kappa)
print(system.v_semiclassical, system.v_nonadiabatic, system.v_adiabatic)
```

### Units and naming

Energies are in electronvolts, lengths in ångström, and particle masses in
electron masses (atomic units), unless a name says otherwise. Unit conversions
are exported as named constants (`KCAL_TO_EV`, `HARTREE_TO_EV`,
`ANGSTROM_TO_BOHR`, ...), as are the physical constants (`BOLTZMANN`, `HBAR`,
`MASS_PROTON`, `MASS_DEUTERON`, ...).

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
6. `example6_ORCA_ASE` — the whole workflow run in-process through ASE with
   ORCA: donor-acceptor scan, proton endpoint optimizations, proton
   potentials, effective mode, rate constant, and KIE. Unlike the others it
   needs a working ORCA installation (pass its path with `--orca`) and your
   own reactant/product structures.

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

## Command-line tools

Installing the package also installs four helpers that prepare the Gaussian
inputs a calculation like the ones above starts from. They wrap
`autopcet.structure` and `autopcet.gaussian_io`, so anything they do is also
available from Python. Reading structure files in formats other than xyz needs
ASE, from the `ase` extra. Every one of them takes `--help`.

The four run in the order below, which is the order a proton potential is built
in.

### 1. Scan the proton donor-acceptor distance

`autopcet-scan-da` takes the fully optimized reactant and product structures --
the atoms must appear in the same order in both -- and writes, for each
donor-acceptor distance $`R`$, a pair of Gaussian inputs that reoptimize the two
states with $`R`$ frozen:

```bash
autopcet-scan-da -r reac.xyz -p prod.xyz -D 0 -A 2 --start 2.4 --stop 2.8 --points 9
```

`-D` and `-A` are the 0-based indices of the proton donor and acceptor. Charges
and multiplicities default to a neutral singlet reactant and a +1 doublet
product; `--reactant-charge` and friends override them. This assumes the two
diabatic electronic states differ only in the charge of the whole system, as
they do in homogeneous electrochemical PCET or in photoexcited PCET with an
external photoreceptor.

`--template` points at a file holding your own Gaussian header, in place of the
built-in B3LYP/6-31+G(d,p) one. It is formatted with `{state}`, `{charge}`, and
`{multiplicity}`.

> [!NOTE]
> Set `Nosymm` in these constrained optimizations. Without it Gaussian rotates
> the molecule and the later steps will not work.

### 2. Align and average the structures

`autopcet-align-average` overlays the reactant and product so the proton donor
and acceptor are superimposed along Z with their midpoint at the origin, rotates
the product about that axis to minimize the RMSD to the reactant, and averages
the two geometries:

```bash
autopcet-align-average -r reac.xyz -p prod.xyz --r-donor 1 --r-acceptor 2 --p-donor 1 --p-acceptor 2
```

These indices are 1-based, as printed by most quantum chemistry programs. The
aligned intermediates are written alongside their inputs, and the average goes
to `AVERAGE_STRUCTURE.xyz` unless `-o` says otherwise.

From the averaged structure, optimize the proton on the donor for the reactant
and on the acceptor for the product, with every other nucleus frozen. With
Gaussian, set that frozen optimization up by hand; with an ASE calculator,
`optimize_proton` does it in-process (see
[Running the scans with an ASE calculator](#running-the-scans-with-an-ase-calculator)).

### 3. Scan the proton along its transfer axis

`autopcet-scan-proton` builds the axis through the two optimized proton
positions and writes a Gaussian single point for each point along it. Run it
once per state:

```bash
autopcet-scan-proton -r averaged_optH_reac.xyz -p averaged_optH_prod.xyz -H 1 --state reactant --charge 0 --multiplicity 1
```

`-H` is the 0-based index of the transferring proton. The scan spans 1.7 times
the distance between the two equilibrium proton positions, and never less than
1 Å, so that the proton comes close to both the donor and the acceptor.

### 4. Effective donor-acceptor force constant

`autopcet-keff` projects the normal modes of a Gaussian frequency job onto the
donor-acceptor axis and prints the effective force constant (a.u.), reduced mass
(amu), and frequency (cm<sup>-1</sup>) of the donor-acceptor mode:

```bash
autopcet-keff --log freq.log -D 0 -A 1
```

The force constant is what `donor_acceptor_distribution` needs to build the
$`P(R)`$ the rate constant is thermally averaged over.

> [!NOTE]
> Run the frequency job with `#P` and `freq=HPmodes`. The parser reads the
> high-precision normal modes those settings print and will not work without
> them.

## Running the scans with an ASE calculator

Everything the helpers above prepare Gaussian inputs for can also run
in-process against any [ASE](https://ase-lib.org) calculator — ORCA, xTB, or
anything else implementing ASE's `Calculator` interface — through
`autopcet.ase_io`, installed with the `ase` extra. The diabatic electronic
states are selected through the calculator: configure one per state, carrying
that state's charge and multiplicity, and run each scan once per state.

Everything that moves atoms — `run_da_scan`, `optimize_proton`,
`run_vibrations` — needs forces, so a calculator that runs an external program
has to be told to compute the gradient. For ORCA that means `EnGrad` in
`orcasimpleinput`: without it ORCA writes no `.engrad` file, ASE reports no
forces, and the scan stops on its first step. `run_proton_scan` and
`read_scan_energies` need energies only.

```python
from ase.calculators.orca import ORCA, OrcaProfile
from ase.io import read

from autopcet import (
    effective_mode_from_vibrations,
    optimize_proton,
    run_da_scan,
    run_proton_scan,
    run_vibrations,
)

profile = OrcaProfile(command="/opt/orca/orca")
reactant_calc = ORCA(
    profile=profile,
    directory="reac",
    charge=0,
    mult=1,
    orcasimpleinput="B3LYP def2-SVP TightSCF EnGrad",
    # ASE replaces its whole default block, so set the memory here too
    orcablocks="%pal nprocs 8 end\n%maxcore 3000",
)
# ... and a product_calc with that state's charge and multiplicity

# 1. reoptimize the state on a grid of frozen donor-acceptor distances
scan = run_da_scan(read("reac.xyz"), 0, 2, reactant_calc, [2.4, 2.5, 2.6])

# 2. align and average with autopcet.structure as before, then optimize the
#    proton endpoints with the rest of the frame frozen
endpoint = optimize_proton(averaged, 1, reactant_calc)

# 3. the proton potential along the transfer axis, on the same grid the
#    command-line helper writes; feeds PCET directly
potential = run_proton_scan(reactant_endpoint, product_endpoint, 1, reactant_calc)
pcet = PCET((potential.offsets, potential.relative_energies), ...)

# 4. the effective donor-acceptor mode, in place of autopcet-keff
mode = effective_mode_from_vibrations(
    run_vibrations(equilibrium, reactant_calc), donor=0, acceptor=2
)
```

Every grid point reuses the calculator's `directory`, so each job overwrites
the last one's files. Give a scan its own per-point directory if you want to
keep them, or to reuse a converged wavefunction as the next point's guess.

`examples/example6_ORCA_ASE` walks the full pipeline. For jobs run elsewhere —
on a cluster, say — keep writing inputs with `autopcet-scan-proton` and skip
the manual energy table afterwards: `read_scan_energies(directory, state)`
walks the numbered directories the helper created and reads the finished
outputs in any format ASE recognizes, Gaussian logs and ORCA outputs included.
Name each job's output after its input, so grid point `03` of a reactant scan
holds `reactant_sp.out` (or `.log`, `.xyz`, `.traj`) beside its `reactant_sp.gjf`.

> [!NOTE]
> Point `OrcaProfile` at the full path of your ORCA binary. On desktop Linux a
> bare `orca` on the PATH is usually the GNOME screen reader, not ORCA.

> [!NOTE]
> ASE warns `Geometry optimization did not converge!` for every ORCA gradient
> single point, reading the gradient header as the start of a relaxation, and
> prints a caution about reading `.engrad` files from optimizations. Both are
> artefacts of ASE's ORCA reader and say nothing about your calculation.

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
│   └── cli/                   # the autopcet-* command-line helpers
├── build_tools/               # optional Conda setup
├── examples/                  # worked examples with reference outputs
├── tests/                     # behavior-focused tests
└── pyproject.toml             # project metadata and tool configuration
```

## Next steps

1. Broaden the supported Python versions beyond 3.14 and add them to the CI
   matrix.
2. Configure trusted publishing and cut a first PyPI release.

## License

Released under the [MIT License](LICENSE).
