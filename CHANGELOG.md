# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-09-18

### Added

- Support for Python 3.12 and 3.13 alongside 3.14. The declared floor moves
  from `>=3.14` to `>=3.12`, and CI tests all three.

### Fixed

- `import autopcet` raised `NameError` on Python 3.12 and 3.13. Annotations
  naming an import made only under `if TYPE_CHECKING:` are evaluated eagerly
  before 3.14, which defers them under PEP 649; every module doing so now
  carries `from __future__ import annotations`. Moving those imports out of the
  type-checking block would have made `ase` and `matplotlib` mandatory, so the
  `TC004` lint rule guards the pattern instead.

## [1.0.0] - 2026-09-18

First public release.

### Added

- Vibronically nonadiabatic PCET rate constants from diabatic proton
  potentials, with proton vibrational states solved on a Fourier grid
  Hamiltonian (`autopcet.fgh`) and golden-rule rate matrices summed over pairs
  of reactant and product vibronic states (`PCET`).
- Kinetic isotope effects from the same rate machinery
  (`kinetic_isotope_effect`).
- Nonadiabaticity analysis after Georgievskii and Stuchebrukhov
  (`KappaCoupling`).
- An electrical double layer model of the interfacial potential drop for
  electrochemical PCET (`make_edl_model`).
- Analytic and tabulated proton potentials, including Morse, inverted Morse and
  Gaussian forms, with polynomial and B-spline fitting of scan data
  (`autopcet.potentials`).
- Readers for Gaussian output and, through the optional `ase` extra, scan and
  vibrational drivers for any ASE calculator (`autopcet.gaussian_io`,
  `autopcet.ase_io`).
- Structure alignment and averaging helpers for donor-acceptor geometries
  (`autopcet.structure`).
- Four console scripts: `autopcet-align-average`, `autopcet-keff`,
  `autopcet-scan-da` and `autopcet-scan-proton`.
- An optional matplotlib layer under the `plotting` extra
  (`autopcet.plotting`). Importing the package, or the subpackage, never
  requires matplotlib; only calling a plotting function does.
- Inline type annotations, exported through `py.typed`.
- Six worked examples with committed reference output, under `examples/`.

[1.1.0]: https://github.com/LouieSlocombe/autopcet/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/LouieSlocombe/autopcet/releases/tag/v1.0.0
