# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[1.0.0]: https://github.com/LouieSlocombe/autopcet/releases/tag/v1.0.0
