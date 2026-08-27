"""Photochemical KIE for a benzimidazole-phenol (BIP) system, thermally
averaged over the proton donor-acceptor distance."""

from typing import TextIO

import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

from autopcet import (
    KCAL_TO_EV,
    MASS_DEUTERON,
    MASS_PROTON,
    PCET,
    donor_acceptor_distribution,
    fit_poly6,
    fit_poly8,
)

# =========================================================================================
# Define the thermodynamic parameters
# for photochemical oxidation of BIP
# =========================================================================================

# R values sampled in calculations
distances = np.arange(2.37, 2.92, 0.05)

REACTION_FREE_ENERGY = -5.0 * KCAL_TO_EV
REORGANIZATION_ENERGY = 21.4 * KCAL_TO_EV
# the coupling is not needed for a KIE, so use a default value of 1 kcal/mol
ELECTRONIC_COUPLING = 1 * KCAL_TO_EV
TEMPERATURE = 298.15

N_STATES = 20  # how many states to include in the rate constant calculation
STATES_TO_SHOW = 7  # how many states to print

FORCE_CONSTANT = 0.0443  # effective proton donor-acceptor force constant, a.u.
EQUILIBRIUM_DISTANCE = 2.58  # equilibrium proton donor-acceptor distance

# =========================================================================================
# Read data from files
# =========================================================================================

reactant_potentials = []
product_potentials = []

# read proton potentials from .csv files
# the proton potentials are digitized from Figure S39 of
# Huynh et. al. ACS Cent. Sci. 2017, 3, 372-380
for i, distance in enumerate(distances):
    reduced = pd.read_csv(
        f"proton_potentials/Reduced_BIP_{distance:.2f}A.csv",
        sep=", ",
        header=0,
        engine="python",
    )
    oxidized = pd.read_csv(
        f"proton_potentials/Oxidized_BIP_{distance:.2f}A.csv",
        sep=", ",
        header=0,
        engine="python",
    )

    # the energy unit in the .csv files is kcal/mol, convert it to eV
    reduced_energies = reduced["y"] * KCAL_TO_EV
    oxidized_energies = oxidized["y"] * KCAL_TO_EV

    # smooth the data by fitting to polynomials; for these data sets a polynomial
    # fits better than a spline. An 8th order polynomial gives the best fit,
    # except for the reduced BIP at R = 2.37A, where 6th order is used.
    fit_reduced = fit_poly6 if i == 0 else fit_poly8
    reactant_potentials.append(fit_reduced(reduced["x"], reduced_energies))
    product_potentials.append(fit_poly8(oxidized["x"], oxidized_energies))

# =========================================================================================
# Calculate the KIE of photochemical PCET of BIP at different R
# =========================================================================================

# for photochemical PCET, the integration over epsilon is not needed


def write_contribution_table(stream: TextIO, system: PCET, isotope: str) -> None:
    """Tabulate how much each pair of vibronic states contributes to the rate."""
    percentage = 100 * system.rate_contributions / system.total_rate_constant

    stream.write(f"\n{isotope}\n" + "=" * 125 + "\n")
    stream.write(
        "(u, v)\t\tP_u\t\t\t|S_uv|^2\t\tDelta G_uv / eV\t\t"
        "Delta G^#_uv / eV\t% Contrib.\n"
    )
    stream.write("-" * 125 + "\n")
    for u in range(STATES_TO_SHOW):
        for v in range(STATES_TO_SHOW):
            stream.write(
                f"({u:d}, {v:d})\t\t{system.populations[u]:.3e}\t\t"
                f"{system.overlaps[u, v] ** 2:.3e}\t\t"
                f"{system.pair_free_energies[u, v]:+.3f}\t\t\t"
                f"{system.pair_activation_energies[u, v]:.3f}\t\t\t"
                f"{percentage[u, v]:.1f}\n"
            )
    stream.write("=" * 125 + "\n\n")


def make_system(index: int) -> PCET:
    """A PCET system for the proton potentials sampled at distance ``index``."""
    return PCET(
        reactant_potentials[index],
        product_potentials[index],
        reaction_free_energy=REACTION_FREE_ENERGY,
        reorganization_energy=REORGANIZATION_ENERGY,
        electronic_coupling=ELECTRONIC_COUPLING,
        n_states=N_STATES,
        r_min=-1.0,
        r_max=1.0,
    )


rates_h = np.zeros(len(distances))
rates_d = np.zeros(len(distances))

for i, distance in enumerate(distances):
    print(f"Calculating... R = {distance:.2f}A")

    # one PCET instance per isotope, so each keeps its own proton states and the
    # tables below report the isotope they are labelled with
    system_h = make_system(i)
    system_d = make_system(i)

    rates_h[i] = system_h.calculate(mass=MASS_PROTON, temperature=TEMPERATURE)
    rates_d[i] = system_d.calculate(mass=MASS_DEUTERON, temperature=TEMPERATURE)

    # Plotting of the wave functions is omitted in this example. They are the
    # same as in the electrochemical case because the same proton potentials are
    # used. Only the contributions of the vibronic states are printed.
    with open(f"rate_constant_contribution_R{distance:.2f}A.log", "w") as log:
        log.write(f"\nR = {distance:.2f}A, epsilon = 0, eta = 0\n")
        write_contribution_table(log, system_h, "H")
        write_contribution_table(log, system_d, "D")

# Print PCET rate constants for H and D at each R to a file
with open("kPCET_data.log", "w") as log:
    log.write("# R_PT/A\tk_H/s^-1\tk_D/s^-1\n")
    for distance, rate_h, rate_d in zip(distances, rates_h, rates_d, strict=True):
        log.write(f"{distance:.2f}\t\t{rate_h:.4e}\t{rate_d:.4e}\n")

# =========================================================================================
# Thermally average the PCET rate constant over R
# =========================================================================================

# the integration should run from 0 to infinity, but in practice we integrate
# over the interval where the integrand has reached zero at both limits
fine_grid = np.linspace(2.0, 3.0, 200)

# Interpolate k(R). At the smallest sampled R = 2.37A, k_PCET * P(R) is still
# non-zero, so k_PCET has to be extrapolated to integrate from 0 to infinity.
# !!!NOTE!!! Always check if k_PCET * P(R) reaches zero at the limit of your sampled R
#
# Different interpolation and extrapolation methods have been tested and give
# similar KIEs. Here log(k) is interpolated and then exponentiated.
rates_h_fine = np.exp(
    interp1d(distances, np.log(rates_h), kind="quadratic", fill_value="extrapolate")(
        fine_grid
    )
)
rates_d_fine = np.exp(
    interp1d(distances, np.log(rates_d), kind="quadratic", fill_value="extrapolate")(
        fine_grid
    )
)


distribution = donor_acceptor_distribution(
    fine_grid, EQUILIBRIUM_DISTANCE, FORCE_CONSTANT, TEMPERATURE
)
distribution /= simpson(distribution, x=fine_grid)

# perform thermal average and print the final results
dominant_distance_h = fine_grid[find_peaks(distribution * rates_h_fine)[0]]
dominant_distance_d = fine_grid[find_peaks(distribution * rates_d_fine)[0]]

average_rate_h = simpson(distribution * rates_h_fine, x=fine_grid)
average_rate_d = simpson(distribution * rates_d_fine, x=fine_grid)

print()
print(f"Dominant R for H = {dominant_distance_h[0]:.2f}A")
print(f"Dominant R for D = {dominant_distance_d[0]:.2f}A")
print(f"k_H_tot = {average_rate_h:.4e} s^-1")
print(f"k_D_tot = {average_rate_d:.4e} s^-1")
print(f"KIE = {average_rate_h / average_rate_d:.2f}")
