"""Electrochemical KIE for a benzimidazole-phenol (BIP) system, integrated over
the electrode states and thermally averaged over the donor-acceptor distance."""

import colorsys
from typing import TextIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

from autopcet import (
    ANGSTROM_TO_BOHR,
    BOLTZMANN,
    HARTREE_TO_EV,
    KCAL_TO_EV,
    MASS_DEUTERON,
    MASS_PROTON,
    PCET,
    fermi_distribution,
    fit_poly6,
    fit_poly8,
)

# =========================================================================================
# Define the thermodynamic parameters, taken from
# Huynh et. al. ACS Cent. Sci. 2017, 3, 372-380
# =========================================================================================

# R values sampled in calculations
distances = np.arange(2.37, 2.92, 0.05)

OVERPOTENTIAL = 0  # unit in V
REORGANIZATION_ENERGY = 21.4 * KCAL_TO_EV
# the coupling is not needed for a KIE, so use a default value of 1 kcal/mol
ELECTRONIC_COUPLING = 1 * KCAL_TO_EV
TEMPERATURE = 298.15

# beta' and rho_M in Eqs. (S2) and (S3) are also not needed for a KIE; set them to 1
INVERSE_DECAY_LENGTH = 1  # unit in A^-1
ELECTRODE_DOS = 1  # unit in eV^-1

# Delta G depends on the electrode state energy epsilon (relative to the Fermi
# level) and the overpotential eta:
#   Delta G_a/c = Delta G^0 +/- epsilon -/+ e*eta   (a: anodic, c: cathodic)
# Delta G^0 is the reaction free energy at the equilibrium potential (eta = 0),
# which is 0 by definition. We start from 0 and update it per epsilon below.
REACTION_FREE_ENERGY = 0

N_STATES = 20  # how many states to include in the rate constant calculation
STATES_TO_SHOW = 7  # how many states to plot/print

# wave functions and energies share an axis, so scale the wave functions down
WAVEFUNCTION_SCALE = 0.06

FORCE_CONSTANT = 0.0443  # effective proton donor-acceptor force constant, a.u.
EQUILIBRIUM_DISTANCE = 2.58  # equilibrium proton donor-acceptor distance

# =========================================================================================
# Read data from files
# =========================================================================================

# rainbow colors for plotting the proton potentials
colors = [
    colorsys.hls_to_rgb(hue, 0.5, 0.85) for hue in np.linspace(0.0, 0.8, len(distances))
]

reactant_potentials = []
product_potentials = []

rp = np.linspace(-1.0, 1.0, 256)

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

# plot the proton potentials
fig = plt.figure(figsize=(5, 7))
gs = fig.add_gridspec(2, hspace=0)
ax1, ax2 = gs.subplots(sharex=True, sharey=True)

for i in range(len(distances)):
    ax1.plot(rp, reactant_potentials[i](rp), "-", lw=2, color=colors[i])
    ax2.plot(rp, product_potentials[i](rp), "-", lw=2, color=colors[i])

ax2.set_xlim(-1, 1)
ax2.set_ylim(0, 2.2)
ax2.set_xlabel(r"$r_{\rm p}\ /\ \rm\AA$", fontsize=16)
ax1.set_ylabel(r"$E$ / eV", fontsize=16)
ax2.set_ylabel(r"$E$ / eV", fontsize=16)
ax2.set_xticks(np.arange(-1.0, 1.5, 0.5))
ax1.tick_params(labelsize=14)
ax2.tick_params(labelsize=14)
plt.tight_layout()
plt.savefig("Proton_potentials.png", dpi=300)
plt.clf()

# =========================================================================================
# Calculate the KIE of electrochemical PCET of BIP at different R and eta = 0
# The standard rate constant is approximated as the anodic rate constant at eta = 0
# =========================================================================================


def plot_states(
    axis: Axes,
    grid: np.ndarray,
    energies: np.ndarray,
    wavefunctions: np.ndarray,
    shift: float,
    color: str,
) -> None:
    """Draw the lowest vibrational states as wave functions on their levels."""
    for i, (energy, wavefunction) in enumerate(
        zip(energies[:STATES_TO_SHOW], wavefunctions[:STATES_TO_SHOW], strict=True)
    ):
        # flip the wave function so its largest amplitude points up
        sign = 1 if np.abs(np.max(wavefunction)) > np.abs(np.min(wavefunction)) else -1
        level = energy + shift
        curve = level + WAVEFUNCTION_SCALE * sign * wavefunction
        axis.plot(grid, curve, f"{color}-", lw=1, alpha=(1 - 0.12 * i))
        axis.fill_between(grid, curve, level, color=color, alpha=0.4)


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


# sample 100 electrode state energies from -2 eV to 2 eV for the numerical
# integration over electrode states
electrode_energies = np.linspace(-2, 2, 101)

rates_h = np.zeros(len(distances))
rates_d = np.zeros(len(distances))

for i, distance in enumerate(distances):
    print(f"Calculating... R = {distance:.2f}A")

    # one PCET instance per isotope, so each can cache its own proton states
    system_h = make_system(i)
    system_d = make_system(i)

    rates_h_of_energy = np.zeros(len(electrode_energies))
    rates_d_of_energy = np.zeros(len(electrode_energies))

    for j, electrode_energy in enumerate(electrode_energies):
        # update Delta G for a given epsilon; only the anodic rate is computed here
        anodic_free_energy = electrode_energy - OVERPOTENTIAL
        system_h.reaction_free_energy = anodic_free_energy
        system_d.reaction_free_energy = anodic_free_energy

        rates_h_of_energy[j] = system_h.calculate(
            mass=MASS_PROTON, temperature=TEMPERATURE, reuse_states=True
        )
        rates_d_of_energy[j] = system_d.calculate(
            mass=MASS_DEUTERON, temperature=TEMPERATURE, reuse_states=True
        )

        # plot the wave functions and print the state contributions for epsilon = 0;
        # epsilon comes off a linspace, so do not rely on exact equality with 0.0
        if abs(electrode_energy) > 1e-9:
            continue

        fig = plt.figure(figsize=(9, 4.5))
        gs = fig.add_gridspec(ncols=2, wspace=0)
        ax1, ax2 = gs.subplots(sharex=True, sharey=True)

        # align the zero-point energy of the reactant and product states
        zero_point_gap = system_h.product_energies[0] - system_h.reactant_energies[0]
        reactant_shift = max(zero_point_gap, 0.0)
        product_shift = max(-zero_point_gap, 0.0)

        ax1.plot(
            system_h.rp, reactant_potentials[i](system_h.rp) + reactant_shift, "b", lw=2
        )
        plot_states(
            ax1,
            system_h.rp,
            system_h.reactant_energies,
            system_h.reactant_wavefunctions,
            reactant_shift,
            "b",
        )

        ax2.plot(
            system_h.rp, product_potentials[i](system_h.rp) + product_shift, "r", lw=2
        )
        plot_states(
            ax2,
            system_h.rp,
            system_h.product_energies,
            system_h.product_wavefunctions,
            product_shift,
            "r",
        )

        ax2.set_xlim(-1.0, 1.0)
        ax2.set_ylim(0, 1.3)
        ax1.set_xlabel(r"$r_{\rm p}\ /\ \rm\AA$", fontsize=16)
        ax1.set_ylabel(r"$E$ / eV", fontsize=16)
        ax2.set_xlabel(r"$r_{\rm p}\ /\ \rm\AA$", fontsize=16)
        ax2.set_xticks(np.arange(-0.8, 1.2, 0.4))
        ax1.tick_params(labelsize=14)
        ax2.tick_params(labelsize=14)

        plt.tight_layout()
        plt.savefig(f"Proton_states_H_R{distance:.2f}.png", dpi=300)
        plt.clf()

        with open(f"rate_constant_contribution_R{distance:.2f}A.log", "w") as log:
            log.write(f"\nR = {distance:.2f}A, epsilon = 0, eta = 0\n")
            write_contribution_table(log, system_h, "H")
            write_contribution_table(log, system_d, "D")

    # calculate the anodic rate constant according to Eq. (S2) in the paper
    hole_occupancy = (
        ELECTRODE_DOS
        / INVERSE_DECAY_LENGTH
        * (1 - fermi_distribution(electrode_energies, temperature=TEMPERATURE))
    )
    rates_h[i] = simpson(hole_occupancy * rates_h_of_energy, x=electrode_energies)
    rates_d[i] = simpson(hole_occupancy * rates_d_of_energy, x=electrode_energies)

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


def donor_acceptor_distribution(
    distance: np.ndarray, equilibrium: float, force_constant: float, temperature: float
) -> np.ndarray:
    """Harmonic P(R) for the proton donor-acceptor mode."""
    energy = (
        0.5
        * force_constant
        * (distance - equilibrium) ** 2
        * ANGSTROM_TO_BOHR**2
        * HARTREE_TO_EV
    )
    return np.exp(-energy / (BOLTZMANN * temperature))


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
