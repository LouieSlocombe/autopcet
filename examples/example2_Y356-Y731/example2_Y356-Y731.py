"""PCET rate constants for the RNR Y356-Y731 interface as a function of the
proton donor-acceptor distance R, thermally averaged over an umbrella-sampled
P(R) distribution."""

import colorsys
from typing import TextIO

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.integrate import simpson
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from autopcet import KCAL_TO_EV, MASS_PROTON, PCET, fit_bspline

# =========================================================================================
# Define the thermodynamic parameters, taken from
# Zhong et. al. J. Am. Chem. Soc. 2025, 147, 4459-4468
# =========================================================================================

# R values sampled in calculations
distances = np.arange(2.42, 3.2, 0.1)

# For PCET between two tyrosines the reaction free energy vanishes
REACTION_FREE_ENERGY = 0.0 * KCAL_TO_EV
REORGANIZATION_ENERGY = 18.86 * KCAL_TO_EV
ELECTRONIC_COUPLING = 0.8 * KCAL_TO_EV
TEMPERATURE = 298

N_STATES = 7  # how many states to include in the rate constant calculation
STATES_TO_SHOW = 4  # how many states to plot/print

# wave functions and energies share an axis, so scale the wave functions down
WAVEFUNCTION_SCALE = 0.06

# =========================================================================================
# Read data from files
# =========================================================================================

# rainbow colors for plotting the proton potentials
colors = [
    colorsys.hls_to_rgb(hue, 0.5, 0.85) for hue in np.linspace(0.0, 0.8, len(distances))
]

reactant_potentials = []
product_potentials = []
grid_limits = np.zeros(len(distances))

rp = np.linspace(-1.5, 1.5, 256)

# read proton potentials from .dat files
for i, distance in enumerate(distances):
    rp_data, reactant_energies, product_energies = np.loadtxt(
        f"proton_potentials/R{distance:.2f}_potential.dat",
        usecols=(0, 1, 2),
        unpack=True,
    )
    grid_limits[i] = rp_data[-1]

    # the energy unit in the .dat files is kcal/mol, convert it to eV
    reactant_energies *= KCAL_TO_EV
    product_energies *= KCAL_TO_EV

    # smooth the data by splining
    reactant_potentials.append(fit_bspline(rp_data, reactant_energies))
    product_potentials.append(fit_bspline(rp_data, product_energies))

# plot the proton potentials
fig = plt.figure(figsize=(8, 4))
gs = fig.add_gridspec(ncols=2, wspace=0)
ax1, ax2 = gs.subplots(sharex=True, sharey=True)

for i in range(len(distances)):
    ax1.plot(rp, reactant_potentials[i](rp) / KCAL_TO_EV, "-", lw=2, color=colors[i])
    ax2.plot(rp, product_potentials[i](rp) / KCAL_TO_EV, "-", lw=2, color=colors[i])

ax2.set_xlim(-1.2, 1.2)
ax2.set_ylim(0, 100)
ax1.set_xlabel(r"$r_{\rm p} (\rm\AA)$", fontsize=16)
ax2.set_xlabel(r"$r_{\rm p} (\rm\AA)$", fontsize=16)
ax1.set_ylabel(r"$E$ / (kcal/mol)", fontsize=16)
ax2.set_xticks(np.arange(-1.0, 1.5, 0.5))
ax1.tick_params(labelsize=14)
ax2.tick_params(labelsize=14)
plt.tight_layout()
plt.savefig("Proton_potentials.png", dpi=300)
plt.clf()

# =========================================================================================
# Calculate the PCET rate constant between Y356 and Y731
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


def write_contribution_table(stream: TextIO, system: PCET) -> None:
    """Tabulate how much each pair of vibronic states contributes to the rate."""
    percentage = 100 * system.rate_contributions / system.total_rate_constant

    stream.write(
        "(u, v)\t\tP_u\t\t\t|S_uv|\t\tDelta G_uv / eV\t\t"
        "Delta G^#_uv / eV\t% Contrib.\n"
    )
    stream.write("-" * 125 + "\n")
    for u in range(STATES_TO_SHOW):
        for v in range(STATES_TO_SHOW):
            stream.write(
                f"({u:d}, {v:d})\t\t{system.populations[u]:.3e}\t\t"
                f"{np.abs(system.overlaps[u, v]):.3e}\t\t"
                f"{system.pair_free_energies[u, v]:+.3f}\t\t\t"
                f"{system.pair_activation_energies[u, v]:.3f}\t\t\t"
                f"{percentage[u, v]:.1f}\n"
            )
    stream.write("=" * 125 + "\n\n")


rates_h = np.zeros(len(distances))

for i, distance in enumerate(distances):
    print(f"Calculating... R = {distance:.2f}A")
    system = PCET(
        reactant_potentials[i],
        product_potentials[i],
        reaction_free_energy=REACTION_FREE_ENERGY,
        reorganization_energy=REORGANIZATION_ENERGY,
        electronic_coupling=ELECTRONIC_COUPLING,
        n_states=N_STATES,
        r_min=-grid_limits[i],
        r_max=grid_limits[i],
        n_grid=512,
    )

    rates_h[i] = system.calculate(mass=MASS_PROTON, temperature=TEMPERATURE)

    # plot the wave functions and print the state contributions for each R
    fig = plt.figure(figsize=(9, 4.5))
    gs = fig.add_gridspec(ncols=2, wspace=0)
    ax1, ax2 = gs.subplots(sharex=True, sharey=True)

    # align the zero-point energy of the reactant and product states in this plot
    zero_point_gap = system.product_energies[0] - system.reactant_energies[0]
    reactant_shift = max(zero_point_gap, 0.0)
    product_shift = max(-zero_point_gap, 0.0)

    ax1.plot(system.rp, reactant_potentials[i](system.rp) + reactant_shift, "b", lw=2)
    plot_states(
        ax1,
        system.rp,
        system.reactant_energies,
        system.reactant_wavefunctions,
        reactant_shift,
        "b",
    )

    ax2.plot(system.rp, product_potentials[i](system.rp) + product_shift, "r", lw=2)
    plot_states(
        ax2,
        system.rp,
        system.product_energies,
        system.product_wavefunctions,
        product_shift,
        "r",
    )

    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(0, 2.5)
    ax1.set_xlabel(r"$r_{\rm p}\ /\ \rm\AA$", fontsize=16)
    ax1.set_ylabel(r"$E$ / eV", fontsize=16)
    ax2.set_xlabel(r"$r_{\rm p}\ /\ \rm\AA$", fontsize=16)
    ax2.set_xticks(np.arange(-1.0, 1.5, 0.5))
    ax1.tick_params(labelsize=14)
    ax2.tick_params(labelsize=14)

    plt.tight_layout()
    plt.savefig(f"Proton_states_H_R{distance:.2f}.png", dpi=300)
    plt.clf()

    with open(f"rate_constant_contribution_R{distance:.2f}A.log", "w") as log:
        log.write(f"\nR = {distance:.2f}A\n")
        write_contribution_table(log, system)

# Print PCET rate constants for H at each R to a file
with open("kPCET_data.log", "w") as log:
    log.write("# R_PT/A\tk_H/s^-1\n")
    for distance, rate in zip(distances, rates_h, strict=True):
        log.write(f"{distance:.2f}\t\t{rate:.4e}\n")

# =========================================================================================
# Thermally average the PCET rate constant over R
# =========================================================================================

RATE_COLOR = "#ff7700"
DISTRIBUTION_COLOR = (0.1, 0.6, 0.2)

# the integration should run from 0 to infinity, but in practice we integrate
# over the interval where the integrand has reached zero at both limits
fine_grid = np.linspace(2.0, 4.0, 500)


# Fit k(R) by fitting log k(R) to a quadratic function
# !!!NOTE!!! Always check if k_PCET * P(R) reaches zero at the limit of your sampled R
def quadratic(x, a, b, c):
    return a * x * x + b * x + c


rate_fit = curve_fit(quadratic, distances, np.log(rates_h))[0]
rates_h_fine = np.exp(quadratic(fine_grid, *rate_fit))


# P(R) is calculated from umbrella sampling
# Read the umbrella sampling data from file and fit log P(R) to a 4th order polynomial
sampled_distances, sampled_distribution = np.loadtxt("p_R_umbrella_A.dat", unpack=True)


def poly4(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e


distribution_fit = curve_fit(poly4, sampled_distances, np.log(sampled_distribution))[0]
distribution = np.exp(poly4(fine_grid, *distribution_fit))

# re-normalize the distribution
distribution /= simpson(distribution, x=fine_grid)
equilibrium_distance = fine_grid[find_peaks(distribution)[0]]

# perform thermal average and print the final results
weighted_rate = distribution * rates_h_fine
dominant_distance = fine_grid[find_peaks(weighted_rate)[0]]
average_rate_h = simpson(weighted_rate, x=fine_grid)

print()
print(f"Dominant R for H = {dominant_distance[0]:.2f}A")
print(f"k_H_tot = {average_rate_h:.4e} s^-1")

# =========================================================================================
# plot k(R), P(R), k(R)*P(R) for H
# =========================================================================================

plt.figure(figsize=(4.8, 4.5))

plt.title("H", fontsize=20)
plt.plot(
    fine_grid,
    rates_h_fine / np.max(rates_h_fine),
    "-",
    lw=2,
    color=RATE_COLOR,
    label=r"$k(R)$",
)
plt.plot(
    fine_grid,
    distribution / np.max(distribution),
    "-",
    lw=2,
    color=DISTRIBUTION_COLOR,
    label="$P(R)$",
)
plt.plot(
    fine_grid,
    weighted_rate / np.max(weighted_rate),
    "-",
    lw=2,
    color="k",
    label=r"$P(R)k(R)$",
)

plt.axvline(
    x=equilibrium_distance, linewidth=1.5, color="darkgray", linestyle=(0, (3, 3))
)
plt.axvline(x=dominant_distance[0], linewidth=1.5, color="k", linestyle=(0, (3, 3)))

plt.legend(fontsize=16, frameon=False, loc=1)
plt.xlim(2.10, 3.7)
plt.ylim(0, 1.2)
plt.xlabel(r"$R$ / $\rm \AA$", fontsize=18)
plt.xticks(np.arange(2.25, 3.75, 0.25), fontsize=16)
plt.yticks([])

plt.tight_layout()
plt.savefig("kR-PR.png", dpi=300)
plt.clf()
