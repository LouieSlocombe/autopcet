"""Heterogeneous electrochemical PCET for CoTPP on graphene, combining the EDL
model with a density-of-states average over the electrode levels."""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from autopcet import (
    BOLTZMANN,
    HARTREE_TO_EV,
    KCAL_TO_EV,
    MASS_DEUTERON,
    MASS_PROTON,
    PCET,
    FloatArray,
    ScalarOrArrayFunction,
    fermi_distribution,
    fit_poly6,
    make_edl_model,
    write_contribution_table,
)
from autopcet.plotting import distance_colors, figure_of, plot_edl_profile, use_style

# ===========================================================
# Define the thermodynamic parameters, taken from
# Hutchison et. al. ACS Catal. 2024, 19, 14363-14372.
# ===========================================================

# Donor-Acceptor distance values sampled in calculations
distances = np.array(
    [
        3.057,
        3.157,
        3.207,
        3.257,
        3.307,
        3.357,
        3.379,
        3.407,
        3.457,
        3.507,
        3.557,
        3.607,
        3.657,
        3.757,
        3.857,
        3.957,
        4.057,
        4.157,
        4.257,
    ]
)

REORGANIZATION_ENERGY = 0.83  # eV, inner sphere contribution only
ELECTRONIC_COUPLING = 0.10  # eV, not needed for the KIE
TEMPERATURE = 300  # K

# electronic density of states for a pristine graphene slab from a periodic
# planewave DFT calculation, in N_states eV^-1 atom^-1
graphene_dos = np.genfromtxt(
    "graphene_DOS_norm_gauss.csv", delimiter=",", skip_header=1
)
# electronic energy levels to numerically integrate over
electrode_energies = graphene_dos[:, 0]
density_of_states = graphene_dos[:, 1]

# Delta G0 is the free energy change of 1/2 H_2 + CoTPP --> CoHTPP. Further
# corrections follow from the applied electrode potential via the EDL model.
FREE_ENERGY_H = 0.55  # eV
FREE_ENERGY_D = 0.55 - 0.009296397  # eV

N_STATES = 10  # how many states to include in the rate constant calculation
STATES_TO_SHOW = 9  # how many states to print

# ===========================================================
# Define the electric double layer model and the non-bonded
# parameters used for the work terms
# ===========================================================
D_IHL = 3.6  # angstrom
D_OHL = 3.5  # angstrom
EPS_IHL = 2.7
EPS_STATIC = 78.0
EPS_OPTICAL = 1.78
DIPOLE: float | Literal["calculate"] = "calculate"
WATER_DENSITY = 0.9970470  # g/cm^3
WATER_MOLAR_MASS = 18.01528  # g/mol
ION_CONCENTRATION = 0.5  # mol/L
EDL_CAPACITANCE = 15  # microfarad/cm^2
PZFC_VS_SHE = 0.04  # V


def make_potential_drop(potential_vs_she: float) -> ScalarOrArrayFunction:
    """EDL potential drop as a function of distance, at a given electrode potential."""
    return make_edl_model(
        potential_vs_she,
        D_IHL,
        D_OHL,
        EPS_IHL,
        EPS_STATIC,
        EPS_OPTICAL,
        DIPOLE,
        WATER_DENSITY,
        WATER_MOLAR_MASS,
        ION_CONCENTRATION,
        EDL_CAPACITANCE,
        PZFC_VS_SHE,
        verbose=False,
    )


use_style()

survey_potentials = np.arange(-1.0, 0.0, 0.2)
survey_grid = np.arange(0, 10, 0.1)
colors = distance_colors(len(survey_potentials))

# the Helmholtz planes are the same for every applied potential, so only the
# last call needs to mark them
edl_axes = None
for potential_vs_she, color in zip(survey_potentials, colors, strict=True):
    last = potential_vs_she == survey_potentials[-1]
    edl_axes = plot_edl_profile(
        make_potential_drop(potential_vs_she),
        survey_grid,
        edl_axes,
        d_ihl=D_IHL if last else None,
        d_ohl=D_OHL if last else None,
        label=f"$E = {potential_vs_she:.1f}$V",
        color=color,
    )

assert edl_axes is not None
edl_axes.set_xlim(0, 10)
edl_axes.legend(loc=4, frameon=True, framealpha=1)

figure_of(edl_axes).savefig("EDL_model.png")
plt.close("all")


# Buckingham potentials using coefficients determined from fitting DFT data.
def reactant_work[T: (float, FloatArray)](distance: T) -> T:
    """CoTPP + H_3O^+ non-bonded interaction, in eV."""
    energy: T = KCAL_TO_EV * (
        692272.09 * np.exp(-distance / 0.25730094) - 3699.5922 / distance**6
    )
    return energy


def product_work[T: (float, FloatArray)](distance: T) -> T:
    """CoHTPP + H_2O non-bonded interaction, in eV."""
    energy: T = KCAL_TO_EV * (
        54305.39 * np.exp(-distance / 0.39709137) - 19539.245 / distance**6
    )
    return energy


def tafel(
    applied_potential: FloatArray, prefactor: float, intercept: float
) -> FloatArray:
    """General Tafel equation; the prefactor is alpha*F/RT.

    Fitting ln[k_H](E) with this yields the transfer coefficient alpha.
    """
    values: FloatArray = -prefactor * applied_potential + intercept
    return values


# ===========================================================
# Read data from files
# ===========================================================

reactant_potentials = []
product_potentials = []

# read proton potentials from .csv files
# the proton potentials are directly from the published work
for distance in distances:
    reactant = pd.read_csv(
        f"proton_potentials/rDA_{distance:.3f}_R.csv",
        sep=",",
        header=0,
        engine="python",
    )
    product = pd.read_csv(
        f"proton_potentials/rDA_{distance:.3f}_P.csv",
        sep=",",
        header=0,
        engine="python",
    )

    # the energy unit in the .csv files is Hartree atomic units, convert it to eV
    reactant_energies = reactant["Reactant"] * HARTREE_TO_EV
    product_energies = product["Product"] * HARTREE_TO_EV

    # smooth the data by fitting to polynomials; for these data sets a 6th order
    # polynomial fits better than a spline
    reactant_potentials.append(fit_poly6(reactant["x"], reactant_energies))
    product_potentials.append(fit_poly6(product["x"], product_energies))

# ===========================================================
# Calculate the KIE of electrochemical PCET of CoTPP at different
# R and applied potentials
# ===========================================================


def make_system(index: int, free_energy: float) -> PCET:
    """A PCET system for the proton potentials sampled at distance ``index``."""
    return PCET(
        reactant_potentials[index],
        product_potentials[index],
        reaction_free_energy=free_energy,
        reorganization_energy=REORGANIZATION_ENERGY,
        electronic_coupling=ELECTRONIC_COUPLING,
        n_states=N_STATES,
        r_min=-1.5,
        r_max=1.5,
    )


rates_h = np.zeros(len(distances))
rates_d = np.zeros(len(distances))

VOLT_PER_DECADE = 8.31446 * TEMPERATURE / 96485.33  # RT/F, units J/C = V
PH = 0  # converts between the RHE and SHE scales, on which the drop is defined

# the occupied density of states does not depend on R or the applied potential
occupied_dos = density_of_states * fermi_distribution(
    electrode_energies, temperature=TEMPERATURE
)

applied_potentials = np.arange(-0.7, -0.49, 0.02)

# proton donor-acceptor distance grid spanning the sampled distances
fine_grid = np.linspace(distances[0], distances[-1], 200)

log_rates_h = np.zeros(len(applied_potentials))
log_rates_d = np.zeros(len(applied_potentials))

# Truncate the rate-constant log and write its header once, up front; every
# applied potential below appends its own block of rows to it.
with open("kPCET_data.log", "w") as log:
    log.write("# E / V\t R_PT/A\tk_H/s^-1\tk_D/s^-1\n")

# Loop over applied potential
for n, applied_potential in enumerate(applied_potentials):
    potential_drop = make_potential_drop(applied_potential)

    # Loop over proton-donor acceptor distances
    for i, distance in enumerate(distances):
        work_reactant = reactant_work(distance) + potential_drop(distance)
        work_product = product_work(distance)

        print(f"Calculating... R = {distance:.3f}A")
        # reaction_free_energy is set per electrode level inside the loop below;
        # these instances just need a placeholder to be constructed with.
        system_h = make_system(i, FREE_ENERGY_H)
        system_d = make_system(i, FREE_ENERGY_D)

        rates_h_of_energy = np.zeros(len(electrode_energies))
        rates_d_of_energy = np.zeros(len(electrode_energies))

        # shifts shared by every electrode level at this R and applied potential
        work_shift = (
            applied_potential
            + work_product
            - work_reactant
            + VOLT_PER_DECADE * np.log(10) * PH
        )
        deuterium_shift = work_shift - VOLT_PER_DECADE * np.log(10) * (14.0 - 14.87)

        for j, electrode_energy in enumerate(electrode_energies):
            # update Delta G for a given epsilon
            system_h.reaction_free_energy = (
                FREE_ENERGY_H + work_shift - electrode_energy
            )
            system_d.reaction_free_energy = (
                FREE_ENERGY_D + deuterium_shift - electrode_energy
            )

            rates_h_of_energy[j] = system_h.calculate(
                mass=MASS_PROTON, temperature=TEMPERATURE, reuse_states=True
            )
            rates_d_of_energy[j] = system_d.calculate(
                mass=MASS_DEUTERON, temperature=TEMPERATURE, reuse_states=True
            )

            if (
                np.abs(applied_potential - -0.66) <= 1e-3
                and np.abs(electrode_energy - 0.005) <= 1e-3
            ):
                with open(
                    f"rate_constant_contribution_R{distance:.3f}A.log", "w"
                ) as log:
                    log.write(
                        f"\nR = {distance:.3f}A, epsilon = 0.005, E_appl = -0.66\n"
                    )
                    write_contribution_table(
                        log, system_h, label="H", n_states=STATES_TO_SHOW
                    )
                    write_contribution_table(
                        log, system_d, label="D", n_states=STATES_TO_SHOW
                    )

        # Numerical integration over the electrode energy levels
        rates_h[i] = simpson(occupied_dos * rates_h_of_energy, x=electrode_energies)
        rates_d[i] = simpson(occupied_dos * rates_d_of_energy, x=electrode_energies)

    # Append the PCET rate constants for H and D at each R to the log
    with open("kPCET_data.log", "a") as log:
        for distance, rate_h, rate_d in zip(distances, rates_h, rates_d, strict=True):
            log.write(
                f"{applied_potential:.2f}\t\t{distance:.3f}\t\t{rate_h:.4e}\t{rate_d:.4e}\n"
            )

    # ===========================================================
    # Calculate P(R): the potential-dependent concentration of proton donors at R
    # ===========================================================

    BULK_CONCENTRATION = 1  # molar, for pH = 0 conditions
    work_of_distance = reactant_work(fine_grid) + potential_drop(fine_grid)
    distribution = (
        np.exp(-work_of_distance / (BOLTZMANN * TEMPERATURE)) * BULK_CONCENTRATION
    )

    # ===========================================================
    # Thermally average the PCET rate constant over R
    # ===========================================================

    # the integration should run from 0 to infinity, but in practice we integrate
    # over the interval where the integrand has reached zero at both limits
    rates_h_fine = interp1d(
        distances, rates_h, kind="linear", fill_value="extrapolate"
    )(fine_grid)
    rates_d_fine = interp1d(
        distances, rates_d, kind="linear", fill_value="extrapolate"
    )(fine_grid)

    # perform thermal average and print the final results
    dominant_distance_h = fine_grid[find_peaks(distribution * rates_h_fine)[0]]
    dominant_distance_d = fine_grid[find_peaks(distribution * rates_d_fine)[0]]

    average_rate_h = simpson(distribution * rates_h_fine, x=fine_grid)
    average_rate_d = simpson(distribution * rates_d_fine, x=fine_grid)

    print()
    print(f"Applied Potential= {applied_potential:.2f} V vs SHE")
    print(f"Dominant R for H = {dominant_distance_h[0]:.2f}A")
    print(f"Dominant R for D = {dominant_distance_d[0]:.2f}A")
    print(f"k_H_tot = {average_rate_h:.4e} s^-1")
    print(f"k_D_tot = {average_rate_d:.4e} s^-1")
    print(f"KIE = {average_rate_h / average_rate_d:.2f}")
    print()

    log_rates_h[n] = np.log(average_rate_h)
    log_rates_d[n] = np.log(average_rate_d)

tafel_h = curve_fit(tafel, applied_potentials, log_rates_h)[0]
tafel_d = curve_fit(tafel, applied_potentials, log_rates_d)[0]

# Remove the factor of F/RT from the Tafel prefactor
transfer_coefficient_h = tafel_h[0] * VOLT_PER_DECADE
transfer_coefficient_d = tafel_d[0] * VOLT_PER_DECADE

print(f"The transfer coefficient for protons is: {transfer_coefficient_h:.4f}")
print(f"The transfer coefficient for deuterons is: {transfer_coefficient_d:.4f}")
