"""Rate constant, vibronic-state analysis, and H/D KIE for a first-principles
double-well proton potential."""

import sys

import matplotlib.pyplot as plt
import numpy as np

from autopcet import (
    MASS_DEUTERON,
    MASS_PROTON,
    PCET,
    fit_poly8,
    write_contribution_table,
)
from autopcet.plotting import (
    figure_of,
    plot_proton_states,
    plot_state_pair_map,
    use_style,
)

# temperature, electronic coupling, reaction free energy, and reorganization
# energy; energies in eV
TEMPERATURE = 298
ELECTRONIC_COUPLING = 0.0434
REACTION_FREE_ENERGY = -0.50
REORGANIZATION_ENERGY = 1.00

# double well potentials calculated from first principles, in eV against the
# proton coordinate in angstrom
rp_data, reactant_energies, product_energies = np.loadtxt(
    "double_well_potentials.dat", unpack=True
)

reactant_potential = fit_poly8(rp_data, reactant_energies)
product_potential = fit_poly8(rp_data, product_energies)

# set up the system and do a calculation
system = PCET(
    reactant_potential,
    product_potential,
    REACTION_FREE_ENERGY,
    REORGANIZATION_ENERGY,
    electronic_coupling=ELECTRONIC_COUPLING,
)
system.calculate(MASS_PROTON, temperature=TEMPERATURE)

# ===========================================================
# Plot proton vibrational wave functions
# ===========================================================

STATES_TO_PLOT = 6

use_style()

# the reactant states on the left panel and the product states on the right,
# with the two potentials shifted so their zero-point levels line up
reactant_axes, product_axes = plot_proton_states(system, n_states=STATES_TO_PLOT)
reactant_axes.set_xlim(-0.8, 0.8)
product_axes.set_xticks(np.arange(-0.6, 0.8, 0.2))
figure_of(reactant_axes).savefig("Proton_states.png")

# which pairs of vibronic states carry the rate constant, as a map of the same
# numbers the table below prints
pair_axes = plot_state_pair_map(system, "contribution", n_states=STATES_TO_PLOT)
figure_of(pair_axes).savefig("State_pair_contributions.png")

plt.close("all")

# ===========================================================
# Analyze the contribution of each pair of vibronic states
# ===========================================================

STATES_TO_PRINT = 4

write_contribution_table(sys.stdout, system, n_states=STATES_TO_PRINT)

# ===========================================================
# Calculate rate constants for H and D, and the KIE
# ===========================================================

rate_h = system.calculate(MASS_PROTON, TEMPERATURE)
rate_d = system.calculate(MASS_DEUTERON, TEMPERATURE)

print(f"At {TEMPERATURE:d}K, k_tot(H) = {rate_h:.2e} s^-1")
print(f"At {TEMPERATURE:d}K, k_tot(D) = {rate_d:.2e} s^-1")
print(f"At {TEMPERATURE:d}K, KIE = {rate_h / rate_d:.2f}")
