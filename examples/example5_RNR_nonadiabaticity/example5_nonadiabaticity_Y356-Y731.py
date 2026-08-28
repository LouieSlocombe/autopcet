"""Vibronic couplings and nonadiabaticity analysis for the RNR Y356-Y731
interface, in the gas phase or in the protein environment.

Run once per configuration; each writes its own figures:

    python example5_nonadiabaticity_Y356-Y731.py --config gas
    python example5_nonadiabaticity_Y356-Y731.py --config env
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.interpolate import CubicSpline

from autopcet import KCAL_TO_EV, MASS_PROTON, KappaCoupling, fit_bspline
from autopcet.plotting import (
    figure_of,
    plot_crossing,
    plot_diabats_and_adiabats,
    use_style,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--config",
    choices=("gas", "env"),
    default="gas",
    help="gas phase or protein environment (default: %(default)s)",
)
config = parser.parse_args().config

# double well potentials and electronic coupling read from a file
# In this file, all energies are in kcal/mol
rp_data, reactant_energies, product_energies, coupling = np.loadtxt(
    f"Y356_Y731_config2_{config}.dat", unpack=True
)
reactant_energies *= KCAL_TO_EV
product_energies *= KCAL_TO_EV
coupling *= KCAL_TO_EV

# Spline the proton potential; this works better for smooth data
reactant_potential = fit_bspline(rp_data, reactant_energies)
product_potential = fit_bspline(rp_data, product_energies)
coupling_of_rp = CubicSpline(rp_data, coupling)

# define a finer rp grid
rp = np.linspace(-0.7, 0.7, 512)

# set up the system and perform the calculation
system = KappaCoupling(
    rp, reactant_potential(rp), product_potential(rp), coupling_of_rp(rp)
)
system.calculate(MASS_PROTON)

# ===========================================================
# Print nonadiabaticity data
# ===========================================================

print(f"tau_p = {system.tau_proton:.3e} s")
print(f"tau_e = {system.tau_electron:.3e} s")
print(f"p = {system.adiabaticity:.3e}")
print(f"kappa = {system.kappa:.3f}")

for label, coupling_value in (
    ("V_ad", system.v_adiabatic),
    ("V_nad", system.v_nonadiabatic),
    ("V_sc", system.v_semiclassical),
):
    print(
        f"{label} = {coupling_value:.2e} eV "
        f"= {coupling_value / KCAL_TO_EV:.2e} kcal/mol"
    )

# ===========================================================
# Plot the proton potentials
# ===========================================================

# this example draws its labels a size up from the house default
use_style(**{"axes.labelsize": 18, "xtick.labelsize": 16, "ytick.labelsize": 16})


def style_axes(axis: Axes, bottom: float) -> None:
    """The shared limits and ticks both figures here are drawn on."""
    axis.set_xlim(-0.75, 0.75)
    axis.set_ylim(bottom, 2.0)
    axis.set_xticks(np.arange(-0.6, 0.8, 0.2))


# the diabats, where they cross, the slopes through it, and the ground states
crossing_axes = plot_crossing(system)
style_axes(crossing_axes, bottom=0.0)
figure_of(crossing_axes).savefig(f"Proton_pot_w_slope_Y356_Y731_{config}.png")

# ===========================================================
# Plot adiabatic proton potentials
# ===========================================================

adiabat_axes = plot_diabats_and_adiabats(system, annotate_splitting=False)
style_axes(adiabat_axes, bottom=-0.2)
figure_of(adiabat_axes).savefig(f"Proton_pot_adiabatic_Y356_Y731_{config}.png")

plt.close("all")
