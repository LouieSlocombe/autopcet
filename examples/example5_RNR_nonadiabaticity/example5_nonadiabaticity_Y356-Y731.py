"""Vibronic couplings and nonadiabaticity analysis for the RNR Y356-Y731
interface, in the gas phase or in the protein environment.

Run once per configuration; each writes its own figures:

    python example5_nonadiabaticity_Y356-Y731.py --config gas
    python example5_nonadiabaticity_Y356-Y731.py --config env
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

from autopcet import KCAL_TO_EV, MASS_PROTON, KappaCoupling, fit_bspline

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

# wave functions and energies share an axis, so scale the wave functions down
WAVEFUNCTION_SCALE = 0.06


def plot_ground_state(energy: float, wavefunction: np.ndarray, color: str) -> None:
    """Draw the ground vibrational wave function sitting on its energy level."""
    # flip the wave function so its largest amplitude points up
    sign = 1 if np.abs(np.max(wavefunction)) > np.abs(np.min(wavefunction)) else -1
    curve = energy + WAVEFUNCTION_SCALE * sign * wavefunction
    plt.plot(rp, curve, f"{color}-", lw=1, alpha=1)
    plt.fill_between(rp, curve, energy, color=color, alpha=0.4)


def style_axes(bottom: float) -> None:
    """Apply the shared axis limits, labels, and tick sizes."""
    plt.xlim(-0.75, 0.75)
    plt.ylim(bottom, 2.0)
    plt.xlabel(r"$r_{\rm p}\ /\ \rm\AA$", fontsize=18)
    plt.ylabel(r"$E$ / eV", fontsize=18)
    plt.xticks(np.arange(-0.6, 0.8, 0.2), fontsize=16)
    plt.yticks(fontsize=16)


plt.plot(rp, system.shifted_reactant_potential, "b", lw=2)
plt.plot(rp, system.shifted_product_potential, "r", lw=2)

# mark the crossing point and the slopes of the two diabats through it
plt.plot(system.crossing_rp, system.crossing_energy, "o", ms=5, mew=2, mfc="k", mec="k")

tangent_rp = np.linspace(system.crossing_rp - 0.1, system.crossing_rp + 0.1, 100)
for slope in (system.reactant_slope, system.product_slope):
    tangent = slope * (tangent_rp - system.crossing_rp) + system.crossing_energy
    plt.plot(tangent_rp, tangent, "k--", lw=1.5)

plot_ground_state(
    system.shifted_reactant_energies[0], system.reactant_wavefunctions[0], "b"
)
plot_ground_state(
    system.shifted_product_energies[0], system.product_wavefunctions[0], "r"
)

style_axes(bottom=0.0)
plt.tight_layout()
plt.savefig(f"Proton_pot_w_slope_Y356_Y731_{config}.png", dpi=300)
plt.clf()

# ===========================================================
# Plot adiabatic proton potentials
# ===========================================================

plt.plot(rp, system.shifted_reactant_potential, "b", lw=2)
plt.plot(rp, system.shifted_product_potential, "r", lw=2)
plt.plot(rp, system.ground_adiabat, "k--", lw=1.5)
plt.plot(rp, system.excited_adiabat, "k--", lw=1.5)

style_axes(bottom=-0.2)
plt.tight_layout()
plt.savefig(f"Proton_pot_adiabatic_Y356_Y731_{config}.png", dpi=300)
