"""Rate constant, vibronic-state analysis, and H/D KIE for a first-principles
double-well proton potential."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from autopcet import MASS_DEUTERON, MASS_PROTON, PCET, fit_poly8

# temperature, electronic coupling, reaction free energy, and reorganization
# energy; energies in eV
TEMPERATURE = 298
ELECTRONIC_COUPLING = 0.0434
REACTION_FREE_ENERGY = -0.50
REORGANIZATION_ENERGY = 1.00

# double well potentials calculated from first principles
rp_data = np.array(
    [
        0.614,
        0.550,
        0.485,
        0.420,
        0.356,
        0.291,
        0.226,
        0.162,
        0.097,
        0.032,
        -0.032,
        -0.097,
        -0.162,
        -0.226,
        -0.291,
        -0.356,
        -0.420,
        -0.485,
        -0.550,
        -0.614,
    ]
)
reactant_energies = np.array(
    [
        5.283,
        4.534,
        4.061,
        3.794,
        3.673,
        3.646,
        3.680,
        3.688,
        3.634,
        3.646,
        3.602,
        3.513,
        3.392,
        3.257,
        3.138,
        3.078,
        3.140,
        3.413,
        4.022,
        5.144,
    ]
)
product_energies = np.array(
    [
        4.847,
        4.134,
        3.706,
        3.495,
        3.452,
        3.509,
        3.620,
        3.801,
        3.949,
        4.073,
        4.157,
        4.194,
        4.189,
        4.157,
        4.128,
        4.144,
        4.270,
        4.597,
        5.250,
        6.411,
    ]
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

# wave functions and energies share an axis, so scale the wave functions down
WAVEFUNCTION_SCALE = 0.06

rp = system.rp

# align the zero-point energy of the reactant and product states in this plot
zero_point_gap = system.product_energies[0] - system.reactant_energies[0]
reactant_shift, product_shift = max(zero_point_gap, 0.0), max(-zero_point_gap, 0.0)


def plot_states(
    axis: Axes,
    energies: np.ndarray,
    wavefunctions: np.ndarray,
    shift: float,
    color: str,
) -> None:
    """Draw the lowest vibrational states as wave functions on their levels."""
    for i, (energy, wavefunction) in enumerate(
        zip(energies[:STATES_TO_PLOT], wavefunctions[:STATES_TO_PLOT], strict=True)
    ):
        # flip the wave function so its largest amplitude points up
        sign = 1 if np.abs(np.max(wavefunction)) > np.abs(np.min(wavefunction)) else -1
        level = energy + shift
        curve = level + WAVEFUNCTION_SCALE * sign * wavefunction
        axis.plot(rp, curve, f"{color}-", lw=1, alpha=(1 - 0.12 * i))
        axis.fill_between(rp, curve, level, color=color, alpha=0.4)


fig = plt.figure(figsize=(9, 4.5))
gs = fig.add_gridspec(ncols=2, wspace=0)
ax1, ax2 = gs.subplots(sharex=True, sharey=True)

for axis in (ax1, ax2):
    axis.plot(rp, reactant_potential(rp) + reactant_shift, "b", lw=2)
    axis.plot(rp, product_potential(rp) + product_shift, "r", lw=2)
    axis.set_xlabel(r"$r_{\rm p}\ /\ \rm\AA$", fontsize=16)
    axis.tick_params(labelsize=14)

plot_states(
    ax1, system.reactant_energies, system.reactant_wavefunctions, reactant_shift, "b"
)
plot_states(
    ax2, system.product_energies, system.product_wavefunctions, product_shift, "r"
)

ax1.set_xlim(-0.8, 0.8)
ax1.set_ylim(0, 1.3)
ax1.set_ylabel(r"$E$ / eV", fontsize=16)

ax2.set_xlim(-0.65, 0.75)
ax2.set_ylim(0, 1.3)
ax2.set_xticks(np.arange(-0.6, 0.8, 0.2))

plt.tight_layout()
plt.savefig("Proton_states.png", dpi=300)
plt.clf()

# ===========================================================
# Analyze the contribution of each pair of vibronic states
# ===========================================================

STATES_TO_PRINT = 4

fractional_contribution = system.rate_contributions / system.total_rate_constant

print("\n" + "=" * 130)
print("(u, v)\t\tP_u\t\t\t|S_uv|^2\t\tDelta G_uv / eV\t\tDelta G^#_uv / eV\t% Contrib.")
print("-" * 130)
for u in range(STATES_TO_PRINT):
    for v in range(STATES_TO_PRINT):
        print(
            f"({u:d}, {v:d})\t\t{system.populations[u]:.3e}\t\t"
            f"{system.overlaps[u, v] ** 2:.3e}\t\t"
            f"{system.pair_free_energies[u, v]:+.3f}\t\t\t"
            f"{system.pair_activation_energies[u, v]:.3f}\t\t\t"
            f"{fractional_contribution[u, v]:.3f}"
        )

print("=" * 130 + "\n")

# ===========================================================
# Calculate rate constants for H and D, and the KIE
# ===========================================================

rate_h = system.calculate(MASS_PROTON, TEMPERATURE)
rate_d = system.calculate(MASS_DEUTERON, TEMPERATURE)

print(f"At {TEMPERATURE:d}K, k_tot(H) = {rate_h:.2e} s^-1")
print(f"At {TEMPERATURE:d}K, k_tot(D) = {rate_d:.2e} s^-1")
print(f"At {TEMPERATURE:d}K, KIE = {rate_h / rate_d:.2f}")
