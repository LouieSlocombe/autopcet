"""Publication-ready figures for the quantities :mod:`autopcet` computes.

Every function takes an optional axes to draw on and returns the axes it drew,
so figures compose: build the grid, hand each panel to a function, then label
and save it yourself. Nothing here saves, shows, or closes a figure, and
nothing sets an axis limit that is not read off the data.

matplotlib is an optional dependency, so this subpackage is imported
explicitly rather than re-exported from :mod:`autopcet`::

    from autopcet.plotting import plot_proton_states, use_style

Importing it costs nothing without matplotlib; the error, naming the
``plotting`` extra, comes when a function that needs it is called.
"""

from ._mpl import MATPLOTLIB_HINT, add_legend, prepare_axes, prepare_axes_grid
from .analysis import (
    plot_crossing,
    plot_diabats_and_adiabats,
    plot_distance_scan,
    plot_edl_profile,
    plot_proton_scan,
)
from .averaging import plot_thermal_average
from .potentials import (
    plot_isotope_states,
    plot_potential_family,
    plot_potential_fit,
    plot_proton_states,
    plot_state_ladder,
)
from .states import (
    QUANTITIES,
    plot_populations,
    plot_state_pair_grid,
    plot_state_pair_map,
)
from .style import (
    ADIABAT_COLOR,
    DISTANCE_LABEL,
    DISTRIBUTION_COLOR,
    ENERGY_LABEL,
    MIN_STATE_ALPHA,
    PRODUCT_COLOR,
    PROTON_LABEL,
    RATE_COLOR,
    RATE_LABEL,
    REACTANT_COLOR,
    STYLE,
    WAVEFUNCTION_FADE,
    WAVEFUNCTION_SCALE,
    distance_colors,
    style_context,
    use_style,
)
from .sweeps import (
    plot_arrhenius,
    plot_kie_vs_temperature,
    plot_rate_vs_driving_force,
)

__all__ = [
    "ADIABAT_COLOR",
    "DISTANCE_LABEL",
    "DISTRIBUTION_COLOR",
    "ENERGY_LABEL",
    "MATPLOTLIB_HINT",
    "MIN_STATE_ALPHA",
    "PRODUCT_COLOR",
    "PROTON_LABEL",
    "QUANTITIES",
    "RATE_COLOR",
    "RATE_LABEL",
    "REACTANT_COLOR",
    "STYLE",
    "WAVEFUNCTION_FADE",
    "WAVEFUNCTION_SCALE",
    "add_legend",
    "distance_colors",
    "plot_arrhenius",
    "plot_crossing",
    "plot_diabats_and_adiabats",
    "plot_distance_scan",
    "plot_edl_profile",
    "plot_isotope_states",
    "plot_kie_vs_temperature",
    "plot_populations",
    "plot_potential_family",
    "plot_potential_fit",
    "plot_proton_scan",
    "plot_proton_states",
    "plot_rate_vs_driving_force",
    "plot_state_ladder",
    "plot_state_pair_grid",
    "plot_state_pair_map",
    "plot_thermal_average",
    "prepare_axes",
    "prepare_axes_grid",
    "style_context",
    "use_style",
]
