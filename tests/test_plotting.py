"""Tests for the figures in :mod:`autopcet.plotting`.

These assert on the artists a function put on the axes -- lines, filled bands,
images, bars, texts -- rather than on rendered pixels, so they are fast and say
something legible when they fail. Two of them rasterize, to prove that a figure
matplotlib accepted the instructions for is one it can actually draw.
"""

import io
import sys
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from autopcet import (
    BOLTZMANN,
    MASS_DEUTERON,
    MASS_PROTON,
    PCET,
    KappaCoupling,
    donor_acceptor_distribution,
)
from autopcet._types import FloatArray, PotentialFunction
from autopcet.ase_io import DistanceScan, ProtonScan
from autopcet.plotting import (
    MIN_STATE_ALPHA,
    QUANTITIES,
    STYLE,
    add_legend,
    distance_colors,
    figure_of,
    plot_arrhenius,
    plot_crossing,
    plot_diabats_and_adiabats,
    plot_distance_scan,
    plot_edl_profile,
    plot_isotope_states,
    plot_kie_vs_temperature,
    plot_populations,
    plot_potential_family,
    plot_potential_fit,
    plot_proton_scan,
    plot_proton_states,
    plot_rate_vs_driving_force,
    plot_state_ladder,
    plot_state_pair_grid,
    plot_state_pair_map,
    plot_thermal_average,
    prepare_axes,
    prepare_axes_grid,
    style_context,
    use_style,
)
from autopcet.plotting.potentials import _zero_point_shifts

GRID: FloatArray = np.linspace(-1.0, 1.0, 64)


# matplotlib's accessors are typed very loosely -- a line's data comes back as
# a union wide enough that numpy will not take it, and anything optional comes
# back as ``| None``. These narrow it once, here, rather than at every call.


def xdata(line: Line2D) -> FloatArray:
    """A line's x data, as a float array."""
    return np.asarray(line.get_xdata(), dtype=np.float64)


def ydata(line: Line2D) -> FloatArray:
    """A line's y data, as a float array."""
    return np.asarray(line.get_ydata(), dtype=np.float64)


def legend_labels(ax: Axes) -> set[str]:
    """The labels on the axes' legend, which has to exist by the time this runs."""
    legend = ax.get_legend()
    assert legend is not None
    return {text.get_text() for text in legend.get_texts()}


def alphas(ax: Axes) -> list[float]:
    """The alpha of every line on the axes, unset counting as fully opaque."""
    values = [line.get_alpha() for line in ax.lines]
    return [1.0 if value is None else float(value) for value in values]


def renders(ax: Axes) -> None:
    """Draw the figure the axes belongs to, failing if matplotlib cannot."""
    figure_of(ax).savefig(io.BytesIO(), format="png")


# ===========================================================================
# Style
# ===========================================================================


def test_house_style_carries_the_settings_the_examples_set_by_hand() -> None:
    """use_style applies the 300 dpi and the label sizes, globally."""
    with plt.rc_context():
        use_style()
        assert plt.rcParams["savefig.dpi"] == STYLE["savefig.dpi"] == 300
        assert plt.rcParams["axes.labelsize"] == 16
        assert plt.rcParams["xtick.labelsize"] == 14


def test_house_style_takes_overrides() -> None:
    """An override wins over the house setting it replaces."""
    with plt.rc_context():
        use_style(**{"axes.labelsize": 22})
        assert plt.rcParams["axes.labelsize"] == 22


def test_style_context_puts_the_previous_settings_back() -> None:
    """The context manager leaves the global rcParams as it found them."""
    before = plt.rcParams["axes.labelsize"]

    with style_context(**{"axes.labelsize": 30}):
        assert plt.rcParams["axes.labelsize"] == 30

    assert plt.rcParams["axes.labelsize"] == before


def test_distance_colors_ramps_over_the_requested_number() -> None:
    """One colour per distance, running from one end of the ramp to the other."""
    colors = distance_colors(8)

    assert len(colors) == 8
    assert len(set(colors)) == 8
    assert all(len(colour) == 3 for colour in colors)


def test_distance_colors_needs_something_to_colour() -> None:
    """An empty ramp is a mistake, not an empty list."""
    with pytest.raises(ValueError, match="at least one colour"):
        distance_colors(0)


# ===========================================================================
# Axis plumbing
# ===========================================================================


def test_prepare_axes_makes_a_figure_when_it_is_given_none() -> None:
    """Called without axes, it draws on a fresh single-panel figure."""
    ax = prepare_axes()

    assert figure_of(ax) is not None


def test_prepare_axes_draws_on_the_axes_it_is_given() -> None:
    """Called with axes, it hands back that same object untouched."""
    _, existing = plt.subplots()

    assert prepare_axes(existing) is existing


def test_prepare_axes_grid_makes_the_panels_it_was_asked_for() -> None:
    """A grid comes back flattened, row by row."""
    panels = prepare_axes_grid(None, "Testing", 2, 3)

    assert len(panels) == 6


def test_prepare_axes_grid_refuses_a_grid_of_the_wrong_size() -> None:
    """Handing in three panels for a four-panel figure is an error."""
    _, given = plt.subplots(1, 3)

    with pytest.raises(ValueError, match="Expected 4 axes"):
        prepare_axes_grid(list(given), "Testing", 2, 2)


def test_a_legend_is_only_added_once_something_carries_a_label() -> None:
    """matplotlib warns about an empty legend, and warnings are errors here."""
    _, ax = plt.subplots()
    ax.plot(GRID, GRID)
    add_legend(ax)
    assert ax.get_legend() is None

    ax.plot(GRID, GRID, label="labelled")
    add_legend(ax)
    assert ax.get_legend() is not None


def test_plotting_without_matplotlib_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The error says which extra installs matplotlib, as the ASE one does."""
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)

    with pytest.raises(ImportError, match=r"autopcet\[plotting\]"):
        prepare_axes(None, "Drawing a figure")


# ===========================================================================
# Proton potentials and their vibrational states
# ===========================================================================


def test_a_state_ladder_draws_a_curve_and_a_band_for_every_state() -> None:
    """Each state contributes one line and one filled band."""
    _, ax = plt.subplots()
    energies = np.arange(4.0)
    wavefunctions = np.tile(np.exp(-(GRID**2)), (4, 1))

    plot_state_ladder(ax, GRID, energies, wavefunctions, "b", n_states=3)

    assert len(ax.lines) == 3
    assert len(ax.collections) == 3


def test_a_state_ladder_flips_each_wave_function_largest_lobe_up() -> None:
    """A state whose largest lobe points down is drawn the other way up."""
    _, ax = plt.subplots()
    downward = -np.exp(-8 * GRID**2)

    plot_state_ladder(
        ax, GRID, np.array([0.5]), downward[np.newaxis, :], "b", n_states=1
    )

    assert np.max(ydata(ax.lines[0])) > 0.5


def test_a_state_ladder_stops_fading_before_the_alpha_runs_out() -> None:
    """The examples' fade would ask for a negative alpha from the tenth state.

    ``1 - 0.12 * i`` is -0.08 at ``i = 9``, which matplotlib rejects outright,
    so the shared helper floors it.
    """
    _, ax = plt.subplots()
    wavefunctions = np.tile(np.exp(-(GRID**2)), (12, 1))

    plot_state_ladder(ax, GRID, np.arange(12.0), wavefunctions, "b", n_states=12)

    faded = alphas(ax)
    assert all(0.0 <= alpha <= 1.0 for alpha in faded)
    assert faded[-1] == pytest.approx(MIN_STATE_ALPHA)
    renders(ax)


def test_a_state_ladder_needs_at_least_one_state() -> None:
    """Asking for no states is a mistake, not an empty figure."""
    _, ax = plt.subplots()

    with pytest.raises(ValueError, match="at least one state"):
        plot_state_ladder(ax, GRID, np.zeros(2), np.zeros((2, 64)), "b", n_states=0)


def test_proton_states_draws_both_diabats_with_their_ladders(
    calculated_system: PCET,
) -> None:
    """Each panel carries its potential plus one line per state."""
    reactant_axes, product_axes = plot_proton_states(calculated_system, n_states=4)

    for panel in (reactant_axes, product_axes):
        assert len(panel.lines) == 5
        assert len(panel.collections) == 4

    assert reactant_axes.get_ylabel() == "$E$ / eV"
    renders(reactant_axes)


@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
def test_proton_states_stacks_the_panels_either_way(
    calculated_system: PCET, orientation: Literal["horizontal", "vertical"]
) -> None:
    """Both layouts return the two panels, reactant first."""
    panels = plot_proton_states(calculated_system, n_states=3, orientation=orientation)

    assert len(panels) == 2


def test_proton_states_draws_on_the_panels_it_is_given(
    calculated_system: PCET,
) -> None:
    """Given a grid, it uses it rather than making its own figure."""
    _, given = plt.subplots(1, 2)

    panels = plot_proton_states(calculated_system, list(given), n_states=3)

    assert panels == list(given)


def test_proton_states_fits_the_energy_axis_around_the_states(
    calculated_system: PCET,
) -> None:
    """Autoscaling keeps the diabats' walls from squashing the ladder flat."""
    fitted = plot_proton_states(calculated_system, n_states=4)[0]
    whole = plot_proton_states(calculated_system, n_states=4, autoscale=False)[0]

    fitted_span = np.ptp(fitted.get_ylim())
    assert fitted_span < np.ptp(whole.get_ylim()) / 2


def test_aligning_the_zero_point_puts_the_two_ground_levels_together(
    calculated_system: PCET,
) -> None:
    """The shift lines up the zero-point levels, and is zero when turned off."""
    reactant_shift, product_shift = _zero_point_shifts(calculated_system, True)

    assert calculated_system.reactant_energies[0] + reactant_shift == pytest.approx(
        calculated_system.product_energies[0] + product_shift
    )
    assert _zero_point_shifts(calculated_system, False) == (0.0, 0.0)


def test_plotting_a_system_with_no_states_says_to_calculate_it_first(
    system: PCET,
) -> None:
    """A system straight out of the constructor has nothing to draw yet."""
    with pytest.raises(ValueError, match="no proton states yet"):
        plot_proton_states(system)


def test_isotope_states_draws_a_ladder_per_mass(calculated_system: PCET) -> None:
    """H and D go on the same panels in different line styles."""
    reactant_axes, _ = plot_isotope_states(calculated_system, n_states=3)

    # three states per mass, twice over, plus the potential itself
    assert len(reactant_axes.lines) == 7
    assert legend_labels(reactant_axes) == {"H", "D"}


def test_isotope_states_needs_a_label_for_every_mass(
    calculated_system: PCET,
) -> None:
    """Masses and labels are zipped, so a mismatch is caught up front."""
    with pytest.raises(ValueError, match="masses but"):
        plot_isotope_states(
            calculated_system, masses=(MASS_PROTON, MASS_DEUTERON), labels=("H",)
        )


def test_a_potential_family_draws_one_curve_per_member(
    reactant_potential: PotentialFunction, product_potential: PotentialFunction
) -> None:
    """Two potentials, two curves, each in its own colour off the ramp."""
    ax = plot_potential_family(
        GRID,
        [reactant_potential, product_potential],
        labels=["reactant", "product"],
    )

    assert len(ax.lines) == 2
    assert ax.lines[0].get_color() != ax.lines[1].get_color()
    assert ax.get_legend() is not None


def test_a_potential_family_scales_the_energies_off_electronvolts(
    reactant_potential: PotentialFunction,
) -> None:
    """energy_scale is how example 2 plots the same curve in kcal/mol."""
    plain = plot_potential_family(GRID, [reactant_potential])
    scaled = plot_potential_family(
        GRID,
        [reactant_potential],
        energy_scale=10.0,
    )

    assert np.allclose(ydata(scaled.lines[0]), 10.0 * ydata(plain.lines[0]))


def test_a_potential_family_needs_a_colour_for_every_member(
    reactant_potential: PotentialFunction, product_potential: PotentialFunction
) -> None:
    """A hand-picked palette has to be as long as the family it colours."""
    with pytest.raises(ValueError, match="potentials but"):
        plot_potential_family(
            GRID,
            [reactant_potential, product_potential],
            colors=["b"],
        )


@pytest.mark.parametrize("residuals", [True, False])
def test_a_fit_is_drawn_over_the_data_it_was_fitted_to(
    reactant_potential: PotentialFunction, residuals: bool
) -> None:
    """The data goes on as markers, the fit as a curve, the error underneath."""
    rp = np.linspace(-0.6, 0.6, 12)
    energies = np.asarray(reactant_potential(rp))

    panels = plot_potential_fit(
        rp,
        energies,
        reactant_potential,
        residuals=residuals,
    )

    assert len(panels) == (2 if residuals else 1)
    assert len(panels[0].lines) == 2
    if residuals:
        # the zero line the residuals are measured against, and the residuals
        assert len(panels[1].lines) == 2
    assert panels[-1].get_xlabel().startswith("$r_")


# ===========================================================================
# Vibronic state-pair maps
# ===========================================================================


@pytest.mark.parametrize("quantity", sorted(QUANTITIES))
def test_every_state_pair_matrix_can_be_mapped(
    calculated_system: PCET, quantity: str
) -> None:
    """Each named quantity draws a square image over the states shown."""
    ax = plot_state_pair_map(calculated_system, quantity, n_states=4)

    (image,) = ax.images
    values = image.get_array()
    assert values is not None
    assert values.shape == (4, 4)
    assert ax.get_xlabel() == "product state $v$"


def test_a_state_pair_map_refuses_a_quantity_it_does_not_know(
    calculated_system: PCET,
) -> None:
    """The error lists what it could have been asked for instead."""
    with pytest.raises(ValueError, match="Unknown quantity 'kappa'"):
        plot_state_pair_map(calculated_system, "kappa")


def test_a_state_pair_map_writes_the_value_into_every_cell(
    calculated_system: PCET,
) -> None:
    """Annotation is one text per cell, and can be turned off."""
    annotated = plot_state_pair_map(calculated_system, n_states=3)
    plain = plot_state_pair_map(calculated_system, n_states=3, annotate=False)

    assert len(annotated.texts) == 9
    assert len(plain.texts) == 0
    renders(annotated)


def test_a_state_pair_map_without_a_colour_bar_titles_the_axes(
    calculated_system: PCET,
) -> None:
    """Dropping the colour bar has to leave the quantity named somewhere."""
    ax = plot_state_pair_map(calculated_system, "overlap", colorbar=False)

    assert ax.get_title() == QUANTITIES["overlap"].label


def test_a_state_pair_map_needs_a_calculated_system(system: PCET) -> None:
    """There are no matrices to map before calculate has run."""
    with pytest.raises(ValueError, match="no proton states yet"):
        plot_state_pair_map(system)


def test_shares_of_a_vanished_rate_constant_are_refused(
    calculated_system: PCET,
) -> None:
    """Dividing by a total that underflowed would quietly produce NaNs."""
    calculated_system.total_rate_constant = 0.0

    with pytest.raises(ValueError, match="total rate constant is zero"):
        plot_state_pair_map(calculated_system, "contribution")


def test_a_state_pair_grid_maps_every_quantity(calculated_system: PCET) -> None:
    """The default grid is one panel per known quantity."""
    panels = plot_state_pair_grid(calculated_system, n_states=3, annotate=False)

    assert len(panels) == len(QUANTITIES)
    assert all(len(panel.images) == 1 for panel in panels)


def test_a_state_pair_grid_leaves_the_panel_it_does_not_need_empty(
    calculated_system: PCET,
) -> None:
    """Three quantities need four panels, so the fourth is blanked."""
    panels = plot_state_pair_grid(
        calculated_system,
        ["contribution", "overlap", "free_energy"],
        n_states=2,
        annotate=False,
    )

    assert len(panels) == 4
    assert len(panels[3].images) == 0


def test_a_state_pair_grid_needs_something_to_draw(
    calculated_system: PCET,
) -> None:
    """An empty list of quantities is a mistake, not an empty figure."""
    with pytest.raises(ValueError, match="at least one quantity"):
        plot_state_pair_grid(calculated_system, [])


def test_populations_get_a_bar_each(calculated_system: PCET) -> None:
    """One bar per reactant state, on a logarithmic axis."""
    ax = plot_populations(calculated_system, n_states=5)

    assert len(ax.containers[0]) == 5
    assert ax.get_yscale() == "log"


# ===========================================================================
# Sweeps
# ===========================================================================


def test_an_arrhenius_plot_recovers_the_activation_energy_it_was_given() -> None:
    """Rates built from a known E_a fit back to it, and it is named in the legend."""
    temperatures = np.linspace(250.0, 400.0, 12)
    activation_energy = 0.25
    rates = 1e12 * np.exp(-activation_energy / (BOLTZMANN * temperatures))

    ax = plot_arrhenius(temperatures, rates, label="H")

    assert any(f"{activation_energy:.3f}" in label for label in legend_labels(ax))


def test_an_arrhenius_plot_can_leave_the_fit_off() -> None:
    """Without the fit there is nothing on the axes but the points."""
    temperatures = np.linspace(250.0, 400.0, 6)
    rates = np.full(6, 1e8)

    ax = plot_arrhenius(temperatures, rates, fit=False)

    assert len(ax.lines) == 1
    assert ax.get_legend() is None


def test_an_arrhenius_plot_refuses_a_rate_that_underflowed() -> None:
    """A zero rate would take the logarithm of zero; say so instead."""
    with pytest.raises(ValueError, match="'rates' has to be positive"):
        plot_arrhenius(np.array([300.0, 400.0]), np.array([1e8, 0.0]))


def test_a_kinetic_isotope_effect_is_drawn_against_a_line_at_unity() -> None:
    """The ratio goes on with a dotted line marking no isotope effect."""
    temperatures = np.linspace(250.0, 400.0, 5)
    light = np.full(5, 4.0)
    heavy = np.full(5, 2.0)

    ax = plot_kie_vs_temperature(temperatures, light, heavy)

    ratio = next(line for line in ax.lines if len(xdata(line)) == 5)
    assert np.allclose(ydata(ratio), 2.0)
    assert ax.get_xlabel() == r"$T$ / K"


def test_a_kinetic_isotope_effect_refuses_a_rate_that_underflowed() -> None:
    """Neither isotope may have reached zero."""
    with pytest.raises(ValueError, match="'rates_heavy'"):
        plot_kie_vs_temperature(np.array([300.0]), np.array([1e8]), np.array([0.0]))


@pytest.mark.parametrize("reorganization_energy", [None, 1.0])
def test_a_marcus_curve_marks_the_reorganization_energy(
    reorganization_energy: float | None,
) -> None:
    """Given lambda, the curve carries a line at -lambda; otherwise just the curve."""
    free_energies = np.linspace(-2.0, 0.5, 40)
    rates = 1e10 * np.exp(-((free_energies + 1.0) ** 2))

    ax = plot_rate_vs_driving_force(
        free_energies, rates, reorganization_energy=reorganization_energy
    )

    assert len(ax.lines) == (2 if reorganization_energy is not None else 1)
    assert ax.get_yscale() == "log"


# ===========================================================================
# Averaging over the donor-acceptor distance
# ===========================================================================


def thermal_average_inputs() -> tuple[FloatArray, FloatArray, FloatArray]:
    """A falling k(R) against a harmonic P(R), as the examples combine them."""
    distances = np.linspace(2.2, 3.4, 80)
    rates = 1e10 * np.exp(-8.0 * (distances - 2.2))
    distribution = donor_acceptor_distribution(distances, 2.7, 0.0443)
    return distances, rates, distribution


def test_a_thermal_average_draws_the_rate_the_distribution_and_their_product() -> None:
    """Three curves, and two guides at the equilibrium and dominant distances."""
    distances, rates, distribution = thermal_average_inputs()

    ax = plot_thermal_average(distances, rates, distribution)

    curves = [line for line in ax.lines if len(xdata(line)) == len(distances)]
    assert len(curves) == 3
    assert all(np.max(ydata(curve)) == pytest.approx(1.0) for curve in curves)
    assert len(ax.lines) == 5


def test_a_thermal_average_pulls_the_dominant_distance_in_from_equilibrium() -> None:
    """A rate that rises as R closes moves the product's peak to shorter R."""
    distances, rates, distribution = thermal_average_inputs()

    ax = plot_thermal_average(distances, rates, distribution)

    equilibrium, dominant = (xdata(line)[0] for line in ax.lines[3:5])
    assert dominant < equilibrium


def test_a_thermal_average_can_keep_the_raw_scale() -> None:
    """Without normalizing, k(R) keeps the magnitude it came in with."""
    distances, rates, distribution = thermal_average_inputs()

    ax = plot_thermal_average(
        distances, rates, distribution, normalize=False, mark_maxima=False
    )

    assert len(ax.lines) == 3
    assert np.max(ydata(ax.lines[0])) == pytest.approx(np.max(rates))


def test_a_thermal_average_refuses_a_curve_that_never_leaves_zero() -> None:
    """Normalizing an all-zero curve would divide by zero."""
    distances = np.linspace(2.2, 3.4, 10)

    with pytest.raises(ValueError, match="zero everywhere"):
        plot_thermal_average(distances, np.zeros(10), np.ones(10))


# ===========================================================================
# Nonadiabaticity, the double layer, and the scans
# ===========================================================================


@pytest.mark.parametrize("show_states", [True, False])
def test_a_crossing_is_marked_with_both_tangents(
    kappa_system: KappaCoupling, show_states: bool
) -> None:
    """Two diabats, the crossing marker, two tangents, and maybe two states."""
    ax = plot_crossing(kappa_system, show_states=show_states)

    assert len(ax.lines) == (7 if show_states else 5)
    assert len(ax.collections) == (2 if show_states else 0)


@pytest.mark.parametrize("annotate_splitting", [True, False])
def test_the_adiabats_go_on_over_the_diabats_they_came_from(
    kappa_system: KappaCoupling, annotate_splitting: bool
) -> None:
    """Two diabats and two adiabats, with the splitting named on request."""
    ax = plot_diabats_and_adiabats(kappa_system, annotate_splitting=annotate_splitting)

    assert len(ax.lines) == 4
    assert (ax.get_legend() is not None) == annotate_splitting


def test_plotting_an_unrun_coupling_analysis_says_to_calculate_it_first() -> None:
    """A KappaCoupling straight out of the constructor has no adiabats yet."""
    grid = np.linspace(-0.55, 0.55, 512)
    analysis = KappaCoupling(grid, grid**2, (grid - 0.1) ** 2, 0.04)

    with pytest.raises(ValueError, match="no results yet"):
        plot_diabats_and_adiabats(analysis)


@pytest.mark.parametrize(
    ("d_ihl", "d_ohl", "guides"), [(None, None, 0), (3.0, None, 1), (3.0, 2.0, 2)]
)
def test_an_edl_profile_marks_the_helmholtz_planes(
    d_ihl: float | None, d_ohl: float | None, guides: int
) -> None:
    """Each plane that was given gets a dashed line at its distance."""
    distances = np.linspace(0.0, 10.0, 100)

    ax = plot_edl_profile(
        lambda distance: 0.5 * np.exp(-distance / 3.0),
        distances,
        d_ihl=d_ihl,
        d_ohl=d_ohl,
        label="$E = -0.6$V",
    )

    assert len(ax.lines) == 1 + guides
    assert ax.get_ylabel() == r"$\phi(R, E)$ / V"


def test_an_outer_helmholtz_plane_needs_an_inner_one_to_measure_from() -> None:
    """d_ohl is a thickness beyond d_ihl, so it cannot stand on its own."""
    with pytest.raises(ValueError, match="measured from 'd_ihl'"):
        plot_edl_profile(lambda distance: distance, np.linspace(0.0, 1.0, 5), d_ohl=2.0)


@pytest.mark.parametrize("relative", [True, False])
def test_a_distance_scan_can_be_shifted_to_its_lowest_point(
    relative: bool,
) -> None:
    """Relative energies start at zero; absolute ones keep the calculator's."""
    scan = DistanceScan(
        distances=np.linspace(2.4, 3.0, 4),
        energies=np.array([-10.5, -10.9, -10.7, -10.2]),
        structures=[],
    )

    ax = plot_distance_scan(scan, relative=relative)

    assert np.min(ydata(ax.lines[0])) == pytest.approx(0.0 if relative else -10.9)
    assert ax.get_xlabel() == r"$R\ /\ \rm\AA$"


@pytest.mark.parametrize("relative", [True, False])
def test_a_proton_scan_can_be_shifted_to_its_lowest_point(relative: bool) -> None:
    """The same for a proton potential straight off the calculator."""
    scan = ProtonScan(
        offsets=np.linspace(-0.5, 0.5, 5),
        energies=np.array([-8.0, -8.6, -8.3, -8.7, -8.1]),
        positions=np.zeros((5, 1, 3)),
    )

    ax = plot_proton_scan(scan, relative=relative)

    assert np.min(ydata(ax.lines[0])) == pytest.approx(0.0 if relative else -8.7)
