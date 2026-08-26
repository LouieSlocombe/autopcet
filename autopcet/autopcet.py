"""Vibronically nonadiabatic proton-coupled electron transfer (PCET) rates.

Implements Fourier grid Hamiltonian (FGH) proton vibrational states,
golden-rule PCET rate constants (:class:`PCET`), vibronic couplings and
nonadiabaticity analysis (:class:`KappaCoupling`), and an electrical double
layer (EDL) potential-drop model for electrochemical PCET.
"""

import warnings
from collections.abc import Callable
from typing import Protocol, cast, overload

import numpy as np
import numpy.typing as npt
from numba import jit
from scipy.constants import (
    N_A,
    angstrom,
    calorie,
    centi,
    elementary_charge,
    nano,
    speed_of_light,
    value,
)
from scipy.integrate import simpson
from scipy.interpolate import BSpline, splrep
from scipy.optimize import curve_fit, fsolve
from scipy.special import gamma

type FloatArray = npt.NDArray[np.float64]
type PotentialFunction = Callable[[FloatArray], FloatArray]


class _ScalarArrayFunction(Protocol):
    """A function evaluating elementwise on a scalar or a 1D array."""

    @overload
    def __call__(self, R: float) -> float: ...
    @overload
    def __call__(self, R: FloatArray) -> FloatArray: ...


kB: float = value("Boltzmann constant in eV/K")
h: float = value("Planck constant in eV/Hz")
hbar: float = h / 2 / np.pi
c: float = speed_of_light / angstrom
massH: float = value("proton-electron mass ratio")
massD: float = value("deuteron-electron mass ratio") + 1

Ha2eV: float = value("Hartree energy in eV")
Ha2kcal: float = value("Hartree energy") * N_A / (1000 * calorie)
kcal2Ha: float = 1 / Ha2kcal
eV2Ha: float = 1 / Ha2eV
eV2kcal: float = eV2Ha * Ha2kcal
kcal2eV: float = 1 / eV2kcal

A2Bohr: float = angstrom / value("Bohr radius")
A2nm: float = angstrom / nano
A2cm: float = angstrom / centi
Bohr2A: float = 1 / A2Bohr
cm2A: float = 1 / A2cm
nm2A: float = 1 / A2nm

wn2eV: float = h * c * A2cm
eV2wn: float = 1 / wn2eV

Da2me: float = value("atomic mass constant") / value("electron mass")
me2Da: float = 1 / Da2me

au2s: float = value("atomic unit of time")

Debye2au: float = (1e-21 / speed_of_light) / value(
    "atomic unit of electric dipole mom."
)


@overload
def morse(r: float, r0: float, De: float, beta: float) -> float: ...
@overload
def morse(r: FloatArray, r0: float, De: float, beta: float) -> FloatArray: ...
def morse(
    r: float | FloatArray, r0: float, De: float, beta: float
) -> float | FloatArray:
    """Morse potential with minimum at ``r0``, well depth ``De``, and width ``beta``."""
    # cast: numpy ufuncs are typed as returning Any for scalar inputs
    return cast("float | FloatArray", De * (1 - np.exp(-beta * (r - r0))) ** 2)


@overload
def inverted_morse(r: float, r0: float, De: float, beta: float) -> float: ...
@overload
def inverted_morse(r: FloatArray, r0: float, De: float, beta: float) -> FloatArray: ...
def inverted_morse(
    r: float | FloatArray, r0: float, De: float, beta: float
) -> float | FloatArray:
    """Mirror image of :func:`morse` about its minimum at ``r0``."""
    return cast("float | FloatArray", De * (1 - np.exp(beta * (r - r0))) ** 2)


@overload
def gaussian(x: float, x0: float, sigma2: float) -> float: ...
@overload
def gaussian(x: FloatArray, x0: float, sigma2: float) -> FloatArray: ...
def gaussian(x: float | FloatArray, x0: float, sigma2: float) -> float | FloatArray:
    """Normalized Gaussian centered at ``x0`` with variance ``sigma2``."""
    return cast(
        "float | FloatArray",
        1 / np.sqrt(2 * np.pi * sigma2) * np.exp(-((x - x0) ** 2) / (2 * sigma2)),
    )


@overload
def poly6(
    x: float,
    c6: float,
    c5: float,
    c4: float,
    c3: float,
    c2: float,
    c1: float,
    c0: float,
) -> float: ...
@overload
def poly6(
    x: FloatArray,
    c6: float,
    c5: float,
    c4: float,
    c3: float,
    c2: float,
    c1: float,
    c0: float,
) -> FloatArray: ...
def poly6(
    x: float | FloatArray,
    c6: float,
    c5: float,
    c4: float,
    c3: float,
    c2: float,
    c1: float,
    c0: float,
) -> float | FloatArray:
    """Sixth-order polynomial with coefficients in descending order."""
    return c6 * x**6 + c5 * x**5 + c4 * x**4 + c3 * x**3 + c2 * x**2 + c1 * x + c0


@overload
def poly8(
    x: float,
    c8: float,
    c7: float,
    c6: float,
    c5: float,
    c4: float,
    c3: float,
    c2: float,
    c1: float,
    c0: float,
) -> float: ...
@overload
def poly8(
    x: FloatArray,
    c8: float,
    c7: float,
    c6: float,
    c5: float,
    c4: float,
    c3: float,
    c2: float,
    c1: float,
    c0: float,
) -> FloatArray: ...
def poly8(
    x: float | FloatArray,
    c8: float,
    c7: float,
    c6: float,
    c5: float,
    c4: float,
    c3: float,
    c2: float,
    c1: float,
    c0: float,
) -> float | FloatArray:
    """Eighth-order polynomial with coefficients in descending order."""
    return (
        c8 * x**8
        + c7 * x**7
        + c6 * x**6
        + c5 * x**5
        + c4 * x**4
        + c3 * x**3
        + c2 * x**2
        + c1 * x
        + c0
    )


def make_morse(r0: float, De: float, beta: float) -> PotentialFunction:
    """Return a Morse potential as a function of the coordinate only."""

    def morse_potential(r: FloatArray) -> FloatArray:
        return morse(r, r0, De, beta)

    return morse_potential


def make_inverted_morse(r0: float, De: float, beta: float) -> PotentialFunction:
    """Return an inverted Morse potential as a function of the coordinate only."""

    def inverted_morse_potential(r: FloatArray) -> FloatArray:
        return inverted_morse(r, r0, De, beta)

    return inverted_morse_potential


def make_double_well(
    De1: float,
    De2: float,
    beta1: float,
    beta2: float,
    R0: float,
    Delta: float,
    VPT0: float,
    smooth: str = "poly8",
    s: float = 8,
    NGrid: int = 500,
) -> PotentialFunction:
    """Build a smoothed double-well potential from two coupled Morse potentials."""
    r = np.linspace(-1.2 * R0, 1.2 * R0, NGrid)
    ED = morse(r, -R0 / 2, De1, beta1)
    EA = inverted_morse(r, R0 / 2, De2, beta2) + Delta

    roots = find_roots(r, ED - EA)
    if len(roots) != 3:
        raise RuntimeError(
            "These input parameters lead to a single-well potential. "
            "Please use a different set of parameters. "
        )
    VPT = VPT0 * gaussian(r, roots[1], R0 * R0)

    E1 = 0.5 * (EA + ED - np.sqrt((EA - ED) ** 2 + 4 * VPT**2))
    E2 = 0.5 * (EA + ED + np.sqrt((EA - ED) ** 2 + 4 * VPT**2))

    switch = np.heaviside(r - roots[0], 0 * r) + np.heaviside(roots[2] - r, 0 * r) - 1

    E = switch * E1 + (1 - switch) * E2

    fitted_E: PotentialFunction
    if smooth == "poly8":
        fitted_E = fit_poly8(r, E)
    elif smooth == "poly6":
        fitted_E = fit_poly6(r, E)
    elif smooth == "bspline":
        tck = splrep(r, E, s=s)

        def bspline_E(rr: FloatArray) -> FloatArray:
            # cast: scipy is untyped
            return cast("FloatArray", BSpline(*tck)(rr))

        fitted_E = bspline_E
    else:
        raise ValueError("'smooth' must be one of 'poly6', 'poly8', or 'bspline'. ")

    def double_well_potential(rr: FloatArray) -> FloatArray:
        return fitted_E(rr) - np.min(fitted_E(r))

    return double_well_potential


def fit_poly6(xdata: FloatArray, ydata: FloatArray) -> PotentialFunction:
    """Fit a sixth-order polynomial to the data, shifted to a minimum of zero."""
    coeffs: FloatArray = curve_fit(poly6, xdata, ydata - np.min(ydata))[0]
    xdata_tmp = np.linspace(np.min(xdata), np.max(xdata), 500)
    ydata_tmp = poly6(xdata_tmp, *coeffs)
    coeffs = curve_fit(poly6, xdata_tmp, ydata_tmp - np.min(ydata_tmp))[0]

    def poly6_fit(x: FloatArray) -> FloatArray:
        return poly6(x, *coeffs)

    return poly6_fit


def fit_poly8(xdata: FloatArray, ydata: FloatArray) -> PotentialFunction:
    """Fit an eighth-order polynomial to the data, shifted to a minimum of zero."""
    coeffs: FloatArray = curve_fit(poly8, xdata, ydata - np.min(ydata))[0]
    xdata_tmp = np.linspace(np.min(xdata), np.max(xdata), 500)
    ydata_tmp = poly8(xdata_tmp, *coeffs)
    coeffs = curve_fit(poly8, xdata_tmp, ydata_tmp - np.min(ydata_tmp))[0]

    def poly8_fit(x: FloatArray) -> FloatArray:
        return poly8(x, *coeffs)

    return poly8_fit


def fit_bspline(
    xdata: FloatArray, ydata: FloatArray, s: float = 1e-4
) -> PotentialFunction:
    """Fit a smoothing B-spline to the data, shifted to a minimum of zero."""
    tck = splrep(xdata, ydata, s=s)
    xdata_tmp = np.linspace(np.min(xdata), np.max(xdata), 500)

    def bspline_fit(x: FloatArray) -> FloatArray:
        # cast: scipy is untyped
        return cast("FloatArray", BSpline(*tck)(x) - np.min(BSpline(*tck)(xdata_tmp)))

    return bspline_fit


def find_roots(xdata: FloatArray, ydata: FloatArray) -> list[float]:
    """Locate the grid points closest to the sign changes of ``ydata``."""
    xo = xdata[0]
    yo = ydata[0]
    roots: list[float] = []
    for xi, yi in zip(xdata[1:], ydata[1:], strict=True):
        if np.sign(yi) != np.sign(yo):
            if np.abs(yi) < np.abs(yo):
                roots.append(xi)
            else:
                roots.append(xo)
        xo = xi
        yo = yi

    return roots


def is_number(dat: object) -> bool:
    """Return True if ``dat`` is a real (non-boolean) scalar number."""
    return isinstance(dat, (int, float, np.integer, np.floating)) and not isinstance(
        dat, bool
    )


def is_array(dat: object) -> bool:
    """Return True if ``dat`` is a tuple, list, or NumPy array."""
    return isinstance(dat, (tuple, list, np.ndarray))


@jit(nopython=True)
def fgh_1d(
    ngrid: int, sgrid: float, potential: FloatArray, mass: float
) -> tuple[FloatArray, FloatArray]:
    """Solve a 1D Schroedinger equation with the Fourier grid Hamiltonian method.

    All quantities are in atomic units: ``sgrid`` is the grid length,
    ``potential`` the potential energy on the grid, and ``mass`` the particle
    mass. Returns the eigenvalues and eigenvectors of the Hamiltonian.
    """
    nx = ngrid
    dx = sgrid / (ngrid - 1)
    k = np.pi / dx

    vmat = np.zeros((nx, nx))
    tmat = np.zeros((nx, nx))
    hmat = np.zeros((nx, nx))

    for i in range(nx):
        for j in range(nx):
            if i == j:
                vmat[i, j] = potential[j]
                tmat[i, j] = (k**2) / 3
            else:
                dji = j - i
                vmat[i, j] = 0
                tmat[i, j] = (2 * k**2) / (np.pi**2) * (((-1) ** dji) / (dji**2))

            hmat[i, j] = (1 / (2 * mass)) * tmat[i, j] + vmat[i, j]

    hmat_soln = np.linalg.eigh(hmat)
    return hmat_soln


def _solve_vibrational_states(
    rp: FloatArray, potential_eV: FloatArray, mass: float, nstates: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Solve for proton vibrational states on the grid ``rp`` (in Angstrom).

    Returns the lowest ``nstates`` energy levels in eV, the corresponding
    normalized wave functions, and the raw FGH eigenvalues in Hartree.
    """
    rp_in_Bohr = rp * A2Bohr
    ngrid = len(rp_in_Bohr)
    sgrid = rp_in_Bohr[-1] - rp_in_Bohr[0]

    eigvals, eigvecs = fgh_1d(ngrid, sgrid, potential_eV * eV2Ha, mass)
    energy_levels = eigvals[:nstates] * Ha2eV

    unnormalized_wfcs = np.transpose(eigvecs)[:nstates]
    wave_functions = np.array(
        [wfci / np.sqrt(simpson(wfci * wfci, x=rp)) for wfci in unnormalized_wfcs]
    )
    return energy_levels, wave_functions, eigvals


def _find_first_crossing(y: FloatArray, x: FloatArray) -> tuple[int, float]:
    """Return the index and midpoint position of the first sign change of ``y``."""
    for i in range(1, len(x)):
        if y[i] * y[i - 1] <= 0:
            return i, (x[i] + x[i - 1]) / 2
    raise RuntimeError(
        "The reactant and product proton potentials do not cross within the rp grid. "
    )


def make_edl_model(
    EvsSHE: float,
    dIHL: float,
    dOHL: float,
    eps_IHL: float,
    eps_st: float,
    eps_op: float,
    dipole: float | str,
    rho_solvent: float,
    m_solvent: float,
    c_ions: float,
    C_EDL: float,
    PZFCvsSHE: float,
    T: float = 298.15,
    eta_Kirkwood: float = 1,
    g_Kirkwood: float = 2.4,
    print_data: bool = False,
) -> _ScalarArrayFunction:
    """Build an electrical double layer (EDL) model of the potential drop.

    Returns a function mapping the distance from the electrode (in Angstrom)
    to the potential drop (in V).
    """
    EvsPZFC = EvsSHE - PZFCvsSHE
    if print_data:
        print(f"E vs. SHE = {EvsSHE:.2f} V")

    sigma_M = (C_EDL * EvsPZFC) / (elementary_charge * 1e6 * (cm2A * A2Bohr) ** 2)

    if print_data:
        print(f"sigma_M = {sigma_M:.6e} a.u.")

    n_solvent_au = rho_solvent / m_solvent * N_A / (cm2A * A2Bohr) ** 3

    if dipole == "calculate":
        if eta_Kirkwood != 1:
            eta_Kirkwood = 2 * (2 * eps_st + eps_op) / (3 * g_Kirkwood * eps_st)
        dipole_au = (
            3
            / (2 + eps_op)
            * np.sqrt(
                3
                * kB
                * T
                * eV2Ha
                * (eps_st - eps_op)
                * eta_Kirkwood
                / (8 * np.pi * n_solvent_au)
            )
        )
        if print_data:
            print(f"dipole = {dipole_au:.6f} a.u.")
    elif is_number(dipole):
        dipole_au = float(dipole) * Debye2au
    else:
        raise ValueError("'dipole' must be a number (in Debye) or 'calculate'. ")

    n_ions_au = (c_ions * N_A * 1000) / (1e10 * A2Bohr) ** 3
    dIHL_Bohr = dIHL * A2Bohr
    dOHL_Bohr = dOHL * A2Bohr

    phi_OHP_au = (
        2
        * kB
        * T
        * eV2Ha
        * np.arcsinh(
            sigma_M / np.sqrt((2 * kB * T * eV2Ha * eps_st * n_ions_au) / np.pi)
        )
    )
    phi_OHP = phi_OHP_au * Ha2eV

    if print_data:
        print(f"phi_OHP = {phi_OHP:.6f} V")

    def langevin(u: float | FloatArray) -> float | FloatArray:
        # cast: numpy ufuncs are typed as returning Any for scalar inputs
        return cast("float | FloatArray", 1 / np.tanh(u) - 1 / u)

    def eps_ohl(E_OHL_au: float | FloatArray) -> float | FloatArray:
        return cast(
            "float | FloatArray",
            eps_op
            + (4 * np.pi * (2 + eps_op))
            / (3 * E_OHL_au)
            * n_solvent_au
            * dipole_au
            * langevin(((2 + eps_op) * dipole_au * E_OHL_au) / (2 * kB * T * eV2Ha)),
        )

    def ohl_field_equation(E_OHL_au: float | FloatArray) -> float | FloatArray:
        return cast(
            "float | FloatArray",
            E_OHL_au * ((dIHL_Bohr * eps_ohl(E_OHL_au) / eps_IHL) + dOHL_Bohr)
            - EvsPZFC * eV2Ha
            + phi_OHP_au,
        )

    E_solution_au = fsolve(ohl_field_equation, x0=0.1)

    E_OHL_au = float(E_solution_au[0])
    E_OHL = E_OHL_au * Ha2eV / Bohr2A

    E_IHL_au = float(E_OHL_au * eps_ohl(E_OHL_au) / eps_IHL)
    E_IHL = E_IHL_au * Ha2eV / Bohr2A

    if print_data:
        print(f"eps_OHL = {eps_ohl(E_OHL_au):.6f}")
        print(f"E_OHL = {E_OHL:.6f} V/A")
        print(f"E_IHL = {E_IHL:.6f} V/A")
        print()

    kappa = float(np.sqrt((8 * np.pi * n_ions_au) / (eps_st * kB * T * eV2Ha)) / Bohr2A)

    @overload
    def edl_potential_drop(R: float) -> float: ...
    @overload
    def edl_potential_drop(R: FloatArray) -> FloatArray: ...
    def edl_potential_drop(R: float | FloatArray) -> float | FloatArray:
        if is_number(R):
            R_num = float(R)
            if R_num <= dIHL:
                return EvsPZFC - R_num * E_IHL
            elif (dIHL < R_num) and (R_num <= dIHL + dOHL):
                return EvsPZFC - dIHL * E_IHL - (R_num - dIHL) * E_OHL
            else:
                return float(
                    4
                    * kB
                    * T
                    * np.arctanh(
                        np.tanh(phi_OHP / (4 * kB * T))
                        * np.exp(-kappa * (R_num - dIHL - dOHL))
                    )
                )
        elif is_array(R):
            R_arr = np.asarray(R, dtype=np.float64)
            result = np.zeros(len(R_arr))
            for i, Ri in enumerate(R_arr):
                if Ri <= dIHL:
                    result[i] = EvsPZFC - Ri * E_IHL
                elif (dIHL < Ri) and (Ri <= dIHL + dOHL):
                    result[i] = EvsPZFC - dIHL * E_IHL - (Ri - dIHL) * E_OHL
                else:
                    result[i] = (
                        4
                        * kB
                        * T
                        * np.arctanh(
                            np.tanh(phi_OHP / (4 * kB * T))
                            * np.exp(-kappa * (Ri - dIHL - dOHL))
                        )
                    )
            return result
        else:
            raise TypeError("'R' must be a number or a 1D array. ")

    return edl_potential_drop


@overload
def fermi_distribution(E: float, E_Fermi: float = 0, T: float = 298.15) -> float: ...
@overload
def fermi_distribution(
    E: FloatArray, E_Fermi: float = 0, T: float = 298.15
) -> FloatArray: ...
def fermi_distribution(
    E: float | FloatArray, E_Fermi: float = 0, T: float = 298.15
) -> float | FloatArray:
    """Fermi-Dirac occupation of a state at energy ``E`` (in eV)."""
    return cast("float | FloatArray", 1 / (np.exp((E - E_Fermi) / kB / T) + 1))


class KappaCoupling:
    """Vibronic couplings and nonadiabaticity analysis for a PCET reaction.

    Takes the reactant and product proton potentials (in eV) tabulated on the
    proton coordinate grid ``rp`` (in Angstrom) and the electronic coupling
    ``Vel`` (in eV), and computes the semiclassical vibronic coupling and the
    nonadiabaticity parameters of Georgievskii and Stuchebrukhov.
    """

    rp: FloatArray
    ReacProtonPot: FloatArray
    ProdProtonPot: FloatArray
    Vel: FloatArray
    NStates: int
    mu: int
    nu: int
    ReacProtonEnergyLevels: FloatArray
    ReacProtonWaveFunctions: FloatArray
    ProdProtonEnergyLevels: FloatArray
    ProdProtonWaveFunctions: FloatArray
    ShiftedReacProtonPot: FloatArray
    ShiftedProdProtonPot: FloatArray
    ShiftedReacProtonEnergyLevels: FloatArray
    ShiftedProdProtonEnergyLevels: FloatArray
    rp_crossing: float
    E_crossing: float
    Vel_crossing: float
    slope_reac: float
    slope_prod: float
    tau_p: float
    tau_e: float
    p: float
    kappa: float
    AdiabaticProtonPotGS: FloatArray
    AdiabaticProtonPotES: FloatArray
    AdiabaticGSProtonEnergyLevels: FloatArray
    AdiabaticGSProtonWaveFunctions: FloatArray
    V_ad: float
    V_nad: float
    V_sc: float

    def __init__(
        self,
        rp: FloatArray,
        ReacProtonPot: FloatArray,
        ProdProtonPot: FloatArray,
        Vel: float | FloatArray,
        NStates: int = 10,
        mu: int = 0,
        nu: int = 0,
    ) -> None:
        if not (is_array(rp) and is_array(ReacProtonPot) and is_array(ProdProtonPot)):
            raise TypeError(
                "'rp', 'ReacProtonPot', and 'ProdProtonPot' must be 1D arrays."
            )
        if is_array(Vel):
            Vel_arr = np.asarray(Vel, dtype=np.float64)
        elif is_number(Vel):
            Vel_arr = np.ones(len(rp)) * Vel
        else:
            raise TypeError("'Vel' must be a number or a 1D array.")

        if (
            len(ReacProtonPot) != len(rp)
            or len(ProdProtonPot) != len(rp)
            or len(Vel_arr) != len(rp)
        ):
            raise ValueError(
                "'rp', 'ReacProtonPot', 'ProdProtonPot', and 'Vel' must have the same dimension. "
            )

        if np.abs(np.log2(len(rp)) - int(np.log2(len(rp)))) > 1e-4:
            raise ValueError(
                "The number of grid points should be some integer power of 2. "
            )

        if not (0 <= mu < NStates and 0 <= nu < NStates):
            raise ValueError("'mu' and 'nu' must satisfy 0 <= mu, nu < NStates. ")

        self.rp = np.asarray(rp, dtype=np.float64)
        self.ReacProtonPot = np.asarray(ReacProtonPot, dtype=np.float64)
        self.ProdProtonPot = np.asarray(ProdProtonPot, dtype=np.float64)
        self.Vel = Vel_arr
        self.NStates = NStates
        self.mu = mu
        self.nu = nu

    def calc_proton_vibrational_states(self, mass: float = massH) -> None:
        """Solve for the proton vibrational states in both diabatic potentials."""
        (
            self.ReacProtonEnergyLevels,
            self.ReacProtonWaveFunctions,
            _,
        ) = _solve_vibrational_states(self.rp, self.ReacProtonPot, mass, self.NStates)
        (
            self.ProdProtonEnergyLevels,
            self.ProdProtonWaveFunctions,
            _,
        ) = _solve_vibrational_states(self.rp, self.ProdProtonPot, mass, self.NStates)

    def analyze_proton_potentials(self, mass: float = massH) -> None:
        """Shift the potentials to degeneracy and locate their crossing point."""
        self.calc_proton_vibrational_states(mass)
        E_reac = self.ReacProtonEnergyLevels[self.mu]
        E_prod = self.ProdProtonEnergyLevels[self.nu]

        if E_prod < E_reac:
            dEr = 0
            dEp = -E_prod + E_reac
        else:
            dEr = E_prod - E_reac
            dEp = 0

        self.ShiftedReacProtonPot = self.ReacProtonPot + dEr
        self.ShiftedProdProtonPot = self.ProdProtonPot + dEp
        self.ShiftedReacProtonEnergyLevels = self.ReacProtonEnergyLevels + dEr
        self.ShiftedProdProtonEnergyLevels = self.ProdProtonEnergyLevels + dEp

        deltaE = self.ShiftedReacProtonPot - self.ShiftedProdProtonPot

        rp_crossing_index, self.rp_crossing = _find_first_crossing(deltaE, self.rp)
        self.E_crossing = (
            self.ShiftedReacProtonPot[rp_crossing_index]
            + self.ShiftedReacProtonPot[rp_crossing_index - 1]
            + self.ShiftedProdProtonPot[rp_crossing_index]
            + self.ShiftedProdProtonPot[rp_crossing_index - 1]
        ) / 4
        self.Vel_crossing = (
            self.Vel[rp_crossing_index] + self.Vel[rp_crossing_index - 1]
        ) / 2

        self.slope_reac = (
            self.ShiftedReacProtonPot[rp_crossing_index]
            - self.ShiftedReacProtonPot[rp_crossing_index - 1]
        ) / (self.rp[rp_crossing_index] - self.rp[rp_crossing_index - 1])
        self.slope_prod = (
            self.ShiftedProdProtonPot[rp_crossing_index]
            - self.ShiftedProdProtonPot[rp_crossing_index - 1]
        ) / (self.rp[rp_crossing_index] - self.rp[rp_crossing_index - 1])

    def calculate(self, mass: float = massH, overlap_thresh: float = 0.8) -> None:
        """Compute the nonadiabaticity parameters and vibronic couplings."""
        self.analyze_proton_potentials(mass)

        E0 = self.ShiftedReacProtonEnergyLevels[self.mu]

        if self.E_crossing < E0:
            raise RuntimeError(
                "The tunneling energy is higher than the energy at the crossing point. "
            )

        vt = np.sqrt(2 * (self.E_crossing - E0) * eV2Ha / mass) * Bohr2A / au2s

        self.tau_p = self.Vel_crossing / (
            np.abs(self.slope_reac - self.slope_prod) * vt
        )
        self.tau_e = hbar / self.Vel_crossing
        self.p = self.tau_p / self.tau_e
        self.kappa = (
            np.sqrt(2 * np.pi * self.p)
            * np.exp(self.p * np.log(self.p) - self.p)
            / gamma(self.p + 1)
        )

        self.AdiabaticProtonPotGS = 0.5 * (
            self.ShiftedReacProtonPot
            + self.ShiftedProdProtonPot
            - np.sqrt(
                (self.ShiftedProdProtonPot - self.ShiftedReacProtonPot) ** 2
                + 4 * self.Vel_crossing**2
            )
        )
        self.AdiabaticProtonPotES = 0.5 * (
            self.ShiftedReacProtonPot
            + self.ShiftedProdProtonPot
            + np.sqrt(
                (self.ShiftedProdProtonPot - self.ShiftedReacProtonPot) ** 2
                + 4 * self.Vel_crossing**2
            )
        )

        (
            self.AdiabaticGSProtonEnergyLevels,
            self.AdiabaticGSProtonWaveFunctions,
            eigvals,
        ) = _solve_vibrational_states(
            self.rp, self.AdiabaticProtonPotGS, mass, 2 * self.NStates
        )

        Smunu = simpson(
            self.ReacProtonWaveFunctions[self.mu]
            * self.ProdProtonWaveFunctions[self.nu],
            x=self.rp,
        )
        sign = 1 if Smunu > 0 else -1
        wfc_symm = (
            self.ReacProtonWaveFunctions[self.mu]
            + sign * self.ProdProtonWaveFunctions[self.nu]
        ) / np.sqrt(2)
        wfc_anti = (
            self.ReacProtonWaveFunctions[self.mu]
            - sign * self.ProdProtonWaveFunctions[self.nu]
        ) / np.sqrt(2)
        wfc_symm /= np.sqrt(simpson(wfc_symm**2, x=self.rp))
        wfc_anti /= np.sqrt(simpson(wfc_anti**2, x=self.rp))

        overlap_w_symm = np.array(
            [
                np.abs(simpson(wfci * wfc_symm, x=self.rp))
                for wfci in self.AdiabaticGSProtonWaveFunctions
            ]
        )
        overlap_w_anti = np.array(
            [
                np.abs(simpson(wfci * wfc_anti, x=self.rp))
                for wfci in self.AdiabaticGSProtonWaveFunctions
            ]
        )

        index_max_overlap_symm = 0
        for i in range(2 * self.NStates):
            if overlap_w_symm[i] > overlap_w_symm[index_max_overlap_symm]:
                index_max_overlap_symm = i

        index_max_overlap_anti = 0
        for i in range(2 * self.NStates):
            if (
                overlap_w_anti[i] > overlap_w_anti[index_max_overlap_anti]
                and i != index_max_overlap_symm
            ):
                index_max_overlap_anti = i

        if (
            overlap_w_symm[index_max_overlap_symm] < overlap_thresh
            or overlap_w_anti[index_max_overlap_anti] < overlap_thresh
        ):
            warnings.warn(
                "The maximum overlap between the proton vibrational wave functions in the "
                "adiabatic potential and the symmetric/antisymmetric combinations of the wave "
                f"functions in diabatic potentials is less than {overlap_thresh:.1f}. ",
                stacklevel=2,
            )
        if index_max_overlap_anti < index_max_overlap_symm:
            warnings.warn(
                "The identified antisymmetric state is lower in energy than the identified "
                f"symmetric state. The symmetric state is state {index_max_overlap_symm:d} and "
                f"the antisymmetric state is state {index_max_overlap_anti:d}. ",
                stacklevel=2,
            )
        if np.abs(index_max_overlap_anti - index_max_overlap_symm) > 1:
            warnings.warn(
                "There are multiple states lying in between the identified symmetric and "
                f"antisymmetric states. The symmetric state is state {index_max_overlap_symm:d} "
                f"and the antisymmetric state is state {index_max_overlap_anti:d}. ",
                stacklevel=2,
            )

        tunneling_splitting = (
            eigvals[index_max_overlap_anti] - eigvals[index_max_overlap_symm]
        ) * Ha2eV
        self.V_ad = 0.5 * tunneling_splitting

        self.V_nad = self.Vel_crossing * Smunu
        self.V_sc = self.kappa * self.V_ad

    def get_reactant_proton_states(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return the shifted reactant potential, energy levels, and wave functions."""
        return (
            self.ShiftedReacProtonPot,
            self.ShiftedReacProtonEnergyLevels,
            self.ReacProtonWaveFunctions,
        )

    def get_product_proton_states(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return the shifted product potential, energy levels, and wave functions."""
        return (
            self.ShiftedProdProtonPot,
            self.ShiftedProdProtonEnergyLevels,
            self.ProdProtonWaveFunctions,
        )

    def get_adiabatic_proton_potentials(self) -> tuple[FloatArray, FloatArray]:
        """Return the ground- and excited-state adiabatic proton potentials."""
        return self.AdiabaticProtonPotGS, self.AdiabaticProtonPotES

    def get_ground_adiabatic_proton_states(self) -> tuple[FloatArray, FloatArray]:
        """Return the energy levels and wave functions in the ground adiabatic potential."""
        return self.AdiabaticGSProtonEnergyLevels, self.AdiabaticGSProtonWaveFunctions

    def get_nonadiabaticity_parameters(self) -> tuple[float, float, float, float]:
        """Return ``tau_e``, ``tau_p``, the adiabaticity parameter ``p``, and ``kappa``."""
        return self.tau_e, self.tau_p, self.p, self.kappa

    def get_vibronic_couplings(self) -> tuple[float, float, float]:
        """Return the semiclassical, nonadiabatic, and adiabatic vibronic couplings."""
        return self.V_sc, self.V_nad, self.V_ad


def _smoothed_potential(
    pot: PotentialFunction | tuple[FloatArray, FloatArray] | list[FloatArray],
    smooth: str,
    name: str,
) -> tuple[PotentialFunction, float | None, float | None]:
    """Turn a tabulated ``(rp, V)`` potential into a callable via smoothing.

    Callables pass through unchanged. Returns the potential function and the
    bounds of the tabulated data (``None`` for callables).
    """
    if callable(pot):
        return pot, None, None
    if is_array(pot) and len(pot) == 2:
        r = np.asarray(pot[0], dtype=np.float64)
        v = np.asarray(pot[1], dtype=np.float64)
        if smooth == "poly6":
            fitted = fit_poly6(r, v)
        elif smooth == "poly8":
            fitted = fit_poly8(r, v)
        elif smooth == "bspline":
            fitted = fit_bspline(r, v)
        else:
            raise ValueError("'smooth' must be one of 'poly6', 'poly8', or 'bspline'. ")
        return fitted, float(np.min(r)), float(np.max(r))
    raise TypeError(f"'{name}' must be a callable or a pair (rp, V) of 1D arrays. ")


class PCET:
    """Golden-rule PCET rate constants from diabatic proton potentials.

    Takes the reactant and product proton potentials (callables in eV, or
    tabulated ``(rp, V)`` pairs to be smoothed), the reaction free energy
    ``DeltaG`` (eV), the reorganization energy ``Lambda`` (eV), and the
    electronic coupling ``Vel`` (eV), and computes the vibronically
    nonadiabatic PCET rate constant.
    """

    rp: FloatArray
    ReacProtonPot: PotentialFunction
    ProdProtonPot: PotentialFunction
    DeltaG: float
    Lambda: float
    Vel: float
    NStates: int
    Pu: FloatArray
    Suv: FloatArray
    dGuv: FloatArray
    kuv: FloatArray
    Iuv: FloatArray
    k_tot: float
    MassUsedPreviously: float | None
    ReacProtonEnergyLevels: FloatArray
    ReacProtonWaveFunctions: FloatArray
    ProdProtonEnergyLevels: FloatArray
    ProdProtonWaveFunctions: FloatArray

    def __init__(
        self,
        ReacProtonPot: PotentialFunction
        | tuple[FloatArray, FloatArray]
        | list[FloatArray],
        ProdProtonPot: PotentialFunction
        | tuple[FloatArray, FloatArray]
        | list[FloatArray],
        DeltaG: float,
        Lambda: float,
        Vel: float = 0.0434,
        NStates: int = 10,
        NGridPot: int = 256,
        smooth: str = "bspline",
        rmin: float | None = None,
        rmax: float | None = None,
    ) -> None:
        self.ReacProtonPot, rmin1, rmax1 = _smoothed_potential(
            ReacProtonPot, smooth, "ReacProtonPot"
        )
        self.ProdProtonPot, rmin2, rmax2 = _smoothed_potential(
            ProdProtonPot, smooth, "ProdProtonPot"
        )

        if rmin is None:
            rmin = (
                min(rmin1, rmin2) if rmin1 is not None and rmin2 is not None else -0.8
            )
        if rmax is None:
            rmax = max(rmax1, rmax2) if rmax1 is not None and rmax2 is not None else 0.8
        self.rp = np.linspace(rmin, rmax, NGridPot)

        self.DeltaG = DeltaG
        self.Lambda = Lambda
        self.Vel = Vel
        self.NStates = NStates

        self.Pu = np.zeros(NStates)
        self.Suv = np.zeros((NStates, NStates))
        self.dGuv = np.zeros((NStates, NStates))
        self.kuv = np.zeros((NStates, NStates))
        self.Iuv = np.zeros((NStates, NStates))
        self.k_tot = 0.0
        self.MassUsedPreviously = None

    def calc_proton_vibrational_states(self, mass: float = massH) -> None:
        """Solve for the proton vibrational states in both diabatic potentials."""
        self.MassUsedPreviously = mass
        E_reac = np.asarray(self.ReacProtonPot(self.rp), dtype=np.float64)
        E_prod = np.asarray(self.ProdProtonPot(self.rp), dtype=np.float64)

        (
            self.ReacProtonEnergyLevels,
            self.ReacProtonWaveFunctions,
            _,
        ) = _solve_vibrational_states(self.rp, E_reac, mass, self.NStates)
        (
            self.ProdProtonEnergyLevels,
            self.ProdProtonWaveFunctions,
            _,
        ) = _solve_vibrational_states(self.rp, E_prod, mass, self.NStates)

    def calc_reactant_state_distribution(self, T: float = 298.15) -> FloatArray:
        """Compute the Boltzmann populations of the reactant proton states."""
        Boltzmann_factors = np.exp(-self.ReacProtonEnergyLevels / kB / T)
        partition_func = np.sum(Boltzmann_factors)
        self.Pu = Boltzmann_factors / partition_func
        return self.Pu

    def calc_proton_overlap_matrix(self) -> FloatArray:
        """Compute the overlap matrix of reactant and product proton states."""
        for u in range(self.NStates):
            for v in range(self.NStates):
                self.Suv[u, v] = simpson(
                    self.ReacProtonWaveFunctions[u] * self.ProdProtonWaveFunctions[v],
                    x=self.rp,
                )
        return self.Suv

    def calc_reaction_free_energy_matrix(self) -> FloatArray:
        """Compute the reaction free energy for each pair of proton states."""
        for u in range(self.NStates):
            for v in range(self.NStates):
                self.dGuv[u, v] = (
                    self.DeltaG
                    + (self.ProdProtonEnergyLevels[v] - self.ProdProtonEnergyLevels[0])
                    - (self.ReacProtonEnergyLevels[u] - self.ReacProtonEnergyLevels[0])
                )
        return self.dGuv

    def calc_rate_contribution_matrix(self, T: float = 298.15) -> FloatArray:
        """Compute the rate contribution of each pair of proton states."""
        k0 = 2 * np.pi / hbar * self.Vel * self.Vel
        self.Iuv = (
            1
            / np.sqrt(4 * np.pi * self.Lambda * kB * T)
            * np.exp(-((self.dGuv + self.Lambda) ** 2) / (4 * self.Lambda * kB * T))
        )
        self.kuv = k0 * np.matmul(np.diag(self.Pu), self.Suv * self.Suv * self.Iuv)
        return self.kuv

    def calculate(
        self,
        mass: float = massH,
        T: float = 298.15,
        reuse_saved_proton_states: bool = False,
    ) -> float:
        """Compute the total PCET rate constant at temperature ``T``."""
        if self.MassUsedPreviously != mass:
            reuse_saved_proton_states = False

        if not reuse_saved_proton_states:
            self.calc_proton_vibrational_states(mass)
            self.calc_proton_overlap_matrix()

        self.calc_reactant_state_distribution(T)
        self.calc_reaction_free_energy_matrix()
        self.calc_rate_contribution_matrix(T)

        self.k_tot = np.sum(self.kuv)
        return self.k_tot

    def set_parameters(
        self,
        DeltaG: float | None = None,
        Lambda: float | None = None,
        Vel: float | None = None,
    ) -> None:
        """Update the reaction free energy, reorganization energy, or coupling."""
        if DeltaG is not None:
            self.DeltaG = DeltaG
        if Lambda is not None:
            self.Lambda = Lambda
        if Vel is not None:
            self.Vel = Vel

    def get_reactant_proton_states(self) -> tuple[FloatArray, FloatArray]:
        """Return the reactant proton energy levels and wave functions."""
        return self.ReacProtonEnergyLevels, self.ReacProtonWaveFunctions

    def get_product_proton_states(self) -> tuple[FloatArray, FloatArray]:
        """Return the product proton energy levels and wave functions."""
        return self.ProdProtonEnergyLevels, self.ProdProtonWaveFunctions

    def get_reactant_state_distribution(self) -> FloatArray:
        """Return the Boltzmann populations of the reactant proton states."""
        return self.Pu

    def get_proton_overlap_matrix(self) -> FloatArray:
        """Return the overlap matrix of reactant and product proton states."""
        return self.Suv

    def get_reaction_free_energy_matrix(self) -> FloatArray:
        """Return the reaction free energy matrix."""
        return self.dGuv

    def get_activation_free_energy_matrix(self) -> FloatArray:
        """Return the Marcus activation free energy matrix."""
        return (self.dGuv + self.Lambda) ** 2 / (4 * self.Lambda)

    def get_rate_contribution_matrix(self) -> FloatArray:
        """Return the rate contribution matrix."""
        return self.kuv

    def get_total_rate_constant(self) -> float:
        """Return the total PCET rate constant."""
        return self.k_tot
