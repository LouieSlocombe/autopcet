import functools
import types

import numpy as np
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
from scipy.interpolate import BSpline, splrep
from scipy.optimize import curve_fit, fsolve
from scipy.special import gamma

try:
    from scipy.integrate import simps
except ImportError:
    from scipy.integrate import simpson as simps

__all__ = ['morse', 'inverted_morse', 'gaussian', 'poly6', 'poly8',
           'make_morse', 'make_inverted_morse', 'make_double_well', 'fit_poly6', 'fit_poly8', 'fit_bspline',
           'find_roots', 'is_number', 'is_array', 'copy_function',
           'fgh_1d', 'pcet', 'kappa_coupling',
           'make_edl_model', 'fermi_distribution',
           'kB', 'h', 'hbar', 'c', 'massH', 'massD',
           'Ha2eV', 'Ha2kcal', 'kcal2Ha', 'eV2Ha', 'eV2kcal', 'kcal2eV',
           'A2Bohr', 'A2nm', 'A2cm', 'Bohr2A', 'cm2A', 'nm2A', 'wn2eV', 'eV2wn', 'Da2me', 'me2Da', 'au2s',
           ]

kB = value('Boltzmann constant in eV/K')
h = value('Planck constant in eV/Hz')
hbar = h / 2 / np.pi
c = speed_of_light / angstrom
massH = value('proton-electron mass ratio')
massD = value('deuteron-electron mass ratio') + 1

Ha2eV = value('Hartree energy in eV')
Ha2kcal = value('Hartree energy') * N_A / (1000 * calorie)
kcal2Ha = 1 / Ha2kcal
eV2Ha = 1 / Ha2eV
eV2kcal = eV2Ha * Ha2kcal
kcal2eV = 1 / eV2kcal

A2Bohr = angstrom / value('Bohr radius')
A2nm = angstrom / nano
A2cm = angstrom / centi
Bohr2A = 1 / A2Bohr
cm2A = 1 / A2cm
nm2A = 1 / A2nm

wn2eV = h * c * A2cm
eV2wn = 1 / wn2eV

Da2me = value('atomic mass constant') / value('electron mass')
me2Da = 1 / Da2me

au2s = value('atomic unit of time')

Debye2au = (1e-21 / speed_of_light) / value('atomic unit of electric dipole mom.')

def morse(r, r0, De, beta):
    return De * (1 - np.exp(-beta * (r - r0))) ** 2


def inverted_morse(r, r0, De, beta):
    return De * (1 - np.exp(beta * (r - r0))) ** 2


def gaussian(x, x0, sigma2):
    return 1 / np.sqrt(2 * np.pi * sigma2) * np.exp(-(x - x0) ** 2 / (2 * sigma2))


def poly6(x, a, b, c, d, e, f, g):
    return a * x ** 6 + b * x ** 5 + c * x ** 4 + d * x ** 3 + e * x ** 2 + f * x + g


def poly8(x, a, b, c, d, e, f, g, h, i):
    return a * x ** 8 + b * x ** 7 + c * x ** 6 + d * x ** 5 + e * x ** 4 + f * x ** 3 + g * x ** 2 + h * x + i


def make_morse(r0, De, beta):

    def morse_potential(r):
        return De * (1 - np.exp(-beta * (r - r0))) ** 2

    return morse_potential


def make_inverted_morse(r0, De, beta):

    def inverted_morse_potential(r):
        return De * (1 - np.exp(beta * (r - r0))) ** 2

    return inverted_morse_potential


def make_double_well(De1, De2, beta1, beta2, R0, Delta, VPT0, smooth='poly8', s=8, NGrid=500):
    r = np.linspace(-1.2 * R0, 1.2 * R0, NGrid)
    ED = morse(r, -R0 / 2, De1, beta1)
    EA = inverted_morse(r, R0 / 2, De2, beta2) + Delta

    roots = find_roots(r, ED - EA)
    if len(roots) != 3:
        raise RuntimeError(
            "These input parameters lead to a single-well potential. Please use a different set of parameters. ")
    else:
        VPT = VPT0 * gaussian(r, roots[1], R0 * R0)

    E1 = 0.5 * (EA + ED - np.sqrt((EA - ED) ** 2 + 4 * VPT ** 2))
    E2 = 0.5 * (EA + ED + np.sqrt((EA - ED) ** 2 + 4 * VPT ** 2))

    switch = np.heaviside(r - roots[0], 0 * r) + np.heaviside(roots[2] - r, 0 * r) - 1

    E = switch * E1 + (1 - switch) * E2

    if smooth == 'poly8':
        fitted_E = fit_poly8(r, E)

        def double_well_potential(rr):
            return fitted_E(rr) - np.min(fitted_E(r))
    elif smooth == 'poly6':
        fitted_E = fit_poly6(r, E)

        def double_well_potential(rr):
            return fitted_E(rr) - np.min(fitted_E(r))
    elif smooth == 'BSpline':
        tck = splrep(r, E, s=s)

        def double_well_potential(rr):
            return BSpline(*tck)(rr) - np.min(BSpline(*tck)(r))
    else:
        raise ValueError("Unrecogonized input, please set smooth = 'poly8', 'poly6', or 'BSpline'. ")

    return double_well_potential


def fit_poly6(xdata, ydata):
    aa, bb, cc, dd, ee, ff, gg = curve_fit(poly6, xdata, ydata - np.min(ydata))[0]
    xdata_tmp = np.linspace(np.min(xdata), np.max(xdata), 500)
    ydata_tmp = poly6(xdata_tmp, aa, bb, cc, dd, ee, ff, gg)
    aa, bb, cc, dd, ee, ff, gg = curve_fit(poly6, xdata_tmp, ydata_tmp - np.min(ydata_tmp))[0]

    def poly6_fit(x):
        return aa * x ** 6 + bb * x ** 5 + cc * x ** 4 + dd * x ** 3 + ee * x ** 2 + ff * x + gg

    return poly6_fit


def fit_poly8(xdata, ydata):
    aa, bb, cc, dd, ee, ff, gg, hh, ii = curve_fit(poly8, xdata, ydata - np.min(ydata))[0]
    xdata_tmp = np.linspace(np.min(xdata), np.max(xdata), 500)
    ydata_tmp = poly8(xdata_tmp, aa, bb, cc, dd, ee, ff, gg, hh, ii)
    aa, bb, cc, dd, ee, ff, gg, hh, ii = curve_fit(poly8, xdata_tmp, ydata_tmp - np.min(ydata_tmp))[0]

    def poly8_fit(x):
        return aa * x ** 8 + bb * x ** 7 + cc * x ** 6 + dd * x ** 5 + ee * x ** 4 + ff * x ** 3 + gg * x ** 2 + hh * x + ii

    return poly8_fit


def fit_bspline(xdata, ydata, s=1e-4):
    tck = splrep(xdata, ydata, s=s)
    xdata_tmp = np.linspace(np.min(xdata), np.max(xdata), 500)

    def bspline_fit(x):
        return BSpline(*tck)(x) - np.min(BSpline(*tck)(xdata_tmp))

    return bspline_fit


def find_roots(xdata, ydata):
    xo = xdata[0]
    yo = ydata[0]
    roots = []
    for xi, yi in zip(xdata[1:], ydata[1:]):
        if np.sign(yi) != np.sign(yo):
            if np.abs(yi) < np.abs(yo):
                roots.append(xi)
            else:
                roots.append(xo)
        else:
            pass
        xo = xi
        yo = yi

    return roots


def is_number(dat):
    return (isinstance(dat, int) or isinstance(dat, float))


def is_array(dat):
    return (isinstance(dat, tuple) or isinstance(dat, list) or isinstance(dat, np.ndarray))


def copy_function(f):
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


@jit(nopython=True)
def fgh_1d(ngrid, sgrid, potential, mass):
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
                tmat[i, j] = (k ** 2) / 3
            else:
                dji = j - i
                vmat[i, j] = 0
                tmat[i, j] = (2 * k ** 2) / (np.pi ** 2) * (((-1) ** dji) / (dji ** 2))

            hmat[i, j] = (1 / (2 * mass)) * tmat[i, j] + vmat[i, j]

    hmat_soln = np.linalg.eigh(hmat)
    return hmat_soln


def make_edl_model(EvsSHE, dIHL, dOHL, eps_IHL, eps_st, eps_op, dipole, rho_solvent, m_solvent, c_ions, C_EDL, PZFCvsSHE,
                   T=298.15, eta_Kirkwood=1, g_Kirkwood=2.4, print_data=True):

    EvsPZFC = EvsSHE - PZFCvsSHE
    if print_data:
        print(f"E vs. SHE = {EvsSHE:.2f} V")

    sigma_M = (C_EDL * EvsPZFC) / (elementary_charge * 1e6 * (cm2A * A2Bohr) ** 2)

    if print_data:
        print(f"sigma_M = {sigma_M:.6e} a.u.")

    n_solvent_au = rho_solvent / m_solvent * N_A / (cm2A * A2Bohr) ** 3

    if dipole == 'calculate':
        if eta_Kirkwood == 1:
            eta_Kirkwood = 1
        else:
            eta_Kirkwood = 2 * (2 * eps_st + eps_op) / (3 * g_Kirkwood * eps_st)
        dipole_au = 3 / (2 + eps_op) * np.sqrt(
            3 * kB * T * eV2Ha * (eps_st - eps_op) * eta_Kirkwood / (8 * np.pi * n_solvent_au))
        if print_data:
            print(f"dipole = {dipole_au:.6f} a.u.")
    elif is_number(dipole):
        dipole_au = dipole * Debye2au
    else:
        raise ValueError("'dipole' must be a number (in Debye) or 'calculate'. ")

    n_ions_au = (c_ions * N_A * 1000) / (1e10 * A2Bohr) ** 3
    dIHL_Bohr = dIHL * A2Bohr
    dOHL_Bohr = dOHL * A2Bohr

    phi_OHP_au = 2 * kB * T * eV2Ha * np.arcsinh(sigma_M / np.sqrt((2 * kB * T * eV2Ha * eps_st * n_ions_au) / np.pi))
    phi_OHP = phi_OHP_au * Ha2eV

    if print_data:
        print(f"phi_OHP = {phi_OHP:.6f} V")

    def langevin(u):
        return 1 / np.tanh(u) - 1 / u

    def eps_ohl(E_OHL_au):
        return eps_op + (4 * np.pi * (2 + eps_op)) / (3 * E_OHL_au) * n_solvent_au * dipole_au * langevin(
            ((2 + eps_op) * dipole_au * E_OHL_au) / (2 * kB * T * eV2Ha))

    def ohl_field_equation(E_OHL_au):
        return E_OHL_au * ((dIHL_Bohr * eps_ohl(E_OHL_au) / eps_IHL) + dOHL_Bohr) - EvsPZFC * eV2Ha + phi_OHP_au

    E_solution_au = fsolve(ohl_field_equation, x0=0.1)


    E_OHL_au = E_solution_au[np.argsort(np.abs(E_solution_au))[-1]]
    E_OHL = E_OHL_au * Ha2eV / Bohr2A

    E_IHL_au = E_OHL_au * eps_ohl(E_OHL_au) / eps_IHL
    E_IHL = E_IHL_au * Ha2eV / Bohr2A

    if print_data:
        print(f"eps_OHL = {eps_ohl(E_OHL_au):.6f}")
        print(f"E_OHL = {E_OHL:.6f} V/A")
        print(f"E_IHL = {E_IHL:.6f} V/A")
        print()

    kappa = np.sqrt((8 * np.pi * n_ions_au) / (eps_st * kB * T * eV2Ha)) / Bohr2A

    def edl_potential_drop(R):
        if is_number(R):
            if R <= dIHL:
                return EvsPZFC - R * E_IHL
            elif (dIHL < R) and (R <= dIHL + dOHL):
                return EvsPZFC - dIHL * E_IHL - (R - dIHL) * E_OHL
            else:
                return 4 * kB * T * np.arctanh(np.tanh(phi_OHP / (4 * kB * T)) * np.exp(-kappa * (R - dIHL - dOHL)))
        elif is_array(R):
            result = np.zeros(len(R))
            for i, Ri in enumerate(R):
                if Ri <= dIHL:
                    result[i] = EvsPZFC - Ri * E_IHL
                elif (dIHL < Ri) and (Ri <= dIHL + dOHL):
                    result[i] = EvsPZFC - dIHL * E_IHL - (Ri - dIHL) * E_OHL
                else:
                    result[i] = 4 * kB * T * np.arctanh(
                        np.tanh(phi_OHP / (4 * kB * T)) * np.exp(-kappa * (Ri - dIHL - dOHL)))
            return result
        else:
            return None

    return edl_potential_drop


def fermi_distribution(E, E_Fermi=0, T=298.15):
    return 1 / (np.exp((E - E_Fermi) / kB / T) + 1)


class kappa_coupling(object):

    def __init__(self, rp, ReacProtonPot, ProdProtonPot, Vel, NStates=10, mu=0, nu=0):

        if not (is_array(rp) and is_array(ReacProtonPot) and is_array(ProdProtonPot)):
            raise TypeError("'rp', 'ReacProtonPot', and 'ProdProtonPot' must be 1D arrays.")
        if is_array(Vel):
            pass
        elif is_number(Vel):
            Vel = np.ones(len(rp)) * Vel
        else:
            raise TypeError("'Vel' must be a number or an 1D array.")

        if len(ReacProtonPot) != len(rp) or len(ProdProtonPot) != len(rp) or len(Vel) != len(rp):
            raise ValueError("'rp', 'ReacProtonPot', 'ProdProtonPot', and 'Vel' must have the same dimension. ")

        if np.abs(np.log2(len(rp)) - int(np.log2(len(rp)))) > 1e-4:
            raise ValueError("The number of grid points should be some interger power of 2. ")

        if mu > NStates or nu > NStates:
            NState = np.max(NStates, mu, nu)

        self.rp = rp
        self.ReacProtonPot = ReacProtonPot
        self.ProdProtonPot = ProdProtonPot
        self.Vel = Vel
        self.NStates = NStates
        self.mu = mu
        self.nu = nu

    def calc_proton_vibrational_states(self, mass=massH):
        rp_in_Bohr = self.rp * A2Bohr
        ngrid = len(rp_in_Bohr)
        sgrid = rp_in_Bohr[-1] - rp_in_Bohr[0]
        dx = sgrid / (ngrid - 1)
        E_reac_in_Ha = self.ReacProtonPot * eV2Ha
        E_prod_in_Ha = self.ProdProtonPot * eV2Ha

        eigvals_reac, eigvecs_reac = fgh_1d(ngrid, sgrid, E_reac_in_Ha, mass)
        self.ReacProtonEnergyLevels = eigvals_reac[:self.NStates] * Ha2eV

        unnormalized_wfcs_reac = np.transpose(eigvecs_reac)[:self.NStates]
        normalized_wfcs_reac = np.array(
            [wfci / np.sqrt(simps(wfci * wfci, self.rp)) for wfci in unnormalized_wfcs_reac])
        self.ReacProtonWaveFunctions = normalized_wfcs_reac

        eigvals_prod, eigvecs_prod = fgh_1d(ngrid, sgrid, E_prod_in_Ha, mass)
        self.ProdProtonEnergyLevels = eigvals_prod[:self.NStates] * Ha2eV

        unnormalized_wfcs_prod = np.transpose(eigvecs_prod)[:self.NStates]
        normalized_wfcs_prod = np.array(
            [wfci / np.sqrt(simps(wfci * wfci, self.rp)) for wfci in unnormalized_wfcs_prod])
        self.ProdProtonWaveFunctions = normalized_wfcs_prod

    def analyze_proton_potentials(self, mass=massH):

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

        def find_first_crossing(y, x):
            for i in range(1, len(x)):
                if y[i] * y[i - 1] <= 0:
                    return i, (x[i] + x[i - 1]) / 2

        rp_crossing_index, self.rp_crossing = find_first_crossing(deltaE, self.rp)
        self.E_crossing = (self.ShiftedReacProtonPot[rp_crossing_index] + self.ShiftedReacProtonPot[
            rp_crossing_index - 1] + \
                           self.ShiftedProdProtonPot[rp_crossing_index] + self.ShiftedProdProtonPot[
                               rp_crossing_index - 1]) / 4
        self.Vel_crossing = (self.Vel[rp_crossing_index] + self.Vel[rp_crossing_index - 1]) / 2

        self.slope_reac = (self.ShiftedReacProtonPot[rp_crossing_index] - self.ShiftedReacProtonPot[
            rp_crossing_index - 1]) / (self.rp[rp_crossing_index] - self.rp[rp_crossing_index - 1])
        self.slope_prod = (self.ShiftedProdProtonPot[rp_crossing_index] - self.ShiftedProdProtonPot[
            rp_crossing_index - 1]) / (self.rp[rp_crossing_index] - self.rp[rp_crossing_index - 1])

    def calculate(self, mass=massH, overlap_thresh=0.8):
        self.analyze_proton_potentials(mass)


        E0 = self.ShiftedReacProtonEnergyLevels[self.mu]

        if E0 > self.E_crossing:
            raise RuntimeError("The tunneling energy is higher than the energy at the crossing point. ")

        vt = np.sqrt(2 * (self.E_crossing - E0) * eV2Ha / mass) * Bohr2A / au2s

        self.tau_p = self.Vel_crossing / (np.abs(self.slope_reac - self.slope_prod) * vt)
        self.tau_e = hbar / self.Vel_crossing
        self.p = self.tau_p / self.tau_e
        self.kappa = np.sqrt(2 * np.pi * self.p) * np.exp(self.p * np.log(self.p) - self.p) / gamma(self.p + 1)


        self.AdiabaticProtonPotGS = 0.5 * (self.ShiftedReacProtonPot + self.ShiftedProdProtonPot - np.sqrt(
            (self.ShiftedProdProtonPot - self.ShiftedReacProtonPot) ** 2 + 4 * self.Vel_crossing ** 2))
        self.AdiabaticProtonPotES = 0.5 * (self.ShiftedReacProtonPot + self.ShiftedProdProtonPot + np.sqrt(
            (self.ShiftedProdProtonPot - self.ShiftedReacProtonPot) ** 2 + 4 * self.Vel_crossing ** 2))


        rp_in_Bohr = self.rp * A2Bohr
        Eg_au = self.AdiabaticProtonPotGS * eV2Ha
        ngrid = len(rp_in_Bohr)
        sgrid = rp_in_Bohr[-1] - rp_in_Bohr[0]
        dx = sgrid / (ngrid - 1)

        eigvals, eigvecs = fgh_1d(ngrid, sgrid, Eg_au, mass)
        self.AdiabaticGSProtonEnergyLevels = eigvals[:2 * self.NStates] * Ha2eV

        unnormalized_wfcs_adia = np.transpose(eigvecs)[:2 * self.NStates]
        normalized_wfcs_adia = np.array(
            [wfci / np.sqrt(simps(wfci * wfci, self.rp)) for wfci in unnormalized_wfcs_adia])
        self.AdiabaticGSProtonWaveFunctions = normalized_wfcs_adia

        Smunu = simps(self.ReacProtonWaveFunctions[self.mu] * self.ProdProtonWaveFunctions[self.nu], self.rp)
        sign = 1 if Smunu > 0 else -1
        wfc_symm = (self.ReacProtonWaveFunctions[self.mu] + sign * self.ProdProtonWaveFunctions[self.nu]) / np.sqrt(2)
        wfc_anti = (self.ReacProtonWaveFunctions[self.mu] - sign * self.ProdProtonWaveFunctions[self.nu]) / np.sqrt(2)
        wfc_symm /= np.sqrt(simps(wfc_symm ** 2, self.rp))
        wfc_anti /= np.sqrt(simps(wfc_anti ** 2, self.rp))

        overlap_w_symm = np.array(
            [np.abs(simps(wfci * wfc_symm, self.rp)) for wfci in self.AdiabaticGSProtonWaveFunctions])
        overlap_w_anti = np.array(
            [np.abs(simps(wfci * wfc_anti, self.rp)) for wfci in self.AdiabaticGSProtonWaveFunctions])

        index_max_overlap_symm = 0
        for i in range(2 * self.NStates):
            if overlap_w_symm[i] > overlap_w_symm[index_max_overlap_symm]:
                index_max_overlap_symm = i
            else:
                pass

        index_max_overlap_anti = 0
        for i in range(2 * self.NStates):
            if overlap_w_anti[i] > overlap_w_anti[index_max_overlap_anti] and i != index_max_overlap_symm:
                index_max_overlap_anti = i
            else:
                pass

        if overlap_w_symm[index_max_overlap_symm] < overlap_thresh or overlap_w_anti[
            index_max_overlap_anti] < overlap_thresh:
            print(
                f"WARNING: The maximum overlap between the proton vibrational wave functions in the adiabatic potential and the symmetric/antisymmetric combinations of the wave functions in diabtic potentials is less than {overlap_thresh:.1f}. ")
        if index_max_overlap_anti < index_max_overlap_symm:
            print(
                f"WARNING: The identified antisymmetric state is lower in energy than the identified symmetric state. The symmetric state is state {index_max_overlap_symm:d} and the antisymmetric state is state {index_max_overlap_anti:d}. ")
        if np.abs(index_max_overlap_anti - index_max_overlap_symm) > 1:
            print(
                f"WARNING: There are multiple states lies in between the identified symmetric and antisymmetyric states. The symmetric state is state {index_max_overlap_symm:d} and the antisymmetric state is state {index_max_overlap_anti:d}. ")

        tunneling_splitting = (eigvals[index_max_overlap_anti] - eigvals[index_max_overlap_symm]) * Ha2eV
        self.V_ad = 0.5 * tunneling_splitting

        self.V_nad = self.Vel_crossing * Smunu
        self.V_sc = self.kappa * self.V_ad

    def get_reactant_proton_states(self):
        return self.ShiftedReacProtonPot, self.ShiftedReacProtonEnergyLevels, self.ReacProtonWaveFunctions

    def get_product_proton_states(self):
        return self.ShiftedProdProtonPot, self.ShiftedProdProtonEnergyLevels, self.ProdProtonWaveFunctions

    def get_adiabatic_proton_potentials(self):
        return self.AdiabaticProtonPotGS, self.AdiabaticProtonPotES

    def get_ground_adiabatic_proton_states(self):
        return self.AdiabaticGSProtonEnergyLevels, self.AdiabaticGSProtonWaveFunctions

    def get_nonadiabaticity_parameters(self):
        return self.tau_e, self.tau_p, self.p, self.kappa

    def get_vibronic_couplings(self):
        return self.V_sc, self.V_nad, self.V_ad


class pcet(object):

    def __init__(self, ReacProtonPot, ProdProtonPot, DeltaG, Lambda, Vel=0.0434, NStates=10, NGridPot=256,
                 Smooth='bspline', **kwargs):
        if callable(ReacProtonPot):
            self.ReacProtonPot = ReacProtonPot
        elif is_array(ReacProtonPot) and len(ReacProtonPot) == 2:
            r = ReacProtonPot[0]
            pot = ReacProtonPot[1]
            rmin1 = np.min(r)
            rmax1 = np.max(r)
            if Smooth == 'fit_poly6':
                self.ReacProtonPot = fit_poly6(r, pot)
            elif Smooth == 'fit_poly8':
                self.ReacProtonPot = fit_poly8(r, pot)
            elif Smooth == 'bspline':
                self.ReacProtonPot = fit_bspline(r, pot)
            else:
                raise ValueError(
                    "'Smooth' must be set to one of the followings: 'fit_poly6', 'fit_poly8', or 'bspline'")
        else:
            raise TypeError("'ReacProtonPot' must be a 2D array with shape (N, 2) or a callable function")

        if callable(ProdProtonPot):
            self.ProdProtonPot = ProdProtonPot
        elif is_array(ProdProtonPot) and len(ProdProtonPot) == 2:
            r = ProdProtonPot[0]
            pot = ProdProtonPot[1]
            rmin2 = np.min(r)
            rmax2 = np.max(r)
            if Smooth == 'fit_poly6':
                self.ProdProtonPot = fit_poly6(r, pot)
            elif Smooth == 'fit_poly8':
                self.ProdProtonPot = fit_poly8(r, pot)
            elif Smooth == 'bspline':
                self.ProdProtonPot = fit_bspline(r, pot)
            else:
                raise ValueError(
                    "'Smooth' must be set to one of the followings: 'fit_poly6', 'fit_poly8', or 'bspline'")
        else:
            raise TypeError("'ProdProtonPot' must be a 2D array with shape (N, 2) or a callable function")

        if 'rmin' in kwargs.keys():
            rmin = kwargs['rmin']
        elif 'rmin1' in locals() and 'rmin2' in locals():
            rmin = np.min([rmin1, rmin2])
        else:
            rmin = -0.8

        if 'rmax' in kwargs.keys():
            rmax = kwargs['rmax']
        elif 'rmax1' in locals() and 'rmax2' in locals():
            rmax = np.max([rmax1, rmax2])
        else:
            rmax = 0.8
        self.rp = np.linspace(rmin, rmax, NGridPot)

        self.DeltaG = DeltaG
        self.Lambda = Lambda
        self.Vel = Vel
        self.NStates = NStates

        self.Pu = np.zeros(NStates)
        self.Suv = np.zeros((NStates, NStates))
        self.dGuv = np.zeros((NStates, NStates))
        self.kuv = np.zeros((NStates, NStates))

    def calc_proton_vibrational_states(self, mass=massH):
        rp_in_Bohr = self.rp * A2Bohr
        ngrid = len(rp_in_Bohr)
        sgrid = rp_in_Bohr[-1] - rp_in_Bohr[0]
        dx = sgrid / (ngrid - 1)
        E_reac_in_Ha = self.ReacProtonPot(self.rp) * eV2Ha
        E_prod_in_Ha = self.ProdProtonPot(self.rp) * eV2Ha

        self.MassUsedPreviously = mass

        eigvals_reac, eigvecs_reac = fgh_1d(ngrid, sgrid, E_reac_in_Ha, mass)
        self.ReacProtonEnergyLevels = eigvals_reac[:self.NStates] * Ha2eV

        unnormalized_wfcs_reac = np.transpose(eigvecs_reac)[:self.NStates]
        normalized_wfcs_reac = np.array(
            [wfci / np.sqrt(simps(wfci * wfci, self.rp)) for wfci in unnormalized_wfcs_reac])
        self.ReacProtonWaveFunctions = normalized_wfcs_reac

        eigvals_prod, eigvecs_prod = fgh_1d(ngrid, sgrid, E_prod_in_Ha, mass)
        self.ProdProtonEnergyLevels = eigvals_prod[:self.NStates] * Ha2eV

        unnormalized_wfcs_prod = np.transpose(eigvecs_prod)[:self.NStates]
        normalized_wfcs_prod = np.array(
            [wfci / np.sqrt(simps(wfci * wfci, self.rp)) for wfci in unnormalized_wfcs_prod])
        self.ProdProtonWaveFunctions = normalized_wfcs_prod

    def calc_reactant_state_distribution(self, T=298):
        Boltzmann_factors = np.exp(-self.ReacProtonEnergyLevels / kB / T)
        partition_func = np.sum(Boltzmann_factors)
        self.Pu = Boltzmann_factors / partition_func
        return self.Pu

    def calc_proton_overlap_matrix(self):
        for u in range(self.NStates):
            for v in range(self.NStates):
                self.Suv[u, v] = simps(self.ReacProtonWaveFunctions[u] * self.ProdProtonWaveFunctions[v], self.rp)
        return self.Suv

    def calc_reaction_free_energy_matrix(self):
        for u in range(self.NStates):
            for v in range(self.NStates):
                self.dGuv[u, v] = self.DeltaG + (self.ProdProtonEnergyLevels[v] - self.ProdProtonEnergyLevels[0]) - (
                    self.ReacProtonEnergyLevels[u] - self.ReacProtonEnergyLevels[0])
        return self.dGuv

    def calc_rate_contribution_matrix(self, T=298):
        k0 = 2 * np.pi / hbar * self.Vel * self.Vel
        self.Iuv = 1 / np.sqrt(4 * np.pi * self.Lambda * kB * T) * np.exp(
            -(self.dGuv + self.Lambda) ** 2 / (4 * self.Lambda * kB * T))
        self.kuv = k0 * np.matmul(np.diag(self.Pu), self.Suv * self.Suv * self.Iuv)
        return self.kuv

    def calculate(self, mass=massH, T=298, reuse_saved_proton_states=False):
        if not (hasattr(self, 'ReacProtonEnergyLevels') and hasattr(self, 'ProdProtonEnergyLevels') and hasattr(self,
                                                                                                                'ReacProtonWaveFunctions') and hasattr(
            self, 'ProdProtonWaveFunctions') and hasattr(self, 'MassUsedPreviously')):
            reuse_saved_proton_states = False
        elif self.MassUsedPreviously != mass:
            reuse_saved_proton_states = False

        if not reuse_saved_proton_states:
            self.calc_proton_vibrational_states(mass)
            self.calc_proton_overlap_matrix()

        self.calc_reactant_state_distribution(T)
        self.calc_reaction_free_energy_matrix()
        self.calc_rate_contribution_matrix(T)

        self.k_tot = np.sum(self.kuv)
        return self.k_tot

    def set_parameters(self, **kwargs):
        if 'DeltaG' in kwargs.keys():
            self.DeltaG = kwargs['DeltaG']
        if 'Lambda' in kwargs.keys():
            self.Lambda = kwargs['Lambda']
        if 'Vel' in kwargs.keys():
            self.Vel = kwargs['Vel']

    def get_reactant_proton_states(self):
        return self.ReacProtonEnergyLevels, self.ReacProtonWaveFunctions

    def get_product_proton_states(self):
        return self.ProdProtonEnergyLevels, self.ProdProtonWaveFunctions

    def get_reactant_state_distribution(self):
        return self.Pu

    def get_proton_overlap_matrix(self):
        return self.Suv

    def get_reaction_free_energy_matrix(self):
        return self.dGuv

    def get_activation_free_energy_matrix(self):
        return (self.dGuv + self.Lambda) ** 2 / (4 * self.Lambda)

    def get_rate_contribution_matrix(self):
        return self.kuv

    def get_total_rate_constant(self):
        return self.k_tot
