from .parameters import physical_constants as const
from .neutrinos import NeutrinoMassSplits, Omeganuh2
from .pspec import TransferFunctions, MatterPowerSpectra, BaryonPowerSpectra
from .pk2d import DefaultPowerSpectrum, Pk2D
from .interpolate import Interpolator1D
from .errors import CCLError

import numpy as np
import warnings
import copy


__all__ = ("Cosmology", "CosmologyVanillaLCDM", "CosmologyCalculator")


class CosmologyParams:
    """Objects of this class store cosmological parameters."""
    # Density parameters (ρ_x / ρ_crit).
    Omega_c = None        # CDM
    Omega_b = None        # baryons
    Omega_m = None        # all matter
    Omega_k = None        # curvature
    sqrtk = None          # square root of the magnitude of curvature, k
    k_sign = None         # sign of the curvature

    # Dark energy.
    w0 = None
    wa = None

    # Hubble parameter.
    h = None

    # Neutrino properties.
    Neff = None           # No. of rel. neutrino species in the early universe.
    N_nu_mass = None      # No. of nonrel. neutrino species today.
    N_nu_rel = None       # No. of rel. neutrino species today.
    m_nu = None           # Neutrino masses.
    sum_nu_masses = None  # Sum of neutrino masses.
    Omega_nu_mass = None  # Ω_nu for massive neutrinos.
    Omega_nu_rel = None   # Ω_nu for massless neutrinos.
    T_ncdm = None         # Non-CDM temperature in units of photon temperature.

    # Primordial power spectra.
    A_s = None
    n_s = None

    # Radiation parameters.
    Omega_g = None
    T_CMB = None

    # mu-Sigma quasistatic parametrization of modified gravity.
    mu_0 = None
    sigma_0 = None
    c1_mg = None
    c2_mg = None
    lambda_mg = None

    # Derived parameters.
    sigma8 = None         # σ8: RMS variance in 8 Mpc/h top-hat spheres.
    Omega_l = None        # Ω_Λ: Density parameter of dark energy.


class CosmologyConfig:
    """Objects of this class store the cosmological configuration."""
    transfer_function = None
    matter_power_spectrum = None
    baryons_power_spectrum = None
    extra_parameters = None


class CosmologyData:
    """Objects of this class store slines for distance and acceleration
    equations. All entries are functions of the scale factor.
    """
    # Distances (in Mpc).
    E = None                           # E(z) = H(z)/H_0
    chi = None                         # chi(a)
    achi = None                        # a(chi)
    lookback = None                    # t(a=1) - t(a)
    age0 = None                        # t(a=1)

    # Growth of density perturbations.
    growth = None                      # D(a)
    fgrowth = None                     # d log(a) / da
    growth0 = None                     # normalization

    # Variance of density perturbations.
    logsigma = None                    # log(σ(M))

    # Real-space splines for RSD.
    rsd_splines = [None, None, None]   #
    rsd_splines_scalefactor = None     #


def _methods_of_cosmology(cls=None, *, modules=[]):
    """Assign all functions in ``modules`` which take ``cosmo`` as their
    first argument as methods of the class ``cls``.
    """
    import functools

    if cls is None:
        # called with parentheses
        return functools.partial(_methods_of_cosmology, modules=modules)

    from importlib import import_module
    from inspect import getmembers, isfunction, signature

    pkg = __name__.rsplit(".")[0]
    modules = [import_module(f".{module}", pkg) for module in modules]
    funcs = [getmembers(module, isfunction) for module in modules]
    funcs = [func for sublist in funcs for func in sublist]

    for name, func in funcs:
        pars = signature(func).parameters
        if pars and list(pars)[0] == "cosmo":
            setattr(cls, name, func)

    return cls


_modules = ["background", "boltzmann", "cells", "musigma", "power", "pspec"]


@_methods_of_cosmology(modules=_modules)
class Cosmology:
    """
    """

    def __init__(self, *,
                 Omega_c, Omega_b, h,
                 n_s, sigma8=None, A_s=None,
                 Omega_k=0.,
                 Omega_g=None, T_CMB=2.725,
                 Neff=const.NEFF, m_nu=0., m_nu_type="normal", T_ncdm=0.71611,
                 w0=-1., wa=0.,
                 mu_0=0., sigma_0=0., c1_mg=1., c2_mg=1., lambda_mg=0.,
                 transfer_function="camb",
                 matter_power_spectrum="hmcode",
                 baryons_power_spectrum="nobaryons",
                 extra_parameters={}):

        self.params = CosmologyParams()
        self.config = CosmologyConfig()
        self.data = CosmologyData()
        self._pkl, self._pknl = {}, {}

        self._fill_config(
            transfer_function=TransferFunctions(transfer_function),
            matter_power_spectrum=MatterPowerSpectra(matter_power_spectrum),
            baryons_power_spectrum=BaryonPowerSpectra(baryons_power_spectrum),
            extra_parameters=extra_parameters)

        TRF, trf = TransferFunctions, transfer_function
        MPS, mps = MatterPowerSpectra, matter_power_spectrum
        if MPS(mps) == MPS.CAMB and TRF(trf) != TRF.CAMB:
            raise CCLError(
                "To compute the non-linear matter power spectrum with CAMB "
                "the transfer function should be 'boltzmann_camb'.")
        if Omega_k < -1.0135:
            raise ValueError("Omega_k must be more than -1.0135.")
        if not (bool(sigma8) ^ bool(A_s)):
            raise ValueError("Provide either sigma8 or A_s.")
        if bool(Omega_g) and bool(T_CMB):
            raise ValueError("Provide either Omega_g or T_CMB.")

        self._initialize_curvature(Omega_k, h)
        self._initialize_radiation(Omega_g, T_CMB, T_ncdm, h)
        self._initialize_neutrinos(Neff, m_nu, m_nu_type, T_CMB, T_ncdm, h)
        self._initialize_late_times(Omega_b, Omega_c, Omega_k)

        self._fill_params(
            Omega_c=Omega_c, Omega_b=Omega_b, h=h, n_s=n_s, sigma8=sigma8,
            A_s=A_s, Omega_k=Omega_k, w0=w0, wa=wa, mu_0=mu_0, sigma_0=sigma_0,
            c1_mg=c1_mg, c2_mg=c2_mg, lambda_mg=lambda_mg)

    def _initialize_curvature(self, Omega_k, h):
        """Parameters related to curvature."""
        k_sign = -np.sign(Omega_k) if abs(Omega_k) > 1e-6 else 0.
        sqrtk = np.sqrt(abs(Omega_k)) * h / const.CLIGHT_HMPC
        self._fill_params(k_sign=k_sign, sqrtk=sqrtk)

    def _initialize_radiation(self, Omega_g, T_CMB, T_ncdm, h):
        """Parameters related to radiation."""
        rho_crit = (const.RHO_CRITICAL * h**2
                    * const.SOLAR_MASS / const.MPC_TO_METER**3)
        if Omega_g is None:
            rho_g = 4 * const.STBOLTZ / const.CLIGHT**3 * T_CMB**4
            Omega_g = rho_g / rho_crit
        else:
            rho_g = Omega_g * rho_crit
            T_CMB = (const.CLIGHT**3 * rho_g / (4 * const.STBOLTZ))**(0.25)
        self._fill_params(Omega_g=Omega_g, T_CMB=T_CMB, T_ncdm=T_ncdm)

    def _mass_split_newton(self, m_nu, m_nu_list, params, atol=1e-15):
        """Split the neutrino masses using the Newton-Raphson method."""
        m_nu_list, params = map(np.asarray, [m_nu_list, params])
        with warnings.catch_warnings():
            # Suppress `RuntimeWarning` during root finding.
            warnings.simplefilter("ignore")
            while np.abs(m_nu - np.sum(m_nu_list)) > atol:
                deriv = 1 + np.sum(m_nu_list[0] / m_nu_list[1:])
                m_nu_list[0] -= (np.sum(m_nu_list) - m_nu) / deriv
                m_nu_list[1:] = np.sqrt(m_nu_list[0]**2 + params)
        return m_nu_list.tolist()

    def _split_neutrino_mass(self, m_nu, m_nu_type):
        """Split the sum of neutrino masses into a 3-sequence."""
        if hasattr(m_nu, "__len__"):
            if len(m_nu) != 3:
                raise ValueError("m_nu must be float or a 3-sequence.")
            if m_nu_type in ["normal", "inverted", "equal"]:
                raise ValueError("Sequence of neutrino masses not compatible "
                                 f"with m_nu_type={m_nu_type}.")
            return m_nu.copy()

        if m_nu < 1e-15:
            return [0., 0., 0.]

        split = NeutrinoMassSplits
        ΔM = [const.DELTAM12_sq, const.DELTAM13_sq_pos, const.DELTAM13_sq_neg]
        if split(m_nu_type) == split.NORMAL:
            if m_nu < np.sqrt(ΔM[0]) + np.sqrt(ΔM[1]):
                raise ValueError("m_nu must be larger than 0.0592 "
                                 "for the normal mass hierarchy.")
            p0 = np.sqrt([0, ΔM[0], ΔM[1]])
            params = [ΔM[0], ΔM[1]]
            return self._mass_split_newton(m_nu, p0, params)

        if split(m_nu_type) == split.INVERTED:
            if m_nu < np.sqrt(-(ΔM[0] + ΔM[2])) + np.sqrt(-ΔM[2]):
                raise ValueError("m_nu must be larger than 0.0978 "
                                 "for the inverted mass hierarchy.")
            p0 = np.sqrt([0, -(ΔM[0] + ΔM[2]), -ΔM[2]])
            params = [ΔM[0], ΔM[2]]
            return self._mass_split_newton(m_nu, p0, params)

        if split(m_nu_type) == split.EQUAL:
            return [m_nu/3, m_nu/3, m_nu/3]

        if split(m_nu_type) == split.SINGLE:
            return [m_nu, 0., 0.]

    def _initialize_neutrinos(self, Neff, m_nu, m_nu_type, T_CMB, T_ncdm, h):
        """Split neutrino mass and calculate number of each species."""
        m_nu_list = self._split_neutrino_mass(m_nu, m_nu_type)

        # Calculate the number of relativistic neutrino species today.
        N_nu_mass, N_nu_rel = 0, Neff
        if (np.abs(m_nu) > 1e-15).any():
            N_nu_mass = sum([m > 0.00017 for m in m_nu_list])
            N_nu_rel = Neff - N_nu_mass * T_ncdm**4 * (4/11)**(-4/3)
            if N_nu_rel < 0:
                raise ValueError("Negative number of relativistic neutrino "
                                 "species for Neff and m_nu combination.")

        rho_crit = (const.RHO_CRITICAL * h**2
                    * const.SOLAR_MASS / const.MPC_TO_METER**3)
        T_nu = T_CMB * (4/11)**(1/3)
        rho = N_nu_rel * (7/8) * 4 * const.STBOLTZ / const.CLIGHT**3 * T_nu**4
        Omega_nu_rel = rho / rho_crit

        # Fill an array with the non-relativistic neutrino masses.
        # TODO: tidy-up this block
        m_nu_final = [0., ]
        if N_nu_mass > 0:
            m_nu_final = [0 for i in range(N_nu_mass)]
            relativistic = [0, 0, 0]
            for i in range(N_nu_mass):
                for j in range(3):
                    if m_nu_list[j] > 0.00017 and relativistic[j] == 0:
                        relativistic[j] = 1
                        m_nu_final[i] = m_nu_list[j]
                        break
        sum_nu_masses = sum(m_nu_final)

        # Phase-space integral for non-relativistic neutrinos.
        Omega_nu_mass = 0.
        if N_nu_mass > 0:
            T_CMB, T_ncdm = self.params.T_CMB, self.params.T_ncdm
            Omega_nu_mass = Omeganuh2(a=1.,
                                      m_nu=m_nu_final,
                                      T_CMB=T_CMB,
                                      T_ncdm=T_ncdm) / h**2

        self._fill_params(
            Omega_nu_mass=Omega_nu_mass, Omega_nu_rel=Omega_nu_rel,
            Neff=Neff, N_nu_rel=N_nu_rel, N_nu_mass=N_nu_mass,
            m_nu=m_nu_final, sum_nu_masses=sum_nu_masses)

    def _initialize_late_times(self, Omega_b, Omega_c, Omega_k):
        params = self.params
        Omega_m = Omega_b + Omega_c + params.Omega_nu_mass
        Omega_l = 1 - Omega_m - params.Omega_g - params.Omega_nu_rel - Omega_k
        self._fill_params(Omega_m=Omega_m, Omega_l=Omega_l)

    def _fill_params(self, **kwargs):
        """Fill cosmological parameters from a ``kwargs`` dictionary."""
        [setattr(self.params, param, value) for param, value in kwargs.items()]

    def _fill_config(self, **kwargs):
        """Fill the cosmological configuration from a ``kwargs`` dictionary."""
        [setattr(self.config, param, value) for param, value in kwargs.items()]

    def get_extra_parameters(self, model=None):
        """Get the dictionary of extra parameters for a particular model."""
        if model is None:
            return self.config.extra_parameters
        try:
            return self.config.extra_parameters[model]
        except KeyError:
            return {}

    def update_extra_parameters(self, **kwargs):
        self.config.extra_parameters.update(kwargs)
        # TODO: Make BCM parameters updateable.

    @property
    def has_distances(self):
        """Check if the distances have been computed."""
        return None not in [self.data.E, self.data.chi, self.data.achi,
                            self.data.lookback]

    @property
    def has_growth(self):
        """Check if the growth function has been computed."""
        return None not in [self.data.growth, self.data.fgrowth,
                            self.data.growth0]

    @property
    def has_linear_power(self):
        """Check if the linear power spectra have been computed."""
        return bool(self._pkl)

    @property
    def has_nonlin_power(self):
        """Checks if the non-linear power spectra have been computed."""
        return bool(self._pknl)

    @property
    def has_sigma(self):
        """Check if sigma(M) is computed."""
        return self.data.logsigma is not None

    def __getitem__(self, key):
        return getattr(self.params, key)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return

    def __getstate__(self):
        return copy.deepcopy(self.__dict__)

    def __setstate__(self, state):
        self.__dict__ = state

    def __repr__(self):
        """TODO"""
        # TODO
        return "TODO"

    def copy(self):
        """Return a copy of this object."""
        return copy.copy(self)


def CosmologyVanillaLCDM(**kwargs):
    """A cosmology with typical flat Lambda-CDM parameters.

    .. code-block:: python

        Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.67, n_s=0.96, sigma8=0.81)

    Arguments
    ---------
    **kwargs : dict
        Dictionary of other cosmological parameters.
        It should not contain any of the ΛCDM parameters:
        :obj:`["Omega_c", "Omega_b", "h", "n_s", "sigma8", "A_s"]`.
    """
    p = {'Omega_c': 0.25, 'Omega_b': 0.05, 'h': 0.67, 'n_s': 0.96,
         'sigma8': 0.81, 'A_s': None}
    if any(k in p for k in kwargs):
        raise ValueError(f"Can't set ΛCDM parameters: {list(p.keys())}.")
    return Cosmology(**{**p, **kwargs})


class CosmologyCalculator(Cosmology):
    """
    """

    def __init__(self, *,
                 Omega_c, Omega_b, h,
                 n_s=None, sigma8=None, A_s=None,
                 Omega_k=0.,
                 Omega_g=None, T_CMB=2.725,
                 Neff=3.044, m_nu=0., m_nu_type="normal", T_ncdm=0.71611,
                 w0=-1., wa=0.,
                 background={}, growth={},
                 pk_linear={}, pk_nonlin={}, nonlinear_model=None):

        super().__init__(
            Omega_c=Omega_c, Omega_b=Omega_b, h=h,
            n_s=n_s, sigma8=sigma8, A_s=A_s,
            Omega_k=Omega_k, Omega_g=Omega_g, T_CMB=T_CMB,
            Neff=Neff, m_nu=m_nu, m_nu_type=m_nu_type, T_ncdm=T_ncdm,
            w0=w0, wa=wa,
            transfer_function='calculator' if pk_linear else None,
            matter_power_spectrum=('calculator' if pk_nonlin
                                   or nonlinear_model else None))

        if background:
            self._initialize_background(background)
        if growth:
            self._initialize_growth(growth)
        if pk_linear:
            self._parse_power_spectrum_dictionary(pk_linear, self._pkl)
        if pk_nonlin:
            self._parse_power_spectrum_dictionary(pk_nonlin, self._pknl)
        if nonlinear_model is not None:
            self._apply_nonlinear_model(nonlinear_model)

    def _check_scale_factor(self, a):
        # Scale factor array should contain today.
        from background import _eps
        if not a[0] - _eps <= 1 <= a[-1] + _eps:
            raise ValueError("Input scale factor should contain today at a=1.")

    def _initialize_background(self, background):
        a = background["a"]
        self._check_scale_factor()

        # E(z)
        hoh0 = background["h_over_h0"]
        self.data.E = Interpolator1D(a, hoh0)

        # chi(z) & a(chi)
        if "chi" in background:
            self.data.chi = Interpolator1D(a, background["chi"])
            self.data.achi = Interpolator1D(background["chi"], a)
        else:
            integrand = const.CLIGHT_HMPC / (hoh0 * a**2 * self["h"])
            integral = Interpolator1D(a, integrand).f.antiderivative()
            chi = integral(1.0) - integral(a)
            self.data.chi = Interpolator1D(a, chi)
            self.data.achi = Interpolator1D(chi[::-1], a[::-1])

        # lookback time
        if "lookback" in background:
            self.data.lookback = Interpolator1D(a, background["lookback"])
        else:
            t_H = const.MPC_TO_METER / 1e14 / const.YEAR / self["h"]
            integral = Interpolator1D(a, 1/(a*hoh0)).f.antiderivative()
            a_eval = np.r[1.0, a]  # make a single call to the spline
            vals = integral(a_eval)
            time = t_H * (vals[0] - vals[1:])
            self.data.lookback = Interpolator1D(a, time)

    def _initalize_growth(self, growth):
        a = growth["a"]
        self._check_scale_factor()
        gf = growth["growth_factor"]
        self.data.growth = Interpolator1D(a, gf)
        if "growth_rate" in growth:
            self.data.fgrowth = Interpolator1D(a, growth["growth_rate"])
        else:
            fg = Interpolator1D(np.log(a), np.log(gf)).f.derivative()
            self.data.fgrowth = Interpolator1D(a, fg(np.log(a)))
        self.data.growth0 = 1

    def _parse_power_spectrum_dictionary(self, dic, destination):
        self._check_scale_factor(dic["a"])
        pks = {name: pk for name, pk in dic.items() if name not in ["a", "k"]}
        for name, pk in pks.items():
            if name.count(":") != 1:
                raise ValueError("Power spectrum label should be 'q1:q2'.")

            is_logp = (pk > 0).all()
            if is_logp:
                np.log(pk, out=pk)

            destination[name] = Pk2D(a_arr=dic["a"], lk_arr=np.log(dic["lk"]),
                                     pk_arr=pk, is_logp=is_logp)

    def _apply_nonlinear_model(self, model):
        if not isinstance(model, [str, dict]):
            raise TypeError("`nonlinear_model` must be str, dict, or None.")
        if isinstance(model, str) and not self.has_linear_power:
            raise ValueError("No linear P(k) to apply the non-linear model.")

        nl = {n: model for n in self._pkl} if isinstance(model, str) else model
        for name, model in nl.items():
            if name in self._pknl:
                continue
            if name not in self._pkl:
                raise ValueError(f"{name} is not a known linear P(k).")
            if model is None:
                if name == DefaultPowerSpectrum:
                    raise ValueError("Non-linear matter P(k) does not exist.")
                continue

            pkl = self._pkl[name]
            self._pknl[name] = Pk2D.apply_model(self, pkl, model)
