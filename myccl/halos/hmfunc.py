from .massdef import MassDef, dc_NakamuraSuto, Dv_BryanNorman
from ..pyutils import get_broadcastable
from ..parameters import physical_constants as const
from ..interpolate import Interpolator1D

import numpy as np
from abc import ABC, abstractmethod


__all__ = ("MassFunc", "MassFuncPress74", "MassFuncSheth99",
           "MassFuncJenkins01", "MassFuncTinker08", "MassFuncTinker10",
           "MassFuncDespali16", "MassFuncBocquet16", "MassFuncWatson13",
           "MassFuncAngulo12")


class MassFunc(ABC):
    r"""Calculations of halo mass functions.

    We assume that all mass functions can be written as

    .. math::

        \frac{\mathrm{d}n}{\mathrm{d}\log_{10}M} =
        f(\sigma_M) \, \frac{\rho_M}{M} \,
        \frac{\mathrm{d}\log \sigma_M}{\mathrm{d}\log_{10} M}

    where :math:`\sigma_M^2` is the overdensity variance on spheres with a
    radius given by the Lagrangian radius for mass :math:`M`.

    Specific mass function parametrizations can be created by subclassing
    and implementing ``_get_fsigma``.

    Parameters
    ----------
    mass_def : :class:`~pyccl.halos.massdef.MassDef`
        Mass definition to use when computing the mass function.
    """

    def __init__(self, mass_def=None):
        if self._check_mass_def(mass_def):
            raise ValueError(
                f"Mass definition {mass_def.Delta}-{mass_def.rho_type} "
                f"is not compatible with mass function {self.name}.")
        self.mass_def = mass_def
        self._setup()

    @abstractmethod
    def _check_mass_def(self, mass_def):
        """Runs before ``__init__`` to flag mass definition inconsistencies.

        Arguments
        ---------
        mass_def : :class:`~pyccl.halos.massdef.MassDef`
            Mass definition.

        Returns
        -------
        check : bool
            ``True`` if the input mass definition is inconsistent with this
            parametrization. ``False`` otherwise.
        """

    def _setup(self):
        """Runs after ``__init__`` to initialize internal attributes."""

    def _get_Delta_m(self, cosmo, a):
        r"""Translate SO mass definitions to :math:`\rho_{\rm m}`.
        This method is used in the Tinker mass functions.
        """
        Δ = self.mass_def.get_Delta(cosmo, a, squeeze=False)
        if self.mass_def.rho_type == 'matter':
            return Δ

        Ω_this = cosmo.omega_x(a, self.mass_def.rho_type, squeeze=False)
        Ω_m = cosmo.omega_x(a, "matter", squeeze=False)
        return Δ * Ω_this / Ω_m

    @abstractmethod
    def _get_fsigma(self, cosmo, sigM, a, lnM):
        r"""Specific implementation of mass function :math:`f(\sigma_M)`.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        sigM : (na, nsigM) ndarray
            Standard deviation of the overdensity field at the scale of a halo.
            This is calculated at the values of the scale factor in ``a``.
        a : (na, 1) ndarray
            Scale factor, corresponding to the rows of ``sigM``.
        lnM : (1, nlnM) ndarray
            Natural logarithm of the halo mass in :math:`\rm M_\odot`.
            This is provided in addition to ``sigM`` for convenience.

        Returns
        -------
        fsigma : (na, nsigM) ndarray
            :math:`f(\sigma_M)` function.
        """

    def get_mass_function(self, cosmo, M, a, *, squeeze=True):
        r"""Compute the mass function.

        .. math::

            \frac{\mathrm{d}n}{\mathrm{d} \log_{10} M}

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        M : float or (nM,) array_like
            Halo mass in units of :math:`\rm M_\odot`.
        a : float or (na,) array_like
            Scale factor.
        squeeze : bool, optional
            Squeeze extra dimensions of size (1,) in the output.
            The default is True.

        Returns
        -------
        mf : float or (na, nM,) array_like
            Mass function in units of :math:`\rm Mpc^{-3}` (comoving).
        """
        a, M = map(np.asarray, [a, M])
        a, M = get_broadcastable(a, M)
        sigM = cosmo.sigmaM(M, a, squeeze=False)
        dlns_dlogM = cosmo.dlnsigM_dlogM(M, a, squeeze=False)

        f = self._get_fsigma(cosmo, sigM, a, np.log(10) * np.log(M))
        ρ = const.RHO_CRITICAL * cosmo['Omega_m'] * cosmo['h']**2
        mf = f * ρ * dlns_dlogM / M
        return mf.squeeze()[()] if squeeze else mf

    @classmethod
    def from_name(cls, name):
        """ Returns mass function subclass from name string

        Arguments
        ---------
        name : str
            Mass function name.

        Returns
        -------
        MassFunc : :class:`~pyccl.halos.hmfunc.MassFunc`
            ``MassFunc`` subclass corresponding to the input name.
        """
        mass_functions = {c.name: c for c in cls.__subclasses__()}
        return mass_functions[name]


class MassFuncPress74(MassFunc):
    """Mass function described in ``1974ApJ...187..425P``.
    This parametrization is only valid for 'fof' masses.

    Parameters
    ----------
    mass_def : :class:`~pyccl.halos.massdef.MassDef`, optional
        Mass definition.
        **Note**: This parametrization is only valid for FoF masses.
    """
    name = 'Press74'

    def __init__(self, mass_def=MassDef('fof', 'matter')):
        super().__init__(mass_def=mass_def)

    def _setup(self):
        self.norm = np.sqrt(2/np.pi)

    def _check_mass_def(self, mass_def):
        return mass_def.Delta != "fof"

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        nu = const.DELTA_C / sigM
        return self.norm * nu * np.exp(-0.5 * nu**2)


class MassFuncSheth99(MassFunc):
    r"""Mass function described in :arXiv:astro-ph/9901122.
    This parametrization is only valid for 'fof' masses.

    Parameters
    ----------
    mass_def : :class:`~pyccl.halos.massdef.MassDef`, optional
        Mass definition.
        **Note**: This parametrization is only valid for FoF masses.
    delta_c_fit : bool, optional
        Whether to use :math:`\delta_{\rm c}` from the fit of Nakamura & Suto
        (1997). If False, use :math:`\delta_c` from linear spherical collapse.
        The default is False.
    """
    name = 'Sheth99'

    def __init__(self, mass_def=MassDef('fof', 'matter'), delta_c_fit=False):
        self.delta_c_fit = delta_c_fit
        super().__init__(mass_def)

    def _setup(self):
        self.A = 0.21615998645
        self.p = 0.3
        self.a = 0.707

    def _check_mass_def(self, mass_def):
        return mass_def.Delta != "fof"

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        δc = const.DELTA_C
        if self.delta_c_fit:
            δc = dc_NakamuraSuto(cosmo, a, squeeze=False)
        nu = δc / sigM
        anu2 = self.a * nu**2
        return nu * self.A * (1. + anu2**(-self.p)) * np.exp(-anu2/2.)


class MassFuncJenkins01(MassFunc):
    """Mass function described in :arXiv:astro-ph/0005260.

    Parameters
    ----------
    mass_def : :class:`~pyccl.halos.massdef.MassDef`, optional
        Mass definition.
        **Note**: This parametrization is only valid for FoF masses.
    """
    name = 'Jenkins01'

    def __init__(self, mass_def=MassDef('fof', 'matter')):
        super().__init__(mass_def=mass_def)

    def _setup(self):
        self.A = 0.315
        self.b = 0.61
        self.q = 3.8

    def _check_mass_def(self, mass_def):
        return mass_def.Delta != "fof"

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        return self.A * np.exp(-np.fabs(-np.log(sigM) + self.b)**self.q)


class MassFuncTinker08(MassFunc):
    r"""Mass function described in :arXiv:0803.2706.

    .. note::

        This parametrization is valid for SO-matter based mass definitions with
        :math:`200 < \Delta < 3200`. CCL will internally translate SO-critical
        based mass definitions to SO-matter.

    Parameters
    ----------
    mass_def : :class:`~pyccl.halos.massdef.MassDef`, optional
        Mass definition for this mass function parametrization.
        The default is 200m.
    """
    name = 'Tinker08'

    def __init__(self, mass_def=MassDef(200, "matter")):
        super().__init__(mass_def=mass_def)

    def _pd(self, ld):
        return 10.**(-(0.75/(ld - 1.8750612633))**1.2)

    def _setup(self):
        delta = np.array([200.0, 300.0, 400.0, 600.0, 800.0,
                          1200.0, 1600.0, 2400.0, 3200.0])
        alpha = np.array([0.186, 0.200, 0.212, 0.218, 0.248,
                          0.255, 0.260, 0.260, 0.260])
        beta = np.array([1.47, 1.52, 1.56, 1.61, 1.87,
                         2.13, 2.30, 2.53, 2.66])
        gamma = np.array([2.57, 2.25, 2.05, 1.87, 1.59,
                          1.51, 1.46, 1.44, 1.41])
        phi = np.array([1.19, 1.27, 1.34, 1.45, 1.58,
                        1.80, 1.97, 2.24, 2.44])
        ldelta = np.log10(delta)
        self.pA0 = Interpolator1D(ldelta, alpha)
        self.pa0 = Interpolator1D(ldelta, beta)
        self.pb0 = Interpolator1D(ldelta, gamma)
        self.pc = Interpolator1D(ldelta, phi)

    def _check_mass_def(self, mass_def):
        Δ = mass_def.Delta
        return not ((isinstance(Δ, (int, float)) and 200 <= Δ <= 3200)
                    or Δ == "vir")

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        ld = np.log10(self._get_Delta_m(cosmo, a))
        pA = self.pA0(ld) * a**0.14
        pa = self.pa0(ld) * a**0.06
        pb = self.pb0(ld) * a**self._pd(ld)
        return pA * ((pb / sigM)**pa + 1) * np.exp(-self.pc(ld) / sigM**2)


class MassFuncDespali16(MassFunc):
    """Mass function described in :arXiv:1507.05627.

    Parameters
    ----------
    mass_def : :class:`~pyccl.halos.massdef.MassDef`, optional
        Mass definition. This parametrization accepts any SO masses.
        The default is 200m.
    """
    name = 'Despali16'

    def __init__(self, mass_def=MassDef(200, "matter"), ellipsoidal=False):
        self.ellipsoidal = ellipsoidal
        super().__init__(mass_def=mass_def)

    def _check_mass_def(self, mass_def):
        return mass_def.Delta == "fof"

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        δc = dc_NakamuraSuto(cosmo, a, squeeze=False)
        Δv = Dv_BryanNorman(cosmo, a, squeeze=False)

        Ω = cosmo.omega_x(a, self. mass_def.rho_type, squeeze=False)
        x = np.log10(Ω * self.mass_def.get_Delta(cosmo, a, squeeze=False) / Δv)

        if self.ellipsoidal:
            A = -0.1768 * x + 0.3953
            a = 0.3268 * x**2 + 0.2125 * x + 0.7057
            p = -0.04570 * x**2 + 0.1937 * x + 0.2206
        else:
            A = -0.1362 * x + 0.3292
            a = 0.4332 * x**2 + 0.2263 * x + 0.7665
            p = -0.1151 * x**2 + 0.2554 * x + 0.2488

        nu = δc / sigM
        nu_p = a * nu**2

        return (2.0 * A * np.sqrt(nu_p / 2.0 / np.pi)
                * np.exp(-0.5 * nu_p) * (1.0 + nu_p**-p))


class MassFuncTinker10(MassFunc):
    r"""Mass function described in :arXiv:1001.3162.

    .. note::

        This parametrization is valid for SO-matter based mass definitions with
        :math:`200 < \Delta < 3200`. CCL will internally translate SO-critical
        based mass definitions to SO-matter.

    Parameters
    ----------
    mass_def : :class:`~pyccl.halos.massdef.MassDef`, optional
        Mass definition for this mass function parametrization.
        The default is 200m.
    norm_all_z : bool, optional
        Whether the mass function should be normalized
        at all :math:`z` or just at :math:`z=0`.
        The default is False.
    """
    name = 'Tinker10'

    def __init__(self, mass_def=MassDef(200, "matter"), norm_all_z=False):
        self.norm_all_z = norm_all_z
        super().__init__(mass_def=mass_def)

    def _setup(self):
        delta = np.array([200.0, 300.0, 400.0, 600.0, 800.0,
                          1200.0, 1600.0, 2400.0, 3200.0])
        alpha = np.array([0.368, 0.363, 0.385, 0.389, 0.393,
                          0.365, 0.379, 0.355, 0.327])
        beta = np.array([0.589, 0.585, 0.544, 0.543, 0.564,
                         0.623, 0.637, 0.673, 0.702])
        gamma = np.array([0.864, 0.922, 0.987, 1.09, 1.20,
                          1.34, 1.50, 1.68, 1.81])
        phi = np.array([-0.729, -0.789, -0.910, -1.05, -1.20,
                        -1.26, -1.45, -1.50, -1.49])
        eta = np.array([-0.243, -0.261, -0.261, -0.273, -0.278,
                        -0.301, -0.301, -0.319, -0.336])

        ldelta = np.log10(delta)
        self.pA0 = Interpolator1D(ldelta, alpha)
        self.pa0 = Interpolator1D(ldelta, eta)
        self.pb0 = Interpolator1D(ldelta, beta)
        self.pc0 = Interpolator1D(ldelta, gamma)
        self.pd0 = Interpolator1D(ldelta, phi)

        if self.norm_all_z:
            p = np.array([-0.158, -0.195, -0.213, -0.254, -0.281,
                          -0.349, -0.367, -0.435, -0.504])
            q = np.array([0.0128, 0.0128, 0.0143, 0.0154, 0.0172,
                          0.0174, 0.0199, 0.0203, 0.0205])
            self.pp0 = Interpolator1D(ldelta, p)
            self.pq0 = Interpolator1D(ldelta, q)

    def _check_mass_def(self, mass_def):
        Δ = mass_def.Delta
        return not ((isinstance(Δ, (int, float)) and 200 <= Δ <= 3200)
                    or Δ == "vir")

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        ld = np.log10(self._get_Delta_m(cosmo, a))
        nu = const.DELTA_C / sigM
        # redshift evolution only up to z=3
        a = np.clip(a, 0.25, 1)
        pa = self.pa0(ld) * a**(-0.27)
        pb = self.pb0(ld) * a**(-0.20)
        pc = self.pc0(ld) * a**0.01
        pd = self.pd0(ld) * a**0.08
        pA0 = self.pA0(ld)

        if self.norm_all_z:
            z = 1./a - 1
            pp = self.pp0(ld)
            pq = self.pq0(ld)
            pA0 *= np.exp(z*(pp+pq*z))

        return (nu * pA0 * (1 + (pb * nu)**(-2 * pd))
                * nu**(2 * pa) * np.exp(-0.5 * pc * nu**2))


class MassFuncBocquet16(MassFunc):
    r"""Mass function described in :arXiv:1502.07357.

    Parameters
    ----------
    mass_def :class:`~pyccl.halos.massdef.MassDef`, optional
        Mass definition.
        **Note**: This parametrization is valid for SO masses with
        :math:`\Delta = 200 \, [\rm {m|c}]` and :math:`500 \rm c`.
        The default is 200m.
    hydro : bool, optional
        Whether to use the parametrization derived from baryonic feedback
        simulations (True), or from DM-only simulations (False).
        The default is True.
    """
    name = 'Bocquet16'

    def __init__(self, mass_def=MassDef(200, "matter"), hydro=True):
        self.hydro = hydro
        super().__init__(mass_def=mass_def)

    def _setup(self, cosmo):
        if self.mass_def.name == '200m':
            if self.hydro:
                self.A0 = 0.228
                self.a0 = 2.15
                self.b0 = 1.69
                self.c0 = 1.30
                self.Az = 0.285
                self.az = -0.058
                self.bz = -0.366
                self.cz = -0.045
            else:
                self.A0 = 0.175
                self.a0 = 1.53
                self.b0 = 2.55
                self.c0 = 1.19
                self.Az = -0.012
                self.az = -0.040
                self.bz = -0.194
                self.cz = -0.021
        elif self.mass_def.name == '200c':
            if self.hydro:
                self.A0 = 0.202
                self.a0 = 2.21
                self.b0 = 2.00
                self.c0 = 1.57
                self.Az = 1.147
                self.az = 0.375
                self.bz = -1.074
                self.cz = -0.196
            else:
                self.A0 = 0.222
                self.a0 = 1.71
                self.b0 = 2.24
                self.c0 = 1.46
                self.Az = 0.269
                self.az = 0.321
                self.bz = -0.621
                self.cz = -0.153
        elif self.mass_def.name == '500c':
            if self.hydro:
                self.A0 = 0.180
                self.a0 = 2.29
                self.b0 = 2.44
                self.c0 = 1.97
                self.Az = 1.088
                self.az = 0.150
                self.bz = -1.008
                self.cz = -0.322
            else:
                self.A0 = 0.241
                self.a0 = 2.18
                self.b0 = 2.35
                self.c0 = 2.02
                self.Az = 0.370
                self.az = 0.251
                self.bz = -0.698
                self.cz = -0.310

    def _check_mass_def(self, mass_def):
        return self.mass_def.name not in ["200m", "200c", "500c"]

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        zp1 = 1./a
        AA = self.A0 * zp1**self.Az
        aa = self.a0 * zp1**self.az
        bb = self.b0 * zp1**self.bz
        cc = self.c0 * zp1**self.cz

        f = AA * ((sigM / bb)**-aa + 1.0) * np.exp(-cc / sigM**2)
        z = 1./a-1
        Omega_m = cosmo.omega_x(a, "matter", squeeze=False)

        if self.mass_def.name == '200c':
            gamma0 = 3.54E-2 + Omega_m**0.09
            gamma1 = 4.56E-2 + 2.68E-2 / Omega_m
            gamma2 = 0.721 + 3.50E-2 / Omega_m
            gamma3 = 0.628 + 0.164 / Omega_m
            delta0 = -1.67E-2 + 2.18E-2 * Omega_m
            delta1 = 6.52E-3 - 6.86E-3 * Omega_m
            gamma = gamma0 + gamma1 * np.exp(-((gamma2 - z) / gamma3)**2)
            delta = delta0 + delta1 * z
            M200c_M200m = gamma + delta * lnM
            f *= M200c_M200m
        elif self.mass_def.name == '500c':
            alpha0 = 0.880 + 0.329 * Omega_m
            alpha1 = 1.00 + 4.31E-2 / Omega_m
            alpha2 = -0.365 + 0.254 / Omega_m
            alpha = alpha0 * (alpha1 * z + alpha2) / (z + alpha2)
            beta = -1.7E-2 + 3.74E-3 * Omega_m
            M500c_M200m = alpha + beta * lnM
            f *= M500c_M200m
        return f


class MassFuncWatson13(MassFunc):
    """Mass function described in :arXiv:1212.0095.

    Parameters
    ----------
    mass_def : :class:`~pyccl.halos.massdef.MassDef`, optional
        Mass definition. This parametrization accepts FoF and any SO masses.
        The default is 200m.
    """
    name = 'Watson13'

    def __init__(self, mass_def=MassDef(200, "matter")):
        super().__init__(mass_def=mass_def)

    def _setup(self, cosmo):
        self.is_fof = self.mass_def.Delta == 'fof'

    def _check_mass_def(self, mass_def):
        return not (mass_def.Delta == "fof"
                    or isinstance(mass_def.Delta, (int, float)))

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        if self.is_fof:
            pA = 0.282
            pa = 2.163
            pb = 1.406
            pc = 1.210
            return pA * ((pb / sigM)**pa + 1.) * np.exp(-pc / sigM**2)

        om = cosmo.omega_x(a, "matter", squeeze=False)
        Delta_178 = self.mass_def.Delta / 178.0

        if a == 1.0:
            pA, pa, pb, pc = 0.194, 1.805, 2.267, 1.287
        elif a < 0.14285714285714285:  # z > 6
            pA, pa, pb, pc = 0.563, 3.810, 0.874, 1.453
        else:
            pA = om * (1.097 * a**3.216 + 0.074)
            pa = om * (5.907 * a**3.058 + 2.349)
            pb = om * (3.136 * a**3.599 + 2.344)
            pc = 1.318

        f_178 = pA * ((pb / sigM)**pa + 1.) * np.exp(-pc / sigM**2)
        C = np.exp(0.023 * (Delta_178 - 1.))
        d = -0.456 * om - 0.139
        Γ = (C * Delta_178**d * np.exp(0.072 * (1-Delta_178) / sigM**2.130))
        return f_178 * Γ


class MassFuncAngulo12(MassFunc):
    """Mass function described in :arXiv:1203.3216.
    This parametrization is only valid for 'fof' masses.

    mass_def : :class:`~pyccl.halos.massdef.MassDef`, optional
        Mass definition.
        **Note**: This parametrization is only valid for FoF masses.
    """
    name = 'Angulo12'

    def __init__(self, mass_def=MassDef('fof', 'matter')):
        super().__init__(mass_def=mass_def)

    def _setup(self, cosmo):
        self.A = 0.201
        self.a = 2.08
        self.b = 1.7
        self.c = 1.172

    def _check_mass_def(self, mass_def):
        return mass_def.Delta != "fof"

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        return (self.A * ((self.a / sigM)**self.b + 1.)
                * np.exp(-self.c / sigM**2))
