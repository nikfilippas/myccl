from .massdef import MassDef, dc_NakamuraSuto
from ..parameters import physical_constants as const
from ..parameters import accuracy_params as acc
from ..pyutils import get_broadcastable

import numpy as np
from scipy.optimize import brentq, root_scalar
from abc import ABC, abstractmethod, abstractproperty


__all__ = ("Concentration", "ConcentrationDiemer15",
           "ConcentrationBhattacharya13", "ConcentrationPrada12",
           "ConcentrationKlypin11", "ConcentrationDuffy08",
           "ConcentrationIshiyama21", "ConcentrationConstant")


class Concentration(ABC):
    """Calculations of halo concentrations.

    Parameters
    ----------
    mass_def : :class:`~pyccl.halos.massdef.MassDef`
        Mass definition for this :math:`c(M)` parametrization.
    """

    def __init__(self, mass_def):
        if self._check_mass_def(mass_def):
            raise ValueError(
                f"Mass definition {mass_def.Delta}-{mass_def.rho_type} "
                f"is not compatible with c(M) {self.name} configuration.")
        self.mass_def = mass_def
        self._setup()

    @abstractproperty
    def name(self):
        """Give a name to the concentration-mass relation."""

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

    @abstractmethod
    def _concentration(cosmo, M, a):
        r"""Specific implementation of the concentration-mass relation.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        M : (1, nM) ndarray
            Halo mass in :math:`\rm M_\odot` at every ``a``.
        a : (na, 1) ndarray
            Scale factor.

        Returns
        -------
        cM : (na, nM) ndarray
            Concentration.
        """

    def get_concentration(self, cosmo, M, a, *, squeeze=True):
        r"""Compute the concentration.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        M : float or (nM,) array_like
            Halo mass in :math:`\rm M_\odot`.
        a : float or (na,) array_like
            Scale factor.
        squeeze : bool
            Squeeze extra dimensions of size (1,) in the output.
            The default is True.

        Returns
        -------
        cM : float or (na, nM,) array_like
            Concentration.
        """
        a, M = map(np.asarray, [a, M])
        a, M = get_broadcastable(a, M)
        shp = np.broadcast_shapes(a.shape, M.shape)
        c = self._concentration(cosmo, M, a).reshape(shp)
        return c.squeeze()[()] if squeeze else c

    @classmethod
    def from_name(cls, name):
        """Get the ``Concentration`` class from its name string.

        Arguments
        ---------
        name : str
            Name of concentration.

        Returns
        -------
        Concentration : :class:`~pyccl.halos.concentration.Concentration`
            Concentration subclass corresponding to the input name.
        """
        concentrations = {c.name: c for c in cls.__subclasses__()}
        return concentrations[name]


class ConcentrationDiemer15(Concentration):
    r"""Concentration-mass relation by Diemer & Kravtsov (2015)
    :arXiv:1407.4730. Valid only for S.O. :math:`\Delta = 200c`
    mass definitions.

    Parameters
    ----------
    mass_def : :class:`~pyccl.halos.massdef.MassDef`
        Mass definition for this :math:`c(M)` parametrization.
        **Note**: can't be changed for this ``Concentration`` subclass.
    """
    name = 'Diemer15'

    def __init__(self, mass_def=MassDef(200, 'critical')):
        super().__init__(mass_def=mass_def)

    def _setup(self):
        self.kappa = 1.0
        self.phi_0 = 6.58
        self.phi_1 = 1.27
        self.eta_0 = 7.28
        self.eta_1 = 1.56
        self.alpha = 1.08
        self.beta = 1.77

    def _check_mass_def(self, mass_def):
        return mass_def.name != "200c"

    def _concentration(self, cosmo, M, a):
        cosmo.compute_linear_power()
        pk = cosmo.get_linear_power()

        R = cosmo.m2r(M, species="matter", comoving=False, squeeze=False)
        kR = 2 * np.pi / R * self.kappa
        # kR in strictly increasing order
        n = pk(kR[:, ::-1], a, derivative=True, grid=False)[:, ::-1]

        σM = cosmo.sigmaM(M, a, squeeze=False)
        nu = const.DELTA_C / σM

        floor = self.phi_0 + n * self.phi_1
        nu0 = self.eta_0 + n * self.eta_1
        return 0.5 * floor * ((nu0 / nu)**self.alpha + (nu / nu0)**self.beta)


class ConcentrationBhattacharya13(Concentration):
    r"""Concentration-mass relation by Bhattacharya et al. (2013)
    :arXiv:1112.5479. Valid only for S.O. masses with
    :math:`\Delta=200m` and :math:`\Delta=200c`.

    Parameters
    ----------
    mass_def : :class:`~pyccl.halos.massdef.MassDef`
        Mass definition for this :math:`c(M)` parametrization.
        The default is :math:`\Delta=200c`.
    """
    name = 'Bhattacharya13'

    def __init__(self, mass_def=MassDef(200, 'critical')):
        super().__init__(mass_def=mass_def)

    def _check_mass_def(self, mass_def):
        return mass_def.name not in ["200m", "200c"]

    def _setup(self):
        if self.mass_def.name == "vir":
            self.A = 7.7
            self.B = 0.9
            self.C = -0.29
        elif self.mass_def.name == "200m":
            self.A = 9.0
            self.B = 1.15
            self.C = -0.29
        else:  # Now, it has to be 200c.
            self.A = 5.9
            self.B = 0.54
            self.C = -0.35

    def _concentration(self, cosmo, M, a):
        gz = cosmo.growth_factor(a, squeeze=False)
        δc = dc_NakamuraSuto(cosmo, a, squeeze=False)
        sig = cosmo.sigmaM(M, a, squeeze=False)
        nu = δc / sig
        return self.A * gz**self.B * nu**self.C


class ConcentrationPrada12(Concentration):
    r"""Concentration-mass relation by Prada et al. (2012)
    :arXiv:1104.5130. Valid only for S.O. masses with :math:`\Delta = 200c`.

    Parameters
    ---------
    mass_def : :class:`~pyccl.halos.massdef.MassDef`
        Mass definition for this :math:`c(M)` parametrization.
        **Note**: can't be changed for this ``Concentration`` subclass.
    """
    name = 'Prada12'

    def __init__(self, mass_def=MassDef(200, 'critical')):
        super().__init__(mass_def=mass_def)

    def _check_mass_def(self, mass_def):
        return mass_def.name != "200c"

    def _setup(self):
        self.c0 = 3.681
        self.c1 = 5.033
        self.al = 6.948
        self.x0 = 0.424
        self.i0 = 1.047
        self.i1 = 1.646
        self.be = 7.386
        self.x1 = 0.526
        self.cnorm = 1 / self._cmin(1.393)
        self.inorm = 1 / self._imin(1.393)

    def _vmin(self, x, x0, v0, v1, v2):
        return v0 + (v1 - v0) * (np.arctan(v2 * (x - x0)) / np.pi + 0.5)

    def _cmin(self, x):
        return self._vmin(x, x0=self.x0, v0=self.c0, v1=self.c1, v2=self.al)

    def _imin(self, x):
        return self._vmin(x, x0=self.x1, v0=self.i0, v1=self.i1, v2=self.be)

    def _concentration(self, cosmo, M, a):
        x = a * (cosmo["Omega_l"] / cosmo["Omega_m"])**(1/3)
        B0 = self._cmin(x) * self.cnorm
        B1 = self._imin(x) * self.inorm

        sig_p = B1 * cosmo.sigmaM(M, a, squeeze=False)
        Cc = 2.881 * ((sig_p / 1.257)**1.022 + 1) * np.exp(0.060 / sig_p**2)
        return B0 * Cc


class ConcentrationKlypin11(Concentration):
    r"""Concentration-mass relation by Klypin et al. (2011)
    :arXiv:1002.3660. Only valid for S.O. masses with
    :math:`\Delta = \Delta_{\rm vir}`.

    Parameters
    ---------
    mass_def : :class:`~pyccl.halos.massdef.MassDef`
        Mass definition for this :math:`c(M)` parametrization.
        **Note**: can't be changed for this ``Concentration`` subclass.
    """
    name = 'Klypin11'

    def __init__(self, mass_def=MassDef('vir', 'critical')):
        super().__init__(mass_def=mass_def)

    def _check_mass_def(self, mass_def):
        return mass_def.name != "vir"

    def _concentration(self, cosmo, M, a):
        c = 9.6 * (M * cosmo["h"] * 1e-12)**-0.075
        return np.repeat(c, repeats=a.size, axis=0)


class ConcentrationDuffy08(Concentration):
    r"""Concentration-mass relation by Duffy et al. (2008)
    :arXiv:0804.2486. Only valid for S.O. masses with
    :math:`\Delta = \Delta_{\rm vir}`,
    :math:`\Delta = 200m`,
    or :math:`\Delta = 200c`.

    Parameters
    ---------
    mass_def : :class:`~pyccl.halos.massdef.MassDef`
        Mass definition for this :math:`c(M)` parametrization.
    """
    name = 'Duffy08'

    def __init__(self, mass_def=MassDef(200, 'critical')):
        super().__init__(mass_def=mass_def)

    def _check_mass_def(self, mass_def):
        return mass_def.name not in ["vir", "200m", "200c"]

    def _setup(self):
        if self.mass_def.name == 'vir':
            self.A = 7.85
            self.B = -0.081
            self.C = -0.71
        elif self.mass_def.name == "200m":
            self.A = 10.14
            self.B = -0.081
            self.C = -1.01
        else:  # Now, it has to be 200c.
            self.A = 5.71
            self.B = -0.084
            self.C = -0.47

    def _concentration(self, cosmo, M, a):
        return self.A * (M * cosmo["h"] * 5e-13)**self.B * a**(-self.C)


class ConcentrationIshiyama21(Concentration):
    r"""Concentration-mass relation by Ishiyama et al. (2021)
    :arXiv:2007.14720. Only valid for S.O. masses with
    :math:`\Delta = \Delta_{\rm vir}`,
    :math:`\Delta = 200c`,
    or :math:`\Delta = 500c`.

    Parameters
    ----------
    mass_def : :class:`~pyccl.halos.massdef.MassDef`
        Mass definition for this :math:`c(M)` parametrization.
    relaxed : bool
        If True, use concentration for relaxed halos. Otherwise,
        use concentration for all halos. The default is False.
    Vmax : bool
        If True, use the concentration found with the Vmax numerical
        method. Otherwise, use the concentration found with profile
        fitting. The default is False.
    """
    name = 'Ishiyama21'

    def __init__(self, mass_def=MassDef(500, 'critical'),
                 relaxed=False, Vmax=False):
        self.relaxed = relaxed
        self.Vmax = Vmax
        super().__init__(mass_def=mass_def)

    def _check_mass_def(self, mass_def):
        is_500Vmax = mass_def.Delta == 500 and self.Vmax
        return mass_def.name not in ["vir", "200c", "500c"] or is_500Vmax

    def _setup(self):
        if self.Vmax:  # use numerical method
            if self.relaxed:  # fit only relaxed halos
                if self.mass_def.name == 'vir':
                    self.kappa = 2.40
                    self.a0 = 2.27
                    self.a1 = 1.80
                    self.b0 = 0.56
                    self.b1 = 13.24
                    self.c_alpha = 0.079
                else:  # now it's 200c
                    self.kappa = 1.79
                    self.a0 = 2.15
                    self.a1 = 2.06
                    self.b0 = 0.88
                    self.b1 = 9.24
                    self.c_alpha = 0.51
            else:  # fit all halos
                if self.mass_def.name == 'vir':
                    self.kappa = 0.76
                    self.a0 = 2.34
                    self.a1 = 1.82
                    self.b0 = 1.83
                    self.b1 = 3.52
                    self.c_alpha = -0.18
                else:  # now it's 200c
                    self.kappa = 1.10
                    self.a0 = 2.30
                    self.a1 = 1.64
                    self.b0 = 1.72
                    self.b1 = 3.60
                    self.c_alpha = 0.32
        else:  # use profile fitting method
            if self.relaxed:  # fit only relaxed halos
                if self.mass_def.Delta == 'vir':
                    self.kappa = 1.22
                    self.a0 = 2.52
                    self.a1 = 1.87
                    self.b0 = 2.13
                    self.b1 = 4.19
                    self.c_alpha = -0.017
                elif self.mass_def.name == "200c":
                    self.kappa = 0.60
                    self.a0 = 2.14
                    self.a1 = 2.63
                    self.b0 = 1.69
                    self.b1 = 6.36
                    self.c_alpha = 0.37
                else:  # now it's 500c
                    self.kappa = 0.38
                    self.a0 = 1.44
                    self.a1 = 3.41
                    self.b0 = 2.86
                    self.b1 = 2.99
                    self.c_alpha = 0.42
            else:  # fit all halos
                if self.mass_def.name == 'vir':
                    self.kappa = 1.64
                    self.a0 = 2.67
                    self.a1 = 1.23
                    self.b0 = 3.92
                    self.b1 = 1.30
                    self.c_alpha = -0.19
                elif self.mass_def.name == "200c":
                    self.kappa = 1.19
                    self.a0 = 2.54
                    self.a1 = 1.33
                    self.b0 = 4.04
                    self.b1 = 1.21
                    self.c_alpha = 0.22
                else:  # now it's 500c
                    self.kappa = 1.83
                    self.a0 = 1.95
                    self.a1 = 1.17
                    self.b0 = 3.57
                    self.b1 = 0.91
                    self.c_alpha = 0.26

    def _dlsigmaR(self, cosmo, M, a):
        # κ multiplies radius, so in log, κ^3 multiplies mass
        dlnsigM_dlogM = cosmo.dlnsigM_dlogM(M*self.kappa**3, a, squeeze=False)
        return -3/np.log(10) * dlnsigM_dlogM

    def _G(self, x, n_eff):
        fx = np.log(1 + x) - x / (1 + x)
        G = x / fx**((5 + n_eff) / 6)
        return G

    def _G_inv(self, arg, n_eff):
        # Numerical calculation of the inverse of `_G`.
        shp = arg.shape
        out = np.empty(np.product(shp))
        arg, n_eff = map(np.ndarray.flatten, [arg, n_eff])

        # TODO: We can replace the root finding with a spline for speed.
        for i, (val, neff) in enumerate(zip(arg, n_eff)):
            func = lambda x: self._G(x, neff) - val  # noqa: _G_inv Traceback
            EPS, NIT = acc.EPSREL, acc.N_ITERATION_ROOT
            try:
                out[i] = brentq(func, a=0.05, b=200, rtol=EPS, maxiter=NIT)
            except ValueError:
                # No root in [0.05, 200] (rare, but it may happen)
                res = root_scalar(func, x0=1, x1=2, xtol=EPS, maxiter=NIT)
                out[i] = res.root.item()

        return out.reshape(shp).squeeze()[()]

    def _concentration(self, cosmo, M, a):
        nu = const.DELTA_C / cosmo.sigmaM(M, a, squeeze=False)
        n_eff = -2 * self._dlsigmaR(cosmo, M, a) - 3
        α_eff = cosmo.growth_rate(a, squeeze=False)

        A = self.a0 * (1 + self.a1 * (n_eff + 3))
        B = self.b0 * (1 + self.b1 * (n_eff + 3))
        C = 1 - self.c_alpha * (1 - α_eff)
        arg = A / nu * (1 + nu**2 / B)
        G = self._G_inv(arg, n_eff)
        return C * G


class ConcentrationConstant(Concentration):
    """Constant contentration-mass relation.

    .. note::

        The mass definition for this concentration is arbitrary, and is
        internally set to ``None``.

    Parameters
    ---------
    c : float
        Value of the constant concentration.
    """
    name = 'Constant'

    def __init__(self, c=1, mass_def=None):
        # Keep the `mass_def` parameter for consistency.
        super().__init__(mass_def=mass_def)
        self.c = c

    def _check_mass_def(self, mass_def):
        return False

    def _concentration(self, cosmo, M, a):
        return np.full((a.size, M.size), self.c)
