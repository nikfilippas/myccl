from .massdef import MassDef, dc_NakamuraSuto
from ..parameters import physical_constants as const
from ..pyutils import get_broadcastable

import numpy as np
from abc import ABC, abstractmethod


__all__ = ("HaloBias", "HaloBiasSheth99", "HaloBiasSheth01",
           "HaloBiasBhattacharya11", "HaloBiasTinker10")


class HaloBias(ABC):
    r"""Calculations of halo bias functions.

    .. note::

        Currently, we assume that all halo bias parametrizations can be written
        as functions that depend only on :math:`M` through :math:`\sigma_M`,
        the overdensity variance on spheres of the Lagrangian radius that
        corresponds to mass :math:`M`. To that end, specific parametrizations
        may be created by subclassing and implementing ``_get_bsigma``.

    Parameters
    ----------
    mass_def : :class:`~pyccl.halos.massdef.MassDef`
        Mass definition for this halo bias parametrization.
    """

    def __init__(self, mass_def):
        if self._check_mass_def(mass_def):
            raise ValueError(
                f"Mass definition {mass_def.Delta}-{mass_def.rho_type} "
                f"is not compatible with halo bias {self.name}.")
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

    @abstractmethod
    def _get_bsigma(self, cosmo, sigM, a):
        r"""Specific implementation of halo bias as a function of
        :math:`\sigma_{\rm M}`.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        sigM : (na, nsigM) ndarray
            Standard deviation of the overdensity field at the scale of a halo.
            This is calculated at the values of the scale factor in ``a``.
        a : (na, 1) ndarray
            Scale factor, corresponding to the rows of ``sigM``.

        Returns
        -------
        bsigma : (na, nsigM) ndarray
            :math:`f(\sigma_{\rm M})` function.
        """

    def get_halo_bias(self, cosmo, M, a, *, squeeze=True):
        r"""Compute the halo bias function.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        M : float or (nM,) array_like
            Halo mass in :math:`\rm M_\odot`.
        a : float or (na,) array_like
            Scale factor.
        squeeze : bool, optional
            Squeeze extra dimensions of size (1,) in the output.
            The default is True.

        Returns
        -------
        halo_bias : float or (na, nM,) array_like
            Halo bias.
        """
        a, M = map(np.asarray, [a, M])
        a, M = get_broadcastable(a, M)
        sigM = cosmo.sigmaM(M, a, squeeze=False)
        b = self._get_bsigma(cosmo, sigM, a)
        return b.squeeze()[()] if squeeze else b

    @classmethod
    def from_name(cls, name):
        """Returns halo bias subclass from name string

        Arguments
        ---------
        name : str
            Halo bias name.

        Returns
        -------
        HaloBias : :class:`~pyccl.halos.hbias.HaloBias`
            ``HaloBias`` subclass corresponding to the input name.
        """
        bias_functions = {c.name: c for c in HaloBias.__subclasses__()}
        return bias_functions[name]


class HaloBiasSheth99(HaloBias):
    r"""Halo bias described in ``1999MNRAS.308..119S``.
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
    name = "Sheth99"

    def __init__(self, mass_def=MassDef('fof', 'matter'), delta_c_fit=False):
        self.delta_c_fit = delta_c_fit
        super().__init__(mass_def=mass_def)

    def _setup(self):
        self.p = 0.3
        self.a = 0.707

    def _check_mass_def(self, mass_def):
        return mass_def.Delta != "fof"

    def _get_bsigma(self, cosmo, sigM, a):
        δc = const.DELTA_C
        if self.delta_c_fit:
            δc = dc_NakamuraSuto(cosmo, a, squeeze=False)
        nu = δc / sigM
        anu2 = self.a * nu**2
        return 1. + (anu2 - 1. + 2. * self.p / (1. + anu2**self.p)) / δc


class HaloBiasSheth01(HaloBias):
    """Halo bias described in :arXiv:astro-ph/9907024.

    Parameters
    ----------
    mass_def : :class:`~pyccl.halos.massdef.MassDef`, optional
        Mass definition.
        **Note**: This parametrization is only valid for FoF masses.
    """
    name = "Sheth01"

    def __init__(self, mass_def=MassDef('fof', 'matter')):
        super().__init__(mass_def=mass_def)

    def _setup(self):
        self.a = 0.707
        self.b = 0.5
        self.c = 0.6
        self.dc = const.DELTA_C

    def _check_mass_def(self, mass_def):
        return mass_def.Delta != "fof"

    def _get_bsigma(self, cosmo, sigM, a):
        sqrta = np.sqrt(self.a)
        nu = self.dc / sigM
        anu2 = self.a * nu**2
        anu2c = anu2**self.c
        t1 = self.b * (1.0 - self.c) * (1.0 - 0.5 * self.c)
        return 1. + (sqrta * anu2 * (1 + self.b / anu2c) -
                     anu2c / (anu2c + t1)) / (sqrta * self.dc)


class HaloBiasBhattacharya11(HaloBias):
    """Halo bias described in :arXiv:1005.2239.
    This parametrization is only valid for 'fof' masses.

    Parameters
    ----------
    mass_def : :class:`~pyccl.halos.massdef.MassDef`, optional
        Mass definition.
        **Note**: This parametrization is only valid for FoF masses.
    """
    name = "Bhattacharya11"

    def __init__(self, mass_def=MassDef('fof', 'matter')):
        super().__init__(mass_def=mass_def)

    def _setup(self):
        self.a = 0.788
        self.az = 0.01
        self.p = 0.807
        self.q = 1.795
        self.dc = const.DELTA_C

    def _check_mass_def(self, mass_def):
        return mass_def.Delta != "fof"

    def _get_bsigma(self, cosmo, sigM, a):
        nu = self.dc / sigM
        a = self.a * a**self.az
        anu2 = a * nu**2
        return 1. + (anu2 - self.q + 2*self.p / (1 + anu2**self.p)) / self.dc


class HaloBiasTinker10(HaloBias):
    r"""Halo bias described in :arXiv:1001.3162.

    .. note::

        This parametrization is valid for SO-matter based mass definitions with
        :math:`200 < \Delta < 3200`. CCL will internally translate SO-critical
        based mass definitions to SO-matter.

    Parameters
    ----------
    mass_def : :class:`~pyccl.halos.massdef.MassDef`, optional
        Mass definition for this halo bias parametrization.
        The default is 200m.
    """
    name = "Tinker10"

    def __init__(self, mass_def=MassDef(200, "matter")):
        super().__init__(mass_def=mass_def)

    def _AC(self, ld):
        xp = np.exp(-(4./ld)**4.)
        A = 1.0 + 0.24 * ld * xp
        C = 0.019 + 0.107 * ld + 0.19*xp
        return A, C

    def _setup(self):
        self.B = 0.183
        self.b = 1.5
        self.c = 2.4
        self.dc = const.DELTA_C

    def _check_mass_def(self, mass_def):
        Δ = mass_def.Delta
        return not ((isinstance(Δ, (int, float)) and 200 <= Δ <= 3200)
                    or Δ == "vir")

    def _get_Delta_m(self, cosmo, a):
        r"""Translate SO mass definitions to :math:`\rho_{\rm m}`."""
        Δ = self.mass_def.get_Delta(cosmo, a, squeeze=False)
        if self.mass_def.rho_type == 'matter':
            return Δ

        Ω_this = cosmo.omega_x(a, self.mass_def.rho_type, squeeze=False)
        Ω_m = cosmo.omega_x(a, "matter", squeeze=False)
        return Δ * Ω_this / Ω_m

    def _get_bsigma(self, cosmo, sigM, a):
        nu = self.dc / sigM

        ld = np.log10(self._get_Delta_m(cosmo, a))
        A, C = self._AC(ld)
        aa = 0.44 * ld - 0.88
        nupa = nu**aa
        return (1. - A * nupa / (nupa + self.dc**aa) +
                self.B * nu**self.b + C * nu**self.c)
