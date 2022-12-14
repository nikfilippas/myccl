from .profile_base import HaloProfileNumberCounts
import numpy as np
from scipy.special import sici, erf


class HaloProfileHOD(HaloProfileNumberCounts):
    r"""Generic halo occupation distribution (HOD) profile
    describing the number density of galaxies as a function of halo mass.

    The parametrization for the mean profile is:

    .. math::

       \langle n_g(r)|M, a \rangle = \bar{N}_c(M,a)
       \left(f_c(a) + \bar{N}_s(M, a) u_{\rm sat}(r|M, a) \right),

    where :math:`\bar{N}_c` and :math:`\bar{N}_s` are the mean number
    of central and satellite galaxies respectively, :math:`f_c` is the observed
    fraction of central galaxies, and :math:`u_{\rm sat}(r|M, a)` is the
    distribution of satellites as a function of distance to the halo centre.

    These quantities are parametrized as follows:

    .. math::

       \bar{N}_c(M, a)= \frac{1}{2} \left( 1+{\rm erf}
       \left(\frac{\log(M / M_{\rm min})}{\sigma_{{\rm ln}M}}
       \right) \right)

    .. math::

       \bar{N}_s(M, a) = \Theta(M-M_0) \left(\frac{M-M_0}{M_1} \right)^\alpha

    .. math::

       u_s(r|M,a) \propto \frac{\Theta(r_{\rm max} - r)}{(r/r_g)(1 + r/r_g)^2},

    Where :math:`\Theta(x)` is the Heaviside step function, and the
    proportionality constant in the last equation is such that the volume
    integral of :math:`u_s` is 1. The radius :math:`r_g` is related to the NFW
    scale radius :math:`r_s` through :math:`r_g=\beta_g \, r_s`, and the radius
    :math:`r_{\rm max}` is related to the overdensity radius :math:`r_\Delta`
    as :math:`r_{\rm max} = \beta_{\rm max} r_\Delta`.
    The scale radius is related to the comoving overdensity halo radius
    via :math:`R_\Delta(M) = c(M) \, r_s`.

    All the quantities :math:`\log_{10} M_{\rm min}`,
    :math:`\log_{10} M_0`, :math:`\log_{10} M_1`, :math:`\sigma_{{\rm ln}M}`,
    :math:`f_c`, :math:`\alpha`, :math:`\beta_g` and :math:`\beta_{\rm max}`
    are time-dependent via a linear expansion around a pivot scale factor
    :math:`a_*` with offset :math:`X_0` and tilt parameter :math:`X_p`:

    .. math::

       X(a) = X_0 + X_p \, (a - a_*).

    This definition of the HOD profile draws from several papers in the
    literature, including: :arXiv:astro-ph/0408564, :arXiv:1706.05422, and
    :arXiv:1912.08209. The default values used here are roughly compatible
    with those found in the latter paper.

    .. note::
        See :class:`~pyccl.halos.profiles_2pt.Profile2ptHOD` for the Fourier-
        -space two-point correlator of the HOD profile.

    Parameters
    ----------
    mass_def : ~pyccl.halos.massdef.MassDef
        Mass definition of the profile.
    concentration : :obj:`Concentration`
        Concentration-mass relation.
    lMmin_0, lMmin_p : float
        Offset and tilt parameters for :math:`\log_{10} M_{\rm min}`.
    siglM_0, siglM_p : float
        Offset and tilt parameters for :math:`\sigma_{{\rm ln}M}`.
    lM0_0, lM0_p : float
        Offset and tilt parameters for :math:`\log_{10} M_0`.
    lM1_0, lM1_p : float
        Offset and tilt parameters for :math:`\log_{10} M_1`.
    alpha_0, alpha_p : float
        Offset and tilt parameters for :math:`\alpha`.
    fc_0, fc_p : float
        Offset and tilt parameters for :math:`f_c`.
    bg_0, bg_p : float
        Offset and tilt parameters for :math:`\beta_g`.
    bmax_0, bmax_p : float
        Offset and tilt parameters for :math:`\beta_{\rm max}`.
    a_pivot :
        Pivot scale factor :math:`a_*`.
    ns_independent : bool
        Drop requirement to only form satellites when centrals are present.
    """
    name = 'HOD'

    def __init__(self, mass_def, concentration, *,
                 lMmin_0=12, lMmin_p=0, siglM_0=0.4, siglM_p=0,
                 lM0_0=7, lM0_p=0, lM1_0=13.3, lM1_p=0,
                 alpha_0=1, alpha_p=0, fc_0=1, fc_p=0,
                 bg_0=1, bg_p=0, bmax_0=1, bmax_p=0,
                 a_pivot=1, ns_independent=False):
        super().__init__(mass_def=mass_def)
        self._check_consistent_mass(mass_def, concentration)
        self.concentration = concentration

        self.lMmin_0 = lMmin_0
        self.lMmin_p = lMmin_p
        self.lM0_0 = lM0_0
        self.lM0_p = lM0_p
        self.lM1_0 = lM1_0
        self.lM1_p = lM1_p
        self.siglM_0 = siglM_0
        self.siglM_p = siglM_p
        self.alpha_0 = alpha_0
        self.alpha_p = alpha_p
        self.fc_0 = fc_0
        self.fc_p = fc_p
        self.bg_0 = bg_0
        self.bg_p = bg_p
        self.bmax_0 = bmax_0
        self.bmax_p = bmax_p
        self.a_pivot = a_pivot
        self.ns_independent = ns_independent

    def _get_concentration(self, cosmo, M, a):
        return self.concentration.get_concentration(cosmo, M, a, squeeze=False)

    def update_parameters(
            self, lMmin_0=None, lMmin_p=None, siglM_0=None, siglM_p=None,
            lM0_0=None, lM0_p=None, lM1_0=None, lM1_p=None, alpha_0=None,
            alpha_p=None, fc_0=None, fc_p=None, bg_0=None, bg_p=None,
            bmax_0=None, bmax_p=None, a_pivot=None, ns_independent=None,
            **kwargs):
        """Update any of the parameters associated with this profile."""
        super().update_parameters(**kwargs)
        if lMmin_0 is not None:
            self.lMmin_0 = lMmin_0
        if lMmin_p is not None:
            self.lMmin_p = lMmin_p
        if lM0_0 is not None:
            self.lM0_0 = lM0_0
        if lM0_p is not None:
            self.lM0_p = lM0_p
        if lM1_0 is not None:
            self.lM1_0 = lM1_0
        if lM1_p is not None:
            self.lM1_p = lM1_p
        if siglM_0 is not None:
            self.siglM_0 = siglM_0
        if siglM_p is not None:
            self.siglM_p = siglM_p
        if alpha_0 is not None:
            self.alpha_0 = alpha_0
        if alpha_p is not None:
            self.alpha_p = alpha_p
        if fc_0 is not None:
            self.fc_0 = fc_0
        if fc_p is not None:
            self.fc_p = fc_p
        if bg_0 is not None:
            self.bg_0 = bg_0
        if bg_p is not None:
            self.bg_p = bg_p
        if bmax_0 is not None:
            self.bmax_0 = bmax_0
        if bmax_p is not None:
            self.bmax_p = bmax_p
        if a_pivot is not None:
            self.a_pivot = a_pivot
        if ns_independent is not None:
            self.ns_independent = ns_independent

    def _get_parameter(self, name, a):
        """Compute parameter from offset and pivot."""
        offset, pivot = [getattr(self, f"{name}_{t}") for t in ["0", "p"]]
        return offset + pivot * (a - self.a_pivot)

    def _Nc(self, M, a):
        """Number of centrals."""
        Mmin = 10.**(self._get_parameter("lMmin", a))
        siglM = self._get_parameter("siglM", a)
        return 0.5 * (1 + erf(np.log(M/Mmin) / siglM))

    def _Ns(self, M, a):
        """Number of satellites."""
        M0 = 10 ** self._get_parameter("lM0", a)
        M1 = 10 ** self._get_parameter("lM1", a)
        α = self._get_parameter("alpha", a)
        return np.heaviside(M - M0, 1) * (np.abs(M - M0) / M1)**α

    def _usat_real(self, cosmo, r, M, a):
        # Comoving virial radius
        bg = self._get_parameter("bg", a)
        bmax = self._get_parameter("bmax", a)
        R_M = self.mass_def.get_radius(cosmo, M, a, squeeze=False) / a
        c = self._get_concentration(cosmo, M, a) * bmax / bg
        R_s = R_M / c

        norm = 1 / (4 * np.pi * (bg*R_s)**3 * (np.log(1+c) - c/(1+c)))
        x = r / (R_s * bg)
        prof = norm / (x * (1 + x)**2)

        if self.truncated:
            prof[r > R_M*bmax] = 0
        return prof

    def _usat_fourier(self, cosmo, k, M, a):
        # Comoving virial radius
        bg = self._get_parameter("bg", a)
        bmax = self._get_parameter("bmax", a)
        R_M = self.mass_def.get_radius(cosmo, M, a, squeeze=False) / a
        c = self._get_concentration(cosmo, M, a) * bmax / bg
        R_s = R_M / c

        x = k * R_s * bg
        Si1, Ci1 = sici((1 + c) * x)
        Si2, Ci2 = sici(x)

        P1 = 1. / (np.log(1+c) - c/(1+c))
        P2 = np.sin(x) * (Si1 - Si2) + np.cos(x) * (Ci1 - Ci2)
        P3 = np.sin(c * x) / ((1 + c) * x)
        return P1 * (P2 - P3)

    def _real(self, cosmo, r, M, a):
        Nc = self._Nc(M, a)
        Ns = self._Ns(M, a)
        fc = self._get_parameter("fc", a)
        ur = self._usat_real(cosmo, r, M, a)  # NFW profile
        if self.ns_independent:
            return Nc * fc + Ns * ur
        return Nc * (fc + Ns*ur)

    def _fourier(self, cosmo, k, M, a):
        Nc = self._Nc(M, a)
        Ns = self._Ns(M, a)
        fc = self._get_parameter("fc", a)
        uk = self._usat_fourier(cosmo, k, M, a)  # NFW profile
        if self.ns_independent:
            return Nc * fc + Ns * uk
        return Nc * (fc + Ns*uk)

    def _fourier_variance(self, cosmo, k, M, a):
        """Fourier-space variance of the HOD profile."""
        Nc = self._Nc(M, a)
        Ns = self._Ns(M, a)
        fc = self._get_parameter("fc", a)
        uk = self._usat_fourier(cosmo, k, M, a)  # NFW profile

        prof = Ns[:, None] * uk
        if self.ns_independent:
            return 2 * Nc * fc * prof + prof**2
        return Nc * (2 * fc * prof + prof**2)

    def _fourier_2pt(self, cosmo, k, M, a, prof2=None):
        """Fourier-space 1-halo 2-point correlator for the HOD profile."""
        if prof2 is not None and prof2 != self:
            raise ValueError("prof2 must be the same as prof.")
        return self._fourier_variance(cosmo, k, M, a)
