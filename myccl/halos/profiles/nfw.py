from .profile_base import HaloProfile
import numpy as np
from scipy.special import sici
from functools import partial


class HaloProfileNFW(HaloProfile):
    r"""Navarro-Frenk-White profile ``(astro-ph:astro-ph/9508025)``.

    .. math::

       \rho(r) = \frac{\rho_0}{\frac{r}{r_s} \left(1 + \frac{r}{r_s} \right)^2}

    where :math:`r_s` is related to the spherical overdensity halo radius
    :math:`R_\Delta(M)` through the concentration parameter :math:`c(M)` as

    .. math::

       R_\Delta(M) = c(M) \, r_s

    and the normalization :math:`\rho_0` is

    .. math::

       \rho_0 = \frac{M}{4 \pi \, r_s^3 \, \left(\log(1+c) - c/(1+c) \right)}.

    By default, this profile is truncated at :math:`r = R_\\Delta(M)`.

    Parameters
    ----------
    mass_def : ~pyccl.halos.massdef.MassDef
        Mass definition.
    concentration : :class:`~pyccl.halos.concentration.Concentration`
        Concentration-mass relation.
    truncated : bool
        Whether to truncate the profile at :math:`r = R_{\Delta}`.
        The default is True.
    """
    name = 'NFW'
    normprof = True

    def __init__(self, mass_def, concentration, truncated=True):
        super().__init__(mass_def=mass_def)
        self._check_consistent_mass(mass_def, concentration)
        self.concentration = concentration
        self.truncated = truncated

        if not truncated:
            self._projected = self._projected_analytic
            self._cumul2d = self._cumul2d_analytic

        kwargs = {"padding_hi_fftlog": 1e2, "padding_lo_fftlog": 1e-2,
                  "n_per_decade": 1000, "plaw_fourier": -2.}
        self.update_precision_fftlog(**kwargs)

    def _get_concentration(self, cosmo, M, a):
        return self.concentration.get_concentration(cosmo, M, a, squeeze=False)

    def _norm(self, M, Rs, c):
        """NFW normalization from mass, radius and concentration."""
        return M / (4 * np.pi * Rs**3 * (np.log(1+c) - c/(1+c)))

    def _real(self, cosmo, r, M, a):
        # Comoving virial radius
        R_M = self.mass_def.get_radius(cosmo, M, a, squeeze=False) / a
        c_M = self._get_concentration(cosmo, M, a)
        R_s = R_M / c_M

        norm = self._norm(M, R_s, c_M)
        x = r / R_s
        prof = norm / (x * (1 + x)**2)

        if self.truncated:
            prof[r > R_M] = 0
        return prof

    def _fx_projected(self, x):
        def f(x, func):
            # N = +1 for arccosh, N = -1 for arccos.
            N = (-1)**(int(func.__name__ == "arccos"))
            x2m1 = x*x - 1
            return 1 / x2m1 + N * func(1/x) / np.abs(x2m1)**1.5

        f1, f2 = [partial(f, func=func) for func in [np.arccosh, np.arccos]]
        return np.piecewise(x, [x < 1, x > 1], [f1, f2, 1/3])

    def _projected_analytic(self, cosmo, r, M, a):
        # Comoving virial radius
        R_M = self.mass_def.get_radius(cosmo, M, a, squeeze=False) / a
        c_M = self._get_concentration(cosmo, M, a)
        R_s = R_M / c_M

        x = r / R_s
        prof = self._fx_projected(x)
        norm = 2 * R_s * self._norm(M, R_s, c_M)
        return prof * norm

    def _fx_cumul2d(self, x):
        def f(x, func):
            sqx2m1 = np.sqrt(np.abs(x*x - 1))
            return np.log(x/2) + func(1/x) / sqx2m1

        f1, f2 = [partial(f, func=func) for func in [np.arccosh, np.arccos]]
        f = np.piecewise(x, [x < 1, x > 1], [f1, f2, 1 - np.log(2)])
        return 2 * f / x**2

    def _cumul2d_analytic(self, cosmo, r, M, a):
        # Comoving virial radius
        R_M = self.mass_def.get_radius(cosmo, M, a, squeeze=False) / a
        c_M = self._get_concentration(cosmo, M, a)
        R_s = R_M / c_M

        x = r / R_s
        prof = self._fx_cumul2d(x)
        norm = 2 * R_s * self._norm(M, R_s, c_M)
        return prof * norm

    def _fourier(self, cosmo, k, M, a):
        # Comoving virial radius
        R_M = self.mass_def.get_radius(cosmo, M, a, squeeze=False) / a
        c = self._get_concentration(cosmo, M, a)
        R_s = R_M / c

        x = k * R_s
        Si2, Ci2 = sici(x)
        P1 = M / (np.log(1 + c) - c / (1 + c))
        if self.truncated:
            Si1, Ci1 = sici((1 + c) * x)
            P2 = np.sin(x) * (Si1 - Si2) + np.cos(x) * (Ci1 - Ci2)
            P3 = np.sin(c * x) / ((1 + c) * x)
            return P1 * (P2 - P3)
        P2 = np.sin(x) * (0.5 * np.pi - Si2) - np.cos(x) * Ci2
        return P1 * P2
