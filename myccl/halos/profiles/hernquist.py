from .profile_base import HaloProfile
import numpy as np
from scipy.special import sici
from functools import partial


class HaloProfileHernquist(HaloProfile):
    r"""Hernquist halo profile ``(1990ApJ...356..359H)``.

    .. math::

       \rho(r) = \frac{\rho_0}{\frac{r}{r_s} \left(1 + \frac{r}{r_s} \right)^3}

    where :math:`r_s` is related to the spherical overdensity halo radius
    :math:`R_{\Delta(M)}` through the concentration parameter :math:`c(M)` as

    .. math::

       R_{\Delta(M)} = c(M) \, r_s

    and the normalization :math:`\rho_0` is the mean density
    inside :math:`R_{\Delta(M)}`.

    Parameters
    ----------
    mass_def : ~pyccl.halos.massdef.MassDef
        Mass definition of the profile.
    concentration : :obj:`Concentration`
        Concentration-mass relation.
    truncated : bool
        Whether to truncate the profile at :math:`r = R_{\Delta}`.
        The default is True.
    """
    name = 'Hernquist'
    normprof = True

    def __init__(self, mass_def, concentration, *, truncated=True):
        super().__init__(mass_def=mass_def)
        self._check_consistent_mass(mass_def, concentration)
        self.concentration = concentration
        self.truncated = truncated

        if not truncated:
            self._projected = self._projected_analytic
            self._cumul2d = self._cumul2d_analytic

        kwargs = {"padding_hi_fftlog": 1e2, "padding_lo_fftlog": 1e-4,
                  "n_per_decade": 1000, "plaw_fourier": -2.}
        self.update_precision_fftlog(**kwargs)

    def _get_concentration(self, cosmo, M, a):
        return self.concentration.get_concentration(cosmo, M, a, squeeze=False)

    def _norm(self, M, Rs, c):
        """Hernquist normalization from mass, radius and concentration."""
        return M / (2 * np.pi * Rs**3 * (c / (1 + c))**2)

    def _real(self, cosmo, r, M, a):
        # Comoving virial radius
        R_M = self.mass_def.get_radius(cosmo, M, a, squeeze=False) / a
        c_M = self._get_concentration(cosmo, M, a)
        R_s = R_M / c_M

        norm = self._norm(M, R_s, c_M)
        x = r / R_s
        prof = norm / (x * (1 + x)**3)

        if self.truncated:
            prof[r > R_M] = 0
        return prof

    def _fx_projected(self, x):
        def f(x, func):
            x2m1 = x*x - 1
            return -3/(2*x2m1**2) + (x2m1+3)*func(1/x)/(2*np.abs(x2m1)**2.5)

        f1, f2 = [partial(f, func=func) for func in [np.arccosh, np.arccos]]
        return np.piecewise(x, [x < 1, x > 1], [f1, f2, 2/15])

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
            # N = +1 for arccosh, N = -1 for arccos.
            N = (-1)**(int(func.__name__ == "arccos"))
            x2m1 = x*x - 1
            return 1 + 1/x2m1 + N * (x2m1 + 1) * func(1/x) / np.abs(x2m1)**1.5

        f1, f2 = [partial(f, func=func) for func in [np.arccosh, np.arccos]]
        f = np.piecewise(x, [x < 1, x > 1], [f1, f2, 1./3.])
        return f / x / x

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
        c_M = self._get_concentration(cosmo, M, a)
        R_s = R_M / c_M

        x = k / R_s
        Si2, Ci2 = sici(x)
        P1 = M / ((c_M / (c_M + 1))**2 / 2)
        if self.truncated:
            c_Mp1 = c_M + 1
            Si1, Ci1 = sici(c_Mp1 * x)
            P2 = x * np.sin(x) * (Ci1 - Ci2) - x * np.cos(x) * (Si1 - Si2)
            P3 = (-1 + np.sin(c_M * x) / (c_Mp1**2 * x)
                  + c_Mp1 * np.cos(c_M * x) / (c_Mp1**2))
            return P1 * (P2 - P3) / 2
        P2 = (-x * (2 * np.sin(x) * Ci2 + np.pi * np.cos(x))
              + 2 * x * np.cos(x) * Si2 + 2) / 4
        return P1 * P2
