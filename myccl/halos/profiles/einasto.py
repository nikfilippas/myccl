from .profile_base import HaloProfile
from ..massdef import MassDef
from ...parameters import physical_constants as const
import numpy as np
from scipy.special import gamma, gammainc


class HaloProfileEinasto(HaloProfile):
    r"""Einasto profile ``(1965TrAlm...5...87E)``.

    .. math::

       \rho(r) = \rho_0 \, \exp(-2 ((r / r_s)^\alpha-1) / \alpha)

    where :math:`r_s` is related to the spherical overdensity halo radius
    :math:`R_\Delta(M)` through the concentration parameter :math:`c(M)` as

    .. math::

       R_\Delta(M) = c(M) \, r_s

    and the normalization :math:`\rho_0` is the mean density inside
    :math:`R_\Delta(M)`. :math:`\alpha` depends on halo mass and redshift,
    following the parameterization of Diemer & Kravtsov (:arXiv:1401.1216).

    By default, this profile is truncated at :math:`r = R_\\Delta(M)`.

    Parameters
    ----------
    mass_def : ~pyccl.halos.massdef.MassDef
        Mass definition of the profile.
    concentration : :obj:`Concentration`
        Concentration-mass relation.
    alpha : float, 'cosmo'
        Value of :math:`\alpha`.
        The default is ``'cosmo'`` to calculate it from cosmology.
    truncated : bool
        Whether to truncate the profile at :math:`r = R_{\Delta}`.
        The default is True.
    """
    name = 'Einasto'
    normprof = True

    def __init__(self, mass_def, concentration, *,
                 alpha="cosmo", truncated=True):
        super().__init__(mass_def=mass_def)
        self._check_consistent_mass(mass_def, concentration)
        self.concentration = concentration
        self.alpha = alpha
        self.truncated = truncated

        kwargs = {"padding_hi_fftlog": 1e2, "padding_lo_fftlog": 1e-2,
                  "n_per_decade": 1000, "plaw_fourier": -2.}
        self.update_precision_fftlog(**kwargs)

    def update_parameters(self, alpha=None, **kwargs):
        """Update any of the parameters associated with this profile."""
        super().update_parameters(**kwargs)
        if alpha is not None:
            self.alpha = alpha

    def _get_concentration(self, cosmo, M, a):
        return self.concentration.get_concentration(cosmo, M, a, squeeze=False)

    def _get_alpha(self, cosmo, M, a):
        if self.alpha == "cosmo":
            mass_def_vir = MassDef('vir', 'matter')
            translate_mass = self.mass_def.translate_mass
            Mvir = translate_mass(cosmo, M, a, mass_def_vir, squeeze=False)
            sM = cosmo.sigmaM(Mvir, a, squeeze=False)
            nu = const.DELTA_C / sM
            return 0.155 + 0.0095 * nu * nu
        return self.alpha

    def _norm(self, M, Rs, c, alpha):
        """Einasto normalization from mass, radius, concentration and alpha."""
        return M / (np.pi * Rs**3 * 2**(2-3/alpha) * alpha**(-1+3/alpha)
                    * np.exp(2/alpha)
                    * gamma(3/alpha) * gammainc(3/alpha, 2/alpha*c**alpha))

    def _real(self, cosmo, r, M, a):
        # Comoving virial radius
        R_M = self.mass_def.get_radius(cosmo, M, a, squeeze=False) / a
        c_M = self._get_concentration(cosmo, M, a)
        R_s = R_M / c_M

        α = self._get_alpha(cosmo, M, a)
        norm = self._norm(M, R_s, c_M, α)
        x = r / R_s
        prof = norm * np.exp(-2. * (x**α - 1) / α)

        if self.truncated:
            prof[r > R_M] = 0
        return prof
