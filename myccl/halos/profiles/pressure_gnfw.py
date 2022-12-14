from .profile_base import HaloProfile
from ..massdef import MassDef500c
from ...interpolate import Interpolator1D
from ...integrate import IntegratorFunction
import numpy as np


class HaloProfilePressureGNFW(HaloProfile):
    r"""Generalized NFW pressure profile, Arnaud et al. (2010A&A...517A..92A).

    The parametrization is:

    .. math::

       P_e(r) = C \times P_0 h_{70}^E (c_{500} x)^{-\gamma}
       \left(1 + (c_{500} x)^\alpha \right)^{(\gamma - \beta) / \alpha},

    where

    .. math::

       C = 1.65 \, h_{70}^2 \left(\frac{H(z)}{H_0}\right)^{8/3}
       \left(\frac{h_{70} \tilde{M}_{500}}
       {3 \times10^{14} \, M_\odot} \right)^{2/3 + \alpha_{\rm P}},

    :math:`x = r / \tilde{r}_{500}`, :math:`h_{70} = h/0.7`, and the exponent
    :math:`E` is -1 for SZ-based profile normalizations and -1.5 for X-ray-
    based normalizations.

    The biased mass :math:`\tilde{M}_{500}` is related to the true overdensity
    mass :math:`M_{500}` via the mass bias parameter :math:`(1-b)`
    as :math:`\tilde{M}_{500} = (1-b) M_{500}`. :math:`\tilde{r}_{500}`
    is the overdensity halo radius associated with :math:`\tilde{M}_{500}`
    (note the intentional tilde).

    .. note::
        The profile is defined for :math:`\Delta = 500c`.
        The default parameters (except ``mass_bias``), correspond to the
        profile parameters used in the ``Planck 2013 (V)`` paper. The profile
        is computed in physical units of :math:`\rm eV \, cm^{-3}`.

    Parameters
    ----------
    mass_def : ~pyccl.halos.massdef.MassDef
        Mass definition of the profile.
        The default is :math:`500c`.
    mass_bias : float, optional
        The mass bias parameter :math:`1-b`.
        The default is 0.8.
    P0 : float, optional
        Profile normalization.
        The default is 6.41.
    c500 : float, optional
        Concentration parameter.
        The default is 1.81.
    alpha, beta, gamma : float, optional
        Profile shape parameters.
        The defaults are (α, β, γ) = (1.33, 4.13, 0.31).
    alpha_P : float, optional
        Additional mass dependence exponent
        The default is 0.12.
    P0_hexp : float, optional
        Power of :math:`h` with which the normalization parameter scales.
        Equal to :math:`-1` for SZ-based normalizations,
        and :math:`-3/2` for X-ray-based normalizations.
        The default is -1.
    x_out : float, optional
        Profile threshold, in units of :math:`R_{\mathrm{500c}}`.
        The default is :math:`+\infty`.
    qrange : (2,) sequence, optional
        Limits of integration used when computing the Fourier-space
        profile template, in units of :math:`R_{\mathrm{vir}}`.
        The default is ``(1e-3, 1e+3)``.
    nq : int, optional
        Number of sampling points of the Fourier-space profile template.
        The default is 128.
    """
    name = 'GNFW'
    normprof = False

    def __init__(self, mass_def=MassDef500c(), *,
                 mass_bias=0.8, P0=6.41, c500=1.81,
                 alpha=1.33, beta=4.13, gamma=0.31, alpha_P=0.12,
                 P0_hexp=-1, x_out=np.inf,
                 qrange=(1e-3, 1e3), nq=128):
        super().__init__(mass_def=mass_def)
        self.mass_bias = mass_bias
        self.P0 = P0
        self.c500 = c500
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.alpha_P = alpha_P
        self.P0_hexp = P0_hexp
        self.x_out = x_out
        self.qrange = qrange
        self.nq = nq

        # Interpolator for dimensionless Fourier-space profile
        self._integrator = IntegratorFunction("fourier_transform")
        self._fourier_interp = None

    def update_parameters(self, mass_bias=None, P0=None,
                          c500=None, alpha=None, beta=None, gamma=None,
                          alpha_P=None, P0_hexp=None, x_out=None, **kwargs):
        """Update any of the parameters associated with this profile."""
        super().update_parameters(**kwargs)
        if mass_bias is not None:
            self.mass_bias = mass_bias
        if alpha_P is not None:
            self.alpha_P = alpha_P
        if P0 is not None:
            self.P0 = P0
        if P0_hexp is not None:
            self.P0_hexp = P0_hexp

        # Check if we need to recompute the Fourier profile.
        re_fourier = False
        if alpha is not None and alpha != self.alpha:
            re_fourier = True
            self.alpha = alpha
        if beta is not None and beta != self.beta:
            re_fourier = True
            self.beta = beta
        if gamma is not None and gamma != self.gamma:
            re_fourier = True
            self.gamma = gamma
        if c500 is not None and c500 != self.c500:
            re_fourier = True
            self.c500 = c500
        if x_out is not None and x_out != self.x_out:
            re_fourier = True
            self.x_out = x_out

        if re_fourier and (self._fourier_interp is not None):
            self._fourier_interp = self._integ_interp()

    def _form_factor(self, x):
        # Scale-dependent factor of the GNFW profile.
        f1 = (self.c500*x)**(-self.gamma)
        exponent = -(self.beta-self.gamma) / self.alpha
        f2 = (1 + (self.c500*x)**self.alpha)**exponent
        return f1 * f2

    def _integ_interp(self):
        # Precompute the Fourier transform of the profile in terms
        # of the scaled radius x and create a spline interpolator.
        # Use the Fourier transform integrator of QUADPACK (qawfe) instead
        # of FFTLog to avoid complicating things.
        integrand = lambda x: self._form_factor(x) * x  # noqa
        q_arr = np.geomspace(self.qrange[0], self.qrange[1], self.nq)
        f_arr = self._integrator(integrand, a=1e-4, b=self.x_out, wvar=q_arr)
        return Interpolator1D(np.log(q_arr), f_arr/q_arr, extrap_orders=[1, 1])

    def _norm(self, cosmo, M, a, mb):
        # Computes the normalisation factor of the GNFW profile.
        # Normalisation factor is given in units of eV/cm^3.
        # (Bolliet et al. 2017).
        h70 = cosmo["h"] / 0.7
        C0 = 1.65 * h70**2
        CM = (h70 * M * mb / 3e14)**(2/3 + self.alpha_P)   # M dependence
        Cz = cosmo.h_over_h0(a, squeeze=False)**(8/3)      # z dependence
        P0_corr = self.P0 * h70**self.P0_hexp  # h-corrected P_0
        return P0_corr * C0 * CM * Cz

    def _real(self, cosmo, r, M, a):
        r"""Real-space profile in :math:`\rm eV \, cm^{-3}."""
        # Comoving virial radius
        mb = self.mass_bias
        R = self.mass_def.get_radius(cosmo, M * mb, a, squeeze=False) / a

        norm = self._norm(cosmo, M, a, mb)
        prof = self._form_factor(r / R)
        return prof * norm

    def _fourier(self, cosmo, k, M, a):
        r"""Fourier-space profile in :math:`\rm eV \, Mpc^3 \, cm^{-3}."""
        # Tabulate if not done yet
        if self._fourier_interp is None:
            self._fourier_interp = self._integ_interp()

        mb = self.mass_bias
        R = self.mass_def.get_radius(cosmo, M * mb, a, squeeze=False) / a

        ff = self._fourier_interp(np.log(k * R))
        nn = self._norm(cosmo, M, a, mb)
        return (4*np.pi*R**3 * nn) * ff
