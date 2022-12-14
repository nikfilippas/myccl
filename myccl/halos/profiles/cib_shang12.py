from .nfw import HaloProfileNFW
from .profile_base import HaloProfile
from ...integrate import IntegratorSamples
from ...parameters import spline_params as sparams
from ...pyutils import get_broadcastable

import numpy as np
from numpy import pi
from scipy.special import lambertw


class HaloProfileCIBShang12(HaloProfile):
    r"""CIB profile of the model by Shang et al. (2012MNRAS.421.2832S).

    The parametrization for the mean profile is:

    .. math::

        j_\nu(r) = \frac{1}{4\pi} \left(
        L^{\rm cen}_{\nu(1+z)}(M) + L^{\rm sat}_{\nu(1+z)} u_{\rm sat}(r|M)
        \right),

    where the luminosity from centrals and satellites is
    modelled as:

    .. math::

        L^{\rm cen}_{\nu}(M) = L^{\rm gal}_\nu(M) \, N_{\rm cen}(M),

    .. math::

        L^{\rm sat}_{\nu}(M) = \int_{M_{\rm min}}^{M} \mathrm{d}m
        \frac{\mathrm{d}N_{\rm sub}}{\mathrm{d}m} \, L^{\rm gal}_\nu(m).

    Here, :math:`\frac{\mathrm{d}N_{\rm sub}}{\mathrm{d}m}` is the subhalo
    mass function, :math:`u_{\rm sat}` is the satellite galaxy density profile
    (modelled as a truncated NFW profile), and the infrared
    galaxy luminosity is parametrized as

    .. math::

        L^{\rm gal}_{\nu}(M,z) = L_0(1+z)^{s_z} \, \Sigma(M) \, S_\nu,

    where the mass dependence is lognormal:

    .. math::

        \Sigma(M) = \frac{M}{\sqrt{2 \pi \sigma_{LM}^2}}
        \exp \left(-\frac{\log_{10}^2(M/M_{\rm eff})} {2\sigma_{LM}^2} \right),

    and the spectrum is a modified black-body:

    .. math::

        S_\nu \propto \left\{
        \begin{array}{cc}
           \nu^\beta \, B_\nu(T_d) & \nu < \nu_0 \\
           \nu^\gamma              & \nu \geq \nu_0
        \end{array}
        \right.,

    with the normalization fixed by :math:`S_{\nu_0} = 1`, and :math:`\nu_0`
    defined so the spectrum has a continuous derivative at all :math:`\nu`.

    Finally, the dust temperature is assumed to have a redshift
    dependence of the form :math:`T_d = T_0(1+z)^\alpha`.

    Parameters
    ----------
    mass_def : ~pyccl.halos.massdef.MassDef
        Mass definition.
    concentration : :obj:`Concentration`
        Concentration-mass relation.
    nu_GHz : float
        Frequency in :math:`\rm GHz`.
    T0 : float
        Dust temperature at :math:`z = 0` in :math:`\rm K`.
    alpha : float, optional
        Dust temperature evolution parameter.
    beta : float, optional
        Dust spectral index.
    gamma : float, optional
        High frequency slope.
    s_z : float, optional
        Luminosity evolution slope.
    log10meff : float, optional
        :math:`\log10` of the most efficient mass.
    sigLM : float, optional
        Logarithmic scatter in mass.
    Mmin : float, optional
        Minimum subhalo mass.
    L0 : float, optional
        Luminosity scale in :math:`\rm Jy \, Mpc^2 \, M_\odot^{-1}`.
    """
    name = 'CIBShang12'
    normprof = True

    def __init__(self, mass_def, concentration, *, nu_GHz,
                 T0=24.4, alpha=0.36, beta=1.75, gamma=1.7,
                 s_z=3.6, log10meff=12.6, sigLM=0.707, Mmin=1e10, L0=6.4e-8):
        super().__init__(mass_def=mass_def)
        self.pNFW = HaloProfileNFW(mass_def, concentration)
        self.concentration = concentration

        self.nu = nu_GHz
        self.T0 = T0
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.s_z = s_z
        self.l10meff = log10meff
        self.sigLM = sigLM
        self.Mmin = Mmin
        self.L0 = L0

        self._integrator = IntegratorSamples("simpson")

    def update_parameters(self, nu_GHz=None, T0=None, alpha=None, beta=None,
                          gamma=None, s_z=None, log10meff=None, sigLM=None,
                          Mmin=None, L0=None, **kwargs):
        """Update the profile parameters."""
        super().update_parameters(**kwargs)
        if nu_GHz is not None:
            self.nu = nu_GHz
        if alpha is not None:
            self.alpha = alpha
        if T0 is not None:
            self.T0 = T0
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
        if s_z is not None:
            self.s_z = s_z
        if log10meff is not None:
            self.l10meff = log10meff
        if sigLM is not None:
            self.sigLM = sigLM
        if Mmin is not None:
            self.Mmin = Mmin
        if L0 is not None:
            self.L0 = L0

    def _spectrum(self, nu, a):
        # h*nu_GHZ / k_B / Td_K
        h_GHz_o_kB_K = 0.0479924466
        Td = self.T0 / a**self.alpha
        x = h_GHz_o_kB_K * nu / Td

        # Find nu_0
        q = self.beta + 3 + self.gamma
        x0 = q + np.real(lambertw(-q * np.exp(-q), k=0))

        mBB = lambda x: x**(3 + self.beta) / (np.exp(x)-1)  # noqa
        mBB0 = mBB(x0)
        plaw = lambda x: mBB0*(x0/x)**self.gamma  # noqa
        return np.piecewise(x, [x <= x0], [mBB, plaw]) / mBB0

    def dNsub_dlnM_TinkerWetzel10(self, Msub, Mparent, *, squeeze=True):
        r"""Subhalo mass function of Tinker & Wetzel (2010ApJ...719...88T)

        Arguments
        ---------
        Msub : float or (nMsub,) array_like
            Sub-halo mass in :math:`\rm M_\odot`.
        Mparent : float or (nMparent,) array_like
            Prent halo mass in :math:`\rm M_\odot`.
        squeeze : bool
            Squeeze extra dimensions of size (1,) in the output.
            The default is True.

        Returns:
        dN_dlnM : float or (nMparent, nMsub) ndarray
            Average number of subhalos.
        """
        Mparent, Msub = map(np.asarray, [Mparent, Msub])
        Mparent, Msub = get_broadcastable(Mparent, Msub)
        dN_dlnM = 0.30*(Msub/Mparent)**(-0.7)*np.exp(-9.9*(Msub/Mparent)**2.5)
        return dN_dlnM.squeeze()[()] if squeeze else dN_dlnM

    def _Lum(self, l10M, a):
        # Redshift evolution
        phi_z = a**(-self.s_z)
        # Mass dependence
        sig_pref = 10**l10M / (np.sqrt(2*pi) * self.sigLM)
        sigma_m = sig_pref * np.exp(-0.5*((l10M - self.l10meff)/self.sigLM)**2)
        return self.L0 * phi_z * sigma_m

    def _Lumcen(self, M, a):
        Lum = self._Lum(np.log10(M), a)
        return np.heaviside(M - self.Mmin, 1) * Lum

    def _Lumsat(self, M, a):
        # TODO: dims here won't work
        shp = np.broadcast_shapes(M.shape, a.shape)
        res = np.zeros(shp)

        if np.max(M) <= self.Mmin:
            return res

        idx = M >= self.Mmin  # relevant masses

        # Set up integration samples.
        logMmax = np.log10(M).max()
        LOGM_MIN = np.log10(self.Mmin)
        nm = max(2, sparams.N_M_PER_DECADE*int(np.max(logMmax) - LOGM_MIN))
        msub = np.linspace(LOGM_MIN, np.max(logMmax), nm+1)

        # Integrate over msub, so this will be the trailing integration axis.
        a = np.expand_dims(a, -1)
        M = np.expand_dims(M, -1)
        shp_integ = (1,)*len(shp) + (-1,)
        msub = msub.reshape(shp_integ)

        Lum = self._Lum(msub, a)
        mfsub = self.dNsub_dlnM_TinkerWetzel10(10**msub, M[idx], squeeze=False)
        integ = mfsub * Lum
        Lumsat = self._integrator(integ, x=np.log(10)*msub)

        idx = np.broadcast_to(idx, res.shape)
        np.place(res, idx, Lumsat)
        return res

    def _real(self, cosmo, r, M, a):
        spec_nu = self._spectrum(self.nu / a, a)  # redshifted nu-dependence
        Ls = self._Lumsat(M, a)
        ur = self.pNFW._real(cosmo, r, M, a) / M
        return Ls * ur * spec_nu / (4 * pi)

    def _fourier(self, cosmo, k, M, a):
        spec_nu = self._spectrum(self.nu/a, a)  # redshifted nu-dependence
        Lc = self._Lumcen(M, a)
        Ls = self._Lumsat(M, a)
        uk = self.pNFW._fourier(cosmo, k, M, a) / M
        return (Lc + Ls*uk) * spec_nu / (4 * pi)

    def _fourier_variance(self, cosmo, k, M, a, nu_other=None):
        spec = self._spectrum(self.nu/a, a)
        spec *= spec if nu_other is None else self._spectrum(nu_other / a, a)

        Lc = self._Lumcen(M, a)
        Ls = self._Lumsat(M, a)
        uk = self.pNFW._fourier(cosmo, k, M, a) / M

        prof = Ls*uk
        return (2*Lc*prof + prof**2) * spec / (4*pi)**2

    def fourier_2pt(self, cosmo, k, M, a, prof2=None):
        """Fourier-space 2-point moment between two CIB profiles."""
        if prof2 is not None or not isinstance(prof2, HaloProfileCIBShang12):
            raise TypeError("prof must be of type `HaloProfileCIB`")
        nu2 = None if prof2 is None else prof2.nu
        return self._fourier_variance(cosmo, k, M, a, nu_other=nu2)
