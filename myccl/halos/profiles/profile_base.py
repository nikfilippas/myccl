from ...parameters import _FFTLogParams
from ...pyutils import get_broadcastable, resample_array, _fftlog_transform
from ...errors import CCLWarning

import numpy as np
from abc import ABC, abstractproperty
from functools import partial
import warnings


class HaloProfile(ABC):
    """Functionality associated to halo profiles.

    This abstract class contains methods to compute a halo profile in 3-D
    real and Fourier space, as well as projected (2-D) and the cumulative
    mean surface density.

    A minimal profile implementation should contain a method ``_real``
    or a ``_fourier`` method with signatures as in ``real`` and ``fourier``
    of this class, respectively. Fast Hankel transforms from real to fourier
    space and vice versa are performed internally with ``FFTLog``. Subclasses
    may contain analytic implementations of any of those methods to bypass
    the ``FFTLog`` calculation.

    Parameters
    ----------
    mass_def : ~pyccl.halos.massdef.MassDef
        Mass definition of the profile.
    r_corr : float, optional
        Tuning knob for the 2-point moment. Details in ``_fourier_2pt``.
        Examples in :arXiv:1909.09102 and :arXiv:2102.07701
        The default is 0, returning the product of the profiles.
        **Note**: For particular implementations of the 2-point moment,
        the behavior of this parameter can be overriden, or ignored.
    """
    is_number_counts = False

    def __init__(self, mass_def, *, r_corr=0):
        self.precision_fftlog = _FFTLogParams()
        self.mass_def = mass_def
        self.r_corr = r_corr

    @abstractproperty
    def normprof(self) -> bool:
        """Normalize the profile in auto- and cross-correlations by
        :math:`I^0_1(k\\rightarrow 0, a|u)`
        (see :meth:`~pyccl.halos.halo_model.HMCalculator.I_0_1`).
        """

    def update_parameters(self, r_corr=None, **kwargs) -> None:
        """Update any of the parameters associated with this profile.

        .. note::

            Subclasses implementing ``update_parameters`` must collect
            any remaining keyword arguments and pipe them to ``super``
            to enable updating of the parent class parameters.
        """
        # super().update_parameters(**kwargs)  # Example: Update parent class.
        if r_corr is not None:
            self.r_corr = r_corr

    def update_precision_fftlog(self, **kwargs):
        self.precision_fftlog.update_parameters(**kwargs)

    update_precision_fftlog.__doc__ = _FFTLogParams.update_parameters.__doc__

    _get_plaw_fourier = partial(_FFTLogParams._get_plaw, name="plaw_fourier")
    _get_plaw_projected = partial(_FFTLogParams._get_plaw, name="plaw_projected")  # noqa
    _get_plaw_fourier.__doc__ = _get_plaw_projected.__doc__ = _FFTLogParams._get_plaw.__doc__  # noqa

    def real(self, cosmo, r, M, a, *, squeeze=True):
        """3-D real-space profile.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        r : float or (nr,) array_like
            Comoving radius in :math:`\\mathrm{Mpc}`.
        M : float or (nM,) array_like
            Halo mass in :math:`\\mathrm{M}_{\\odot}`.
        a : float or (na,) array_like
            Scale factor.
        squeeze : bool
            Squeeze extra dimensions of size (1,) in the output.
            The default is True.

        Returns
        -------
        P_r : float or array_like
            Real halo profile.
        """
        # Axes are [a, r, M].
        a, r, M = map(np.atleast_1d, [a, r, M])
        a, r, M = get_broadcastable(a, r, M)
        if self._has_implementation("_real"):
            out = self._real(cosmo, r, M, a)
            return out.squeeze()[()] if squeeze else out
        if self._has_implementation("_fourier"):
            out = self._fftlog_wrap(cosmo, r, M, a, fourier_out=False)
            return out.squeeze()[()] if squeeze else out
        raise NotImplementedError

    def fourier(self, cosmo, k, M, a, *, squeeze=True):
        r"""3-D Fourier-space profile.

        .. math::

           \rho(k) = \frac{1}{2 \pi^2} \int \mathrm{d}r \, r^2 \,
           \rho(r) \, j_0(k \, r)

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        k : float or array_like
            Comoving wavenumber in :math:`\\mathrm{Mpc}^{-1}`.
        M : float or array_like
            Halo mass in :math:`\\mathrm{M}_{\\odot}`.
        a : float or array_like
            Scale factor.
        squeeze : bool
            Squeeze extra dimensions of size (1,) in the output.
            The default is True.

        Returns
        -------
        P_r : float or array_like
            Fourier halo profile.
        """
        # Axes are [a, k, M].
        a, k, M = map(np.atleast_1d, [a, k, M])
        a, k, M = get_broadcastable(a, k, M)
        if self._has_implementation("_fourier"):
            out = self._fourier(cosmo, k, M, a)
            return out.squeeze()[()] if squeeze else out
        if self._has_implementation("_real"):
            out = self._fftlog_wrap(cosmo, k, M, a, fourier_out=True)
            return out.squeeze()[()] if squeeze else out
        raise NotImplementedError

    def projected(self, cosmo, r, M, a, *, squeeze=True):
        """2-D projected profile.

        .. math::

           \\Sigma(R) = \\int \\mathrm{d}r_\\parallel \\,
           \\rho( \\sqrt{ r_\\parallel^2 + R^2} )

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        r : float or array_like
            Comoving radius in :math:`\\mathrm{Mpc}`.
        M : float or array_like
            Halo mass in :math:`\\mathrm{M}_{\\odot}`.
        a : float or array_like
            Scale factor.
        squeeze : bool
            Squeeze extra dimensions of size (1,) in the output.
            The default is True.

        Returns
        -------
        P_proj : float or array_like
            Projected halo profile.
        """
        # Axes are [a, r, M].
        a, r, M = map(np.asarray, [a, r, M])
        a, r, M = get_broadcastable(a, r, M)
        if self._has_implementation("_projected"):
            out = self._projected(cosmo, r, M, a)
            return out.squeeze()[()] if squeeze else out
        out = self._projected_fftlog_wrap(cosmo, r, M, a, is_cumul2d=False)
        return out.squeeze()[()] if squeeze else out

    def cumul2d(self, cosmo, r, M, a, *, squeeze=True):
        r"""2-D cumulative surface density.

        .. math::

           \Sigma(<R) = \frac{2}{R^2} \int \mathrm{d}R' \, R' \, \Sigma(R')

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        r : float or array_like
            Comoving radius in :math:`\\mathrm{Mpc}`.
        M : float or array_like
            Halo mass in :math:`\\mathrm{M}_{\\odot}`.
        a : float or array_like
            Scale factor.
        squeeze : bool
            Squeeze extra dimensions of size (1,) in the output.
            The default is True.

        Returns
        -------
        P_cumul : float or array_like
            Cumulative halo profile.
        """
        # Axes are [a, r, M].
        a, r, M = map(np.asarray, [a, r, M])
        a, r, M = get_broadcastable(a, r, M)
        if self._has_implementation("_cumul2d"):
            out = self._cumul2d(cosmo, r, M, a)
            return out.squeeze()[()] if squeeze else out
        out = self._projected_fftlog_wrap(cosmo, r, M, a, is_cumul2d=True)
        return out.squeeze()[()] if squeeze else out

    def convergence(self, cosmo, r, M, a_lens, a_source, *, squeeze=True):
        r"""Profile onvergence.

        .. math::

           \kappa(R) = \frac{\Sigma(R)}{\Sigma_{\rm crit}},

        where :math:`\Sigma(R)` is the 2D projected surface mass density.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        r : float or array_like
            Comoving radius in :math:`\\mathrm{Mpc}`.
        M : float or array_like
            Halo mass in :math:`\\mathrm{M}_{\\odot}`.
        a_lens, a_source : float or array_like
            Scale factors of the lens and the source, respectively.
            If ``a_source`` is array_like, ``r.shape == a_source.shape``.
        squeeze : bool
            Squeeze extra dimensions of size (1,) in the output.
            The default is True.

        Returns
        -------
        P_conv : float or array_like
            Convergence :math:`\kappa` of the profile.
        """
        Sigma = self.projected(cosmo, r, M, a_lens, squeeze=False) / a_lens**2
        Sigma_crit = cosmo.sigma_critical(a_lens, a_source, squeeze=False)
        out = Sigma / Sigma_crit
        return out.squeeze()[()] if squeeze else out

    def shear(self, cosmo, r, M, a_lens, a_source, *, squeeze=False):
        r"""Tangential shear of a profile.

        .. math::

           \gamma(R) = \frac{\Delta \Sigma(R)}{\Sigma_{\rm crit}} =
           \frac{\overline{\Sigma}(< R) - \Sigma(R)}{\Sigma_{\rm crit}},

        where :math:`\overline{\Sigma}(< R)` is the average surface density
        within R.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        r : float or array_like
            Comoving radius in :math:`\\mathrm{Mpc}`.
        M : float or array_like
            Halo mass in :math:`\\mathrm{M}_{\\odot}`.
        a_lens, a_source : float or array_like
            Scale factors of the lens and the source, respectively.
            If ``a_source`` is array_like, ``r.shape == a_source.shape``.
        squeeze : bool
            Squeeze extra dimensions of size (1,) in the output.
            The default is True.

        Returns
        -------
        P_shear : float or array_like
            Tangential shear :math:`\\gamma` of the profile.
        """
        Sigma = self.projected(cosmo, r, M, a_lens, squeeze=False)
        Sigma_bar = self.cumul2d(cosmo, r, M, a_lens, squeeze=False)
        Sigma_crit = cosmo.sigma_critical(a_lens, a_source, squeeze=False)
        out = (Sigma_bar - Sigma) / (Sigma_crit * a_lens**2)
        return out.squeeze()[()] if squeeze else out

    def reduced_shear(self, cosmo, r, M, a_lens, a_source, *, squeeze=False):
        r"""Reduced shear of a profile.

        .. math::

           g_t (R) = \frac{\gamma(R)}{(1 - \kappa(R))},

        where :math:`\gamma(R)` is the shear and :math:`\kappa(R)` is the
        convergence.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        r : float or array_like
            Comoving radius in :math:`\\mathrm{Mpc}`.
        M : float or array_like
            Halo mass in :math:`\\mathrm{M}_{\\odot}`.
        a_lens, a_source : float or array_like
            Scale factors of the lens and the source, respectively.
            If ``a_source`` is array_like, ``r.shape == a_source.shape``.
        squeeze : bool
            Squeeze extra dimensions of size (1,) in the output.
            The default is True.

        Returns
        -------
        P_red_shear : float or array_like
            Reduced shear :math:`g_t` of the profile.
        """
        convergence = self.convergence(cosmo, r, M, a_lens, a_source,
                                       squeeze=False)
        shear = self.shear(cosmo, r, M, a_lens, a_source, squeeze=False)
        out = shear / (1.0 - convergence)
        return out.squeeze()[()] if squeeze else out

    def magnification(self, cosmo, r, M, a_lens, a_source, *, squeeze=False):
        r"""Magnification of a profile.

        .. math::

           \mu (R) = \frac{1}{\left( (1 - \kappa(R))^2 -
           \vert \gamma(R) \vert^2 \right)},

        where :math:`\gamma(R)` is the shear and :math:`\kappa(R)` is the
        convergence.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        r : float or array_like
            Comoving radius in :math:`\mathrm{Mpc}`.
        M : float or array_like
            Halo mass in :math:`\mathrm{M}_{\odot}`.
        a_lens, a_source : float or array_like
            Scale factors of the lens and the source, respectively.
            If ``a_source`` is array_like, ``r.shape == a_source.shape``.
        squeeze : bool
            Squeeze extra dimensions of size (1,) in the output.
            The default is True.

        Returns
        -------
        P_magn : float or array_like
            Magnification :math:`\mu` of the profile.
        """
        convergence = self.convergence(cosmo, r, M, a_lens, a_source,
                                       squeeze=False)
        shear = self.shear(cosmo, r, M, a_lens, a_source, squeeze=False)
        out = 1.0 / ((1.0 - convergence)**2 - np.abs(shear)**2)
        return out.squeeze()[()] if squeeze else out

    def _fourier_2pt(self, cosmo, k, M, a, prof2):
        r"""Default implementation for the Fourier-space 1-halo 2-point
        correlator between two halo profiles:

        .. math::

            (1 + \rho_{u_1, u_2}) \langle u_1(k) \rangle \langle u_2(k) \rangle

        In the simplest scenario, the second-order cumulant is the product
        of the individual Fourier-space profiles, scaled by ``r_corr`` as
        :math:`(1 + \rho_{u_1, u_2})` for profiles not fully correlated.
        """
        if prof2 is None:
            prof2 = self

        # Warn if profiles implement the same correlator with different r_corr.
        cl1, cl2 = self.__class__, prof2.__class__
        same_correlator = cl1._fourier_2pt == cl2._fourier_2pt
        if same_correlator and self.r_corr != prof2.r_corr:
            warnings.warn("Correlating profiles with different r_corr. "
                          "Using r_corr of the first profile.", CCLWarning)

        uk = self.fourier(cosmo, k, M, a, squeeze=False)
        uk *= uk if prof2 == self else prof2.fourier(cosmo, k, M, a, squeeze=False)  # noqa
        return uk * (1 + self.r_corr)

    def fourier_2pt(self, cosmo, k, M, a, prof2=None, *, squeeze=True):
        r"""Fourier-space 2-point moment between two profiles.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        k : float or (nk,) array_like
            Comoving wavenumber in :math:`\rm Mpc^{-1}`.
        M : float or (nM,) array_like
            Halo mass in :math:`\rm M_\odot`.
        a : float or (na,) array_like
            Scale factor.
        prof, prof2 : :class:`~pyccl.halos.profiles.HaloProfile`
            Halo profiles. If ``prof2 is None``, ``prof`` will be used
            (i.e. profile will be auto-correlated).
        squeeze : bool, optional
            Squeeze extra dimensions of size (1,) in the output.
            The default is True.

        Returns
        -------
        F_var : float or (na, nk, nM) ndarray
            Second-order Fourier-space moment between the input profiles.
        """
        a, k, M = map(np.atleast_1d, [a, k, M])
        a, k, M = get_broadcastable(a, k, M)
        # Try `self ⊗ prof2`. If it doesn't work do `prof2 ⊗ self`.
        try:
            out = self._fourier_2pt(cosmo, k, M, a, prof2)
        except ValueError:
            out = prof2._fourier_2pt(cosmo, k, M, a, self)
        return out.squeeze()[()] if squeeze else out

    def _fftlog_wrap(self, cosmo, k, M, a,
                     fourier_out=False,
                     large_padding=True):
        # COmpute the 3D Hankel transform ρ(x) = K ∫ dx x² ρ(x) j₀(x x̃),
        # where K is 1/(2π²) or 4π for real and fourier profiles, respectively.

        # Select which profile should be the input
        if fourier_out:
            p_func = self._real
        else:
            p_func = self._fourier
        k_use = np.atleast_1d(k)
        M_use = np.atleast_1d(M)
        lk_use = np.log(k_use)
        nM = len(M_use)

        # k/r ranges to be used with FFTLog and its sampling.
        if large_padding:
            k_min = self.precision_fftlog['padding_lo_fftlog'] * np.amin(k_use)
            k_max = self.precision_fftlog['padding_hi_fftlog'] * np.amax(k_use)
        else:
            k_min = self.precision_fftlog['padding_lo_extra'] * np.amin(k_use)
            k_max = self.precision_fftlog['padding_hi_extra'] * np.amax(k_use)
        n_k = (int(np.log10(k_max / k_min)) *
               self.precision_fftlog['n_per_decade'])
        r_arr = np.geomspace(k_min, k_max, n_k)

        p_k_out = np.zeros([nM, k_use.size])
        # Compute real profile values
        p_real_M = p_func(cosmo, r_arr, M_use, a)
        # Power-law index to pass to FFTLog.
        plaw_index = self._get_plaw_fourier(cosmo, a)

        # Compute Fourier profile through fftlog
        k_arr, p_fourier_M = _fftlog_transform(r_arr, p_real_M,
                                               3, 0, plaw_index)
        lk_arr = np.log(k_arr)

        for im, p_k_arr in enumerate(p_fourier_M):
            # Resample into input k values
            p_fourier = resample_array(lk_arr, p_k_arr, lk_use,
                                       self.precision_fftlog['extrapol'],
                                       self.precision_fftlog['extrapol'],
                                       0, 0)
            p_k_out[im, :] = p_fourier
        if fourier_out:
            p_k_out *= (2 * np.pi)**3

        if np.ndim(k) == 0:
            p_k_out = np.squeeze(p_k_out, axis=-1)
        if np.ndim(M) == 0:
            p_k_out = np.squeeze(p_k_out, axis=0)
        return p_k_out

    def _projected_fftlog_wrap(self, cosmo, r_t, M, a, is_cumul2d=False):
        # This computes Σ(R) from the Fourier-space profile as
        # Σ(R) = 1/(2π) ∫ dk k J₀(k R) ρ(k).
        r_t_use = np.atleast_1d(r_t)
        M_use = np.atleast_1d(M)
        lr_t_use = np.log(r_t_use)
        nM = len(M_use)

        # k/r range to be used with FFTLog and its sampling.
        r_t_min = self.precision_fftlog['padding_lo_fftlog'] * np.amin(r_t_use)
        r_t_max = self.precision_fftlog['padding_hi_fftlog'] * np.amax(r_t_use)
        n_r_t = (int(np.log10(r_t_max / r_t_min)) *
                 self.precision_fftlog['n_per_decade'])
        k_arr = np.geomspace(r_t_min, r_t_max, n_r_t)

        sig_r_t_out = np.zeros([nM, r_t_use.size])
        # Compute Fourier-space profile
        if getattr(self, '_fourier', None):
            # Compute from `_fourier` if available.
            p_fourier = self._fourier(cosmo, k_arr, M_use, a)
        else:
            # Compute with FFTLog otherwise.
            lpad = self.precision_fftlog['large_padding_2D']
            p_fourier = self._fftlog_wrap(cosmo,
                                          k_arr,
                                          M_use, a,
                                          fourier_out=True,
                                          large_padding=lpad)
        if is_cumul2d:
            # The cumulative profile involves a factor 1/(k R) in
            # the integrand.
            p_fourier *= 2 / k_arr[None, :]

        # Power-law index to pass to FFTLog.
        if is_cumul2d:
            i_bessel = 1
            plaw_index = self._get_plaw_projected(cosmo, a) - 1
        else:
            i_bessel = 0
            plaw_index = self._get_plaw_projected(cosmo, a)

        # Compute projected profile through fftlog
        r_t_arr, sig_r_t_M = _fftlog_transform(k_arr, p_fourier,
                                               2, i_bessel,
                                               plaw_index)
        lr_t_arr = np.log(r_t_arr)

        if is_cumul2d:
            sig_r_t_M /= r_t_arr[None, :]
        for im, sig_r_t_arr in enumerate(sig_r_t_M):
            # Resample into input r_t values
            sig_r_t = resample_array(lr_t_arr, sig_r_t_arr,
                                     lr_t_use,
                                     self.precision_fftlog['extrapol'],
                                     self.precision_fftlog['extrapol'],
                                     0, 0)
            sig_r_t_out[im, :] = sig_r_t

        if np.ndim(r_t) == 0:
            sig_r_t_out = np.squeeze(sig_r_t_out, axis=-1)
        if np.ndim(M) == 0:
            sig_r_t_out = np.squeeze(sig_r_t_out, axis=0)
        return sig_r_t_out

    def _check_consistent_mass(self, mass_def, concentration):
        """Check mass definition consistency for profile & concentration."""
        if concentration.mass_def is None:
            # Concentration defined for arbitrary mass definition.
            return

        if mass_def != concentration.mass_def:
            name = self.__class__.__name__
            raise ValueError(f"Inconsistent mass definition for {name}.")

    def _has_implementation(self, func):
        """Check whether ``func`` is implemented or inherited."""
        return func in vars(self) or func in vars(self.__class__)


class HaloProfileNumberCounts(HaloProfile):
    """Abstract profile implementing a number counts quantity."""
    is_number_counts = True
    normprof = True
