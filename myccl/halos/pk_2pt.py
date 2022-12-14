from ..pk2d import Pk2D
from ..parameters import spline_params as sparams
from ..errors import CCLWarning

import numpy as np
from functools import partial
import warnings


__all__ = ("halomod_power_spectrum", "halomod_Pk2D")


def _pk_hm(cosmo, hmc, *, k=None, a=None, prof, prof2=None,
           p_of_k_a="linear",
           get_1h=True, get_2h=True,
           smooth_transition=None, suppress_1h=None,
           extrap_order_lok=1, extrap_order_hik=2,
           pk2d_out=False, use_log=True, squeeze=True):
    r"""Halo model power spectrum of two quantities expressed as halo profiles.
    The halo model power spectrum for two profiles :math:`u` and :math:`v` is:

    .. math::

        P_{u,v}(k, a) = I^0_2(k, a|u,v) +
        I^1_1(k, a|u) \, I^1_1(k, a|v) \, P_{\rm lin}(k, a)

    where :math:`P_{\rm lin}(k ,a)` is the linear matter power spectrum,
    :math:`I^1_1` is defined in :meth:`~HMCalculator.I_1_1`,
    and :math:`I^0_2` is defined in :meth:`~HMCalculator.I_0_2`.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    hmc : :class:`HMCalculator`
        Halo model calculator associated with the profile calculations.
    k : float or (nk,) array_like, optional
        Comoving wavenumber in :math:`\rm Mpc^{-1}`.
        If ``pk2d_out is True``, this should be :math:`log(k)`.
        The default is defined via the spline parameters of ``cosmo``.
    a : float or (na,) array_like, optional
        Scale factor.
        The default is defined via the spline parameters of ``cosmo``.
    prof, prof2 : :class:`~pyccl.halos.profiles.HaloProfile`
        Halo profiles. If ``prof2 is None``, ``prof`` will be used.
    p_of_k_a : :class:`~pyccl.pk2d.Pk2D`, or {'linear', 'nonlinear'}
        Power spectrum.
        The default is the linear matter power spectrum stored in ``cosmo``.
    get_1h, get_2h : bool
        Compute the 1-halo and 2-halo terms, respectively.
    smooth_transition : callable or None
        Modify the halo model 1-halo/2-halo transition region
        via a time-dependent function :math:`\alpha(a)`,
        defined as in ``HMCODE-2020`` (:arXiv:2009.01858). :math:`P(k,a) =
        (P_{1h}^{\alpha(a)}(k)+P_{2h}^{\alpha(a)}(k))^{1/\alpha}`.
        The default is ``None`` for no modification.
    suppress_1h : callable or None
        Suppress the 1-halo large scale contribution by a
        time- and scale-dependent function :math:`k_*(a)`,
        defined as in HMCODE-2020 (:arXiv:2009.01858).
        :math:`\frac{(k/k_*(a))^4}{1+(k/k_*(a))^4}`.
        The default is ``None`` for no damping.
    extrap_order_lok, extrap_order_hik : {0, 1, 2}
        Extrapolation order for low and high ``k``, respectively.
        Provided when ``pk2d_out is True``. The defaults are 1 and 2.
    pk2d_out : bool
        Return the halo model power spectrum array, or a ``Pk2D`` object.
        The default is False.
    use_log : bool
        When ``pk2d_out is True``, interpolate the power spectrum in log.
        The default is True.
    squeeze : bool, optional
        Squeeze extra dimensions of size (1,) in the output.
        Provided when ``pk2d_out is False``. The default is True.

    Returns
    -------
    pk_hm : float, (na, nk) ndarray, or :class:`~pyccl.pk2d.Pk2D`
        The halo model power spectrum.
        If ``pk2d_out is True``, a ``Pk2D`` object is returned.
    """
    if k is None and not pk2d_out:
        raise ValueError("Provide k and a for the power spectrum.")
    if smooth_transition and not (get_1h and get_2h):
        raise ValueError("Transition region can only be modified "
                         "when both 1-halo and 2-halo terms are queried.")
    if suppress_1h and not get_1h:
        raise ValueError("Can't supress the 1-halo term when get_1h is False.")

    # Sampling rate.
    if k is None:
        k = np.exp(sparams.get_pk_spline_lk())
    if a is None:
        a = sparams.get_pk_spline_a()

    # Power spectrum.
    pkf = p_of_k_a
    if not isinstance(p_of_k_a, Pk2D):
        if pkf == "linear":
            cosmo.compute_linear_power()
            pkf = cosmo.get_linear_power()
        elif pkf == "nonlinear":
            cosmo.compute_nonlin_power()
            pkf = cosmo.get_nonlin_power()

    # Profiles.
    if prof2 is None:
        prof2 = prof

    # Normalization.
    norm = hmc.profile_norm(cosmo, a, prof) if prof.normprof else 1
    if prof2.normprof:
        norm *= norm if prof2 == prof else hmc.profile_norm(cosmo, a, prof2)

    # 2-halo term.
    pk2h = 0
    if get_2h:
        i11 = hmc.I_1_1(cosmo, k, a, prof)
        i11 *= i11 if prof2 == prof else hmc.I_1_1(cosmo, k, a, prof2)
        pk2h = pkf(k, a) * i11

    # 1-halo term.
    pk1h = 0
    if get_1h:
        pk1h = hmc.I_0_2(cosmo, k, a, prof, prof2=prof2)
        if suppress_1h:
            # 1-halo large-scale dampening.
            ks = suppress_1h(a)
            pk1h *= (k/ks)**4 / (1+(k/ks)**4)

    # 1h/2h transition smoothing.
    if smooth_transition:
        alpha = smooth_transition(a)
        pk_out = (pk1h**alpha + pk2h**alpha)**(1/alpha) * norm
    else:
        pk_out = (pk1h + pk2h) * norm

    # Return power spectrum array.
    if not pk2d_out:
        return pk_out.squeeze()[()] if squeeze else pk_out

    # Interpolate in log.
    if use_log:
        if (pk_out <= 0).any():
            warnings.warn("Power spectrum is non-positive. "
                          "Will not interpolate in log.", CCLWarning)
            use_log = False
        else:
            np.log(pk_out, out=pk_out)

    return Pk2D(a_arr=a, lk_arr=np.log(k),
                pk_arr=pk_out, is_logp=use_log,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik)


halomod_power_spectrum = partial(_pk_hm, pk2d_out=False)
halomod_power_spectrum.__doc__ = _pk_hm.__doc__

halomod_Pk2D = partial(_pk_hm, pk2d_out=True)
halomod_Pk2D.__doc__ = _pk_hm.__doc__
