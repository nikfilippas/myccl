from .profiles import HaloProfileNFW
from ..pk2d import Pk2D
from ..tk3d import Tk3D
from ..parameters import spline_params as sparams
from ..errors import CCLWarning
import warnings

import numpy as np
from functools import partial


__all__ = ("halomod_trispectrum_1h", "halomod_Tk3D_1h",
           "halomod_Tk3D_SSC_linear_bias", "halomod_Tk3D_SSC")


def _tkk_hm(cosmo, hmc, *, k=None, a=None,
            prof, prof2=None, prof3=None, prof4=None,
            extrap_order_lok=1, extrap_order_hik=1,
            tk3d_out=False, use_log=True, squeeze=True):
    """Halo model 1-halo trispectrum for four different quantities defined
    by their respective halo profiles. The 1-halo trispectrum for four profiles
    :math:`u_{1,2}`, :math:`v_{1,2}` is calculated as:

    .. math::

        T_{u_1, u_2; v_1, v_2}(k_u, k_v, a) =
        I^0_{2,2}(k_u, k_v, a|u_{1,2},v_{1,2})

    where :math:`I^0_{2,2}` is defined in :meth:`~HMCalculator.I_0_22`.

    .. note::

        This approximation assumes that the 4-point profile cumulant is equals
        the product of two 2-point cumulants.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    hmc : :class:`HMCalculator`
        Halo model calculator associated with the profile calculations.
    k : float or (nk,) array_like, optional
        Comoving wavenumber in :math:`\\mathrm{Mpc}^{-1}`.
        If ``tk3d_out is True``, this should be :math:`log(k)`.
        The default is defined via the spline parameters of ``cosmo``.
    a : float or (na,) array_like, optional
        Scale factor.
        The default is defined via the spline parameters of ``cosmo``.
    prof, prof2, prof3, prof4 : :class:`~pyccl.halos.profiles.HaloProfile`
        Halo profiles.
        If ``prof2 is None``, ``prof`` will be used.
        If ``prof3 is None``, ``prof`` will be used.
        If ``prof4 is None``, ``prof2`` will be used.
    extrap_order_lok, extrap_order_hik : {0, 1, 2}
        Extrapolation order for low and high ``k``, respectively.
        Provided when ``pk2d_out is True``. The defaults are 1 and 1.
    pk2d_out : bool
        Return the halo model power spectrum array, or a ``Pk2D`` object.
        The default is False.
    use_log : bool
        When ``tk3d_out is True``, interpolate the trispectrum in log.
        The default is True.
    squeeze : bool, optional
        Squeeze extra dimensions of size (1,) in the output.
        Provided when ``tk3d_out is False``. The default is True.

    Returns
    -------
    tkk_hm : float, (na, nk, nk) ndarray or  :class:`~pyccl.tk3d.Tk3D`
        Imlementation of the halo model trispectrum.
        If ``tk3d_out is True``, a ``Tk3D`` object is returned.
    """
    if k is None and not tk3d_out:
        raise ValueError("Provide k and a for the trispectrum.")

    # Sampling rate.
    if k is None:
        k = np.exp(sparams.get_pk_spline_lk())
    if a is None:
        a = sparams.get_pk_spline_a()

    # Profiles.
    if prof2 is None:
        prof2 = prof
    if prof3 is None:
        prof3 = prof
    if prof4 is None:
        prof4 = prof2

    # Normalization.
    norm2 = norm3 = norm4 = 1
    norm1 = hmc.profile_norm(cosmo, a, prof) if prof.normprof else 1
    if prof2.normprof:
        norm2 = norm1 if prof2 == prof else hmc.profile_norm(cosmo, a, prof2)
    if prof3.normprof:
        norm3 = norm1 if prof3 == prof else hmc.profile_norm(cosmo, a, prof3)
    if prof4.normprof:
        norm4 = norm1 if prof4 == prof2 else hmc.profile_norm(cosmo, a, prof4)
    norm = norm1 * norm2 * norm3 * norm4

    # I_0_22
    tk1h = hmc.I_0_22(cosmo, k, a, prof, prof2, prof3, prof4)
    tk1h *= norm

    # Return trispectrum array.
    if not tk3d_out:
        return tk1h.squeeze()[()] if squeeze else tk1h

    # Interpolate in log.
    if use_log:
        if (tk1h <= 0).any():
            warnings.warn("Trispectrum is non-positive. "
                          "Will not interpolate in log.", CCLWarning)
            use_log = False
        else:
            np.log(tk1h, out=tk1h)

    return Tk3D(a_arr=a, lk_arr=np.log(k),
                tkk_arr=tk1h, is_logt=use_log,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik)


halomod_trispectrum_1h = partial(_tkk_hm, tk3d_out=True)
halomod_trispectrum_1h.__doc__ = _tkk_hm.__doc__

halomod_Tk3D_1h = partial(_tkk_hm, tk3d_out=True)
halomod_Tk3D_1h.__doc__ = _tkk_hm.__doc__


def halomod_Tk3D_SSC_linear_bias(cosmo, hmc, prof,
                                 bias1=1, bias2=1, bias3=1, bias4=1,
                                 p_of_k_a=None, lk_arr=None,
                                 a_arr=None, extrap_order_lok=1,
                                 extrap_order_hik=1, use_log=False):
    r"""Super-sample covariance trispectrum.

    Defined as the tensor product of the power spectrum responses
    associated with the two pairs of quantities correlated.
    Each response is calculated as:

    .. math::

        \frac{\partial P_{u,v}(k)}{\partial\delta_L} = b_u b_v \left( \left(
        \frac{68}{21} - \frac{\mathrm{d} \log k^3 P_L(k)}{\mathrm{d} \log k}
        \right) P_L(k) + I^1_2(k|u,v) - (b_{u} + b_{v}) P_{u, v}(k) \right)

    where :math:`I^1_2` is defined in :meth:`~HMCalculator.I_1_2`
    and :math:`b_{u}`, :math:`b_{v}` are the linear halo biases
    for quantities :math:`u` and :math:`v`, respectively (0 if not clustering).

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    hmc : :class:`HMCalculator`
        Halo model calculator associated with the profile calculations.
    bias1, bias2, bias4, bias4 : float or (na,) array_like
        Linear galaxy bias for quantity 1.
    is_number_counts[1, 2, 3, 4] : bool
        Compute the clustering terms for the respective quantity.
        The default is False.
    p_of_k_a : :class:`~pyccl.pk2d.Pk2D`, or {'linear', 'nonlinear'}
        Power spectrum.
        The default is the linear matter power spectrum stored in ``cosmo``.
    a_arr : float or (na,) array_like, optional
        Scale factor.
        The default is defined via the spline parameters of ``cosmo``.
    lk_arr : float or (nk,) array_like, optional
        Log of comoving wavenumber in :math:`\rm Mpc^{-1}`.
        The default is defined via the spline parameters of ``cosmo``.
    extrap_order_lok, extrap_order_hik : {0, 1}
        Extrapolation order for low and high ``k``, respectively.
    use_log : bool
        When ``tk3d_out is True``, interpolate the trispectrum in log.
        The default is True.

    Returns:
    tkk_ssh_lb : :class:`~pyccl.tk3d.Tk3D`
        SSC effective trispectrum.
    """
    if not isinstance(prof, HaloProfileNFW):
        raise TypeError("prof must be of type `HaloProfileNFW`.")

    # Sampling rate.
    if lk_arr is None:
        lk_arr = sparams.get_pk_spline_lk()
    if a_arr is None:
        a_arr = sparams.get_pk_spline_a()

    # Broadcast the biases to arrays.
    ones = np.ones_like(a_arr)
    biases = [bias1, bias2, bias3, bias4]
    bias1, bias2, bias3, bias3 = map(lambda x: ones*x, biases)

    k, a = np.exp(lk_arr), a_arr

    # Power spectrum.
    pkf = p_of_k_a
    if not isinstance(p_of_k_a, Pk2D):
        if pkf == "linear":
            cosmo.compute_linear_power()
            pkf = cosmo.get_linear_power()
        elif pkf == "nonlinear":
            cosmo.compute_nonlin_power()
            pkf = cosmo.get_nonlin_power()

    norm = hmc.profile_norm(cosmo, a, prof)**2
    i12 = hmc.I_1_2(cosmo, k, a, prof, prof)

    pk = pkf(k, a, squeeze=False)
    dpk = pkf(k, a, derivative=True, squeeze=False)
    dpk12 = ((47/21 - dpk/3)) * pk + i12
    dpk34 = dpk12.copy()

    # Counter terms for clustering: -(bA + bB) * PAB.
    if any([locals()[f"is_number_counts{i}"] for i in range(4)]):
        b1 = b2 = b3 = b4 = 0
        i02 = hmc.I_0_2(cosmo, k, a, prof, prof) * norm
        P_12 = P_34 = pk + i02

        if is_number_counts1:
            b1 = bias1
        if is_number_counts2:
            b2 = bias2
        if is_number_counts3:
            b3 = bias3
        if is_number_counts4:
            b4 = bias4

        dpk12 -= (b1 + b2) * P_12
        dpk34 -= (b3 + b4) * P_34

    dpk12 *= bias1 * bias2
    dpk34 *= bias3 * bias4

    if use_log:
        if (dpk12 <= 0).any() or (dpk34 <= 0).any():
            warnings.warn("Power spectrum is non-positive. "
                          "Will not interpolate in log.", CCLWarning)
            use_log = False
        else:
            np.log(dpk12, out=dpk12)
            np.log(dpk34, out=dpk34)

    return Tk3D(a_arr=a_arr, lk_arr=lk_arr,
                pk1_arr=dpk12, pk2_arr=dpk34, is_logt=use_log,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik)


def halomod_Tk3D_SSC(cosmo, hmc, prof, prof2=None, prof3=None, prof4=None,
                     p_of_k_a=None, lk_arr=None, a_arr=None,
                     extrap_order_lok=1, extrap_order_hik=1, use_log=False):
    r"""Super-sample covariance trispectrum

    Defined as the tensor product of the power spectrum responses
    associated with the two pairs of quantities correlated.
    Each response is calculated as:

    .. math::

        \frac{\partial P_{u,v}(k)}{\partial\delta_L} = \left(
        \frac{68}{21} - \frac{\mathrm{d} \log k^3 P_L(k)}{\mathrm{d} \log k}
        \right) P_L(k) I^1_1(k,|u) I^1_1(k,|v) + I^1_2(k|u,v) - (b_{u} + b_{v})
        P_{u,v}(k)

    where :math:`I^1_n` are defined in :meth:`~HMCalculator.I_1_n`
    and :math:`b_{u}`, :math:`b_{v}` are the linear halo biases
    for quantities :math:`u` and :math:`v`, respectively (0 if not clustering).

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    hmc : :class:`HMCalculator`
        Halo model calculator associated with the profile calculations.
    prof, prof2, prof3, prof4 : :class:`~pyccl.halos.profiles.HaloProfile`
        Halo profiles.
        If ``prof2 is None``, ``prof`` will be used.
        If ``prof3 is None``, ``prof`` will be used.
        If ``prof4 is None``, ``prof2`` will be used.
    p_of_k_a : :class:`~pyccl.pk2d.Pk2D`, or {'linear', 'nonlinear'}
        Power spectrum.
    a_arr : float or (na,) array_like, optional
        Scale factor.
        The default is defined via the spline parameters of ``cosmo``.
    lk_arr : float or (nk,) array_like, optional
        Log of comoving wavenumber in :math:`\\mathrm{Mpc}^{-1}`.
        The default is defined via the spline parameters of ``cosmo``.
    extrap_order_lok, extrap_order_hik : {0, 1}
        Extrapolation order for low and high ``k``, respectively.
    use_log : bool
        When ``tk3d_out is True``, interpolate the trispectrum in log.
        The default is True.

    Returns:
    tk_ssc : :class:`~pyccl.tk3d.Tk3D`
        SSC effective trispectrum.
    """
    if lk_arr is None:
        lk_arr = sparams.get_pk_spline_lk()
    if a_arr is None:
        a_arr = sparams.get_pk_spline_a()

    k_use = np.exp(lk_arr)

    if prof3 is None:
        prof3_bak = prof
    else:
        prof3_bak = prof3

    # Power spectrum
    if isinstance(p_of_k_a, Pk2D):
        pk2d = p_of_k_a
    elif (p_of_k_a is None) or (str(p_of_k_a) == 'linear'):
        pk2d = cosmo.get_linear_power('delta_matter:delta_matter')
    elif str(p_of_k_a) == 'nonlinear':
        pk2d = cosmo.get_nonlin_power('delta_matter:delta_matter')
    else:
        raise TypeError("p_of_k_a must be `None`, \'linear\', "
                        "\'nonlinear\' or a `Pk2D` object")

    def get_norm(prof, sf):
        return hmc.profile_norm(cosmo, sf, prof) if prof.normprof else 1.

    na = len(a_arr)
    nk = len(k_use)
    dpk12 = np.zeros([na, nk])
    dpk34 = np.zeros([na, nk])
    for ia, aa in enumerate(a_arr):
        # Compute profile normalizations
        norm1 = get_norm(prof, aa)
        i11_1 = hmc.I_1_1(cosmo, k_use, aa, prof)
        # Compute second profile normalization
        if prof2 is None:
            norm2 = norm1
            i11_2 = i11_1
        else:
            norm2 = get_norm(prof2, aa)
            i11_2 = hmc.I_1_1(cosmo, k_use, aa, prof2)
        if prof3 is None:
            norm3 = norm1
            i11_3 = i11_1
        else:
            norm3 = get_norm(prof3, aa)
            i11_3 = hmc.I_1_1(cosmo, k_use, aa, prof3)
        if prof4 is None:
            norm4 = norm3
            i11_4 = i11_3
        else:
            norm4 = get_norm(prof4, aa)
            i11_4 = hmc.I_1_1(cosmo, k_use, aa, prof4)

        i12_12 = hmc.I_1_2(cosmo, k_use, aa, prof, prof2)
        if (prof3 is None) and (prof4 is None) and (prof34_2pt is None):
            i12_34 = i12_12
        else:
            i12_34 = hmc.I_1_2(cosmo, k_use, aa, prof3_bak, prof4,
                               prof34_2pt_bak)
        norm12 = norm1 * norm2
        norm34 = norm3 * norm4

        pk = pk2d.eval(k_use, aa, cosmo)
        dpk = pk2d(k_use, aa, derivative=True)
        dpk12[ia, :] = norm12*((47/21 - dpk/3)*i11_1*i11_2*pk+i12_12)
        dpk34[ia, :] = norm34*((47/21 - dpk/3)*i11_3*i11_4*pk+i12_34)

        # Counter terms for clustering (i.e. - (bA + bB) * PAB
        if prof.is_number_counts or (prof2 is None or prof2.is_number_counts):
            b1 = b2 = np.zeros_like(k_use)
            i02_12 = hmc.I_0_2(cosmo, k_use, aa, prof, prof2)
            P_12 = norm12 * (pk * i11_1 * i11_2 + i02_12)

            if prof.is_number_counts:
                b1 = i11_1 * norm1

            if prof2 is None:
                b2 = b1
            elif prof2.is_number_counts:
                b2 = i11_2 * norm2

            dpk12[ia, :] -= (b1 + b2) * P_12

        if prof3_bak.is_number_counts or \
                ((prof3_bak.is_number_counts and prof4 is None) or
                 (prof4 is not None) and prof4.is_number_counts):
            b3 = b4 = np.zeros_like(k_use)
            if (prof3 is None) and (prof4 is None) and (prof34_2pt is None):
                i02_34 = i02_12
            else:
                i02_34 = hmc.I_0_2(cosmo, k_use, aa, prof3_bak, prof4)
            P_34 = norm34 * (pk * i11_3 * i11_4 + i02_34)

            if prof3 is None:
                b3 = b1
            elif prof3.is_number_counts:
                b3 = i11_3 * norm3

            if prof4 is None:
                b4 = b3
            elif prof4.is_number_counts:
                b4 = i11_4 * norm4

            dpk34[ia, :] -= (b3 + b4) * P_34

    if use_log:
        if np.any(dpk12 <= 0) or np.any(dpk34 <= 0):
            warnings.warn(
                "Some values were not positive. "
                "Will not interpolate in log-space.",
                category=CCLWarning)
            use_log = False
        else:
            dpk12 = np.log(dpk12)
            dpk34 = np.log(dpk34)

    return Tk3D(a_arr=a_arr, lk_arr=lk_arr,
                pk1_arr=dpk12, pk2_arr=dpk34, is_logt=use_log,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik)
