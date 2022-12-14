from .pspec import (PowerSpectrum, TransferFunctions,
                    MatterPowerSpectra, BaryonPowerSpectra)
from .pk2d import DefaultPowerSpectrum, Pk2D, parse_pk2d
from .interpolate import Interpolator2D
from .integrate import IntegratorFunction
from .errors import CCLWarning
from .pyutils import get_broadcastable
from .parameters import accuracy_params as acc
from .parameters import spline_params as sparams

import numpy as np
from scipy.special import jv
import warnings
import functools


__all__ = ("get_linear_power", "get_nonlin_power", "linear_power",
           "dlnPk_dlnk", "nonlin_power",
           "sigmaR", "sigmaV", "sigma8", "sigma2B", "kNL", "r2m", "m2r",
           "sigmaM", "dlnsigM_dlogM")


def compute_linear_power(cosmo):
    """Compute the linear power spectrum."""
    if cosmo.has_linear_power:
        return

    cosmo.compute_growth()

    TRF, trf = TransferFunctions, cosmo.config.transfer_function
    MPS, mps = MatterPowerSpectra, cosmo.config.matter_power_spectrum
    model = PowerSpectrum.from_model(trf.value)(cosmo)
    extras = cosmo.get_extra_parameters(trf.value)

    # Compute the CAMB non-linear power spectrum if needed,
    # to avoid repeating the code in `compute_nonlin_power`.
    # Because CAMB power spectra come in pairs with pkl always computed,
    # we set the non-linear power spectrum first, but keep the linear via a
    # status variable to use it later if the transfer function is also CAMB.
    pkl = None
    if MPS(mps) == MPS.CAMB:
        extras_camb = cosmo.get_extra_parameters(mps.value).copy()
        extras_camb["nonlin"] = True
        camb_model = PowerSpectrum.from_model(mps.value)
        pkl, pknl = camb_model(cosmo).get_power_spectrum(**extras_camb)
        cosmo._pknl[DefaultPowerSpectrum] = pknl

    is_computed = pkl is not None and TRF(trf) == TRF.CAMB
    pk = pkl if is_computed else model.get_power_spectrum(**extras)

    pk = cosmo.rescale_power_spectrum(pk,
                                      rescale_s8=model.rescale_s8,
                                      rescale_mg=model.rescale_mg)

    cosmo._pkl[DefaultPowerSpectrum] = pk


def compute_nonlin_power(cosmo):
    """Compute the non-linear power spectrum."""
    if cosmo.has_nonlin_power:
        return

    MPS, mps = MatterPowerSpectra, cosmo.config.matter_power_spectrum
    BPS, bps = BaryonPowerSpectra, cosmo.config.baryons_power_spectrum

    is_MG = (cosmo["mu_0"], cosmo["sigma_0"]) != (0, 0)
    if is_MG and MPS(mps) != MPS.LINEAR:
        warnings.warn("For mu-Sigma modified gravity cosmologies, only the "
                      "linear power spectrum can be consistently used "
                      "as the non-linear power spectrum.", CCLWarning)

    if MPS(mps) != MPS.EMU:
        cosmo.compute_linear_power()

    if MPS(mps) == MPS.CAMB:
        return  # already computed

    cosmo.compute_distances()

    if MPS(mps) == MPS.HALOFIT:
        pkl = cosmo.get_linear_power()
        pk = pkl.apply_model(cosmo, "halofit")
    elif MPS(mps) in [MPS.HMCODE, MPS.EMU]:
        pk = Pk2D.from_model(cosmo, model=mps.value)
    elif MPS(mps) == MPS.LINEAR:
        pk = cosmo.get_linear_power().copy()

    # Baryon correction.
    if BPS(bps) == BPS.BCM:
        pk = pk.apply_model(cosmo, "bcm")

    cosmo._pknl[DefaultPowerSpectrum] = pk


def get_linear_power(cosmo, name=DefaultPowerSpectrum):
    """Get the :class:`~pyccl.pk2d.Pk2D` object associated with
    the linear power spectrum ``name``.

    Arguments
    ---------
    name : str
        Name of the power spectrum to return.
        The default is ``'delta_matter:delta_matter'``.

    Returns
    -------
    pk_linear : :class:`~pyccl.pk2d.Pk2D`
        The power spectrum object.
    """
    return cosmo._pkl[name]


def get_nonlin_power(cosmo, name=DefaultPowerSpectrum):
    """Get the :class:`~pyccl.pk2d.Pk2D` object associated with
    the non-linear power spectrum ``name``.

    Arguments
    ---------
    name : str
        Name of the power spectrum to return.
        The default is ``'delta_matter:delta_matter'``.

    Returns
    -------
    pk_nonlin : :class:`~pyccl.pk2d.Pk2D`
        The power spectrum object.
    """
    return cosmo._pknl[name]


def linear_power(cosmo, k, a, p_of_k_a=DefaultPowerSpectrum, *,
                 derivative=False, squeeze=True):
    """The linear power spectrum (:math:`\\mathrm{Mpc}^3`).

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    k : float or array_like
        Wavenumber (:math:`\\mathrm{Mpc}^{-1}`).
    a : float or array_like
        Scale factor(s), normalized to 1 today.
    p_of_k_a : str
        Which power spectrum to compute (which should be stored in ``cosmo``).
        Defaults to the linear matter power spectrum.
    derivative : bool
        If ``False``, evaluate the power spectrum. If ``True``, evaluate
        the logarithmic derivative of the power spectrum,
        :math:`\\frac{\\mathrm{d} \\log P(k)}{\\mathrm{d} \\log k}`.
    squeeze : bool
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.
    """
    cosmo.compute_linear_power()
    return cosmo.get_linear_power(name=p_of_k_a)(k, a, derivative=derivative,
                                                 squeeze=squeeze)


def dlnPk_dlnk(cosmo, k, a, p_of_k_a=DefaultPowerSpectrum, *, squeeze=True):
    """Helper for the logarithmic derivative of the linear power spectrum.
    See :func:`linear_power` for details.
    """
    return linear_power(cosmo, k, a, p_of_k_a, derivative=True,
                        squeeze=squeeze)


def nonlin_power(cosmo, k, a, p_of_k_a=DefaultPowerSpectrum, *, squeeze=True):
    """The non-linear power spectrum (:math:`\\mathrm{Mpc}^3`).

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    k : float or array_like
        Wavenumber (:math:`\\mathrm{Mpc}^{-1}`).
    a : float or array_like
        Scale factor(s), normalized to 1 today.
    p_of_k_a : str
        Which power spectrum to compute (which should be stored in ``cosmo``).
        Defaults to the non-linear matter power spectrum.
    squeeze : bool
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.
    """
    cosmo.compute_nonlin_power()
    return cosmo.get_nonlin_power(name=p_of_k_a)(k, a, squeeze=squeeze)


def linear_matter_power(cosmo, k, a, *, squeeze=True):
    """The linear matter power spectrum (:math:`\\mathrm{Mpc}^3`).

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    k : float or array_like
        Wavenumber (:math:`\\mathrm{Mpc}^{-1}`).
    a : float or array_like
        Scale factor(s), normalized to 1 today.
    squeeze : bool
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.
    """
    return linear_power(cosmo, k, a,
                        p_of_k_a=DefaultPowerSpectrum,
                        squeeze=squeeze)


def nonlin_matter_power(cosmo, k, a, *, squeeze=True):
    """The nonlinear matter power spectrum (:math:`\\mathrm{Mpc}^3`).

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    k : float or array_like
        Wavenumber (:math:`\\mathrm{Mpc}^{-1}`).
    a : float or array_like
        Scale factor(s), normalized to 1 today.
    squeeze : bool
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.
    """
    return nonlin_power(cosmo, k, a,
                        p_of_k_a=DefaultPowerSpectrum,
                        squeeze=squeeze)


def _w_tophat(kR):
    """Smooth 1-D top-hat window function.

    .. math::

        W(x) = \\frac{3 \\, (\\sin(x) - x\\cos(x))}{x^3}

    Arguments
    ---------
    kR : float or array_like
        Wavenumber multiplied by a smoothing radius.
    """
    kR = np.asarray(kR)
    out = np.empty_like(kR)

    kR2 = kR*kR

    idx = kR < 0.1
    # Maclaurin expansion of this function around kR (up to 10th order).
    # Implemented because we rely on the cancellation of two terms at low kR.
    out[idx] = (1 + kR2[idx]*(-1/10 + kR2[idx]*(1/280 + kR2[idx]*(
        -1/15120 + kR2[idx]*(1/1330560 + kR2[idx]*(-1/172972800))))))

    idx = np.logical_not(idx)
    out[idx] = 3. * (np.sin(kR[idx]) - kR[idx]*np.cos(kR[idx]))/kR[idx]**3

    return out


def _w_tophat_2d(kR):
    """Smooth 2-D top-hat window function.

    .. math::

        W(x) = \\frac{2 J_1(x)}{x}

    Arguments
    ---------
    kR : float or array_like
        Wavenumber multiplied by a smoothing radius.
    """
    kR = np.asarray(kR)
    out = np.empty_like(kR)

    kR2 = kR*kR

    idx = kR < 0.1
    # Maclaurin expansion of this function around kR (up to 10th order).
    # Implemented because we rely on the cancellation of two terms at low kR.
    out[idx] = (1 + kR2[idx]*(-1/8 + kR2[idx]*(1/192 + kR2[idx]*(
        -1/9216 + kR2[idx]*(1/737280 + kR2[idx]*(-1/88473600))))))

    idx = np.logical_not(idx)
    out[idx] = 2 * jv(1, kR) / kR

    return out


def _integrate_sigma(cosmo, integrand):
    """Used internally to integrate σ(k, R)."""
    # Adaptive quadrature (`scipy.integrate.quad_vec`) is O(100) slower.
    # We trade off by using fixed quadrature at the sub-division limit
    # (which runs faster than adaptive quadrature in C).
    integrator = IntegratorFunction("fixed_quadrature", acc.EPSREL_SIGMAR)
    σ = integrator(integrand, np.log10(sparams.K_MIN), np.log10(sparams.K_MAX))
    return np.sqrt(σ * np.log(10) / (2 * np.pi**2))


def _sigmaR(cosmo, R, a=1, p_of_k_a=DefaultPowerSpectrum):
    r"""RMS variance in a top-hat sphere of radius ``R`` :math:\rm Mpc.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    R : float or (nR,) array_like
        Radius of the top-hat sphere.
    a : float or (na,) array_like, optional
        Scale factor, normalized to 1 today. The default is 1.
    p_of_k_a : :class:`~pyccl.pk2d.Pk2D` or str
        3-D power spectrum to integrate.
        If a string is passed, a non-linear power spectrum under that name
        must be stored in ``cosmo`` (e.g.`'delta_matter:delta_matter'`).

    Returns
    -------
    σ_R : float or (na, nR,) ndarray
        RMS variance in a top-hat sphere of radius ``R``.
    """
    a, R = map(np.atleast_1d, [a, R])
    a, R = get_broadcastable(a, R)

    # Integrand axes are [a, R, k]. Integrate over final axis (k).
    shp = np.broadcast_shapes(a.shape, R.shape)
    k_shape = (1,) * len(shp) + (-1,)  # add a trailing axis
    a, R = a[..., None], R[..., None]  # create a trailing integration axis

    pk = parse_pk2d(cosmo, p_of_k_a, linear=True)
    def sigmaR_integrand(lk):
        lk = np.atleast_1d(lk).reshape(k_shape)
        k = 10**lk
        pka = pk(k, a, squeeze=False)
        return pka * k**3 * _w_tophat(k*R)**2

    return _integrate_sigma(cosmo, sigmaR_integrand)


@functools.wraps(_sigmaR)
def sigmaR(cosmo, R, a=1, p_of_k_a=DefaultPowerSpectrum, *, squeeze=True):
    if p_of_k_a == DefaultPowerSpectrum:
        return sigmaM(cosmo, r2m(cosmo, R), a, squeeze=squeeze)
    out = _sigmaR(cosmo, R, a, p_of_k_a=p_of_k_a)
    return out.squeeze()[()] if squeeze else out


def sigmaV(cosmo, R, a=1, p_of_k_a=DefaultPowerSpectrum, *, squeeze=True):
    """RMS variance in the displacement field in a top-hat sphere of radius R.
    The linear displacement field is the gradient of the linear density field.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    R : float or (nR,) array_like
        Radius of the top-hat sphere.
    a : float or (na,), optional
        Scale factor, normalized to 1 today. The default is 1.
    p_of_k_a : :class:`~pyccl.pk2d.Pk2D` or str
        3-D power spectrum to integrate.
        If a string is passed, a non-linear power spectrum under that name
        must be stored in ``cosmo`` (e.g.`'delta_matter:delta_matter'`).
    squeeze : bool, optional
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.

    Returns
    -------
    σ_V : float or (na, nR,) ndarray
        RMS variance of the displacement field in a top-hat sphere of radius R.
    """
    a, R = map(np.atleast_1d, [a, R])
    a, R = get_broadcastable(a, R)

    # Integrand axes are [a, R, k]. Integrate over final axis (k).
    shp = np.broadcast_shapes(a.shape, R.shape)
    k_shape = (1,) * len(shp) + (-1,)  # add a trailing axis
    a, R = a[..., None], R[..., None]  # create a trailing integration axis

    pk = parse_pk2d(cosmo, p_of_k_a, linear=True)
    def sigmaV_integrand(lk):
        lk = np.atleast_1d(lk).reshape(k_shape)
        k = 10**lk
        pka = pk(k, a, squeeze=False)
        return pka * k * _w_tophat(k*R)**2 / 3.

    out = _integrate_sigma(cosmo, sigmaV_integrand)
    return out.squeeze()[()] if squeeze else out


def sigma8(cosmo, p_of_k_a=DefaultPowerSpectrum):
    """RMS variance in a top-hat sphere of radius 8 Mpc/h.

    .. note::

        8 Mpc/h is rescaled based on the chosen value of the Hubble
        constant within `cosmo`.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    p_of_k_a : :class:`~pyccl.pk2d.Pk2D` or str
        3-D power spectrum to integrate.
        If a string is passed, a non-linear power spectrum under that name
        must be stored in ``cosmo`` (e.g.`'delta_matter:delta_matter'`).
    """
    if p_of_k_a == DefaultPowerSpectrum and cosmo["sigma8"] is not None:
        # Return the value stored in cosmo.
        return cosmo["sigma8"]
    out = sigmaR(cosmo, 8/cosmo["h"], 1., p_of_k_a=p_of_k_a)
    if p_of_k_a == DefaultPowerSpectrum:
        # Populate sigma8 in cosmo.
        cosmo.params.sigma8 = out
    return out


def sigma2B(cosmo, R, a, p_of_k_a=DefaultPowerSpectrum, *, squeeze=True):
    """
    # TODO
    """
    a, R = map(np.atleast_1d, [a, R])
    a, R = get_broadcastable(a, R)

    # Integrand axes are [a, R, k]. Integrate over final axis (k).
    shp = np.broadcast_shapes(a.shape, R.shape)
    k_shape = (1,) * len(shp) + (-1,)  # add a trailing axis
    a, R = a[..., None], R[..., None]  # create a trailing integration axis

    pk = parse_pk2d(cosmo, p_of_k_a, linear=True)
    def sigma2B_integrand(lk):
        lk = np.atleast_1d(lk).reshape(k_shape)
        k = 10**lk
        pka = pk(k, a, squeeze=False)
        return pka * k**2 * _w_tophat_2d(k*R)**2

    out = _integrate_sigma(cosmo, sigma2B_integrand)
    return out.squeeze()[()] if squeeze else out


def kNL(cosmo, a, p_of_k_a=DefaultPowerSpectrum):
    r"""Scale for the non-linear cut.

    `k_{\rm NL}` is calculated based on Lagrangian perturbation
    theory, as the inverse of the variance of the displacement field,

    .. math::

        k_{\rm NL} = \frac{1}{\sigma_\eta}
        = \left( \frac{1}{6\pi^2} \int P_{\rm L}(k) \mathrm{d}k \right)^{-1/2}.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
        Scale factor, normalized to 1 today.
    p_of_k_a : :class:`~pyccl.pk2d.Pk2D` or str, optional
        3-D power spectrum to integrate.
        If a string is passed, a non-linear power spectrum under that name
        must be stored in ``cosmo`` (e.g.`'delta_matter:delta_matter'`).

    Returns
    -------
    kNL : float or (na,) ndarray
        Scale of non-linear cut-off (:math:`\rm Mpc^{-1}`).
    """
    pk = parse_pk2d(cosmo, p_of_k_a, linear=True)
    integrator = IntegratorFunction("fixed_quadrature", acc.EPSREL_KNL)
    PL = integrator(pk, sparams.K_MIN, sparams.K_MAX, args=(a,))
    return 1. / np.sqrt(PL / (6*np.pi**2))


def r2m(cosmo, R, a=1, Delta=1, species="matter", comoving=True, *,
        Delta_vectorized=True, squeeze=True):
    r"""Generic function to convert radius to mass via density.

    .. math::

        M = \frac{4\pi}{3} \, \Delta \, \rho_{\rm x}(a) \, R^3

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    R : float or (nR) array_like
        Radius in :math:`\rm Mpc`.
    a : float or (na) array_like
        Scale factor(s). The default is 1.
    Delta : float or (nDelta,) array_like, optional
        Overdensity parameter :math:`\Delta`.
        The default is 1.
    species : str, optional
        Species of which the density is calculated.
        :class:`~pyccl.Species` lists all available species.
        The default is 'matter'.
    comoving : bool, optional
        Comoving or physical coordinates. The default is comoving.
    Delta_vectorized : bool, optional
        Whether to treat ``Delta`` as an extra, vectorized dimension.
        Otherwise, ``Delta`` lies parallel to the dimension of ``a``,
        and the return shape is (na, nR).
        The default is True.
    squeeze : bool, optional
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.

    Returns
    -------
    mass : float or (na, nR, nDelta) ndarray
        Halo mass in :math:`\rm M_\odot`.
    """
    # Axes are [a, R, Delta].
    a, R, Delta = map(np.atleast_1d, [a, R, Delta])
    if Delta_vectorized:
        a, R, Delta = get_broadcastable(a, R, Delta)
    else:
        a, R = get_broadcastable(a, R)
        Delta = np.broadcast_to(Delta, a.shape)
    ρ = cosmo.rho_x(a, species=species, is_comoving=comoving, squeeze=False)
    M = (4/3) * np.pi * Delta * R**3 * ρ
    return M.squeeze()[()] if squeeze else M


def m2r(cosmo, M, a=1, Delta=1, species="matter", comoving=True, *,
        Delta_vectorized=True, squeeze=True):
    r"""Generic function to convert mass to radius via density.

    .. math::

        R = \left(
            \frac{3 M}{4\pi \, \Delta \, \rho_{\rm x}(a)}
            \right)
            ^{\frac{1}{3}}

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    M : float or (nM) array_like
        Halo mass in :math:`\rm M_\odot`.
    a : float or (na,) array_like
        Scale factor(s). The default is 1.
    Delta : float or (nDelta,) array_like, optional
        Overdensity parameter :math:`\Delta`.
        The default is 1.
    species : str, optional
        Species of which the density is calculated.
        :class:`~pyccl.Species` lists all available species.
        The default is 'matter'.
    comoving : bool, optional
        Comoving or physical coordinates. The default is comoving.
    Delta_vectorized : bool, optional
        Whether to treat ``Delta`` as an extra, vectorized dimension.
        Otherwise, ``Delta`` lies parallel to the dimension of ``a``,
        and the return shape is (na, nM).
        The default is True.
    squeeze : bool, optional
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.

    Returns
    -------
    radius : float or (na, nM, nDelta) ndarray
        Radius in :math:`\rm Mpc`.
    """
    # Axes are [a, M, Delta].
    a, M, Delta = map(np.asarray, [a, M, Delta])
    if Delta_vectorized:
        a, M, Delta = get_broadcastable(a, M, Delta)
    else:
        a, M = get_broadcastable(a, M)
        Delta.shape = a.shape
    ρ = cosmo.rho_x(a, species=species, is_comoving=comoving, squeeze=False)
    R = (3 * M / (4 * np.pi * Delta * ρ))**(1/3)
    return R.squeeze()[()] if squeeze else R


def compute_sigma(cosmo):
    """Compute the σ(M) spline."""
    if cosmo.has_sigma:
        return

    cosmo.compute_growth()
    cosmo.compute_linear_power()
    pk = cosmo.get_linear_power()

    a = sparams.get_sm_spline_a()
    m = sparams.get_sm_spline_lm()
    y = np.log(_sigmaR(cosmo, m2r(cosmo, 10**m), a, pk))
    cosmo.data.logsigma = Interpolator2D(a, m, y, extrap_orders=[1, 1, 1, 1])


def sigmaM(cosmo, M, a, *, squeeze=True, grid=True):
    """Root mean squared variance for the given halo mass
    of the linear power spectrum (:math:`\\mathrm{M_{\\odot}}`).

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    M : float or (nM,) array_like
        Halo masses (in units of M_sun).
    a : float or (na,) array_like
        Scale factor, normalized to 1 today.
    squeeze : bool
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.
    grid : bool
        Evaluate on a grid spanned by the input arrays (True),
        or at the points specified by the input arrays (False).
        The default is True.

    Returns
    -------
    σ_Μ : float or (na, nM) ndarray
        RMS variance of the halo masses ``M`` at scale factor ``a``.
    """
    cosmo.compute_sigma()
    out = np.exp(cosmo.data.logsigma(a, np.log10(M), grid=grid))
    return out.squeeze()[()] if squeeze else out


def dlnsigM_dlogM(cosmo, M, a, *, squeeze=True):
    r"""Logarithmic derivative of the RMS variance for the given halo mass
    of the linear power spectrum (:math:`\rm M_\odot`).

    .. math::

        \frac{\mathrm{d} \ln (1 / \sigma(M))}{\mathrm{d} \log_{10} M}

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    M : float or (nM,) array_like
        Halo masses (in units of M_sun).
    a : float or (na,) array_like
        Scale factor, normalized to 1 today.
    squeeze : bool
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.

    Returns
    -------
    dlnσM_dlogM : float or (na, nM,) ndarray
        Logarithmic RMS variance of the halo masses ``M`` at ``a``.
    """
    cosmo.compute_sigma()
    out = -cosmo.data.logsigma.f(a, np.log10(M), dy=1)
    return out.squeeze()[()] if squeeze else out
