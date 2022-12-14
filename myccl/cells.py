from .errors import CCLWarning
from .pk2d import DefaultPowerSpectrum, parse_pk2d
from .interpolate import linlog_spacing
from .integrate import IntegratorFunction
from .parameters import accuracy_params
from .parameters import spline_params as sparams
from .pyutils import get_broadcastable

import numpy as np
import warnings


_ELL_DEFAULT = linlog_spacing(
    sparams.ELL_CLS_MIN, sparams.ELL_CLS_MINLOG, sparams.ELL_CLS_MAX,
    sparams.ELL_CLS_NLIN, sparams.ELL_CLS_NLOG)


def _get_k_interval(cosmo, tracer1, tracer2, ell):
    """Loop through all tracers and find distance bounds."""
    # Maximum of minima and minimum of maxima.
    # (i.e. edges where the produce of all kernels will have support).
    chi_min = max([tracer1.chi_min, tracer2.chi_min])
    chi_max = min([tracer1.chi_max, tracer2.chi_max])

    clip = [sparams.K_MIN, None, sparams.K_MAX]
    kmin = np.clip((ell + 0.5) / chi_max, *clip[:-1])
    if chi_min == 0:  # CCL default
        kmax = np.full_like(ell, sparams.K_MAX)
    else:
        kmax = np.clip(2 * (ell + 0.5) / chi_min, *clip[1:])

    return kmin, kmax


def transfer_limber(cosmo, tr, ell, k, chi, a, pk, ignore_jbes_deriv):
    dd = 0

    # Kernel and transfer evaluated at chi(ell).
    w = tr.get_kernel(chi)
    t = tr.get_transfer(k, a)
    fl = tr.get_f_ell(ell)

    if tr.der_bessel < 1:
        # we don't need (ell + 1)
        dd = w * t
        if tr.der_bessel == -1:
            # if we divide by (chi*k)^2
            dd /= (ell + 0.5)**2
    else:
        # we need (ell + 1)
        if ignore_jbes_deriv:
            dd = 0
        if not ignore_jbes_deriv:
            # Compute chi_{ell+1} and a_{ell+1}.
            lp1h = ell + 0.5
            lp3h = ell + 1.5
            chi_lp = lp3h / k
            a_lp = cosmo.scale_factor_of_chi(chi_lp)

            # Compute power spectrum ratio there.
            pk_ratio = np.abs(pk(k, a_lp) / pk(k, a))

            # Compute kernel and transfer at chi_{ell+1}.
            w_p = tr.get_kernel(chi_lp)
            t_p = tr.get_transfer(k, a_lp)

            sqell = np.sqrt(lp1h/lp3h * pk_ratio)
            if tr.der_bessel == 1:
                dd = ell*w*t / lp1h - sqell*w_p*t_p
            elif tr.der_bessel == 2:
                dd = 2*sqell*w_p*t_p / lp3h - (0.25 + 2*ell)*w*t / (lp1h**2)

    return dd*fl


def transfer_limber_wrap(cosmo, ell, k, chi, a, tracer, pk, ignore_jbes_deriv=False):
    out = np.zeros_like(ell)
    for tr in tracer:
        # FIXME: Check if this is + or x.
        out += transfer_limber(cosmo, tr=tr, ell=ell, k=k, chi=chi,
                               a=a, pk=pk, ignore_jbes_deriv=ignore_jbes_deriv)
    return out


def _angular_cl_limber(cosmo, tracer1, tracer2, ell, pk, integration_method):
    """Compute the angular power spectrum of two tracers
    using the Limber approximation.
    """

    def integrand(l, k):
        chi = (l + 0.5) / k
        a = cosmo.scale_factor_of_chi(chi)
        d = transfer_limber_wrap(cosmo, ell=l, k=k, chi=chi, a=a,
                                 tracer=tracer1, pk=pk)
        if tracer1 == tracer2:
            d **= 2
        else:
            idx = d != 0
            d[idx] *= transfer_limber_wrap(cosmo, ell=l[idx], k=k[idx],
                                           chi=chi[idx], a=a, tracer=tracer2,
                                           pk=pk)
        return k * pk(k, a) * d

    kmin, kmax = _get_k_interval(cosmo, tracer1, tracer2, ell)
    nk = (np.ceil((kmax - kmin) / sparams.DLOGK + 0.5) + 1).astype(int)

    # FIXME: This can become more efficient if we predefine the sampling array.
    # k = np.asarray([np.geomspace(*vals) for vals in zip(kmin, kmax, nk)], dtype=object)
    for l, vals in zip(ell, zip(kmin, kmax, nk)):
        k = np.geomspace(*vals)





    # integrator = IntegratorFunction(integration_method, acc)
    return integrator(integrand, a=lkmin, b=lkmax) / (ell + 0.5)


def angular_cl(cosmo, tracer1, tracer2=None, ell=None,
               p_of_k_a=DefaultPowerSpectrum,
               l_limber=-1., integration_method='spline'):
    r"""Angular (cross-)power spectrum for a pair of tracers:

    .. math::

        C_\ell^{uv} = \int \mathrm{d}\chi \frac{W_u(\chi) W_v(\chi)}{\chi^2}
        P_{UV} \left( k=\frac{\ell+1/2}{\chi}, z(\chi) \right),

    where the 3-D power spectrum is defined as the variance
    of the Fourier-space 3-D quantities :math:`U` and :math:`V`:

    .. math::

        \langle U(\mathbf{k}) V^*(\mathbf{k'}) \rangle =
        (2 \pi)^2  \delta(\mathbf{k}-\mathbf{k'}) \, P_{UV}.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    tracer1, tracer2 : :class:`~pyccl.tracers.Tracer`
        Tracers of an observable quantity of any kind.
        If ``tracer2 is None``, ``tracer1`` will be used for auto-correlation.
    ell : float or (nell,) array_like, optional
        Angular wavenumber for angular power spectrum evaluation.
        By default, use the internally defined array of :math:`\ell`.
    p_of_k_a : :obj:`~pyccl.pk2d.DefaultPowerSpectrum` \
        or :class:`~pyccl.pk2d.Pk2D`, optional
        3-D power spectrum to project.  String input must correspond to a
        non-linear power spectrum in ``cosmo``.
    l_limber : float, optional
        Angular wavenumber beyond which Limber's approximation is used.
        The default is -1.
    limber_integration_method : {'spline', 'adaptive_quadrature'}, optional
        Limber integration method. The default is 'spline'.

    Returns
    -------
    Cls : float or (nell,) ndarray
        Angular (cross-)power spectrum values, :math:`C_\ell`,
        for the pair of tracers.
    """
    if l_limber > -1:
        raise NotImplementedError("Limber integration not yet implemented.")

    if cosmo['Omega_k'] != 0:
        # TODO: CLASS computes hyperspherical Bessel functions.
        warnings.warn("Hyperspherical Bessel functions for Ωk != 0 "
                      "not implemented in CCL.", CCLWarning)

    cosmo.compute_distances()
    pk = parse_pk2d(cosmo, p_of_k_a, linear=False)

    if tracer2 is None:
        tracer2 = tracer1

    if ell is None:
        ell = _ELL_DEFAULT

    cl = _angular_cl_limber(
        cosmo=cosmo, tracer1=tracer1, tracer2=tracer2, ell=ell, pk=pk,
        integration_method=integration_method)

    return cl
