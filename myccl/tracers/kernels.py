from ..interpolate import Interpolator1D
from ..parameters import physical_constants as const
import numpy as np


def _check_background_spline_compatibility(cosmo, z):
    """Helper to check that a redshift array lies within
    the support of the background splines in ``cosmo``.
    """
    z = np.asarray(z)
    cosmo.compute_distances()
    a_bg_min, a_bg_max = cosmo.data.chi._points[0].take([0, -1])
    a_min, a_max = (1/(1+z)).take([0, -1])
    if a_min < a_bg_min or a_max > a_bg_max:
        raise ValueError(
            "Tracer has wider support than splines of cosmo.\n"
            f"Tracer:     z ∈ [{z[0]}, {z[-1]}].\n"
            f"Background: z ∈ [{1/a_bg_max - 1}, {1/a_bg_min - 1}].")


def get_nz_norm(nz_f):
    """Normalization of the :math:`N(z)` spline."""
    zmin, zmax = nz_f._points[0].take([0, -1])
    return nz_f.f.integrate(zmin, zmax)[()]


def get_density_kernel(cosmo, dndz):
    """Radial kernel for tracers of galaxy clustering.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    dndz : (nz, nz) tuple of array_like
        Redshift distribution with arbitrary units.
        :math:`N(z)` is internally normalized to 1.

    Returns
    -------
    chi, w(chi) : (nz, nz) tuple of ndarrays
        Radial kernel expressed as :math:`p(z) \\, H(z)`, where :math:`H(z)`
        is the expansion rate in :math:`\\mathrm{Mpc}^{-1}` and :math:`p(z)`
        is the normalized redshift distribution.
    """
    z, Nz = dndz
    a = 1 / (1+z)
    _check_background_spline_compatibility(cosmo, z)
    chi = cosmo.comoving_radial_distance(a)

    nz_f = Interpolator1D(z, Nz, extrap_orders=[0, 0])
    norm = 1 / get_nz_norm(nz_f)
    h = cosmo["h"] * cosmo.h_over_h0(a) / const.CLIGHT_HMPC
    return chi, h * Nz * norm


def get_lensing_prefactor(cosmo):
    r"""Calculate :math:`\frac{3}{2} \, H_0^2 \, \Omega_{\rm m}`."""
    return 1.5 * (cosmo["h"]*const.CLIGHT_HMPC)**2 * cosmo["Omega_m"]


def get_lensing_kernel(cosmo, *, dndz, mag_bias=None, n_samples):
    r"""Radial kernel for weak lensing:

    .. math::

        \frac{3 H_0^2 \Omega_{\rm m}}{2 a} \,
        \int p(z)

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    dndz : (nz, nz) tuple of array_like
        Redshift distribution.
    mag_bias : (nzb, nzb), optional
        Magnification bias. The default is None, for no bias.
    n_samples : int, optional
        Number of samples used in the kernel integration.

    Returns
    -------
    chi, wchi : (nchi, nchi) tuple of ndarray
        The lensing kernel, where :math:`\chi` is the comoving radial
        distance and :math:`w(\chi)` is the kernel.
    """
    cosmo.compute_distances()

    z, Nz = dndz
    _check_background_spline_compatibility(cosmo, z)
    nz_f = Interpolator1D(z, Nz, extrap_orders=[0, 0])
    norm = 1 / get_nz_norm(nz_f)
    if mag_bias:
        sz_f = Interpolator1D(*mag_bias, extrap_orders=[0, 0])

    z_max = z[-1]
    z_end = np.linspace(0, z_max, n_samples)[:, None]
    chi_end = cosmo.comoving_radial_distance(1/(1+z_end), squeeze=False)
    prefac = (1 + z_end) * get_lensing_prefactor(cosmo) * norm * chi_end

    def integrand(z):
        """TODO"""
        shp = np.broadcast_shapes(chi_end.shape, z.shape)
        out = np.ones(shp)
        idx = z != 0

        pz = nz_f(z)
        qz = 1 - 2.5*sz_f(z) if mag_bias else 1.

        val = cosmo.sinn(chi_int[idx] - chi_end) / cosmo.sinn(chi_int[idx])
        idx = np.broadcast_to(idx, out.shape)
        np.place(out, idx, val)
        return out * pz * qz

    # 1. eval the integrand at all those points
    z_int = np.linspace(0, z_max, 10*n_samples)[None, :]
    chi_int = cosmo.comoving_radial_distance(1/(1+z_int), squeeze=False)
    samples = integrand(z_int)

    # 2. interpolate and integrate

    # 3. evaluate the integral at all those points
    wchi = None

    return chi_end, wchi * prefac


def get_kappa_kernel(cosmo, *, z_source=1100, n_samples):
    r"""Radial kernel for CMB lensing tracers:

    .. math::

        \frac{3}{2} \, H_0^2 \Omega_{\rm m} \,
        \frac{\chi (\chi_s - \chi)}{\chi_s \, a}

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    z_source : float, optional
        Redshift of the source. The default is 1100, for the CMB.
    n_samples : int, optional
        Number of samples used in the kernel integration.
        These are spaced linearly in distance.

    Returns
    -------
    chi, wchi : (n_samples, n_samples) tuple of ndarray
        The :math:`\kappa` kernel, where :math:`\chi` is the comoving radial
        distance and :math:`w(\chi)` is the kernel.
    """
    _check_background_spline_compatibility(cosmo, z_source)
    chi_source = cosmo.comoving_radial_distance(1/(1+z_source))
    chi = np.linspace(0, chi_source, n_samples)

    lens_prefac = get_lensing_prefactor(cosmo) / cosmo.sinn(chi_source)
    a = cosmo.scale_factor_of_chi(chi)
    wchi = lens_prefac * cosmo.sinn(chi_source-chi) * chi / a
    return chi, wchi
