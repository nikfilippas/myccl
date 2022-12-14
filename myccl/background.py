from .interpolate import Interpolator1D, loglin_spacing
from .neutrinos import Omeganuh2
from .parameters import accuracy_params
from .parameters import spline_params as sparams
from .parameters import physical_constants as const
from .errors import CCLWarning

import numpy as np
from scipy.integrate import solve_ivp
import warnings
from enum import Enum
import functools


__all__ = ("Species", "h_over_h0", "comoving_radial_distance",
           "scale_factor_of_chi", "sinn", "comoving_angular_distance",
           "angular_diameter_distance", "luminosity_distance",
           "distance_modulus", "hubble_distance", "comoving_volume_element",
           "comoving_volume", "lookback_time", "age_of_universe",
           "omega_x", "rho_x", "growth_factor", "growth_factor_unnorm",
           "growth_rate", "sigma_critical")


_eps = 1e-8  # ε used to truncate distances


class Species(Enum):
    CRITICAL = 'critical'
    MATTER = 'matter'
    DARK_ENERGY = 'dark_energy'
    RADIATION = 'radiation'
    CURVATURE = 'curvature'
    NEUTRINOS_REL = 'neutrinos_rel'
    NEUTRINOS_MASSIVE = 'neutrinos_massive'


def _h_over_h0(cosmo, a):
    r"""Ratio of Hubble constant at `a` over Hubble constant today.

    .. math::

        E(a) = \frac{H(a)}{H_0}.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
        Scale factor(s) normalized to 1 today.

    Returns
    -------
    Ez : float or (na,) ndarray
    """
    a = np.asarray(a)
    Omega_nu_mass = 0
    if cosmo["N_nu_mass"] > 1e-12:
        Omega_nu_mass = Omeganuh2(a=a,
                                  m_nu=cosmo["m_nu"],
                                  T_CMB=cosmo["T_CMB"],
                                  T_ncdm=cosmo["T_ncdm"],
                                  squeeze=False) / cosmo["h"]**2

    out = ((cosmo["Omega_c"] + cosmo["Omega_b"]
            + cosmo["Omega_l"] * a**(-3*cosmo["w0"] + cosmo["wa"])
            * np.exp(3*cosmo["wa"]*(a-1))
            + cosmo["Omega_k"] * a
            + (cosmo["Omega_g"] + cosmo["Omega_nu_rel"]) / a
            + Omega_nu_mass * a**3) / (a**3))**(0.5)

    return out


def compute_distances(cosmo) -> None:
    """Compute background distance quantities for an input Cosmology."""
    if cosmo.has_distances:
        return

    a = loglin_spacing(
        sparams.A_SPLINE_MINLOG, sparams.A_SPLINE_MIN, sparams.A_SPLINE_MAX,
        sparams.A_SPLINE_NLOG, sparams.A_SPLINE_NA)

    # E(z)
    hoh0 = _h_over_h0(cosmo, a)
    cosmo.data.E = Interpolator1D(a, hoh0)

    # chi(z) & a(chi)
    integrand = const.CLIGHT_HMPC / (hoh0 * a**2 * cosmo["h"])
    integral = Interpolator1D(a, integrand).f.antiderivative()
    chi_arr = integral(1.0) - integral(a)

    cosmo.data.chi = Interpolator1D(a, chi_arr)
    cosmo.data.achi = Interpolator1D(chi_arr[::-1], a[::-1])

    # lookback time
    t_H = const.MPC_TO_METER / 1e14 / const.YEAR / cosmo["h"]
    integral = Interpolator1D(a, 1/(a*hoh0)).f.antiderivative()
    a_eval = np.r_[1.0, a]  # make a single call to the spline
    vals = integral(a_eval)
    t_arr = t_H * (vals[0] - vals[1:])

    cosmo.data.lookback = Interpolator1D(a, t_arr, extrap_orders=[1, None])
    cosmo.data.age0 = cosmo.data.lookback(0)[()]


@functools.wraps(_h_over_h0)
def h_over_h0(cosmo, a, *, squeeze=True):
    a = np.asarray(a)
    out = np.ones_like(a, dtype=float)
    idx = np.logical_or(a < 1 - _eps,  a > 1 + _eps)
    cosmo.compute_distances()
    out[idx] = cosmo.data.E(a[idx])
    return out.squeeze()[()] if squeeze else out


def comoving_radial_distance(cosmo, a, *, squeeze=True):
    r"""Comoving radial distance (in :math:`\rm Mpc`).

    .. math::

        D_{\rm c} = \frac{c}{H_0} \int_0^z \frac{dz'}{E(z')}.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
        Scale factor(s) normalized to 1 today.
    squeeze : bool, optional
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.

    Returns
    -------
    D_C : float or (na,) ndarray
        Comoving radial distance at ``a``.
    """
    a = np.asarray(a)
    out = np.zeros_like(a, dtype=float)
    idx = np.logical_or(a < 1 - _eps,  a > 1 + _eps)
    cosmo.compute_distances()
    out[idx] = cosmo.data.chi(a[idx])
    return out.squeeze()[()] if squeeze else out


def scale_factor_of_chi(cosmo, chi, *, squeeze=True):
    r"""Scale factor at some comoving radial distance, :math:`a(\chi)`.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    chi : float or (nchi,) array_like
        Comoving radial distance :math:`\chi` in :math:`\rm Mpc`.
    squeeze : bool, optional
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.

    Returns
    -------
    a_chi : float or (nchi,) ndarray
        Scale factor at ``chi``.
    """
    chi = np.asarray(chi)
    out = np.ones_like(chi, dtype=float)
    idx = np.logical_or(chi > _eps, chi < -_eps)
    cosmo.compute_distances()
    out[idx] = cosmo.data.achi(chi[idx])
    return out.squeeze()[()] if squeeze else out


def sinn(cosmo, chi, *, squeeze=True):
    r"""Piecewise function related to the geometry of the Universe.

    .. math::

        \mathrm{sinn(x)} = \begin{cases}
        \sin(x)   &  k = +1   \\
            x      &  k = 0   \\
        \sinh(x)  &  k = -1
        \end{cases}

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    chi : float or (nchi,) array_like
        Comoving radial distance :math:`\chi` in :math:`\rm Mpc`.
    squeeze : bool, optional
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.

    Returns
    -------
    sinn : float or (nchi,) ndarray
        Value of the :math:`\rm sinn` function at :math:`\chi`.
    """
    chi = np.asarray(chi)
    k = cosmo["k_sign"]
    if k == 0:
        out = chi
    elif k == 1:
        out = np.sin(cosmo["sqrtk"] * chi) / cosmo["sqrtk"]
    elif k == -1:
        out = np.sinh(cosmo["sqrtk"] * chi) / cosmo["sqrtk"]
    return out.squeeze()[()] if squeeze else out


def comoving_angular_distance(cosmo, a, *, squeeze=True):
    r"""Comoving angular distance (in :math:`\rm Mpc`).

    .. math::
        D_{\rm M} = \mathrm{sinn}(\chi(a)).

    .. note::

        This quantity is otherwise known as the transverse comoving distance,
        and is **not** the angular diameter distance or the angular separation.
        The comoving angular distance is defined such that the comoving
        distance between two objects at a fixed scale factor separated
        by an angle :math:`\theta` is :math:`\theta r_{A}(a)` where
        :math:`r_{A}(a)` is the comoving angular distance.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
        Scale factor(s) normalized to 1 today.
    squeeze : bool, optional
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.

    Returns
    -------
    D_M : float or (na,) ndarray
        Comoving angular distance at ``a``.
    """
    D_c = comoving_radial_distance(cosmo, a, squeeze=False)
    return sinn(cosmo, D_c, squeeze=squeeze)


transverse_comoving_distance = comoving_angular_distance  # alias


def angular_diameter_distance(cosmo, a1, a2=None, *, squeeze=True):
    r"""Angular diameter distance (in :math:`\rm Mpc `).

    Defined as the ratio of an object's physical transverse size to its
    angular size. It is related to the comoving angular distance as:

    .. math::

        D_{\rm A} = \frac{D_{\rm M}}{1 + z}

    .. note::

        If ``a2`` is ``None``, the distance is calculated between 1 and ``a1``.
        Note that ``a2`` has to be smaller than ``a1``. The calculation is
        **not** commutative.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
           Cosmological parameters.
    a1 : float or (na1,) array_like
        Scale factor(s), normalized to 1 today.
    a2 : float or (na2,) array_like, optional
        Scale factor(s), normalized to 1 today.
    squeeze : bool, optional
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.

    Returns
    -------
    D_A : float, (na1, na2,) ndarray or (na,) ndarray
        Angular diameter distance. If ``a1.shape != a2.shape`` but ``a1`` and
        ``a2`` can be broadcast to one another, all distance combinations are
        returned. If the input shapes match, the pairwise distance is returned.
    """
    cosmo.compute_distances()
    if a2 is None:
        a2, chi1 = a1, 0.
    else:
        chi1 = comoving_radial_distance(cosmo, a1, squeeze=False)
    chi2 = comoving_radial_distance(cosmo, a2, squeeze=False)

    out = a2 * sinn(cosmo, chi2-chi1, squeeze=False)
    return out.squeeze()[()] if squeeze else out


def luminosity_distance(cosmo, a, *, squeeze=True):
    r"""Luminosity distance.

    Defined by the relationship between bolometric flux :math:`S` and
    bolometric luminosity :math:`L`.

    .. math::
        D_{\rm L} = \sqrt{\frac{L}{4 \pi S}}

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
        Scale factor(s) normalized to 1 today.
    squeeze : bool, optional
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.

    Returns
    -------
    D_L : float or (na,) ndarray
        Luminosity distance at ``a``.
    """
    a = np.asarray(a)
    out = comoving_angular_distance(cosmo, a, squeeze=False) / a
    return out.squeeze()[()] if squeeze else out


def distance_modulus(cosmo, a, *, squeeze=True):
    r"""Distance modulus.

    Used to convert between apparent and absolute magnitudes
    via :math:`m = M + (\rm dist. \, mod.)` where :math:`m` is the
    apparent magnitude and :math:`M` is the absolute magnitude.

    .. math::

        m - M = 5 * \log_{10}(D_{\rm L} / 10 \, \rm pc).

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
        Scale factor(s) normalized to 1 today.
    squeeze : bool, optional
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.

    Returns
    -------
    D_M : float or (na,) ndarray
        Distance modulus at ``a``.
    """
    a = np.asarray(a)
    out = np.zeros_like(a, dtype=float)
    idx = np.logical_or(a < 1 - _eps,  a > 1 + _eps)
    # m - M = 5 * log10(d) - 5. Since d in CCL is in Mpc,
    # you get 5*log10(10^6) - 5 = 30 - 5 = 25 for the constant.
    D_L = luminosity_distance(cosmo, a[idx], squeeze=False)
    out[idx] = 5 * np.log10(D_L) + 25
    return out.squeeze()[()] if squeeze else out


def hubble_distance(cosmo, a, *, squeeze=True):
    r"""Hubble distance in :math:`\rm Mpc`.

    .. math::

        D_{\rm H} = \frac{cz}{H_0}

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
        Scale factor(s) normalized to 1 today.
    squeeze : bool, optional
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.

    Returns
    -------
    D_H : float or (na,) ndarray
        Hubble distance.
    """
    a = np.asarray(a)
    out = (1/a - 1) * const.CLIGHT_HMPC / cosmo["h"]
    return out.squeeze()[()] if squeeze else out


def comoving_volume_element(cosmo, a, *, squeeze=True):
    r"""Comoving volume element in :math:`\rm Gpc^3 \, sr^{-1}`.

    .. math::

        \frac{\mathrm{d}V}{\mathrm{d}a \, \mathrm{d} \Omega}

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
        Scale factor(s) normalized to 1 today.
    squeeze : bool, optional
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.

    Returns
    -------
    dV : float or (na,) ndarray
        Comoving volume per unit scale factor per unit solid angle.

    See Also
    --------
    comoving_volume : integral of the comoving volume element
    """
    Dm = comoving_angular_distance(cosmo, a, squeeze=False)
    Ez = _h_over_h0(cosmo, a)
    Dh = const.CLIGHT_HMPC / cosmo["h"]
    dV = Dh * Dm**2 / (Ez * a**2) * 1e-9
    return dV.squeeze()[()] if squeeze else dV


def comoving_volume(cosmo, a, *, solid_angle=4*np.pi, squeeze=True):
    r"""Comoving volume, in :math:`\rm Gpc^3`.

    .. math::

        V_{\rm C} = \int_{\Omega} \mathrm{{d}}\Omega \int_z \mathrm{d}z
        D_{\rm H} \frac{(1+z)^2 D_{\mathrm{A}}^2}{E(z)}

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
        Scale factor(s) normalized to 1 today.
    solid_angle : float
        Solid angle subtended in the sky for which
        the comoving volume is calculated.
    squeeze : bool, optional
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.

    Returns
    -------
    V_C : float or (na,) ndarray
        Comoving volume at ``a``.

    See Also
    --------
    comoving_volume_element : comoving volume element
    """
    a = np.asarray(a)
    out = np.zeros_like(a, dtype=float)
    idx = np.logical_or(a < 1 - _eps, a > 1 + _eps)

    Ωk, sqrtk = cosmo["Omega_k"], cosmo["sqrtk"]
    Dm = comoving_angular_distance(cosmo, a[idx], squeeze=False)
    if Ωk == 0:
        out[idx] = solid_angle/3 * Dm**3 * 1e-9
        return out.squeeze()[()] if squeeze else out

    Dh = cosmo.hubble_distance(a[idx], squeeze=False)
    DmDh = Dm / Dh
    arcsinn = np.arcsin if Ωk < 0 else np.arcsinh
    out[idx] = ((solid_angle * Dh**3 / (2 * Ωk))
                * (DmDh * np.sqrt(1 + Ωk * DmDh**2)
                   - arcsinn(sqrtk * DmDh)/sqrtk)) * 1e-9
    return out.squeeze()[()] if squeeze else out


def lookback_time(cosmo, a, *, squeeze=True):
    r"""Difference of the age of the Universe between some scale factor
    and today, in :math:`\rm Gyr`.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
        Scale factor(s) normalized to 1 today.
    squeeze : bool, optional
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.

    Returns
    -------
    t_L : float or (na,) ndarray
        Lookback time at ``a``.
    """
    a = np.asarray(a)
    out = np.zeros_like(a, dtype=float)
    idx = np.logical_or(a < 1 - _eps,  a > 1 + _eps)
    cosmo.compute_distances()
    out[idx] = cosmo.data.lookback(a[idx])
    return out.squeeze()[()] if squeeze else out


def age_of_universe(cosmo, a, *, squeeze=True):
    r"""Age of the Universe at some scale factor, in :math:`\rm Gyr`.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
        Scale factor(s) normalized to 1 today.
    squeeze : bool, optional
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.

    Returns
    -------
    t_age : float or (na,) ndarray
        Age of the Universe at ``a``.
    """
    cosmo.compute_distances()
    out = cosmo.data.age0 - lookback_time(cosmo, a, squeeze=False)
    return out.squeeze()[()] if squeeze else out


def omega_x(cosmo, a, species, *, squeeze=True):
    r"""Density fraction of a given species at a particular scale factor.

    .. math::

        \Omega_x(a) = \frac{\rho_x(a)}{\rho_{\rm crit}(a)}

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
        Scale factor, normalized to 1 today.
    species : str
        Species type. Should be one of:
            - 'matter': cold dark matter, massive neutrinos, and baryons
            - 'dark_energy': cosmological constant or otherwise
            - 'radiation': relativistic species besides massless neutrinos
            - 'curvature': curvature density
            - 'neutrinos_rel': relativistic neutrinos
            - 'neutrinos_massive': massive neutrinos
    squeeze : bool, optional
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.

    Returns
    -------
    Omega_x : float or (na,) ndarray
        Density fraction of a given species at ``a``.
    """
    a = np.asarray(a)
    Omega_nu_mass = 0
    if (Species(species) in [Species.MATTER, Species.NEUTRINOS_MASSIVE]
            and cosmo["N_nu_mass"] > 1e-12):
        Omega_nu_mass = Omeganuh2(a=a,
                                  m_nu=cosmo["m_nu"],
                                  T_CMB=cosmo["T_CMB"],
                                  T_ncdm=cosmo["T_ncdm"],
                                  squeeze=False) / cosmo["h"]**2

    if Species(species) != Species.CRITICAL:
        hnorm = _h_over_h0(cosmo, a)

    if Species(species) == Species.CRITICAL:
        out = np.ones_like(a)

    elif Species(species) == Species.MATTER:
        out = (((cosmo["Omega_c"] + cosmo["Omega_b"]) / a**3
                + Omega_nu_mass) / hnorm**2)

    elif Species(species) == Species.DARK_ENERGY:
        out = (cosmo["Omega_l"] * a**(-3*(1 + cosmo["w0"] + cosmo["wa"]))
               * np.exp(3 * cosmo["wa"] * (a-1)) / hnorm**2)

    elif Species(species) == Species.RADIATION:
        out = cosmo["Omega_g"] / a**4 / hnorm**2

    elif Species(species) == Species.CURVATURE:
        out = cosmo["Omega_k"] / a**2 / hnorm**2

    elif Species(species) == Species.NEUTRINOS_REL:
        out = cosmo["Omega_nu_rel"] / a**4 / hnorm**2

    elif Species(species) == Species.NEUTRINOS_MASSIVE:
        out = Omega_nu_mass / hnorm**2

    return out.squeeze()[()] if squeeze else out


def rho_x(cosmo, a, species, is_comoving=False, *, squeeze=True):
    r"""Physical or comoving density as a function of scale factor.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
        Scale factor(s) normalized to 1 today.
    species : string
        Species type. Should be one of

        - 'matter': cold dark matter, massive neutrinos, and baryons
        - 'dark_energy': cosmological constant or otherwise
        - 'radiation': relativistic species besides massless neutrinos
        - 'curvature': curvature density
        - 'neutrinos_rel': relativistic neutrinos
        - 'neutrinos_massive': massive neutrinos

    is_comoving : bool, optional
        Either physical or comoving. Default is ``False`` for physical.
    squeeze : bool, optional
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.

    Returns
    -------
    rho_x : float or (na,) ndarray
        Physical density of a given species at a scale factor,
        in units of :math:`\rm M_\odot / Mpc^3`.
    """
    a = np.asarray(a)
    comfac = a**3 if is_comoving else np.ones(a.shape)
    hnorm = _h_over_h0(cosmo, a)
    rho_c = const.RHO_CRITICAL * (cosmo["h"] * hnorm)**2 * comfac
    out = rho_c * omega_x(cosmo, a, species, squeeze=False)
    return out.squeeze()[()] if squeeze else out


def compute_growth(cosmo) -> None:
    """Compute the function for the growth of density perturbations."""
    if cosmo.has_growth:
        return

    if cosmo["N_nu_mass"] > 0:
        warnings.warn(
            "CCL does not properly compute the linear growth rate in "
            "cosmological models with massive neutrinos!", CCLWarning)

    # Growth will be evaluated at these points.
    a = loglin_spacing(
        sparams.A_SPLINE_MINLOG, sparams.A_SPLINE_MIN, sparams.A_SPLINE_MAX,
        sparams.A_SPLINE_NLOG, sparams.A_SPLINE_NA)

    def dSdx(a, S):
        # System of coupled equations for the growth.
        if a < a_0:
            # initial conditions
            return [a, 1.]
        hnorm = _h_over_h0(cosmo, a)
        Om = omega_x(cosmo, a, "matter")
        is_MG = abs(cosmo["mu_0"]) > 1e-12
        mu = cosmo.mu_MG(k=0., a=a) if is_MG else 0  # μ(a, k=0) : large scales
        return [S[1] / (a**3 * hnorm),
                1.5 * hnorm * a * Om * S[0] * (1. + mu)]

    # Run the solver.
    a_0 = accuracy_params.EPSREL_GROWTH
    amax = sparams.A_SPLINE_MAX
    S_0 = [a_0, a_0**2 * _h_over_h0(cosmo, a_0)]
    res = solve_ivp(dSdx, y0=S_0, t_span=(a_0, amax), t_eval=a,
                    method="RK45", first_step=0.1*a_0,
                    atol=0, rtol=accuracy_params.EPSREL_GROWTH)

    gf = res.y[0]  # growth
    fg = res.y[1] / (a**2 * _h_over_h0(cosmo, a) * res.y[0])  # growth rate

    unnorm = Interpolator1D(a, gf)
    growth = Interpolator1D(a, gf/unnorm(1.0))
    fgrowth = Interpolator1D(a, fg)

    cosmo.data.growth = growth
    cosmo.data.fgrowth = fgrowth
    cosmo.data.growth0 = unnorm(1.0)[()]


def growth_factor(cosmo, a, *, squeeze=True):
    """Growth factor.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
        Scale factor(s) normalized to 1 today.
    squeeze : bool, optional
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.

    Returns
    -------
    D : float or (na,) ndarray
        Growth factor at ``a``.
    """
    a = np.asarray(a)
    out = np.ones_like(a, dtype=float)
    idx = np.logical_or(a < 1 - _eps,  a > 1 + _eps)
    cosmo.compute_growth()
    out[idx] = cosmo.data.growth(a[idx])
    return out.squeeze()[()] if squeeze else out


def growth_factor_unnorm(cosmo, a, *, squeeze=True):
    """Unnormalized growth factor.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
        Scale factor(s) normalized to 1 today.
    squeeze : bool, optional
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.

    Returns
    -------
    D_unnorm : float or (na,) ndarray
        Unnormalized growth factor at ``a``.
    """
    out = growth_factor(cosmo, a, squeeze=False) * cosmo.data.growth0
    return out.squeeze()[()] if squeeze else out


def growth_rate(cosmo, a, *, squeeze=True):
    r"""Growth rate defined as the logarithmic derivative of the
    growth factor,

    .. math::

        \frac{\mathrm{d}\ln{D}}{\mathrm{d}\ln{a}}.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
        Scale factor(s) normalized to 1 today.
    squeeze : bool, optional
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.

    Returns
    -------
    dlnD_dlna : float or (na,) ndarray
        Growth rate at ``a``.
    """
    a = np.asarray(a)
    cosmo.compute_growth()
    dlnD_dlna = cosmo.data.fgrowth(a)
    return dlnD_dlna.squeeze()[()] if squeeze else dlnD_dlna


def sigma_critical(cosmo, a_lens, a_source, *, squeeze=True):
    r"""Returns the critical surface mass density.

    .. math::

         \Sigma_{\mathrm{crit}} = \frac{c^2}{4 \pi G}
         \frac{D_{\rm s}}{D_{\rm l}D_{\rm ls}},

    where :math:`c` is the speed of light, :math:`G` is the
    gravitational constant, and :math:`D_i` is the angular diameter distance
    The labels :math:`i =` ``s``, ``l`` and ``ls`` denote the distances
    to the source, lens, and between source and lens, respectively.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        A Cosmology object.
    a_lens : float or (na_lens,) array_like
        Scale factor of lens.
    a_source : float or (na_source,) array_like
        Scale factor of source.
    squeeze : bool, optional
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.

    Returns
    -------
    sigma_critical : float, (na_lens, na_source,) ndarray or (na,) ndarray
        :math:`\Sigma_{\rm crit} in units of
        :math:`\rm M_\odot / Mpc^2`.
        If ``a_lens.shape != a_source.shape`` but ``a_lens`` and ``a_source``
        can be broadcast to one another, all distance combinations are
        returned. If the input shapes match, the pairwise distance is returned.
    """
    Ds = angular_diameter_distance(cosmo, a_source, a2=None, squeeze=False)
    Dl = angular_diameter_distance(cosmo, a_lens, a2=None, squeeze=False)
    Dls = angular_diameter_distance(cosmo, a_lens, a_source, squeeze=False)
    A = (const.CLIGHT**2 * const.MPC_TO_METER
         / (4.0 * np.pi * const.GNEWT * const.SOLAR_MASS))
    Sigma_crit = A * Ds / (Dl * Dls)
    return Sigma_crit
