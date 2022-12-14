from .parameters import spline_params as sparams
from .parameters import physical_constants as const
from .interpolate import Interpolator1D
from .integrate import IntegratorSamples

import numpy as np
from enum import Enum
import warnings


__all__ = ("NeutrinoMassSplits", "Omeganuh2", "nu_masses")


class NeutrinoMassSplits(Enum):
    NORMAL = 'normal'
    INVERTED = 'inverted'
    EQUAL = 'equal'
    SUM = 'sum'
    SINGLE = 'single'


# This is where the neutrino phase-space integral spline is stored.
_nu_spline = None


def _compute_nu_phasespace_spline():
    """Get the spline of the result of the phase-space integral
    for massive neutrinos.
    """
    global _nu_spline
    if _nu_spline is not None:
        return

    mnut = np.linspace(np.log(sparams.NU_MNUT_MIN),
                       np.log(sparams.NU_MNUT_MAX),
                       sparams.NU_MNUT_N)

    x_arr = np.linspace(sparams.NU_MOM_MIN,
                        sparams.NU_MOM_MAX,
                        sparams.NU_MOM_N)

    def nu_integrand(x, r):
        x, r = x[None, :], r[:, None]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.sqrt(x*x + r*r) / (np.exp(x) + 1.) + x*x

    integrator = IntegratorSamples("simpson")
    integral = integrator(nu_integrand(x_arr, np.exp(mnut)), x=x_arr)
    _nu_spline = Interpolator1D(mnut, integral/integral[0],
                                extrap_orders=[0, 0])


def _nu_phasespace_integral(mnuOT):
    """Get the value of the phase-space integral at mnuOT
    (the dimensionless mass/temperature of a single massive neutrino).
    """
    idx_lo = mnuOT < sparams.NU_MNUT_MIN
    idx_hi = mnuOT > sparams.NU_MNUT_MAX
    _compute_nu_phasespace_spline()
    out = np.asarray(7/8 * _nu_spline(np.log(mnuOT)))
    out[idx_lo], out[idx_hi] = 7/8, 0.2776566337 * mnuOT[idx_hi]
    return out


def Omeganuh2(a, m_nu, T_CMB, T_ncdm, *, squeeze=True):
    r"""Calculate :math:`\Omega_\nu \, h^2` at a given scale factor given
    the neutrino masses.

    Arguments
    ---------
    a : float or (na,) array-like
        Scale factor(s), normalized to 1 today.
    m_nu : float or sequence
        Neutrino masses in :math:`\rm eV`.
    T_CMB : float
        Temperature of the Cosmic Microwave Background.
    T_ncdm : float
        Non-CDM temperature in units of photon temperature.
    squeeze : bool
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.

    Returns
    -------
    float or (na,) array_like:
        :math:`\Omega_\nu \, h^2` of the neutrino masses at ``a``.
    """
    a = np.asarray(a)
    m_nu = np.atleast_1d(m_nu)
    N_nu_mass = len(m_nu)

    # Tnu_eff is used in the massive case because CLASS uses an effective
    # temperature of non-ΛCDM components to match to mnu / Omeganu = 93.14 eV.
    T_nu = T_CMB * (4/11)**(1/3)
    Tnu_eff = T_CMB * T_ncdm

    # Define the prefix using the effective temperature
    # (to get mnu / Omega = 93.14 eV) for the massive case.
    prefix_massive = const.NU_CONST * Tnu_eff**4

    OmNuh2 = 0.
    for i in range(N_nu_mass):
        # Check whether this species is effectively massless.
        # In this case, invoke the analytic massless limit.
        if m_nu[i] < 0.00017:
            prefix_massless = const.NU_CONST * T_nu**4
            OmNuh2 += 7/8 * N_nu_mass * prefix_massless / a**4
        else:
            # For massive neutrinos, return the density normalized to get Nuh2
            # at a = 0: mass over T (mass (eV) / ((kb eV/s/K) Tnu_eff (K))).
            mnuOT = (m_nu[i] / (Tnu_eff/a)
                     * (const.EV_IN_J/const.KBOLTZ))
            OmNuh2 += prefix_massive * _nu_phasespace_integral(mnuOT) / a**4

    return OmNuh2.squeeze()[()] if squeeze else OmNuh2


def nu_masses(OmNuh2, mass_split):
    """Returns the neutrinos mass(es) for a given OmNuh2, according to the
    splitting convention specified by the user.

    Arguments
    ---------
    OmNuh2 : float or (nΩ,) array_like
        Neutrino energy density at z=0 times h^2.
    mass_split : {'normal', 'iverted', 'equal', 'sum', 'single'}
        Indicates how the masses should be split up.

    Returns
    -------
    nu_masses : float or (N, nΩ) ndarray
        Neutrino mass(es) corresponding to this ``OmeNuh2``.
    """
    sumnu = 93.14 * OmNuh2

    # Now split the sum up into three masses depending on the label given.
    split = NeutrinoMassSplits

    if split(mass_split) in [split.SUM, split.SINGLE]:
        return sumnu

    if split(mass_split) == split.EQUAL:
        return np.full(3, sumnu/3.)

    # See CCL note for how we get these expressions for the
    # neutrino masses in normal and inverted hierarchy.
    if split(mass_split) == split.NORMAL:
        ΔM12 = const.DELTAM12_sq
        ΔM13p = const.DELTAM13_sq_pos
        sqrt = np.sqrt(-6*ΔM12 + 12*ΔM13p + 4*sumnu**2)
        mnu = np.array([
            2/3*sumnu - 1/6*sqrt - 0.25*ΔM12/(2/3*sumnu - 1/6*sqrt),
            2/3*sumnu - 1/6*sqrt + 0.25*ΔM12/(2/3*sumnu - 1/6*sqrt),
            -1/3*sumnu + 1/3*sqrt])

    if split(mass_split) == split.INVERTED:
        ΔM12 = const.DELTAM12_sq
        ΔM13n = const.DELTAM13_sq_neg
        sqrt = np.sqrt(-6*ΔM12 + 12*ΔM13n + 4*sumnu**2)
        mnu = np.array([
            2/3*sumnu - 1/6*sqrt - 0.25*ΔM12/(2/3*sumnu - 1/6*sqrt),
            2/3*sumnu - 1/6*sqrt + 0.25*ΔM12/(2/3*sumnu - 1/6*sqrt),
            -1/3*sumnu + 1/3*sqrt])

    if (mnu < 0).any():
        # Sum is below the physical limit.
        if sumnu < 1e-14:
            return np.zeros(3)
        raise ValueError(
            "Sum of neutrino masses for this `OmegaNu` is incompatible "
            "with the requested mass hierarchy.")
    return mnu
