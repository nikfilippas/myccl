from .tracer_base import Tracer
from ..parameters import physical_constants as const
import numpy as np


def tSZTracer(cosmo, *, z_max=6., n_samples=1024):
    r"""Specific ``Tracer`` associated with the thermal Sunyaev Zel'dovich
    Compton-y parameter. The radial kernel for this tracer is:

    .. math::

       W(\chi) = \frac{\sigma_T}{m_e c^2} \frac{1}{1+z},

    where :math:`\sigma_T` is the Thomson scattering cross-section
    for the electron and :math:`m_e` is the electron mass.

    Any angular power spectra computed with this tracer, should use
    a three-dimensional power spectrum involving the electron pressure
    in physical (non-comoving) units of :math:`\rm eV \, cm^{-3}`.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    z_max : float, optional
        Maximum redshift up to which the tracer is defined.
        The default is 6.
    n_samples : float, optional
        Number of sampling points for the radial kernel.
        The samples are linearly spaced in distance.
        The default is 1024.
    """
    tracer = Tracer()

    chi_max = cosmo.comoving_radial_distance(1./(1+z_max))
    chi_arr = np.linspace(0, chi_max, n_samples)
    a_arr = cosmo.scale_factor_of_chi(chi_arr)
    # This is σ_T / (m_e * c^2), expressed in [eV Mpc / cm^3].
    prefac = const.SIG_THOMSON / (const.ELECTRON_MASS * const.CLIGHT**2)
    prefac *= 100**3 * const.EV_IN_J * const.MPC_TO_METER
    w_arr = prefac * a_arr

    tracer.add_tracer(cosmo, kernel=(chi_arr, w_arr))
    return tracer
