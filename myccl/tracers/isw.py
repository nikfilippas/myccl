from .tracer_base import Tracer
from ..parameters import physical_constants as const
import numpy as np


def ISWTracer(cosmo, *, z_max=6., n_samples=1024):
    r"""Specific ``Tracer`` associated with the integrated Sachs-Wolfe
    effect (ISW). Useful when cross-correlating any low-redshift probe with
    the primary CMB anisotropies. The ISW contribution to the temperature
    fluctuations is:

    .. math::

        \Delta T_{\rm CMB} =
        2T_{\rm CMB} \int_0^{\chi_{LSS}} \mathrm{d}\chi a \, \dot{\phi}.

    Any angular power spectra computed with this tracer, should use a 3-D
    power spectrum involving the matter power spectrum.
    The current implementation of this tracers assumes a standard Poisson
    equation relating :math:`\phi` and :math:`\delta`, and linear structure
    growth. Although this should be valid in :math:`\Lambda`CDM and on
    the large scales the ISW is sensitive to, these approximations must be
    borne in mind.

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
    chi = np.linspace(0, chi_max, n_samples)
    a_arr = cosmo.scale_factor_of_chi(chi)
    H0 = cosmo['h'] / const.CLIGHT_HMPC
    Ωm = cosmo['Omega_m']
    Ez = cosmo.h_over_h0(a_arr)
    fz = cosmo.growth_rate(a_arr)
    w_arr = 3*cosmo['T_CMB']*H0**3*Ωm*Ez*chi**2*(1-fz)

    tracer.add_tracer(kernel=(chi, w_arr), der_bessel=-1)
    return tracer
