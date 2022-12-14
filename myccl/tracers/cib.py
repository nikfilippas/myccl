from .tracer_base import Tracer
import numpy as np


def CIBTracer(cosmo, *, z_min=0., z_max=6., n_samples=1024):
    r"""Specific ``Tracer`` associated with the cosmic infrared
    background (CIB). The radial kernel for this tracer is:

    .. math::

       W(\chi) = \frac{1}{1+z}.

    Any angular power spectra computed with this tracer, should use a 3-D
    power spectrum involving the CIB emissivity density expressed in
    :math:`\rm Jy \, Mpc^{-1} \, sr^{-1}` (or multiples thereof).

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    z_min, z_max : float, optional
        Minimum and maximum redshift up to which the tracer is defined.
        The default range is [0, 6].
    n_samples : float, optional
        Number of sampling points for the radial kernel.
        The samples are linearly spaced in distance.
        The default is 1024.
    """
    tracer = Tracer()

    chi_max = cosmo.comoving_radial_distance(1./(1+z_max))
    chi_min = cosmo.comoving_radial_distance(1./(1+z_min))
    chi_arr = np.linspace(chi_min, chi_max, n_samples)
    a_arr = cosmo.scale_factor_of_chi(chi_arr)

    tracer.add_tracer(kernel=(chi_arr, a_arr))
    return tracer
