from .tracer_base import Tracer
from .kernels import get_kappa_kernel


def CMBLensingTracer(cosmo, *, z_source=1100, n_samples=100):
    """A Tracer for CMB lensing.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    z_source : float
        Redshift of source plane for CMB lensing.
    n_samples : int, optional
        Number of samples used in the kappa kernel integration.
        These are linearly sapced in radial distance.
        The kernel is smooth, so usually :math:`O(100)` samples are enough.
    """
    tracer = Tracer()

    # we need the distance functions at the C layer
    cosmo.compute_distances()
    kernel = get_kappa_kernel(cosmo, z_source=z_source, n_samples=n_samples)
    if (cosmo['sigma_0'] == 0):
        tracer.add_tracer(kernel=kernel, der_bessel=-1, der_angles=1)
    else:
        tracer._MG_add_tracer(cosmo, kernel, z_source,
                              der_bessel=-1, der_angles=1)
    return tracer
