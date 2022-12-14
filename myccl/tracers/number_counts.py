from .tracer_base import Tracer
from .kernels import get_density_kernel, get_lensing_kernel


def NumberCountsTracer(cosmo, *, dndz, bias=None, mag_bias=None,
                       has_rsd, n_samples=256) -> Tracer:
    r"""Specific ``Tracer`` associated to galaxy clustering with linear
    scale-independent bias, including redshift-space distortions and
    magnification.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    dndz : (nz, nz) tuple of array_like
        Redshift distribution. Units are arbitrary; N(z) is normalized to 1.
    bias : (nz, nz) tuple of array_like, optional
        Galaxy bias. The default (None) contains no term proportional to
        the matter density contrast.
    mag_bias : (nz, nz) tuple of array_like, optional
        Magnification bias. The default (None) contains no magnification bias.
    has_rsd : bool
        Flag for whether the tracer has a redshift-space distortion term.
    n_samples : int, optional
        Number of samples used in the lensing magnification kernel integration.
        These are linearly sapced in radial distance.
        The kernel is smooth, so usually :math:`O(100)` samples are enough.
    """
    tracer = Tracer()

    # we need the distance functions at the C layer
    cosmo.compute_distances()

    if bias or has_rsd:
        kernel_d = get_density_kernel(cosmo, dndz=dndz)

    if bias:  # Has density term
        # Transfer
        z_b, b = bias
        # Reverse order for increasing a
        t_a = (1 / (1 + z_b[::-1]), b[::-1])
        tracer.add_tracer(kernel=kernel_d, transfer_a=t_a)

    if has_rsd:  # Has RSDs
        # Transfer (growth rate)
        z_b = dndz[0]
        a_s = 1 / (1 + z_b[::-1])
        t_a = (a_s, -cosmo.growth_rate(a_s))
        tracer.add_tracer(kernel=kernel_d, transfer_a=t_a, der_bessel=2)

    if mag_bias:  # Has magnification bias
        # Kernel
        chi, w = get_lensing_kernel(cosmo, dndz=dndz, mag_bias=mag_bias,
                                    n_samples=n_samples)
        # Multiply by -2 for magnification
        kernel_m = (chi, -2 * w)
        if (cosmo['sigma_0'] == 0):
            # GR case
            tracer.add_tracer(kernel=kernel_m, der_bessel=-1, der_angles=1)
        else:
            # MG case
            tracer._MG_add_tracer(cosmo, kernel_m, dndz[0],
                                  der_bessel=-1, der_angles=1)
    return tracer
