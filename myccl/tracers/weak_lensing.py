from .tracer_base import Tracer
from .kernels import get_density_kernel, get_lensing_kernel
from ..parameters import physical_constants as const


def WeakLensingTracer(cosmo, *, dndz, ia_bias=None,
                      has_shear=True, use_A_ia=True, n_samples=256) -> Tracer:
    r"""Specific ``Tracer`` associated to galaxy shape distortions including
    lensing shear and intrinsic alignments within the L-NLA model.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    dndz : (nz, nz) tuple of array_like
        Redshift distribution. Units are arbitrary; N(z) is normalized to 1.
    ia_bias : (nz, nz) tuple of array_like, optional
        Intrinsic alignment amplitude, :math:`A_{\rm IA}(z)`.
        The default (None) assumes no intrinsic alignments.
    has_shear : bool
        Whether to include the lensing shear contribution of this tracer.
        The default is True.
    use_A_ia : bool
        Whether to use conventional IA normalization.
        If False, use the raw input amplitude, which will usually be 1.
        This is useful for PT IA modeling.
        The default is True.
    n_samples : int, optional
        Number of samples used in the lensing magnification kernel integration.
        These are linearly sapced in radial distance.
        The kernel is smooth, so usually :math:`O(100)` samples are enough.
    """
    tracer = Tracer()

    # we need the distance functions at the C layer
    cosmo.compute_distances()

    if has_shear:
        kernel_l = get_lensing_kernel(cosmo, dndz=dndz, n_samples=n_samples)
        if (cosmo['sigma_0'] == 0):
            # GR case
            tracer.add_tracer(kernel=kernel_l, der_bessel=-1, der_angles=2)
        else:
            # MG case
            tracer._MG_add_tracer(cosmo, kernel_l, dndz[1],
                                  der_bessel=-1, der_angles=2)

    if ia_bias is not None:  # Has intrinsic alignments
        z_a, tmp_a = ia_bias
        # Kernel
        kernel_i = get_density_kernel(cosmo, dndz=dndz)
        if use_A_ia:
            # Normalize so that A_IA=1
            D = cosmo.growth_factor(1./(1+z_a))
            # Transfer
            # See Joachimi et al. (2011), arXiv: 1008.3491, Eq. 6.
            # and note that we use C_1= 5e-14 from arXiv:0705.0166
            rho_m = const.RHO_CRITICAL * cosmo['Omega_m']
            a = - tmp_a * 5e-14 * rho_m / D
        else:
            # use the raw input normalization. Normally, this will be 1
            # to allow nonlinear PT IA models, where normalization is
            # already applied to the power spectrum.
            a = tmp_a
        # Reverse order for increasing a
        t_a = (1./(1+z_a[::-1]), a[::-1])
        tracer.add_tracer(kernel=kernel_i, transfer_a=t_a,
                          der_bessel=-1, der_angles=2)
    return tracer
