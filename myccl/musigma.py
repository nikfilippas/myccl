from .parameters import physical_constants as const
from .background import _h_over_h0
from .pyutils import get_broadcastable
import numpy as np


__all__ = ("mu_MG", "Sigma_MG")


def _musigma(cosmo, k, a, *, c_mg=None):
    """Helper for the μ(k, a) and Σ(k, a) parametrizations."""
    # This function can be extended to include other
    # redshift and scale z-dependencies for μ and Σ in the future.
    a, k = map(np.asarray, [a, k])
    a, k = get_broadcastable(a, k)
    shape = np.broadcast_shapes(a.shape, k.shape)
    out = np.full(shape, c_mg)

    idx = k != 0
    H0 = 100*cosmo["h"]
    hnorm = _h_over_h0(cosmo, a)
    s2_k = 1000 * cosmo["lambda_mg"] * hnorm * H0 / (k[idx] * const.CLIGHT)
    s1_k = (1. + c_mg * s2_k**2) / (1. + s2_k**2)
    vals = (s1_k * cosmo["mu_0"]
            * cosmo.omega_x(a, "dark_energy", squeeze=False)
            / cosmo["Omega_l"])

    idx = np.broadcast_to(idx, out.shape)
    np.place(out, idx, vals)
    return out


def mu_MG(cosmo, k, a, *, squeeze=True):
    """Compute μ(k, a), where μ is one of the parametrizing functions
    of modifications to GR in the quasistatic approximation.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    k : float or array_like
        Wavenumber(s) in units of 1/Mpc.
    a : float or array_like
        Scale factor(s), normalized to 1 today.
    squeeze : bool
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.

    Returns
    -------
    μ_MG : float or array_like
        Modification to Poisson equation under modified gravity.
    """
    out = _musigma(cosmo, k, a, c_mg=cosmo["c1_mg"])
    return out.squeeze()[()] if squeeze else out


def Sigma_MG(cosmo, k, a, *, squeeze=True):
    """Compute Σ(a, k), where Σ is one of the parametrizing functions
    of modifications to GR in the quasistatic approximation.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    k : float or array_like
        Wavenumber(s) in units of 1/Mpc.
    a : float or array_like
        Scale factor(s), normalized to 1 today.
    squeeze : bool
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.

    Returns
    -------
    Σ_MG : float or array_like
        Modification to Poisson equation under modified gravity.
    """
    out = _musigma(cosmo, k, a, c_mg=cosmo["c2_mg"])
    return out.squeeze()[()] if squeeze else out
