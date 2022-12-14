__all__ = ("halomod_mean_profile_1pt", "halomod_bias_1pt")


def halomod_mean_profile_1pt(cosmo, hmc, k, a, prof, *, squeeze=True):
    r"""Mass-weighted mean halo profile.

    .. math::

        I^0_1(k, a|u) = \int \mathrm{d}M \, n(M, a) \,
        \langle u(k, a|M) \rangle,

    where :math:`n(M, a)` is the halo mass function, and
    :math:`\langle u(k, a|M)\rangle` is the halo profile as a
    function of scale, scale factor and halo mass.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    hmc : :class:`HMCalculator`
        Halo model calculator associated with the profile calculations.
    k : float or (nk,) array_like
        Comoving wavenumber in :math:`\rm Mpc^{-1}`.
    a : float or (na,) array_like
        Scale factor.
    prof : :class:`~pyccl.halos.profiles.HaloProfile`
        Halo profile.
    squeeze : bool, optional
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.

    Returns
    -------
    i01 : float or (na, nk,) ndarray
        Mean halo profile (mass-weighted).
    """
    i01 = hmc.I_0_1(cosmo, k, a, prof)
    if prof.normprof:
        i01 *= hmc.profile_norm(cosmo, a, prof)
    return i01.squeeze()[()] if squeeze else i01


def halomod_bias_1pt(cosmo, hmc, k, a, prof, *, squeeze=True):
    r"""Mass-and-bias-weighted mean halo profile.

    .. math::

        I^1_1(k, a|u) = \int \mathrm{d}M \, n(M, a) \, b(M, a) \,
        \langle u(k, a|M) \rangle,

    where :math:`n(M, a)` is the halo mass function, :math:`b(M,a)` is the
    halo bias, and :math:`\langle u(k,a|M)\rangle` is the halo profile as a
    function of scale, scale factor and halo mass.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    hmc : :class:`HMCalculator`
        Halo model calculator associated with the profile calculations.
    k : float or (nk,) array_like
        Comoving wavenumber in :math:`\rm Mpc`.
    a : float or (na,) array_like
        Scale factor.
    prof : :class:`~pyccl.halos.profiles.HaloProfile`
        Halo profile.
    squeeze : bool, optional
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.

    Returns
    -------
    i11 : float or (na, nk,) ndarray
        Mean halo profile (mass-and-bias weighted).
    """
    i11 = hmc.I_1_1(cosmo, k, a, prof)
    if prof.normprof:
        i11 *= hmc.profile_norm(cosmo, a, prof)
    return i11.squeeze()[()] if squeeze else i11
