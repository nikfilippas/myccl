from .massdef import MassDef
from .hmfunc import MassFunc
from .hbias import HaloBias
from ..parameters import spline_params as sparams
from ..parameters import physical_constants as const
from ..integrate import IntegratorSamples
from ..pyutils import get_broadcastable

import numpy as np


__all__ = ("HMCalculator",)


class HMCalculator:
    r"""Implementation of methods used to compute various halo model quantities.
    A lot of these quantities involve integrals of the sort:

    .. math::

       \int \mathrm{d}M \, n(M, a) \, f(M, k, a),

    where :math:`n(M, a)` is the halo mass function, and :math:`f` is
    an arbitrary function of mass, scale factor and Fourier scales.

    Parameters
    ----------
    mass_function : str or :class:`~pyccl.halos.hmfunc.MassFunc`
        Mass function to use for the Halo Model calculations.
    halo_bias : str or :class:`~pyccl.halos.hbias.HaloBias`
        Halo bias to use for the Halo Model calculations.
    mass_def : :class:`~pyccl.halos.massdef.MassDef`
        Mass definition to use for the Halo Model calculations.
    k_large_scale : float
        Wavenumber (in :math:`\rm Mpc^{-1}`) indicating the effective
        'very large scale'. Used to normalize some Halo Model quantities.
        The default is 1e-5.
    """
    def __init__(self, mass_function, halo_bias, mass_def, k_large_scale=1e-5):
        # Initialize halo mass definition.
        if isinstance(mass_def, MassDef):
            self.mass_def = mass_def
        elif isinstance(mass_def, str):
            self.mass_def = MassDef.from_name(mass_def)
        else:
            raise TypeError("mass_def must be `MassDef` or a name string.")

        # Initialize halo mass function.
        if isinstance(mass_function, MassFunc):
            self.mass_function = mass_function
        elif isinstance(mass_function, str):
            nMclass = MassFunc.from_name(mass_function)
            self.mass_function = nMclass(mass_def=self.mass_def)
        else:
            raise TypeError("mass_function must be `MassFunc` or a name string.")  # noqa

        # Initialize halo bias function.
        if isinstance(halo_bias, HaloBias):
            self.halo_bias = halo_bias
        elif isinstance(halo_bias, str):
            bMclass = HaloBias.from_name(halo_bias)
            self.halo_bias = bMclass(mass_def=self.mass_def)
        else:
            raise TypeError("halo_bias must be `HaloBias` or a name string.")

        # Initialize precision parameters.
        self._k_large_scale = k_large_scale
        self._mass = np.geomspace(sparams.M_MIN, sparams.M_MAX, sparams.N_M)
        self._lmass = np.log10(self._mass)
        self._m0 = self._mass[0]

        # Initialize integration methods.
        self._integrator = IntegratorSamples("simpson")
        self._integrator2 = IntegratorSamples("symmetrical", "trapezoid")

        # Cache last results for mass function and halo bias.
        self._cosmo_mf = self._cosmo_bf = None
        self._a_mf = self._a_bf = -1

    def _get_mass_function(self, cosmo, a, ρ0):
        # Compute the mass function at this cosmo and a.
        cached_mf = cosmo == self._cosmo_mf and np.array_equal(a, self._a_mf)
        if not cached_mf:
            massfunc = self.mass_function.get_mass_function
            self._mf = massfunc(cosmo, self._mass, a, squeeze=False)
            self._mf = np.expand_dims(self._mf, axis=1)  # (a, [k], M)
            integ = self._integrator(self._mf*self._mass, self._lmass)
            self._mf0 = (ρ0 - integ) / self._m0
            self._cosmo_mf, self._a_mf = cosmo, a  # cache

    def _get_halo_bias(self, cosmo, a, ρ0):
        # Compute the halo bias at this cosmo and a.
        cached_bf = cosmo == self._cosmo_bf and np.array_equal(a, self._a_bf)
        if not cached_bf:
            hbias = self.halo_bias.get_halo_bias
            self._bf = hbias(cosmo, self._mass, a, squeeze=False)
            self._bf = np.expand_dims(self._bf, axis=1)  # (a, [k], M)
            integ = self._integrator(self._mf*self._bf*self._mass, self._lmass)
            self._mbf0 = (ρ0 - integ) / self._m0
            self._cosmo_bf, self._a_bf = cosmo, a  # cache

    def _get_ingredients(self, cosmo, a, *, get_bf):
        """Compute mass function and halo bias at some scale factor."""
        ρ0 = const.RHO_CRITICAL * cosmo["Omega_m"] * cosmo["h"]**2
        self._get_mass_function(cosmo, a, ρ0)
        if get_bf:
            self._get_halo_bias(cosmo, a, ρ0)

    def _integrate_over_mf(self, array_2, fast_integ=False):
        integrator = self._integrator2 if fast_integ else self._integrator
        i1 = integrator(self._mf * array_2, self._lmass)
        small_mass_contribution = self._mf0 * array_2[..., 0]
        return i1 + small_mass_contribution

    def _integrate_over_mbf(self, array_2):
        i1 = self._integrator(self._mf * self._bf * array_2, self._lmass)
        small_mass_contribution = self._mbf0 * array_2[..., 0]
        return i1 + small_mass_contribution

    def profile_norm(self, cosmo, a, prof):
        r"""Compute :math:`I^0_1(k \rightarrow 0, \, a|u)`.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        a : float or (na,) array_like
            Scale factor.
        prof : :class:`~pyccl.halos.profiles.HaloProfile`
            Halo profile.

        Returns
        -------
        norm : float or (na,) array_like
            Profile normalization.
        """
        return 1 / self.I_0_1(cosmo, self._k_large_scale, a, prof)

    def number_counts(self, cosmo, sel, solid_angle=1,
                      amin=None, amax=None, na=None):
        r"""Compute the number of clusters between some scale factors,
        and for some solid angle, given a selection function:

        .. math::

            \mathrm{n_c}(\mathrm{sel}) =
            \int \mathrm{d}M \int \mathrm{d}a \,
            \frac{\mathrm{d}V}{\mathrm{d}a \, \mathrm{d}\Omega} \,
            n(M, a) \, \mathrm{sel}(M, a),

        where :math:`n(M, a)` is the halo mass function, and
        :math:`\mathrm{sel}(M, a)` is the selection function.

        The selection function represents the probability per unit mass
        per unit scale factor and integrates to :math:`1`.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        sel : callable
            Selection function with signature ``sel(m, a)``,
            The function must be vectorized in both ``m`` and ``a``,
            and the output shape must be ``(na, nm)`` (not squeezed).
        solid_angle : float, optional
            Solid angle (:math:`\rm sr`) subtended in the sky for which
            the number of clusters is calculated.
            The default is :math:`1`, with the output being the number of
            clusters per unit solid angle.
        amin, amax : float, optional
            Minimum and maximum scale factors used in the selection function
            integrals. The defaults are taken from ``ccl.spline_params``,
            ``A_SPLINE_MIN`` and ``A_SPLINE_MAX``.
        na : int, optional
            Number of scale factor samples for the integrals.
            The default is taken from ``ccl.spline_params``,
            ``A_SPLINE_NLOG_PK + A_SPLINE_NA_PK - 1``.

        Returns
        -------
        n_c : float
            The total number of clusters.
        """
        if amin is None:
            amin = sparams.A_SPLINE_MIN
        if amax is None:
            amax = sparams.A_SPLINE_MAX
        if na is None:
            na = sparams.A_SPLINE_NLOG_PK + sparams.A_SPLINE_NA_PK - 1
        a = np.linspace(amin, amax, na)
        a, M = get_broadcastable(a, self._mass)

        self._get_ingredients(cosmo, a, False)
        dV = cosmo.comoving_volume_element(a, squeeze=False) * 1e9
        sel_m = sel(self._mass, a)
        M_int = self._integrator(dV * self._mf * sel_m, self._lmass)

        a.shape = M_int.shape
        return self._integrator(M_int, a)

    def I_0_1(self, cosmo, k, a, prof):
        r"""Compute the integral:

        .. math::

            I^0_1(k,a|u) = \int \mathrm{d}M \, n(M, a) \,
            \langle u(k, a|M) \rangle,

        where :math:`n(M, a)` is the halo mass function, and
        :math:`\langle u(k, a|M) \rangle` is the halo profile as a
        function of wavenumber, scale factor and halo mass.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        k : float or (nk,) array_like
            Comoving wavenumber in :math:`\rm Mpc^{-1}`.
        a : float or (na,) array_like
            Scale factor.
        prof : :class:`~pyccl.halos.profiles.HaloProfile`
            Halo profile.

        Returns
        -------
        I_0_1 : float or (na, nk)ndarray
            Integral value.
        """
        self._get_ingredients(cosmo, a, get_bf=False)
        uk = prof.fourier(cosmo, k, self._mass, a, squeeze=False)
        return self._integrate_over_mf(uk)

    def I_1_1(self, cosmo, k, a, prof):
        r"""Compute the integral:

        .. math::

            I^1_1(k, a|u) = \int \mathrm{d}M \, n(M, a) \, b(M, a) \,
            \langle u(k, a|M) \rangle,

        where :math:`n(M, a)` is the halo mass function, :math:`b(M, a)` is
        the halo bias function, and :math:`\langle u(k, a|M) \rangle` is the
        halo profile as a function of wavenumber, scale factor and halo mass.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        k : float or (nk,) array_like
            Comoving wavenumber in :math:`\rm Mpc^{-1}`.
        a : float or (na,) array_like
            Scale factor.
        prof : :class:`~pyccl.halos.profiles.HaloProfile`
            Halo profile.

        Returns
        -------
        I_1_1 : float or (na, nk) ndarray
            Integral value.
        """
        self._get_ingredients(cosmo, a, get_bf=True)
        uk = prof.fourier(cosmo, k, self._mass, a, squeeze=False)
        return self._integrate_over_mbf(uk)

    def I_0_2(self, cosmo, k, a, prof, prof2=None):
        r"""Compute the integral:

        .. math::

            I^0_2(k, a | u,v) = \int \mathrm{d}M \, n(M, a) \,
            \langle u(k, a|M) v(k, a|M) \rangle,

        where :math:`n(M, a)` is the halo mass function, and
        :math:`\langle u(k,a|M) v(k,a|M)\rangle` is the two-point
        moment of the two halo profiles.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        k : float or (nk,) array_like
            Comoving wavenumber in :math:`\rm Mpc^{-1}`.
        a : float or (na,) array_like
            Scale factor.
        prof, prof2 : :class:`~pyccl.halos.profiles.HaloProfile`
            Halo profiles. If ``prof2 is None``, ``prof`` will be used.

        Returns
        -------
        I_0_2 : float or (na, nk) ndarray
            Integral value.
        """
        self._get_ingredients(cosmo, a, get_bf=False)
        uk = prof.fourier_2pt(cosmo, k, self._mass, a, prof2=prof2,
                              squeeze=False)
        return self._integrate_over_mf(uk)

    def I_1_2(self, cosmo, k, a, prof, prof2=None):
        r"""Compute the integral:

        .. math::

            I^1_2(k, a|u,v) = \int \mathrm{d}M \, n(M, a) \, b(M, a) \,
            \langle u(k, a|M) v(k, a|M) \rangle,

        where :math:`n(M, a)` is the halo mass function, :math:`b(M, a)` is
        the halo bias, and :math:`\langle u(k,a|M) v(k,a|M) \rangle` is the
        two-point moment of the two halo profiles.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        k : float or (.nk,) array_like
            Comoving wavenumber in :math:`\rm Mpc^{-1}`.
        a : float or (na,) array_like
            Scale factor.
        prof, prof2 : :class:`~pyccl.halos.profiles.HaloProfile`
            Halo profiles. If ``prof2 is None``, ``prof`` will be used.

        Returns
        -------
        I_1_2 : float or (na, nk) ndarray
            Integral value.
        """
        self._get_ingredients(cosmo, a, get_bf=True)
        uk = prof.fourier_2pt(cosmo, k, self._mass, a, prof2)
        return self._integrate_over_mbf(uk)

    def I_0_22(self, cosmo, k, a, prof, prof2=None, prof3=None, prof4=None):
        r"""Compute the integral:

        .. math::

            I^0_{2,2}(k_u, k_v, a|u_{1,2},v_{1,2}) =
            \int \mathrm{d}M \, n(M, a) \,
            \langle u_1(k_u, a|M) u_2(k_u, a|M) \rangle
            \langle v_1(k_v, a|M) v_2(k_v, a|M) \rangle,

        where :math:`n(M,a)` is the halo mass function, and
        :math:`\langle u(k,a|M) v(k,a|M) \rangle` is the
        two-point moment of the two halo profiles.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        k : float or (nk,) array_like
            Comoving wavenumber in :math:`\rm Mpc^{-1}`.
        a : float or (na,) array_like
            Scale factor.
        prof, prof2, prof3, prof4 : :class:`~pyccl.halos.profiles.HaloProfile`
            Halo profiles.
            If ``prof2 is None``, ``prof`` will be used.
            If ``prof3 is None``, ``prof`` will be used.
            If ``prof4 is None``, ``prof2`` will be used.

        Returns
        -------
        I_0_22 : float or (na, nk, nk) ndarray
             Integral value.
        """
        if prof3 is None:
            prof3 = prof

        self._get_ingredients(cosmo, a, get_bf=False)
        uk12 = prof.fourier_2pt(cosmo, k, self._mass, a, prof2)
        if (prof, prof2) == (prof3, prof4):
            uk34 = uk12
        else:
            uk34 = prof3.fourier_2pt(cosmo, k, self._mass, a, prof4)

        uk12 = np.expand_dims(uk12, axis=2)            # (a, k, 1, M)
        uk34 = np.expand_dims(uk34, axis=1)            # (a, 1, k, M)
        shp_mf, shp_mf0 = self._mf.shape, self._mf0.shape
        self._mf = np.expand_dims(self._mf, axis=1)    # (a, 1, 1, M)
        self._mf0 = np.expand_dims(self._mf0, axis=1)  # (a, 1, 1)

        try:
            return self._integrate_over_mf(uk12 * uk34, fast_integ=True)
        finally:
            # back to original shape
            self._mf.shape = shp_mf
            self._mf0.shape = shp_mf0
