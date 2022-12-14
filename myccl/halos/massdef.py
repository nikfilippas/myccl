from ..interpolate import Interpolator2D
from ..parameters import accuracy_params
from ..pyutils import get_broadcastable
import numpy as np
from scipy.optimize import newton


__all__ = ("MassDef", "MassDef200m", "MassDef200c", "MassDef500c",
           "MassDefVir", "MassDefFoF", "convert_concentration",
           "dc_NakamuraSuto", "Dv_BryanNorman")


def dc_NakamuraSuto(cosmo, a, *, squeeze=True):
    r"""Compute the peak threshold :math:`\delta_c(z)` assuming ΛCDM.

    Cosmology dependence of the critical linear density according to the
    spherical-collapse model. Fitting function from Nakamura & Suto (1997).

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
        Scale factor(s).
    squeeze : bool
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.

    Returns
    -------
    δ_c : float or (na,) ndarray
        Peak theshold at ``a``.
    """
    a = np.asarray(a)
    Om_mz = cosmo.omega_x(a, "matter", squeeze=False)
    dc0 = (3/20) * (12*np.pi)**(2/3)
    out = dc0 * (1 + 0.012299 * np.log10(Om_mz))
    return out.squeeze()[()] if squeeze else out


def Dv_BryanNorman(cosmo, a, *, squeeze=True):
    """Compute the virial collapse density contrast w.r.t. matter density,
    assuming ΛCDM.

    Cosmology dependence of the critical linear density according to the
    spherical-collapse model. Fitting function from Bryan & Norman (1997).
    :arXiv:`astro-ph/9710107`.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.c
    a : float or (na,) array_like
        Scale factor(s).
    squeeze : bool
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.

    Returns
    -------
    Δ_vir : float or (na,) ndarray
        Virial collapse density contrast w.r.t. matter density.
    """
    a = np.asarray(a)
    Om_mz = cosmo.omega_x(a, "matter", squeeze=False)
    x = Om_mz - 1
    Dv0 = 18 * np.pi * np.pi
    out = (Dv0 + 82*x - 39*x**2) / Om_mz
    return out.squeeze()[()] if squeeze else out


def convert_concentration(c_old, Delta_old, Delta_new, *, squeeze=True):
    r"""Compute the concentration parameter for a different mass definition.
    This is done assuming an NFW profile. The output concentration ``c_new``
    is found by solving the equation:

    .. math::

        f(c_{\rm old}) \Delta_{\rm old} = f(c_{\rm new}) \Delta_{\rm new}

    where

    .. math::

        f(x) = \frac{x^3}{\log(1+x) - x/(1+x)}.

    .. note::

        Vectorization for this function is specialized. It assumes that
        ``c_old`` has shape (na, nM) and ``Delta_old``, ``Delta_new`` both have
        shape (na,), so that the Δ-conversion only depends on scale factor.
        Every mass is only translated for its scale factor. The output has the
        same shape as the input, and all concentrations are translated with
        :math:`\frac{\Delta_{\rm old}}{\Delta_{\rm new}}`.

    Arguments
    ---------
    c_old : float or (na, nM,) array_like
        Concentration values to translate.
    Delta_old, Delta_new : float or (na,) array_like
        Overdensity (:math:`\Delta`) parameters associated with the
        halo mass definition of the old and new concentrations, respectively.
        See :class:`~pyccl.halos.massdef.MassDef` for details.
    squeeze : bool
        Squeeze extra dimensions of size (1,) in the output.
        The default is True.

    Returns
    -------
    c_new : float or ``np.shape(c_old)`` ndarray
        Translated concentration to the new mass definition.
    """
    c_old = np.asarray(c_old, dtype=float)
    c_use = c_old.ravel()  # `ravel` returns a view - 10x faster than `flatten`

    # Δ_fac repeats itself every nM elements in the flattened c_use array.
    Δ_fac = np.asarray(Delta_old / Delta_new)
    Δ_use = np.repeat(Δ_fac, c_use.size/Δ_fac.size)

    # If Δ_new == Δ_old, then c_new == c_old.
    idx = Δ_use != 1
    c_new = c_use.copy()

    _compute_concentration_conversion()
    # Evaluate the spline at zip(c_use, Δ_use).
    c_new[idx] = np.exp(_cM_spline(c_use[idx], Δ_use[idx], grid=False))
    c_new = c_new.reshape(c_old.shape)
    return c_new.squeeze()[()] if squeeze else c_new


# This is where the c(M) Δ-conversion spline is stored.
_cM_spline = None


def _compute_concentration_conversion():
    """Compute the c(M) Δ-conversion spline."""
    global _cM_spline
    if _cM_spline is not None:
        return

    c_old = np.geomspace(0.01, 30, 128)
    Δ_fac = np.geomspace(0.01, 20, 64)

    # Initial guess: Δ_old / Δ_new. || Second guess: 10x larger.
    x0 = np.ones_like(c_old)
    x1 = 10 * x0
    nfw = lambda x: x**3 / (np.log(x+1) - x / (x+1))                     # noqa
    func = lambda c_new, Δ_fac: nfw(c_new) - nfw(c_old) * Δ_fac          # noqa

    res = np.array([
        newton(func, x0=x0, x1=x1, args=(Δ,),
               maxiter=accuracy_params.N_ITERATION_ROOT,
               rtol=accuracy_params.EPSREL)
        for Δ in Δ_fac])

    _cM_spline = Interpolator2D(c_old, Δ_fac, np.log(res).T,
                                extrap_orders=[1, 1, 1, 1])


class MassDef:
    r"""Halo mass definition. Halo masses are defined in terms of an overdensity
    parameter :math:`\Delta` and an associated density :math:`X` (either the
    matter density or the critical density):

    .. math::

        M = \frac{4 \pi}{3} \Delta \, \rho_X \, R^3,

    where :math:`R` is the halo radius. This object also holds methods to
    translate between :math:`R` and :math:`M`, and to translate masses between
    different definitions if a concentration-mass relation is provided.

    Parameters
    ----------
    Delta : float or {'fof', 'vir'}
        Spherical overdensity (S.O.) parameter. ``'fof'`` for friends-of-
        friends masses and ``'vir'`` for Virial masses.
    rho_type : {'critical', 'matter'}
        Associated reference mean density.
    concentration : None, str, :obj:`~pyccl.halos.concentration.Concentration`
        Concentration-mass relation. Provided either as a name string,
        or as a ``Concentration`` object. If ``None``, the mass definition
        object is unable to be translated to another mass definition.
    """

    def __init__(self, Delta, rho_type, concentration=None):
        if Delta not in ["fof", "vir"]:
            if not isinstance(Delta, (int, float)) or Delta <= 0:
                raise ValueError(f"Can't parse Delta = {Delta}.")
        if rho_type not in ['matter', 'critical']:
            raise ValueError("rho_type must be either 'matter' or 'critical'.")

        self.Delta = Delta
        self.rho_type = rho_type
        self._initialize_concentration(concentration)

    @property
    def name(self):
        """Give a name to this mass definition."""
        if isinstance(self.Delta, (int, float)):
            return f"{self.Delta}{self.rho_type[0]}"
        return f"{self.Delta}"

    def __eq__(self, other):
        return (self.Delta, self.rho_type) == (other.Delta, other.rho_type)

    def _initialize_concentration(self, concentration):
        # Associate a concentration to this mass definition.
        if concentration is None:
            self.concentration = None
            return

        from .concentration import Concentration
        if isinstance(concentration, Concentration):
            self.concentration = concentration
            return

        if isinstance(concentration, str):
            cM = Concentration.from_name(concentration)
            self.concentration = cM(mass_def=self)
            return

        raise ValueError("concentration must be `None`, "
                         "a string, or a `Concentration` object.")

    def get_Delta(self, cosmo, a, *, squeeze=True):
        """Compute the overdensity parameter for this mass definition.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.c
        a : float or (na,) array_like
            Scale factor(s).
        squeeze : bool
            Squeeze extra dimensions of size (1,) in the output.
            The default is True.

        Returns
        -------
        Delta : float or (na,) ndarray
            Overdensity parameter at ``a``.
        """
        if self.Delta == 'fof':
            raise ValueError("FoF masses have no associated overdensity, "
                             "and can't be translated into other masses.")
        if self.Delta == 'vir':
            return Dv_BryanNorman(cosmo, a, squeeze=squeeze)
        out = self.Delta * np.ones_like(a)
        return out.squeeze()[()] if squeeze else out

    def get_radius(self, cosmo, M, a, *, squeeze=True):
        r"""Compute the radius corresponding to the input masses and this
        mass definition.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        M : float or (nM,) array_like
            Halo mass in :math:`\rm M_\odot`.
        a : float or (na,) array_like
            Scale factor(s).
        squeeze : bool, optional
            Squeeze extra dimensions of size (1,) in the output.
            The default is True.

        Returns
        -------
        R : float or (na, nM,) ndarray
            Halo radius in physical :math:`\rm Mpc`.
        """
        return cosmo.m2r(M, a,
                         Delta=self.get_Delta(cosmo, a, squeeze=False),
                         species=self.rho_type,
                         comoving=False,
                         Delta_vectorized=False,
                         squeeze=squeeze)

    def translate_mass(self, cosmo, M, a, mass_def_other, *, squeeze=True):
        r"""Translate halo mass in this definition into another definition.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        M : float or (nM,) array_like
            Halo mass in :math:`\rm M_\odot`.
        a : float or (na,) array_like
            Scale factor(s).
        mass_def_other : :class:`~pyccl.halos.massdef.MassDef`
            Mass definition to translate to.
        squeeze : bool, optional
            Squeeze extra dimensions of size (1,) in the output.
            The default is True.

        Returns
        -------
        M_translated : float or (na, nM,) ndarray
            Halo masses in new definition.
        """
        if self.concentration is None:
            raise ValueError("Mass definition has no associated c(M).")

        a, M = map(np.atleast_1d, [a, M])
        a, M = get_broadcastable(a, M)
        if self == mass_def_other:
            shp = np.broadcast_shapes(a.shape, M.shape)
            M = np.broadcast_to(M, shp)
            return M

        kw = {"squeeze": False}  # never squeeze internal function output

        Ω_old = cosmo.omega_x(a, self.rho_type, **kw)
        ρ_old = cosmo.rho_x(a, self.rho_type, **kw)
        Δ_old = self.get_Delta(cosmo, a, **kw) * Ω_old
        c_old = self.concentration.get_concentration(cosmo, M, a, **kw)
        R_old = (3 * M / (4 * np.pi * Δ_old * ρ_old))**(1/3)

        Ω_new = cosmo.omega_x(a, mass_def_other.rho_type, **kw)
        ρ_new = cosmo.rho_x(a, mass_def_other.rho_type, **kw)
        Δ_new = mass_def_other.get_Delta(cosmo, a, **kw) * Ω_new
        c_new = convert_concentration(c_old, Δ_old, Δ_new, **kw)
        R_new = R_old * c_new / c_old
        M_new = (4 * np.pi / 3) * Δ_new * ρ_new * R_new**3

        return M_new.squeeze()[()] if squeeze else M_new

    @classmethod
    def from_name(cls, name):
        r"""Return a mass definition from name string.

        Arguments
        ---------
        name : string
            A mass definition name (e.g. ``'200m'`` for :math:`\Delta=200_m`).

        Returns
        -------
        mass_def : :obj:`~pyccl.halos.massdef.MassDef`
            Mass definition corresponding to the input name.
        """
        if name == "fof":
            return MassDef("fof", "matter")
        if name == "vir":
            return MassDef("vir", "critical")

        parser = {"m": "matter", "c": "critical"}
        Δ, ρ = name[:-1], name[-1]
        return MassDef(int(Δ), parser[ρ])


def MassDef200m():
    r""":math:`\Delta = 200m` mass definition."""
    return MassDef(200, "matter", concentration="Duffy08")


def MassDef200c():
    r""":math:`\Delta = 200c` mass definition."""
    return MassDef(200, "critical", concentration="Duffy08")


def MassDef500c():
    r""":math:`\Delta = 500m` mass definition."""
    return MassDef(500, "critical", concentration="Ishiyama21")


def MassDefVir():
    r""":math:`\Delta = \rm vir` mass definition."""
    return MassDef("vir", "critical", concentration="Klypin11")


def MassDefFoF():
    r""":math:`\Delta = \rm FoF` mass definition."""
    return MassDef("fof", "matter")
