from numpy import pi, log, log10, ceil, linspace
from inspect import isfunction


__all__ = ("physical_constants", "accuracy_params", "spline_params")


class CCLParameters:
    """Base for classes that gold global CCL parameters and their values.

    Parameters
    ----------
    freeze : bool
        Disable parameter mutation.
    """

    def __init_subclass__(cls, freeze=False):
        super().__init_subclass__()
        cls._frozen = freeze

    def __init__(self):
        # Instances use a copy of the parameters in the class dictionary.
        self_dict = {p: v for p, v in self.__class__.__dict__.items()
                     if not p.startswith("_") and not isfunction(v)}
        self.__dict__.update(self_dict)

    def __setattr__(self, param, value):
        if self._frozen:
            name = self.__class__.__name__
            raise AttributeError(f"Instances of {name} are frozen.")
        if not hasattr(self, param):
            raise AttributeError(f"Parameter {param} does not exist.")

        if param == "A_SPLINE_MAX" and value < 1:
            raise ValueError("A_SPLINE_MAX should be >= 1.")
        self.__dict__[param] = value

    def __getattr__(self, name):
        return self.__dict__[name]

    __setitem__ = __setattr__

    def __repr__(self):
        return repr(self.__dict__)

    def reload(self):
        """Reload the default CCL parameters."""
        bak = dict(self.__class__.__dict__)
        for param in self.__dict__:
            self.__dict__[param] = bak[param]


class _PhysicalConstants(CCLParameters, freeze=True):
    """This class holds physical constants."""
    # ~~ ASTRONOMICAL CONSTANTS ~~ #
    # Astronomical Unit, unit conversion (m/au). [exact]
    AU = 149_597_870_800
    # Mean solar day (s/day). [exact]
    DAY = 86400.
    # Sidereal year (days/yr). [IERS2014 in J2000.0]
    YEAR = 365.256_363_004 * DAY

    # ~~ FUNDAMENTAL PHYSICAL CONSTANTS ~~ #
    # Speed of light (m/s). [exact]
    CLIGHT = 299_792_458.
    # Unit conversion (J/eV). [exact]
    EV_IN_J = 1.602_176_634e-19
    ELECTRON_CHARGE = EV_IN_J
    # Electron mass (kg). [CODATA2018]
    ELECTRON_MASS = 9.109_383_7015e-31
    # Planck's constant (J s). [exact]
    HPLANCK = 6.626_070_15e-34
    # Boltzmann's constant (J/K). [exact]
    KBOLTZ = 1.380_649e-23
    # Universal gravitational constant (m^3/kg/s^2). [CODATA2018]
    GNEWT = 6.674_30e-11
    # Vacuum permeability (N / A^2). [CODATA2018]
    MU_0 = 1.256_637_062_12e-6
    # Effective number of massless neutrinos. [2008.01074, 2012.02726]
    NEFF = 3.044

    # ~~ DERIVED CONSTANTS ~~ #
    # Reduced Planck's constant (J s).
    HBAR = HPLANCK / 2 / pi
    # Vacuum permittivity (F / m).
    EPSILON_0 = 1 / (MU_0 * CLIGHT**2)
    # Fine structure constant (~ 1/137).
    ALPHA = ELECTRON_CHARGE**2 / (4*pi*EPSILON_0*HBAR*CLIGHT)
    # Thomson scattering cross-section for the electron (m^2).
    SIG_THOMSON = (8*pi/3) * (ALPHA*HBAR*CLIGHT / (ELECTRON_MASS*CLIGHT**2))**2
    # Speed of light (Mpc/h).
    CLIGHT_HMPC = CLIGHT / 1e5
    # Unit conversion (m/pc).
    PC_TO_METER = 180*60*60/pi * AU
    # Unit conversion (m/Mpc).
    MPC_TO_METER = 1e6 * PC_TO_METER
    # Stefan-Boltzmann's constant (kg m^2 / s).
    STBOLTZ = (pi**2/60) * KBOLTZ**4 / HBAR**3 / CLIGHT**2
    # Solar mass in (kg).
    SOLAR_MASS = 4 * pi*pi * AU**3 / GNEWT / YEAR**2
    # Critical density (100 M_sun/h / (Mpc/h)^3).
    RHO_CRITICAL = 3*1e4/(8*pi*GNEWT) * 1e6 * MPC_TO_METER / SOLAR_MASS
    # Neutrino constant required in Omeganuh2.
    NU_CONST = (8*pi**5 * (KBOLTZ/HPLANCK)**3 * (KBOLTZ/15/CLIGHT**3)
                * (8*pi*GNEWT/3) * (MPC_TO_METER**2/CLIGHT**2/1e10))
    # Linear density contrast of spherical collapse.
    DELTA_C = (3/20) * (12*pi)**(2/3)

    # ~~ OTHER CONSTANTS ~~ #
    # Neutrino mass splitting differences.
    # Lesgourgues & Pastor (2012)
    # Adv. High Energy Phys. 2012 (2012) 608515
    # arXiv:1212.6154, p.13
    DELTAM12_sq = 7.62e-5
    DELTAM13_sq_pos = 2.55e-3
    DELTAM13_sq_neg = -2.43e-3


class _AccuracyParams(CCLParameters):
    """Instances of this class hold the accuracy parameters."""
    # ~~ INTEGRATION ERRORS ~~ #
    N_SPLINE = 256
    EPSREL = 1e-4
    EPSREL_GROWTH = 1e-6
    EPSREL_KNL = 1e-5
    EPSREL_LIMBER = 1e-4
    EPSREL_SIGMAR = 1e-5

    N_ITERATION = 1000
    N_ITERATION_ROOT = N_ITERATION


class _SplineParams(CCLParameters):
    """Instances of this class hold the spline accuracy parameters."""
    # Total number of sampling points from vmin, vmax, and N per decade.
    _Ntot = lambda vmin, vmax, vdex: int(ceil(log10(vmax/vmin) * vdex))  # noqa

    # ~~ GENERIC a SPLINES ~~ #
    A_SPLINE_MINLOG = 0.0001
    A_SPLINE_MIN = 0.1
    A_SPLINE_MAX = 1.0
    A_SPLINE_NLOG = 250
    A_SPLINE_NA = 250

    # ~~ GENERIC k SPLINES ~~ #
    K_MAX_CLASS = 50.0  # only for CLASS
    K_MIN = 5e-05
    K_MAX = 1000.0
    N_K_PER_DECADE = 167
    N_K = _Ntot(K_MIN, K_MAX, N_K_PER_DECADE)

    # ~~ P(k, a) SPLINES ~~ #
    A_SPLINE_MINLOG_PK = 0.01
    A_SPLINE_MIN_PK = 0.1
    A_SPLINE_NLOG_PK = 11
    A_SPLINE_NA_PK = 40

    # ~~ σ(M) SPLINES ~~ #
    A_SPLINE_MINLOG_SM = 0.01
    A_SPLINE_MIN_SM = 0.1
    A_SPLINE_NLOG_SM = 6
    A_SPLINE_NA_SM = 13

    # ~~ MASSES (HALO MODEL & σ(M) SPLINES) ~~ #
    M_MIN = 1e6
    M_MAX = 1e17
    N_M_PER_DECADE = 15
    N_M = _Ntot(M_MIN, M_MAX, N_M_PER_DECADE)

    # ~~ CORRELATIONS ~~ #
    ELL_MIN_CORR = 0.01
    ELL_MAX_CORR = 60000.0
    N_ELL_CORR = 5000
    N_K_3DCOR = 100000

    # ~~ C(ℓ) ANGULAR POWER SPECTRA ~~ #
    ELL_CLS_MIN = 1
    ELL_CLS_MINLOG = 15
    ELL_CLS_MAX = 1000
    ELL_CLS_NLIN = 20
    ELL_CLS_NLOG = 50

    DCHI = 5.0
    DLOGK = 0.025

    # ~~ NEUTRINO PHASESPACE INTEGRAL SPLINE (mass/temperature & momentum) ~~ #
    NU_MNUT_MIN = 1e-4
    NU_MNUT_MAX = 500
    NU_MNUT_N = 1000

    NU_MOM_MIN = 0
    NU_MOM_MAX = 1000
    NU_MOM_N = 256

    # ~~ TRACER KERNELS ~~ #
    N_CHI = 128

    def get_pk_spline_a(self):
        """Get a sampling a-array. Used for P(k) splines."""
        from .interpolate import loglin_spacing
        return loglin_spacing(self.A_SPLINE_MINLOG_PK,
                              self.A_SPLINE_MIN_PK,
                              self.A_SPLINE_MAX,
                              self.A_SPLINE_NLOG_PK,
                              self.A_SPLINE_NA_PK)

    def get_pk_spline_lk(self):
        """Get a sampling log(k)-array. Used for P(k) splines."""
        return linspace(log(self.K_MIN), log(self.K_MAX), self.N_K)

    def get_sm_spline_a(self):
        """Get a sampling a-array. Used for σ(M) splines."""
        from .interpolate import loglin_spacing
        return loglin_spacing(self.A_SPLINE_MINLOG_SM,
                              self.A_SPLINE_MIN_SM,
                              self.A_SPLINE_MAX,
                              self.A_SPLINE_NLOG_SM,
                              self.A_SPLINE_NA_SM)

    def get_sm_spline_lm(self):
        """Get a sampling M-array. Used for σ(M) splines."""
        return linspace(log10(self.M_MIN), log10(self.M_MAX), self.N_M)


physical_constants = _PhysicalConstants()
accuracy_params = _AccuracyParams()
spline_params = _SplineParams()


class _FFTLogParams:
    """Objects of this class store the FFTLog accuracy parameters."""
    padding_lo_fftlog = 0.1   # | Anti-aliasing: multiply the lower boundary.
    padding_hi_fftlog = 10.   # |                multiply the upper boundary.

    n_per_decade = 100        # Samples per decade for the Hankel transforms.
    extrapol = "linx_lny"     # Extrapolation type.

    padding_lo_extra = 0.1    # Padding for the intermediate step of a double
    padding_hi_extra = 10.    # transform. Doesn't have to be as precise.
    large_padding_2D = False  # If True, high precision intermediate transform.

    plaw_fourier = -1.5       # Real <--> Fourier transforms.
    plaw_projected = -1.0     # 2D projected & cumulative density profiles.

    def __getitem__(self, name):
        return getattr(self, name)

    def __setitem__(self, name, value):
        setattr(self, name, value)

    def update_parameters(self, **kwargs):
        """Update the precision of FFTLog for the Hankel transforms.

        Arguments
        ---------
        padding_lo_fftlog, padding_hi_fftlog : float
            Multiply the lower and upper boundary of the input range
            to avoid aliasing. The defaults are 0.1 and 10.0, respectively.
        n_per_decade : float
            Samples per decade for the Hankel transforms.
            The default is 100.
        extrapol : {'linx_liny', 'linx_logy'}
            Extrapolation type when FFTLog has narrower output support.
            The default is 'linx_liny'.
        padding_lo_extra, padding_hi_extra : float
            Padding for the intermediate step of a double Hankel transform.
            Used to compute the 2D projected profile and the 2D cumulative
            density, where the first transform goes from 3D real space to
            Fourier, then from Fourier to 2D real space. Usually, it doesn't
            have to be as precise as ``padding_xx_fftlog``.
            The defaults are 0.1 and 10.0, respectively.
        large_padding_2D : bool
            Override ``padding_xx_extra`` in the intermediate transform,
            and use ``padding_xx_fftlog``. The default is False.
        plaw_fourier, plaw_projected : float
            FFTLog pre-whitens its arguments (makes them flatter) to avoid
            aliasing. The ``plaw`` parameters describe the tilt of the profile,
            :math:`P(r) \\sim r^{\\mathrm{tilt}}`, between real and Fourier
            transforms, and between 2D projected and cumulative density,
            respectively. Subclasses of ``HaloProfile`` may obtain finer
            control via ``_get_plaw_[fourier | projected]``, and some level of
            experimentation with these parameters is recommended.
            The defaults are -1.5 and -1.0, respectively.
        """
        for name, value in kwargs.items():
            setattr(self, name, value)

    def _get_plaw(self, cosmo, a, name):
        """Fine tine ``plaw_fourier`` and ``plaw_projected``,
        used by ``FFTLog``. This function implements default behavior.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        a : float or array_like
            Scale factor.

        Returns
        -------
        plaw : float or ndarray
            Power law index.
        """
        return self.precision_fftlog[name]
