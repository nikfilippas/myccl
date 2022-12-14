from ..pk2d import Pk2D
from ..parameters import spline_params as sparams
from ..errors import CCLWarning

import numpy as np
import warnings
from abc import ABC, abstractproperty, abstractmethod
from enum import Enum


class TransferFunctions(Enum):
    NONE = None
    EISENSTEIN_HU = 'eisenstein_hu'
    EISENSTEIN_HU_NOWIGGLES = 'eisenstein_hu_nowiggles'
    BBKS = 'bbks'
    CLASS = 'class'
    CAMB = 'camb'
    ISITGR = 'isitgr'
    CALCULATOR = 'calculator'


class MatterPowerSpectra(Enum):
    NONE = None
    HALOFIT = 'HALOFIT'
    HMCODE = 'hmcode'
    LINEAR = 'linear'
    EMU = 'emu'
    CAMB = 'camb'
    CALCULATOR = 'calculator'


class BaryonPowerSpectra(Enum):
    NONE = 'nobaryons'
    BCM = 'bcm'


def rescale_power_spectrum(cosmo, pk, rescale_mg=False, rescale_s8=False):
    """
    """
    rescale_mg = rescale_mg and cosmo["mu_0"] > 1e-14
    rescale_s8 = rescale_s8 and cosmo["A_s"] is None
    if not (rescale_mg or rescale_s8):
        return pk

    a_arr, lk_arr, pk_arr = pk.get_spline_arrays()
    rescale_extra_musigma = 1.
    rescale_factor = np.ones_like(a_arr, dtype=float)

    # If scale-independent mu/Sigma modified gravity is in use and mu != 0,
    # get the unnormalized growth factor in MG and the one for the GR case
    # to rescale the CLASS power spectrum.
    if rescale_mg:
        # Set up a copy Cosmology in GR (mu_0 = Sigma_0 = 0) to rescale P(k).

        cosmo_GR = cosmo.copy()
        cosmo_GR.__dict__.update(
            {"mu_0": 0, "sigma_0": 0, "c1_mg": 1, "c2_mg": 1, "lambda_mg": 0})

        D_MG = cosmo.growth_factor_unnorm(a_arr)
        D_GR = cosmo_GR.growth_factor_unnorm(a_arr)
        rescale_factor = (D_MG / D_GR)**2
        rescale_extra_musigma = rescale_factor[-1]

    if rescale_s8:
        renorm = (cosmo["sigma8"] / cosmo.sigma8(pk))**2
        renorm /= rescale_extra_musigma
        rescale_factor *= renorm

    pk_arr *= rescale_factor[:, None]
    if pk.is_logp:
        np.log(pk_arr, out=pk_arr)

    return Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk_arr,
                is_logp=pk.is_logp,
                extrap_order_lok=pk.extrap_order_lok,
                extrap_order_hik=pk.extrap_order_hik)


class PowerSpectrum(ABC):
    """
    """

    def __init__(self, cosmo):
        self.cosmo = cosmo
        self.a_arr = sparams.get_pk_spline_a()
        self.lk_arr = sparams.get_pk_spline_lk()

    @abstractproperty
    def rescale_s8(self):
        """Boolean to indicate whether to perform σ8 rescaling to the
        power spectrum (output of ``get_power_spectrum``).
        """

    @abstractproperty
    def rescale_mg(self):
        """Boolean to indicate whether to perform MG rescaling to the
        power spectrum (output of ``get_power_spectrum``).
        """

    @abstractmethod
    def get_power_spectrum(self):
        """Return a :obj:`~pyccl.pk2d.Pk2D` object of the power spectrum."""

    def apply_model(self, pk):
        """Apply a model onto an input power spectrum."""

    @classmethod
    def _subclasses(cls):
        return set(cls.__subclasses__()).union(
            [sub for cl in cls.__subclasses__() for sub in cl._subclasses()])

    @classmethod
    def from_model(cls, model):
        """
        """
        power_spectra = {p.name: p for p in cls._subclasses()}
        return power_spectra[model]


class PowerSpectrumAnalytic(PowerSpectrum):
    """
    """
    name = "analytic_base"
    rescale_mg = False
    rescale_s8 = False

    def __init__(self, cosmo):
        super().__init__(cosmo=cosmo)
        if cosmo["sigma8"] is None:
            raise ValueError("sigma8 required for analytic power spectra.")
        if cosmo["N_nu_mass"] > 0:
            warnings.warn(f"{self.name} does not properly account for "
                          "massive neutrinos.", CCLWarning)

    def _get_full_analytic_power(self, pk):
        """Expand an analytic P(k) into the time-dimension, scaling by the
        growth factor, and rescale sigma8.
        """
        out = np.full((self.a_arr.size, self.lk_arr.size), np.log(pk))
        out += 2*np.log(self.cosmo.growth_factor(self.a_arr))[:, None]
        pk_out = Pk2D(a_arr=self.a_arr, lk_arr=self.lk_arr, pk_arr=out,
                      is_logp=True, extrap_order_lok=1, extrap_order_hik=2)
        return rescale_power_spectrum(self.cosmo, pk_out, rescale_s8=True)


class PowerSpectrumNonLinear(PowerSpectrum):
    """
    """
    name = "nonlinear_base"
    rescale_s8 = False
    rescale_mg = False
