"""This script provides an interface with HMCode."""
from .pspec_base import PowerSpectrumNonLinear
from ..pk2d import Pk2D

import pyhmcode as hmcode
import numpy as np


class PowerSpectrumHMCode(PowerSpectrumNonLinear):
    """
    """
    name = "hmcode"

    def __init__(self, cosmo):
        super().__init__(cosmo=cosmo)

        # Set the linear power spectrum.
        pk2d = cosmo.get_linear_power()
        a, lk, pk = pk2d.get_spline_arrays()
        h = cosmo["h"]
        z, kh, pkh = 1/a-1, np.exp(lk)*h, pk*h**3

        c = self.get_pyhmcode_cosmology(cosmo)
        c.set_linear_power_spectrum(kh, z[::-1], pkh[::-1])
        self.hmcode_cosmo = c
        self._a_arr, self._lk_arr = a, lk

    @staticmethod
    def get_pyhmcode_cosmology(cosmo):
        """Get a pyhmcode-compatible cosmology."""
        out = hmcode.Cosmology()
        out.om_m = cosmo["Omega_m"]
        out.om_b = cosmo["Omega_b"]
        out.om_v = cosmo["Omega_l"] + cosmo["Omega_g"] + cosmo["Omega_nu_rel"]
        out.h = cosmo["h"]
        out.ns = cosmo["n_s"]
        out.sig8 = cosmo.sigma8()
        out.m_nu = cosmo["sum_nu_masses"]
        return out

    def get_power_spectrum(self, logT_AGN=7.8):
        c = self.hmcode_cosmo
        c.theat = 10**logT_AGN

        v = {"verbose": False}
        hmod = hmcode.Halomodel(hmcode.HMcode2020_feedback, **v)
        pk = hmcode.calculate_nonlinear_power_spectrum(c, hmod, **v)

        pk /= self.cosmo["h"]**3
        np.log(pk, out=pk)
        return Pk2D(a_arr=self._a_arr, lk_arr=self._lk_arr, pk_arr=pk[::-1],
                    is_logp=True, extrap_order_lok=1, extrap_order_hik=2)
