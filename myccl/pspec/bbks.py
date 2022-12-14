from .pspec_base import PowerSpectrumAnalytic
import numpy as np


class PowerSpectrumBBKS(PowerSpectrumAnalytic):
    """
    """
    name = "bbks"

    def __init__(self, cosmo):
        super().__init__(cosmo=cosmo)

    def _transfer(self, k):
        cosmo = self.cosmo
        q = ((cosmo["T_CMB"]/2.7)**2 * k
             / (cosmo["Omega_m"]*cosmo["h"]**2
                * np.exp(-cosmo["Omega_b"]
                         * (1. + np.sqrt(2*cosmo["h"]) / cosmo["Omega_m"]))))
        polynomial = (6.71*q)**4 + (5.46*q)**3 + (16.1*q)**2 + 3.89*q + 1
        return (np.log(1 + 2.34*q) / (2.34*q))**2 / np.sqrt(polynomial)

    def get_power_spectrum(self):
        k_arr = np.exp(self.lk_arr)
        pk_arr = k_arr**self.cosmo["n_s"] * self._transfer(k_arr)
        return self._get_full_analytic_power(pk_arr)
