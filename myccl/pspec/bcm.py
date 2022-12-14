from .pspec_base import PowerSpectrumNonLinear
from ..pk2d import Pk2D
from ..errors import CCLWarning

import numpy as np
import warnings


class PowerSpectrumBCM(PowerSpectrumNonLinear):
    """The 'Baryonic Correction Model' (BCM) of Schneider & Teyssier (2015),
    https://arxiv.org/abs/1510.06034).
    """
    name = "bcm"

    def __init__(self, cosmo, log10Mc=np.log10(1.2e14), etab=0.5, ks=55.):
        if cosmo["N_nu_mass"] > 0:
            warnings.warn("BCM is not calibrated for massive neutrinos.",
                          CCLWarning)

        params = {"log10Mc": log10Mc, "etab": etab, "ks": ks}
        cosmo_params = cosmo.get_extra_parameters(model=self.name)
        params.update(cosmo_params)
        self.__dict__.update(params)
        super().__init__(cosmo=cosmo)

    def _fka(self, k, a):
        """The BCM model correction factor for baryons.

        .. math::

            P_{\\rm new}(k, a) = P(k, a)\\, f_{\\rm bcm}(k, a)

        Arguments
        ---------
        k : float or (nk,) array_like
            Wavenumber (:math:`\\mathrm{Mpc}^{-1}`).
        a : float or (na,) array_like:
            Scale factor(s), normalized to 1 today.

        Returns
        -------
        fka : float or (na, nk) array_lke
            BCM correction factor at the grid defined by ``a`` and ``k``.
        """
        a, k = map(np.atleast_1d, [a, k])
        a, k = a[:, None], k[None, :]

        z = 1 / a - 1
        kh = k / self.cosmo["h"]
        b0 = 0.105 * self.log10Mc - 1.27
        bfunc = b0 / (1. + (z/2.3)**2.5)
        kg = 0.7 * (1-bfunc)**4 * self.etab**(-1.6)
        gf = bfunc / (1. + (kh/kg)**3) + 1. - bfunc  # k [h/Mpc]
        return gf * (1 + (kh / self.ks)**2)

    def get_power_spectrum(self):
        pk = self._fka(np.exp(self.lk_arr), self.a_arr)
        return Pk2D(a_arr=self.a_arr, lk_arr=self.lk_arr, pk_arr=np.log(pk),
                    is_logp=True, extrap_order_lok=0, extrap_order_hik=0)

    def apply_model(self, pk):
        a_arr, lk_arr, pk_arr = pk.get_spline_arrays()
        pk_out = pk_arr * self._fka(np.exp(lk_arr), a_arr).T
        is_log = pk.is_logp
        if is_log:
            np.log(pk_out, out=pk_out)
        return Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk_out, is_logp=is_log,
                    extrap_order_lok=pk.extrap_order_lok,
                    extrap_order_hik=pk.extrap_order_hik)
