from .pspec_base import PowerSpectrum
from ..pk2d import Pk2D
from ..parameters import spline_params as sparams
from ..parameters import physical_constants as const
from ..errors import CCLError, CCLWarning

import numpy as np
import warnings
try:
    import isitgr  # noqa: F401
except ModuleNotFoundError:
    pass  # prevent nans from isitgr


class PowerSpectrumOnCAMB(PowerSpectrum):
    """Base for Boltzmann solvers based on CAMB (e.g. CAMB, ISiTGR)."""
    name = "camb_base"

    def _setup(self, package, nonlin, dark_energy_model, kmax, lmax):
        """Set up the parameters and initialize the power spectrum."""
        cosmo = self.cosmo
        cp = package.model.CAMBparams()

        # Configuration.
        cp.WantCls = False
        cp.DoLensing = False
        cp.Want_CMB = False
        cp.Want_CMB_lensing = False
        cp.Want_cl_2D_array = False
        cp.WantTransfer = True

        # Background parameters.
        h2 = cosmo['h']**2
        cp.H0 = cosmo['h'] * 100
        cp.ombh2 = cosmo['Omega_b'] * h2
        cp.omch2 = cosmo['Omega_c'] * h2
        cp.omk = cosmo['Omega_k']
        cp.TCMB = cosmo['T_CMB']

        # Neutrinos.
        # We maually setup the CAMB neutrinos to match
        # the adjustments CLASS makes to their temperatures.
        cp.share_delta_neff = False
        cp.omnuh2 = cosmo['Omega_nu_mass'] * h2
        cp.num_nu_massless = cosmo['N_nu_rel']
        cp.num_nu_massive = int(cosmo['N_nu_mass'])
        cp.nu_mass_eigenstates = int(cosmo['N_nu_mass'])
        delta_neff = cosmo['Neff'] - const.NEFF  # used for BBN YHe comps

        # * CAMB defines a neutrino degeneracy factor as `T_i = g^(1/4)*T_nu`
        #   where `T_nu` is the standard neutrino temperature from first order
        #   computations.
        # * CLASS defines the temperature of each neutrino species to be
        #   `T_i_eff = T_ncdm * T_cmb` where `T_ncdm` is a fudge factor to get
        #   the total mass in terms of eV to match second-order computations
        #   of the relationship between m_nu and Omega_nu.
        # * Trying to get the codes to use the same neutrino temperature,
        #   we set `T_i_eff = T_i = g^(1/4) * T_nu` and solve for the right
        #   value of g for CAMB. We get `g = (T_ncdm / (11/4)^(-1/3))^4`.
        g = (cosmo["T_ncdm"] / (11/4)**(-1/3))**4

        if cosmo['N_nu_mass'] > 0:
            nu_mass_fracs = cosmo['m_nu'][:cosmo['N_nu_mass']]
            nu_mass_fracs = nu_mass_fracs / np.sum(nu_mass_fracs)

            cp.nu_mass_numbers = np.ones(cosmo['N_nu_mass'], dtype=np.int)
            cp.nu_mass_fractions = nu_mass_fracs
            cp.nu_mass_degeneracies = np.ones(int(cosmo['N_nu_mass'])) * g

        # YHe from BBN.
        cp.bbn_predictor = package.bbn.get_predictor()
        Tr = package.constants.COBE_CMBTemp / cp.TCMB
        cp.YHe = cp.bbn_predictor.Y_He(cp.ombh2 * (Tr) ** 3, delta_neff)

        # Dark energy.
        camb_de_models = ['DarkEnergyPPF', 'ppf', 'DarkEnergyFluid', 'fluid']
        if dark_energy_model not in camb_de_models:
            raise ValueError("CCL only supports fluid and ppf dark energy with"
                             f"{package.__name__}.")
        cp.set_classes(dark_energy_model=dark_energy_model)

        is_ppf = "ppf" in dark_energy_model.lower()
        w0, wa, eps = cosmo["w0"], cosmo["wa"], 1e-6
        if not is_ppf and wa != 0 and (w0 < -(1+eps) or 1+w0+wa < -eps):
            raise ValueError("For w < -1, use the 'ppf' dark energy model.")
        cp.DarkEnergy.set_params(w=cosmo['w0'], wa=cosmo['wa'])

        # Initialize power spectrum.
        zs = 1.0 / self.a_arr - 1
        zs = zs[zs >= 0]
        cp.set_matter_power(redshifts=zs.tolist(), kmax=kmax, nonlinear=nonlin,
                            silent=True)

        # Power spectrum normalization.
        # If A_s is not given, we just get close and CCL will normalize it.
        A_s, σ8 = cosmo["A_s"], cosmo["sigma8"]
        A_s_fid = A_s if A_s is not None else 2.43e-9 * (σ8 / 0.87659)**2

        cp.set_for_lmax(int(lmax))
        cp.InitPower.set_params(As=A_s_fid, ns=cosmo['n_s'])

        return cp

    def _get_power_spectrum(self, res, cosmo, nonlin):
        k, z, pk = res.get_linear_matter_power_spectrum(hubble_units=False,
                                                        k_hunit=False,
                                                        nonlinear=nonlin)

        np.log(k, out=k)
        np.log(pk, out=pk)

        # reverse the time axis because CAMB uses z
        return Pk2D(a_arr=(1/(1+z))[::-1], lk_arr=k, pk_arr=pk[::-1],
                    is_logp=True, extrap_order_lok=1, extrap_order_hik=2)


class PowerSpectrumCAMB(PowerSpectrumOnCAMB):
    """
    """
    name = "camb"
    rescale_s8 = True
    rescale_mg = True

    def get_power_spectrum(self, nonlin=False, kmax=10, lmax=5000,
                           dark_energy_model="fluid",
                           halofit_version="mead2020_feedback",
                           HMCode_A_baryon=3.13, HMCode_eta_baryon=0.603,
                           HMCode_logT_AGN=7.8):
        import camb

        cosmo = self.cosmo
        cp = self._setup(package=camb, nonlin=nonlin,
                         dark_energy_model=dark_energy_model,
                         kmax=kmax, lmax=lmax)

        if nonlin:
            if abs(cosmo["mu_0"]) > 1e-14:
                warnings.warn(
                    "CAMB does not consistently compute the non-linear "
                    "power spectrum for mu_0 > 0.", CCLWarning)
            if cosmo["A_s"] is None:
                raise CCLError("CAMB does not rescale the non-linear "
                               "spectrum consistently with sigma8.")

            # Set non-linear model parameters.
            cp.NonLinearModel = camb.nonlinear.Halofit()
            cp.NonLinearModel.set_params(halofit_version=halofit_version,
                                         HMCode_A_baryon=HMCode_A_baryon,
                                         HMCode_eta_baryon=HMCode_eta_baryon,
                                         HMCode_logT_AGN=HMCode_logT_AGN)

        res = camb.get_results(cp)
        pkl = self._get_power_spectrum(res, cosmo=cosmo, nonlin=False)
        if nonlin:
            pknl = self._get_power_spectrum(res, cosmo=cosmo, nonlin=True)
            return pkl, pknl
        return pkl


class PowerSpectrumISITGR(PowerSpectrumOnCAMB):
    """
    """
    name = "isitgr"
    rescale_s8 = True
    rescale_mg = False

    def get_power_spectrum(self, kmax=10, lmax=5000,
                           dark_energy_model="fluid"):
        import isitgr  # noqa: F811

        cosmo = self.cosmo
        cp = self._setup(package=isitgr, dark_energy_model=dark_energy_model)

        cp.GR = 1  # modified GR
        cp.ISiTGR_muSigma = True
        cp.mu0 = cosmo['mu_0']
        cp.Sigma0 = cosmo['sigma_0']
        cp.c1 = cosmo['c1_mg']
        cp.c2 = cosmo['c2_mg']
        cp.Lambda = cosmo['lambda_mg']

        return self._run(cp=cp, package=isitgr, cosmo=cosmo, nonlin=False,
                         kmax=kmax, lmax=lmax)


class PowerSpectrumCLASS(PowerSpectrum):
    """
    """
    name = "class"
    rescale_s8 = True
    rescale_mg = True

    def get_power_spectrum(self):
        """Run CLASS and return the linear power spectrum."""
        import classy

        cosmo = self.cosmo

        params = {
            "output": "mPk",
            "non linear": "none",
            "P_k_max_1/Mpc": sparams.K_MAX_CLASS,
            "z_max_pk": 1.0 / sparams.A_SPLINE_MINLOG_PK - 1.0,
            "modes": "s",
            "lensing": "no",
            "h": cosmo["h"],
            "Omega_cdm": cosmo["Omega_c"],
            "Omega_b": cosmo["Omega_b"],
            "Omega_k": cosmo["Omega_k"],
            "n_s": cosmo["n_s"],
            "T_cmb": cosmo["T_CMB"]}

        # Dark energy.
        if cosmo['w0'] != -1 or cosmo['wa'] != 0:
            params["Omega_Lambda"] = 0
            params['w0_fld'] = cosmo['w0']
            params['wa_fld'] = cosmo['wa']

        # Massless neutrinos.
        params["N_ur"] = cosmo["N_nu_rel"] if cosmo["N_nu_rel"] > 1e-4 else 0.

        # Massive neutrinos.
        if cosmo["N_nu_mass"] > 0:
            params["N_ncdm"] = cosmo["N_nu_mass"]
            masses = cosmo["m_nu"]
            params["m_ncdm"] = ", ".join(
                ["%g" % m for m in masses[:cosmo["N_nu_mass"]]])

        # Power spectrum normalization.
        # If A_s is not given, we just get close and CCL will normalize it.
        A_s, σ8 = cosmo["A_s"], cosmo["sigma8"]
        A_s_fid = A_s if A_s is not None else 2.43e-9 * (σ8 / 0.87659)**2
        params["A_s"] = A_s_fid

        # Set up sampling.
        a = np.delete(self.a_arr, self.a_arr > 1)
        z = 1 / a - 1

        nk = sparams.get_pk_spline_lk().size
        lkmax = np.log10(sparams.K_MAX_CLASS)
        lk = np.log(np.logspace(-5, lkmax, nk))  # FIXME: get kmin from CLASS
        lk = lk[lk < np.log(sparams.K_MAX_CLASS)]  # cut to max CLASS val

        try:
            model = classy.Class()
            model.set(params)
            model.compute()

            pk = np.array([model.pk_lin(np.exp(κ), α) for α in z for κ in lk])
            pk = pk.reshape((a.size, lk.size))
        finally:
            if model in locals():
                model.struct_cleanup()
                model.empty()

        return Pk2D(a_arr=a, lk_arr=lk, pk_arr=np.log(pk),
                    is_logp=True, extrap_order_lok=1, extrap_order_hik=2)
