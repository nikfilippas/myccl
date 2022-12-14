from .pspec_base import PowerSpectrumAnalytic
from numpy import e, exp, log, sqrt, sin


class PowerSpectrumEistensteinHu(PowerSpectrumAnalytic):
    """Base for the Eisenstein & Hu (1998) fitting formulae."""
    name = "eisenstein_hu_base"

    def get_power_spectrum(self):
        # Compute the Eisenstein & Hu (1998) unnormalized power spectrum.
        cosmo = self.cosmo
        k_arr = exp(self.lk_arr)
        pk_arr = k_arr**cosmo["n_s"] * self._transfer(k_arr / cosmo["h"])
        return self._get_full_analytic_power(pk_arr)


class EisensteinHuWiggles(PowerSpectrumEistensteinHu):
    """Wiggled Eisenstein & Hu (1998) powe spectrum."""
    name = "eisenstein_hu"

    def _setup_baryons(self):
        Ωm, Ωb = self.cosmo["Omega_m"], self.cosmo["Omega_b"]
        h = self.cosmo["h"]
        Ωmh2 = Ωm * h * h
        Ωbh2 = Ωb * h * h
        Θ27 = self.cosmo["T_CMB"] / 2.7

        # Compute Eisenstein & Hu parameters for P_k and r_sound.
        z_eq = 2.5e4 * Ωmh2 / Θ27**4
        self.k_eq = 0.0746*Ωmh2 / (h*Θ27**2)

        # Equation 4
        b1 = 0.313 * Ωmh2**(-0.419) * (1 + 0.607*Ωmh2**0.674)
        b2 = 0.238 * Ωmh2**0.223
        z_drag = 1291*Ωmh2**0.251 * (1 + b1*Ωbh2**b2) / (1 + 0.659*Ωmh2**0.828)

        # Equation 5: Baryon-to-photon ratios and drag epochs.
        R_eq = 31.5 * Ωbh2 * 1000 / (z_eq * Θ27**4)
        R_drag = 31.5 * Ωbh2 * 1000 / ((1 + z_drag) * Θ27**4)
        self.r_sound = (2/(3*self.k_eq) * sqrt(6/R_eq)
                        * log((sqrt(1 + R_drag) + sqrt(R_drag + R_eq))
                              / (1 + sqrt(R_eq))))

        # Equation 7 (in h/Mpc)
        self.k_Silk = (1.6 * Ωbh2**0.52 * Ωmh2**0.73
                       * (1 + (10.4 * Ωmh2)**(-0.95)) / h)

        # Equations 11
        def get_alpha(v1, v2, v3, v4):
            return (v1*Ωmh2)**v2 * (1 + (v3*Ωmh2)**v4)

        a1 = get_alpha(46.9, 0.670, 32.1, -0.532)
        a2 = get_alpha(12.0, 0.424, 45.0, -0.582)
        self.b_frac = Ωbh2 / Ωmh2
        self.α_c = a1**(-self.b_frac) * a2**(-self.b_frac**3)

        # Equations 12
        bb1 = 0.944 / (1 + (458 * Ωmh2)**(-0.708))
        bb2 = (0.395 * Ωmh2)**(-0.0266)
        self.β_c = 1 / (1 + bb1 * ((1 - self.b_frac)**bb2 - 1))

        y = z_eq / (1 + z_drag)
        sqy = sqrt(1 + y)
        gy = y * (-6 * sqy + (2 + 3*y) * log((sqy + 1) / (sqy - 1)))  # Eq15

        # Baryon suppression Eq. 14
        self.α_bar = 2.07*self.k_eq*self.r_sound * (1 + R_drag)**(-0.75) * gy

        # Baryon envelope shift Eq. 24
        self.β_bar = (0.5 + self.b_frac + (3 - 2*self.b_frac)
                      * sqrt((17.2*Ωmh2)**2 + 1))

    def _transfer_0(self, k, a, b):
        # Eisenstein & Hu Tk_0
        q = k / (13.41 * self.k_eq)  # Eq. 10
        c = 14.2/a + 386. / (1 + 69.9*q**1.08)  # Eq. 20
        lg = log(e + 1.8*b*q)  # change of var for Eq. 19
        return lg / (lg + c*q**2)

    def _transfer_b(self, k):
        # Eisenstein & Hu Tk_b (Eq. 21)
        Ωm, h = self.cosmo["Omega_m"], self.cosmo["h"]
        Ωmh2 = Ωm * h * h

        # Node shift parameter Eq. 23
        b_node = 8.41 * Ωmh2**0.435

        x = k * self.r_sound

        # First term of Eq. 21.
        x_bessel = x * (1 + (b_node / x)**3)**(-1/3)
        part1 = self._transfer_0(k, 1, 1) / (1 + (x/5.2)**2)
        # Second term of Eq. 21.
        part2 = (self.α_bar / (1 + (self.β_bar / x)**3)
                 * exp(-(k / self.k_Silk)**1.4))
        return sin(x_bessel) / x_bessel * (part1 + part2)

    def _transfer_c(self, k):
        # Eisenstein & Hu Tk_c
        f = 1 / (1 + (k * self.r_sound / 5.4)**4)  # Eq. 18
        part1 = self._transfer_0(k, a=1, b=self.β_c)
        part2 = self._transfer_0(k, a=self.α_c, b=self.β_c)
        return f * part1 + (1-f) * part2  # Eq. 17

    def _transfer(self, k):
        # Equation 8
        self._setup_baryons()
        part1 = self._transfer_b(k)
        part2 = self._transfer_c(k)
        tk = self.b_frac * part1 + (1 - self.b_frac) * part2
        return tk * tk


class EisensteinHuNoWiggles(PowerSpectrumEistensteinHu):
    """Eisenstein & Hu (1998) power spectrum with no baryons."""
    name = "eisenstein_hu_nowiggles"

    def _transfer(self, k):
        # Section 4.2
        Ωm, Ωb = self.cosmo["Omega_m"], self.cosmo["Omega_b"]
        b_frac = Ωb / Ωm
        h = self.cosmo["h"]
        Ωmh2 = Ωm * h * h
        Ωbh2 = Ωb * h * h
        Θ27 = self.cosmo["T_CMB"] / 2.7

        # Approximation for the sound horizon Eq. 26
        r_sound_approx = (44.5*h*log(9.83/Ωmh2) / sqrt(1+10*Ωbh2**0.75))

        α_γ = (1 - 0.328 * log(431 * Ωmh2) * b_frac
               + 0.38 * log(22.3 * Ωmh2) * b_frac**2)
        γ_eff = (Ωm*h*(α_γ + (1-α_γ) / (1 + (0.43*k*r_sound_approx)**4)))
        q = k * Θ27**2 / γ_eff
        l0 = log(2*e + 1.8*q)
        c0 = 14.2 + 731 / (1 + 62.5*q)
        return (l0 / (l0 + c0*q**2))**2
