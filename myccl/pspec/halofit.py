from ..interpolate import Interpolator1D
from ..parameters import physical_constants as const
import numpy as np


def compute_chi(cosmo, a):
    pass


def z_drag(cosmo):
    """Equation 4 of Eisenstein & Hu (1998)."""
    Ωmh2 = (cosmo["Omega_c"] + cosmo["Omega_b"]) * cosmo["h"]**2
    Ωbh2 = cosmo["Omega_b"] * cosmo["h"]**2
    b1 = 0.313 * Ωmh2**(-0.419) * (1 + 0.607 * Ωmh2**0.674)
    b2 = 0.238 * Ωmh2**0.223
    return 1291 * Ωmh2**0.251 * (1 + b1 * Ωbh2**b2) / (1 + 0.659 * Ωmh2**0.828)


def create_cosmo_w0eff(cosmo, w0eff):
    """Create a cosmology with the same parameters as the input,
    except :math:`w_0 - w_a`. The new :math:`w_0` is :math:`w_{0, \rm eff}`.
    """
    params = {"w0": w0eff, "wa": 0}
    cosmo_new = cosmo.copy()
    cosmo_new.__dict__.update(**params)
    return cosmo_new


def get_w0eff(cosmo, a):
    a_drag = 1 / (1 + z_drag(cosmo))
    a_vals = np.r_[a_drag, a]
    chis = cosmo.comoving_radial_distance(a_vals)
    Δx = chis[0] - chis[1:]

    def minfunc(w0eff):
        cosmo_new = create_cosmo_w0eff(cosmo, w0eff)
        a_drag_new = 1 / (1 + z_drag(cosmo_new))




def halofit_struct_new(cosmo, plin):
    hf = HalofitStruct()

    # TODO: Convert wa != 0 to wa = 0.
    a = plin.get_spline_arrays()[0]


    weff = np.full_like(a, cosmo["w0"])
    hf.weff = Interpolator1D(a, weff)
    hf.omeff = Interpolator1D(a, cosmo.omega_x(a, "matter"))
    hf.deeff = Interpolator1D(a, cosmo.omega_x(a, "dark_energy"))



class HalofitStruct:
    r_sigma: Interpolator1D
    n_eff: Interpolator1D
    C: Interpolator1D


def halofit_power(cosmo, plin, k, a, hf):
    """
    Equations from Takashi et al.
    """
    # Equations A4 - A5.
    rsigma = hf.r_sigma(a)
    neff = hf.n_eff(a)
    C = hf.C(a)

    weff = cosmo["w0"]
    Ωm = cosmo.omega_x(a, "matter")
    ΩΛ = cosmo.omega_x(a, "dark_energy")

    ksigma = 1 / rsigma
    delta2_norm = k**3 / (2*np.pi**2)

    # Compute the present day neutrino massive neutrino fraction.
    # Use all neutrinos even if they are relativistic.
    Ωnu = cosmo["sum_nu_masses"] / (93.14 * cosmo["h"]**2)
    fnu = Ωnu / cosmo["Omega_m"]

    # Equations A6 - A13.
    neff2 = neff*neff
    neff3 = neff*neff2
    neff4 = neff*neff3

    ΩΛeff = ΩΛ * (1 + weff)
    an = 10**(1.522 + 2.8553*neff + 0.9903*neff3 + 0.2250*neff4
              - 0.6038*C + 0.1749*ΩΛeff)
    bn = 10**(-0.5642 + 0.5864*neff + 0.5716*neff2 - 1.5474*C + 0.2279*ΩΛeff)
    cn = 10**(0.3698 + 2.0404*neff + 0.8161*neff2 + 0.5869*C)
    γn = 0.1971 - 0.0843 + 0.8460*C
    αn = np.abs(6.0835 + 1.3373*neff - 0.1959*neff2 - 5.5274*C)
    βn = (2.0379 - 0.7354*neff + 0.3157*neff2 + 1.2490*neff3 + 0.3980*neff3
          - 0.1682*C)
    μn = 0.
    νn = 10**(5.2105 + 3.6902*neff)

    # Equations C17 and C18 from Smith et al.
    idx = np.abs(1 - Ωm) > 0.01
    f1, f2, f3 = [np.ones_like(Ωm) for i in range(3)]
    f1a = Ωm[idx]**(-0.0732)
    f2a = Ωm[idx]**(-0.1423)
    f3a = Ωm[idx]**(0.0725)
    f1b = Ωm[idx]**(-0.0307)
    f2b = Ωm[idx]**(-0.0585)
    f3b = Ωm[idx]**(0.0743)
    fb_frac = ΩΛ[idx] / (1 - Ωm[idx])
    f1[idx] = fb_frac * f1b + (1-fb_frac) * f1a
    f2[idx] = fb_frac * f2b + (1-fb_frac) * f2a
    f3[idx] = fb_frac * f3b + (1-fb_frac) * f3a

    # Correction from Bird et al. (Eq. A10).
    βn += fnu * (1.081 + 0.395*neff2)

    # Equations A1 - A3.
    pkl = plin(k, a)
    y = k / ksigma
    fy = y/4 + y*y/8
    Delta_kl = pkl * delta2_norm

    # Correction to Delta_kl from Bird et al. (Eq A9).
    kh = k / cosmo["h"]
    kh2 = kh * kh
    Delta_kl_tilde = Delta_kl * (1 + fnu*(47.48*kh2) / (1 + 1.5*kh2))
    Delta_kQ = (Delta_kl * (1+Delta_kl_tilde)**βn / (1 + αn*Delta_kl_tilde)
                * np.exp(-fy))
    Delta_kH_prime = an * y**(3*f1) / (1 + bn * y**f2) + (cn*f3*y)**(3 - γn)
    Delta_kH = Delta_kH_prime / (1 + μn/y + νn/y/y)

    # Correction to Delta_kH from Bird et al. (Eq A6 - A7).
    Qnu = fnu * (0.977 - 18.015 * (cosmo["Omega_m"] - 0.3))
    Delta_kH *= (1 + Qnu)

    Delta_knl = Delta_kQ + Delta_kH
    return Delta_knl / delta2_norm
