from .pspec_base import PowerSpectrumNonLinear


class PowerSpectrumCosmicEmu(PowerSpectrumNonLinear):
    """
    """
    name = "emu"
    emulator_neutrinos = "strict"  # TODO

    def __init__(self, cosmo):
        super().__init__(cosmo=cosmo)
        if not 0.55 <= cosmo["h"] <= 0.85:
            raise ValueError("h is outside allowed emulator range.")

        self._setup()

    def _setup(self):
        cosmo = self.cosmo
        if cosmo["N_nu_mass"] > 0:
            cosmo.get_extra_parameters["emu"]
