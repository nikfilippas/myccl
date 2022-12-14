from ..parameters import spline_params as sparams
from ..interpolate import Interpolator1D, Interpolator2D
from ..pyutils import get_broadcastable
import numpy as np


class _tracer:
    """A unique tracer object. The class ``Tracer`` contains instances
    of collections of objects of this class. Do not use this class directly.
    """

    def __init__(self, *, der_bessel=0, der_angles=0,
                 kernel=None, transfer=None, is_logt=False,
                 extrap_order_lok=0, extrap_order_hik=2):
        if der_angles not in [0, 1, 2]:
            raise ValueError("der_angles must be in {0, 1, 2}.")
        if der_bessel not in [-1, 0, 1, 2]:
            raise ValueError("der_bessel must be in {-1, 0, 1, 2}.")

        self.der_angles = der_angles
        self.der_bessel = der_bessel
        self.transfer = transfer
        self.is_logt = is_logt

        # Initialize radial kernel.
        self.kernel = Interpolator1D(*kernel, extrap_orders=[0, 0])
        self.chi_min, self.chi_max = kernel[0].take([0, -1])

        # Initialize transfer function.
        if transfer is not None:
            extrap = [0, 1, extrap_order_lok, extrap_order_hik]
            self.transfer = Interpolator2D(*transfer, extrap_orders=extrap)

    def get_kernel(self, chi):
        r"""Kernel of the tracer, :math:`w(\chi)`."""
        return self.kernel(chi)

    def get_transfer(self, k, a):
        r"""Transfer function, :math:`T(k, a)`."""
        if self.transfer:
            tf = self.transfer(np.log(k), a)
            if self.is_logt:
                np.exp(tf, out=tf)
            return tf
        a, lk = map(np.asarray, [a, k])
        a, lk = get_broadcastable(a, k)
        shp = np.broadcast_shapes(a.shape, k.shape)
        return np.ones(shp)

    def get_f_ell(self, ell):
        r""":math:`\ell`-dependent prefactor , :math:`f(\ell)`."""
        ell = np.asarray(ell)
        if self.der_angles == 1:
            return ell * (ell + 1)
        if self.der_angles == 2:
            return np.sqrt((ell + 2) * (ell + 1) * ell * (ell - 1))
        return np.ones_like(ell)


class Tracer:
    r"""Container for a collection of tracers (:obj:`~pyccl.tracers._tracer`).
    Tracers can be added to a collection via the ``add_tracer`` method.

    Tracers contain the necessary information to describe the contribution of
    a given sky observable to its cross-power spectrum with any other tracer.

    Each tracer is uniquely described by:
        - a radial kernel, :math:`w(\chi)`, which expresses the support in
          redshift/distance,
        - an optional transfer function, :math:`T(k, a)`, which describes
          the connection between the tracer and the power spectrum,
        - the order of the Bessel derivative, ``der_bessel``, with which
          tracers enter the computation of angular power spectra,
        - a flag describing the :math:`f(\ell)` prefactor, associated with
          angular derivatives of a given fundamental quantity.
    """

    def __init__(self):
        self._trc = []

    def __bool__(self):
        """Check if this tracer collection contains any tracers."""
        return bool(self._trc)

    def __len__(self):
        """Number of tracers in this ``Tracer`` collection."""
        return len(self._trc)

    def __iter__(self):
        """Iterate over the collection of tracers."""
        return iter(self._trc)

    @property
    def chi_min(self):
        """Return ``chi_min`` for this ``Tracer``, if it exists. For more than
        one tracers containing a ``chi_min`` in the tracer collection, the
        lowest value is returned.
        """
        chis = [tr.chi_min for tr in self._trc]
        return min(chis) if chis else None

    @property
    def chi_max(self):
        """Return ``chi_max`` for this ``Tracer``, if it exists. For more than
        one tracers containing a ``chi_max`` in the tracer collection, the
        highest value is returned.
        """
        chis = [tr.chi_max for tr in self._trc]
        return max(chis) if chis else None

    def get_bessel_derivative(self):
        """Get Bessel function derivative orders for all tracers contained
        in this `Tracer`.

        Returns
        -------
        der_bessel : list
            Bessel derivative orders for each tracer.
        """
        return [t.der_bessel for t in self._trc]

    def _MG_add_tracer(self, cosmo, kernel, z_b, der_bessel=0, der_angles=0,
                       bias_transfer_a=None, bias_transfer_k=None):
        """MG tracers for different cases: biases, intrinsic alignments etc."""
        mg_transfer = self._get_MG_transfer_function(cosmo, z_b)

        if bias_transfer_a is not None and bias_transfer_k is not None:
            # k- and a-dependent astro bias.
            mg_transfer = (mg_transfer[0], mg_transfer[1],
                           (bias_transfer_a[1] * (bias_transfer_k[1] *
                            mg_transfer[2]).T).T)
        elif bias_transfer_a is not None:
            # a-dependent astro bias.
            mg_transfer = (mg_transfer[0], mg_transfer[1],
                           (bias_transfer_a[1] * mg_transfer[2].T).T)
        elif bias_transfer_k is not None:
            # k-dependent astro bias.
            mg_transfer = (mg_transfer[0], mg_transfer[1],
                           (bias_transfer_k[1] * mg_transfer[2]))

        self.add_tracer(kernel, transfer_ka=mg_transfer,
                        der_bessel=der_bessel, der_angles=der_angles)

    def _get_MG_transfer_function(self, cosmo, z):
        r"""Obtain :math:`\Sigma(k, z)` for the redshifts samples of a redshift
        distribution and some ``k``. The MG parameter becomes a multiplicative
        factor within the MG transfer function.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        z : float or (2,) tuple of array_like
            A single :math:`z` value (e.g. for the CMB) or a tuple of arrays
            ``(z, N(z))`` with the redshift distribution of the objects.
            :math:`N(z)` has arbitrary units and is internally normalized to 1.
        """
        # Sampling scale factor from a very small (at CMB for example)
        # all the way to 1 here and today for the transfer function.
        # For a < a_single it is GR (no early MG).
        if isinstance(z, (int, float)):
            a = np.linspace(1 / (1+z), 1, 100)
        else:
            if z[0] != 0:
                n_samples = int(z[0] / (z[1] - z[0]))
                z_arr = np.linspace(0.0, z[0], n_samples + 1)
            a = 1 / (1+z_arr[::-1])

        # Scale-dependant MG case with an array of k.
        lk = sparams.get_pk_spline_lk()
        k = np.exp(lk)
        # computing MG factor array
        mgfac_1d = 1 + cosmo.Sigma_MG(k, a)
        mgfac_2d = mgfac_1d.reshape(len(a), -1, order='F')  # FIXME: simplify
        return a, lk, mgfac_2d

    def add_tracer(self, *, der_bessel=0, der_angles=0,
                   kernel=None,
                   transfer_k=None, transfer_a=None,
                   transfer_ka=None, is_logt=False,
                   extrap_order_lok=0, extrap_order_hik=2):
        r"""Append a tracer to the ``Tracer`` collection.

        Arguments
        ---------
        der_bessel : {-1, 0, 1, 2}
            Order of the Bessel derivative for the power spectrum calculations.
            -1 corresponds to :math:`\frac{J(x)}{x^2}` and is provided to make
            common tracers more stable at small :math:`k` and :math:`\chi`.
        der_angles : {0, 1, 2}
            Flag for the :math:`\ell`-dependent prefactor.
                - 0: no prefactor
                - 1: :math:`\ell (\ell + 1)` (angular Laplacian)
                - 2: :math:`\sqrt{\frac{(\ell+2)!}{(\ell-2)!}}` (spin-2 fields)
        kernel : (nchi, nchi) tulple of array_like, optional
            A tuple ``(chi, w_chi)`` describing the radial kernel.
            ``chi`` is the comoving radial distance
            and ``w_chi`` is the kernel sampled at ``chi``.
            Values outside of the sampling range are extrapolated to zero.
            The default is None, for :math:`w(\chi) = 1`.
        transfer_ka : (na, nk, (na, nk)) tuple of array_like, optional
            The most general transfer function.
            The wavenumber array holds :math:`\ln (k)`
            in :math:`\mathrm{Mpc}^{-1}` and the transfer function array
            is controlled by ``is_logt``.
            The default is None, for :math:`T(k, a) = 1`.
        transfer_k : (nk, nk) tuple of array_like, optional): a
            Scale-dependent part of a factorizable transfer function.
            The wavenumber array holds :math:`\ln (k)`
            in :math:`\mathrm{Mpc}^{-1}` and the transfer function array
            is controlled by ``is_logt``.
            The default is None, for :math:`T(k) = 1`.
        transfer_a : (na, na) tuple of arrays, optional
            Time-dependent part of a factorizable transfer function.
            The transfer function array controlled by ``is_logt``.
            The default is None, for :math:`T(k, a) = 1`.
        is_logt : bool
            Whether ``transfer_x`` contains the log of the transfer function.
            The default is False.
        extrap_order_lok, extrap_order_hik : int
            Taylor expansion order for low- and high-k extrapolation.
        """
        if transfer_a is not None:
            # Expand transfer function.
            (a, ta), (lk, tk) = transfer_a, transfer_k
            comb = np.add if is_logt else np.multiply
            tk = comb(*get_broadcastable(ta, tk))
            transfer_ka = (a, lk, tk)

        trc = _tracer(der_bessel=der_bessel, der_angles=der_angles,
                      kernel=kernel, transfer=transfer_ka, is_logt=is_logt,
                      extrap_order_lok=extrap_order_lok,
                      extrap_order_hik=extrap_order_hik)

        self._trc.append(trc)
