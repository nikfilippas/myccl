from .interpolate import Interpolator2D
from .errors import CCLWarning

import numpy as np
import warnings
import copy


__all__ = ("DefaultPowerSpectrum", "Pk2D", "parse_pk2d")


DefaultPowerSpectrum = 'delta_matter:delta_matter'


class Pk2D:
    """A power spectrum class holding the information needed to reconstruct an
    arbitrary function of wavenumber and scale factor.

    Parameters
    ----------
    a_arr : array, optional
        An array holding values of the scale factor.
    lk_arr : array, optional
        An array holding values of the natural logarithm
        of the wavenumber (in units of Mpc^-1).
    pk_arr : array, optional
        A 2D array containing the values of the power
        spectrum at the values of the scale factor and the wavenumber
        held by `a_arr` and `lk_arr`. The shape of this array must be
        `[na,nk]`, where `na` is the size of `a_arr` and `nk` is the
        size of `lk_arr`. This array can be provided in a flattened
        form as long as the total size matches `nk*na`. The array can
        hold the values of the natural logarithm of the power
        spectrum, depending on the value of `is_logp`. If `pkfunc`
        is not None, then `a_arr`, `lk_arr` and `pk_arr` are ignored.
        However, either `pkfunc` or all of the last three array must
        be non-None. Note that, if you pass your own Pk array, you
        are responsible of making sure that it is sufficiently well
        sampled (i.e. the resolution of `a_arr` and `lk_arr` is high
        enough to sample the main features in the power spectrum).
        For reference, CCL will use bicubic interpolation to evaluate
        the power spectrum at any intermediate point in k and a.
    pkfunc : :obj:`function`, optional
        A function returning a floating point
        number or numpy array with the signature `f(k,a)`, where k
        is a wavenumber (in units of Mpc^-1) and a is the scale
        factor. The function must able to take numpy arrays as `k`.
        The function must return the value(s) of the power spectrum
        (or its natural logarithm, depending on the value of
        `is_logp`. The power spectrum units should be compatible
        with those used by CCL (e.g. if you're passing a matter power
        spectrum, its units should be Mpc^3). If this argument is not
        `None`, this function will be sampled at the values of k and
        a used internally by CCL to store the linear and non-linear
        power spectra.
    is_logp (boolean): if True, pkfunc/pkarr return/hold the natural
         logarithm of the power spectrum. Otherwise, the true value
         of the power spectrum is expected. Note that arrays will
         always be interpolated in log space.
    extrap_order_lok, extrap_order_hik : int, optional
        Extrapolation orders to be used on k-values below and
        abovethe minimum of the splines (use 0, 1 or 2). Note that
        the extrapolation will always be done in log(P(k)).
    """

    def __init__(self, a_arr=None, lk_arr=None, pk_arr=None,
                 pkfunc=None, is_logp=True,
                 extrap_order_lok=1, extrap_order_hik=2):
        self.is_logp = is_logp
        if pkfunc:
            # TODO: convert to Interpolator2D
            # Check that the input function is usable.
            try:
                pkfunc(k=np.array([1e-2, 2e-2]), a=np.array([0.75, 0.80]))
            except Exception:
                raise ValueError("Can't use input function.")
            self.psp = pkfunc
        else:
            self.psp = Interpolator2D(
                a_arr, lk_arr, pk_arr,
                extrap_orders=[1, 1, extrap_order_lok, extrap_order_hik])

    @property
    def amin(self):
        """Minimum interpolated scale factor."""
        return self.psp._xmin

    @property
    def amax(self):
        """Maximum interpolated scale factor."""
        return self.psp._xmax

    @property
    def kmin(self):
        """Minimum interpolated wavenumber."""
        return np.exp(self.psp._ymin)

    def kmax(self):
        """Maximum interpolated wavenumber."""
        return np.exp(self.psp._ymax)

    @property
    def extrap_order_lok(self):
        """Polynomial order for Taylor extrapolation below ``kmin``."""
        return self.psp.extrap_orders[2] if self.psp else None

    @property
    def extrap_order_hik(self):
        """Polynomial order for Taylor extrapolation above ``kmax``."""
        return self.psp.extrap_orders[3] if self.psp else None

    def __contains__(self, other):
        return other.psp in self.psp

    def __call__(self, k, a, *, derivative=False, squeeze=True, grid=True):
        """Evaluate the power spectrum, or its logarithmic derivative.

        Arguments
        ---------
        k : ``float`` or ``array_like``
            Wavenumber value(s) in units of :math:`\\mathrm{Mpc}^{-1}`.
        a : ``float`` or ``array_like``
            Value of the scale factor.
        derivative : bool
            If ``False``, evaluate the power spectrum. If ``True``, evaluate
            the logarithmic derivative of the power spectrum,
            :math:`\\frac{\\mathrm{d} \\log P(k)}{\\mathrm{d} \\log k}`.
        squeeze : bool
            Squeeze extra dimensions of size (1,) in the output.
            The default is True.
        grid : bool
            Evaluate P(k, a) on a grid spanned by the input arrays (True),
            or at the points specified by the input arrays (False).
            The default is True.

        Returns
        -------
        Pka : float or array_like, shape=(``na``, ``nk``)
            Value(s) of the power spectrum.
        """
        pk = self.psp(a, np.log(k), derivative=[0, int(derivative)], grid=grid)
        if not derivative:
            pk = np.exp(pk)
        return pk.squeeze()[()] if squeeze else pk

    def copy(self):
        """Return a copy of this ``Pk2D`` object."""
        return copy.copy(self)

    def get_spline_arrays(self):
        """Get the spline data arrays.

        Returns:
            a_arr: array_like
                Array of scale factors.
            lk_arr: array_like
                Array of logarithm of wavenumber k.
            pk_arr: array_like
                Array of the power spectrum P(k, z). The shape
                is (a_arr.size, lk_arr.size).
        """
        a_arr, lk_arr, pk_arr = [a.copy() for a in self.psp._points]
        pk_arr = pk_arr.reshape((a_arr.size, lk_arr.size))
        if self.is_logp:
            np.exp(pk_arr, out=pk_arr)
        return [a_arr, lk_arr, pk_arr]

    def _get_binary_operator_arrays(self, other):
        """"Get the necessary ingredients to perform binary operations."""
        if not isinstance(other, Pk2D):
            raise TypeError(
                "Binary operations are only defined between Pk2D objects.")
        if not (self and other):
            raise ValueError("Pk2D object is empty.")
        if self not in other:
            raise ValueError("The 2nd operand is defined over a smaller range "
                             "than the 1st operand. To avoid extrapolation, "
                             "this operation is forbidden. To operate on the "
                             "smaller support try swapping the operands.")

        a_arr_a, lk_arr_a, pk_arr_a = self.get_spline_arrays()
        pk_arr_b = other(np.exp(lk_arr_a), a_arr_a)
        return [a_arr_a, lk_arr_a, pk_arr_a, pk_arr_b]

    def __add__(self, other):
        if isinstance(other, (float, int)):
            arrays = self.get_spline_arrays()
            arrays[-1] += other
        elif isinstance(other, Pk2D):
            *arrays, pk_arr_b = self._get_binary_operator_arrays(other)
            arrays[-1] += pk_arr_b
        else:
            raise TypeError("Addition of Pk2D is only defined for "
                            "floats, ints, and Pk2D objects.")

        if self.is_logp:
            np.log(arrays[-1], out=arrays[-1])

        return Pk2D(*arrays, is_logp=self.is_logp,
                    extrap_order_lok=self.extrap_order_lok,
                    extrap_order_hik=self.extrap_order_hik)

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            arrays = self.get_spline_arrays()
            arrays[-1] *= other
        elif isinstance(other, Pk2D):
            *arrays, pk_arr_b = self._get_binary_operator_arrays(other)
            arrays[-1] *= pk_arr_b
        else:
            raise TypeError("Multiplication of Pk2D is only defined for "
                            "floats, ints, and Pk2D objects.")

        if self.is_logp:
            np.log(arrays[-1], out=arrays[-1])

        return Pk2D(*arrays, is_logp=self.is_logp,
                    extrap_order_lok=self.extrap_order_lok,
                    extrap_order_hik=self.extrap_order_hik)

    def __pow__(self, exponent):
        if not isinstance(exponent, (float, int)):
            raise TypeError("Exponentiation of Pk2D is only defined for "
                            "float and int.")

        arrays = self.get_spline_arrays()
        if (arrays[-1] < 0).any() and exponent % 1 != 0:
            warnings.warn("Taking a non-positive Pk2D object to a non-integer "
                          "power may lead to unexpected results", CCLWarning)

        arrays[-1] **= exponent

        if self.is_logp:
            np.log(arrays[-1], out=arrays[-1])

        return Pk2D(*arrays, is_logp=self.is_logp,
                    extrap_order_lok=self.extrap_order_lok,
                    extrap_order_hik=self.extrap_order_hik)

    def __sub__(self, other):
        return self + (-1)*other

    def __truediv__(self, other):
        return self * other**(-1)

    __radd__ = __add__

    __rmul__ = __mul__

    def __rsub__(self, other):
        return other + (-1)*self

    def __rtruediv__(self, other):
        return other * self**(-1)

    def __iadd__(self, other):
        return self + other

    def __imul__(self, other):
        return self * other

    def __isub__(self, other):
        return self - other

    def __itruediv__(self, other):
        return self / other

    def __ipow__(self, other):
        return self**other

    @classmethod
    def from_model(cls, cosmo, model):
        """Return the ``Pk2D`` object associated with a given numerical model.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            A Cosmology object.
        model : :obj:`str`
            Models are listed in :obj:`~pyccl.TransferFunctions`,
            :obj:`~pyccl.power.MatterPowerSpectra`, and
            :obj:`~pyccl.power.BaryonPowerSpectra`.
        """
        from .pspec import PowerSpectrum
        return PowerSpectrum.from_model(model)(cosmo).get_power_spectrum()

    def apply_model(self, cosmo, model):
        """Apply a (non-linear) transformation to a ``Pk2D`` object.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            A Cosmology object.
        model : :obj:`str`
            See ``Pk2D.from_model`` for a list of available models.
        """


def parse_pk2d(cosmo=None, p_of_k_a=DefaultPowerSpectrum, linear=False):
    """Return the ``Pk2D`` object, as indicated.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`, optional
        A Cosmology object. Not needed if ``pk`` is given.
        The default is None.
    p_of_k_a : {'linear', 'nonlinear'} or :class:`~pyccl.pk2d.Pk2D`, optional
        If a ``Pk2D`` object, its spline is used.
        If a string, the linear or non-linear power spectrum
        stored in ``cosmo`` under this name is used.
    linear : :obj:`bool`, optional
        Specify whether the linear or non-linear power spectrum
        will be used from ``cosmo``. The default is False.
    """
    if isinstance(p_of_k_a, Pk2D):
        return p_of_k_a

    if not isinstance(p_of_k_a, str):
        raise ValueError("`pk` must be a `Pk2D` object, a string, or None.")

    if linear:
        cosmo.compute_linear_power()
        return cosmo.get_linear_power(p_of_k_a)

    cosmo.compute_nonlin_power()
    return cosmo.get_nonlin_power(p_of_k_a)
