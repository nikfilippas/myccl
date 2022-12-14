from .interpolate import Interpolator1D
from .parameters import accuracy_params as acc

import numpy as np
from scipy import integrate
from enum import Enum


__all__ = ("IntegratorSamples", "IntegratorFunction")


class _IntegrationMethodsSamples(Enum):
    TRAPEZOID = "trapezoid"
    SIMPSON = "simpson"
    SYMMETRICAL = "symmetrical"
    SPLINE = "spline"


class _IntegrationMethodsFunction(Enum):
    SPLINE = "spline"
    QAG = "adaptive_quadrature"
    QNG = "fixed_quadrature"
    QAWF = "fourier_transform"


class IntegratorSamples:
    """Generic class to handle 1-D integration methods.

    Parameters
    ----------
    method : {'trapezoid', 'simpson', 'symmetrical', 'spline'}
        Integration method.
    sym_method : {'trapezoid', 'simpson'}
        Integration method for symmetrical grid.
    """

    def __init__(self, method, sym_method=None):
        IM = _IntegrationMethodsSamples
        if IM(method) == IM.TRAPEZOID:
            self._integrator = integrate.trapezoid
        elif IM(method) == IM.SIMPSON:
            self._integrator = integrate.simpson
        elif IM(method) == IM.SYMMETRICAL:
            self.sym_method = IntegratorSamples(IM(sym_method))
            self._integrator = self._symmetrical()
        elif IM(method) == IM.SPLINE:
            self._integrator = self._from_interpolator

    def __call__(self, y, x):
        return self._integrator(y, x)

    def _from_interpolator(self, y, x):
        return Interpolator1D(x, y).f.integrate(x[0], x[-1])[()]

    def _symmetrical(self):
        """Symmetrical integrator.
        Benchmarks show a speedup of ~1.5x relative to full grid integration.
        """
        def symmetrical(y, x):
            """Implement specialized integration for
            :meth:`HMCalculator.I_0_22`.

            Assume:
                - ``y.ndim == 4`` (a, k1, k2, M),
                - The symmetrical axes are ``[1, 2]``.
            """
            # Slice y at only the upper triangular indices.
            sym_size = y.shape[1]
            idxu = np.triu_indices(sym_size)
            idxu = np.s_[:, idxu[0], idxu[1]]
            y_use = y[idxu]

            # Integrate & fill-in the upper triangular indices.
            res = np.empty(y.shape[:-1])
            res[idxu] = self.sym_method(y_use, x)

            # Fill-in the lower triangular indices.
            idxl = np.tril_indices(sym_size)
            idxl = np.s_[:, idxl[0], idxl[1]]
            res[idxl] = res.transpose((0, 2, 1))[idxl]
            return res
        return symmetrical


class IntegratorFunction:
    """Generic class to handle function integration methods.

    Parameters
    ----------
    mathod : {'spline', 'adaptive_quadrature', \
              'fixed_quadrature', 'fourier_transform'}
        Integration method.
    accuracy : int or float
        - ``int``: Number of points for spline integration.
        - ``float``: Relative error (ε) for quadrature integration.
    """

    def __init__(self, method, accuracy=None):
        IM = _IntegrationMethodsFunction
        if IM(method) == IM.SPLINE:
            self._spline_integrator = IntegratorSamples("spline")
            self._integrator = self._spline(accuracy)
        elif IM(method) == IM.QAG:
            self._integrator = self._adaptive_quadrature(accuracy)
        elif IM(method) == IM.QNG:
            self._integrator = self._fixed_quadrature  # fixed accuracy
        elif IM(method) == IM.QAWF:
            self._integrator = self._fourier_transform(accuracy)

    def __call__(self, func, a, b, **kwargs):
        """Generic caller for function integrators. Each ``method`` has
        its own set of ``kwargs``, outlined in ``IntegratorFunction._method``.
        """
        return self._integrator(func, a, b, **kwargs)

    def _spline(self, accuracy=acc.N_SPLINE):
        def spline(func, a, b, args=()):
            x = np.linspace(a, b, num=accuracy)
            y = func(x, *args)
            return self._spline_integrator(y, x)
        return spline

    def _adaptive_quadrature(self, accuracy=acc.EPSREL):
        def adaptive_quadrature(func, a, b, args=()):
            return integrate.quad(func, a, b, epsabs=0., epsrel=accuracy,
                                  limit=acc.N_ITERATION, args=args)[0]
        return adaptive_quadrature

    def _fixed_quadrature(self, func, a, b, args=()):
        return integrate.fixed_quad(func, a, b,
                                    n=acc.N_ITERATION, args=args)[0]

    def _fourier_transform(self, accuracy=acc.EPSREL):
        def fourier_transform(func, a, b, wvar):
            return np.array([
                integrate.quad(func, a, b, weight="sin", wvar=w,
                               epsrel=accuracy, limit=acc.N_ITERATION)[0]
                for w in wvar])
        return fourier_transform
