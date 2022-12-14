from .pyutils import get_broadcastable
import numpy as np
from scipy import interpolate


__all__ = ("linlog_spacing", "loglin_spacing",
           "Interpolator1D", "Interpolator2D", "Interpolator3D")


def linlog_spacing(xmin, logstart, xmax, num_lin, num_log, dtype=float):
    """Create an array spaced first linearly, then logarithmically.

    .. note::

        The number of linearly spaced points used in ``num_lin - 1``
        because the first point of the logarithmically spaced points is the
        same as the end point of the linearly spaced points.

    .. code-block:: text

        |=== num_lin ==|  |============= num_log ================|
      --*--*--*--*--*--*--*--*---*----*------*--------*----------*--> (axis)
        ^                  ^                                       ^
       xmin             logstart                                 xmax
    """
    step = (logstart - xmin) / (num_lin - 1)
    lin = np.arange(xmin, logstart, step)
    log = np.geomspace(logstart, xmax, num_log)
    out = np.concatenate((lin, log))
    if dtype is int:
        return np.unique(out.astype(int))
    return out.astype(dtype)


def loglin_spacing(logstart, xmin, xmax, num_log, num_lin):
    """Create an array spaced first logarithmically, then linearly.

    .. note::

        The number of logarithmically spaced points used is ``num_log - 1``
        because the first point of the linearly spaced points is the same as
        the end point of the logarithmically spaced points.

    .. code-block:: text

        |=== num_log ==|   |============== num_lin ================|
      --*-*--*---*-----*---*---*---*---*---*---*---*---*---*---*---*--> (axis)
        ^                  ^                                       ^
     logstart             xmin                                    xmax
    """
    log = np.geomspace(logstart, xmin, num_log-1, endpoint=False)
    lin = np.linspace(xmin, xmax, num_lin)
    return np.concatenate((log, lin))


class Interpolator:
    _allowed_orders = [None, 0, 1, 2]  # allowed extrapolation orders

    def __init__(self, *, extrap_orders=None):
        # Number of edges from class name. (If it works it ain't stupid!)
        n_edges = 2 * int(self.__class__.__name__[-2])
        if extrap_orders is None:
            extrap_orders = [None] * n_edges
        elif any([ex not in self._allowed_orders for ex in extrap_orders]):
            raise ValueError("`extrap_orders` must be one of "
                             f"{self._allowed_orders}.")
        if len(extrap_orders) != n_edges:
            raise ValueError("Length of `extrap_orders` should be twice "
                             "the number of dimensions of the Interpolator.")
        self.extrap_orders = extrap_orders

    def __contains__(self, other):
        for x1, x2 in zip(self._points[:-1], other._points[:-1]):
            if not (x1[0] <= x2[0] and x1[-1] >= x2[-1]):
                return False
        return True

    def _extrapolate(self, xi, val, extrap, name, pos=0, out=None, idx=None):
        """Extrapolate the interpolant.

        Use a Taylor expansion of order ``extrap`` at the boundary ``val``.

        Arguments
        ---------
        xi : list
            The grid points to compute the extrapolation: ``[x1, ..., xN]``.
        val : float
            Value at the boundary, around which the interpolant
            is Taylor-expanded to extrapolate.
        extrap : {None, 0, 1, 2}
            Order of extrapolation.
        name : str
            Name of the keyword argument specifying the partial derivative.
        pos : int
            0-index position of the extrapolated dimension in ``xi``
        out : :class:``numpy.ndarray``
            Output array.
        idx : :class:``numpy.ndarray``
            Indexes of the extrapolated values in the output array.
        """
        if extrap is None:
            raise ValueError("Value(s) outside of interpolation range.")

        xi_bak = xi[pos]  # store original xi
        dx, xi[pos] = xi[pos][idx] - val, val

        taylor = self.f(*xi) * np.ones_like(dx)
        if extrap > 0:
            deriv = {name: 1}
            taylor += self.f(*xi, **deriv) * dx
        if extrap > 1:
            deriv[name] += 1
            taylor += 0.5 * self.f(*xi, **deriv) * dx * dx

        idx = np.broadcast_to(idx, out.shape)
        np.place(out, idx, taylor)
        xi[pos] = xi_bak  # restore original xi

    def _extrapolate_axis(self, xi, vals, extraps, name, pos, out, idxs):
        for val, extrap, idx in zip(vals, extraps, idxs):
            if idx.any():
                self._extrapolate(xi, val, extrap, name, pos, out, idx)


class Interpolator1D(Interpolator):
    interpolator = interpolate.Akima1DInterpolator

    def __init__(self, x, y, *, extrap_orders=None):
        super().__init__(extrap_orders=extrap_orders)
        self._points = x, y
        self._xmin, self._xmax = x[0], x[-1]
        self.f = self.interpolator(x, y)

    def __call__(self, x):
        x = np.asarray(x)
        shape = x.shape
        x = x.squeeze()
        res = self.f(x)

        idx_xlo = x < self._xmin
        idx_xhi = x > self._xmax
        extrap = idx_xlo.any() or idx_xhi.any()

        if extrap:
            self._extrapolate_axis(
                xi=[x],
                vals=(self._xmin, self._xmax),
                extraps=self.extrap_orders,
                name="nu", pos=0,
                out=res,
                idxs=(idx_xlo, idx_xhi))

        return res.reshape(shape)


class Interpolator2D(Interpolator):
    interpolator = interpolate.RectBivariateSpline

    def __init__(self, x, y, z, *, extrap_orders=None):
        super().__init__(extrap_orders=extrap_orders)
        self._points = x, y, z
        self._xmin, self._xmax = x[0], x[-1]
        self._ymin, self._ymax = y[0], y[-1]
        self.f = self.interpolator(x, y, z)

    def __call__(self, x, y, derivative=[0, 0], grid=True):
        if not grid:
            # TODO: For now, this option only does constant extrapolation.
            return self.f(x, y, *derivative, grid=False)

        x, y = map(np.atleast_1d, [x, y])
        x, y = get_broadcastable(x, y)
        # shape = x.shape + y.shape  # FIXME
        shape = np.broadcast_shapes(x.shape, y.shape)  # TODO: delete if ok
        x, y = x.squeeze(), y.squeeze()
        res = self.f(x, y, *derivative)

        idx_xlo = x < self._xmin
        idx_xhi = x > self._xmax
        extrap_x = idx_xlo.any() or idx_xhi.any()

        idx_ylo = y < self._ymin
        idx_yhi = y > self._ymax
        extrap_y = idx_ylo.any() or idx_yhi.any()

        if extrap_x:
            y_ev = y
            if extrap_y:
                y_ev = y_ev.copy()
                y_ev[idx_ylo], y_ev[idx_yhi] = self._ymin, self._ymax

            self._extrapolate_axis(
                xi=[x, y_ev],
                vals=(self._xmin, self._xmax),
                extraps=self.extrap_orders[0: 2],
                name="dx", pos=0,
                out=res,
                idxs=(idx_xlo, idx_xhi))

        if extrap_y:
            x_ev = x
            if extrap_x:
                x_ev = x_ev.copy()
                x_ev[idx_xlo], x_ev[idx_xhi] = self._xmin, self._xmax

            self._extrapolate_axis(
                xi=[x_ev, y],
                extraps=self.extrap_orders[2: 4],
                vals=(self._ymin, self._ymax),
                name="dy", pos=1,
                out=res,
                idxs=(idx_ylo, idx_yhi))

        return res.reshape(shape)


class Interpolator3D(Interpolator):
    interpolator = interpolate.RBFInterpolator

    def __init__(self, xi, d, **kwargs):
        self.f = self.interpolator(xi, d, neighbors=20)
        super().__init__(**kwargs)

    def __call__(self, xi):
        return self.f(xi)
