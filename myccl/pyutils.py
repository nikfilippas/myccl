import numpy as np
from inspect import signature
import functools


def expand_dim(arr, ndim, pos) -> np.ndarray:
    """Expand array an arbitrary number of dimensions.

    .. note::

        Similar to ``numpy.expand_dims``, with enhanced functionality.

    Arguments
    ---------
    arr : :class:`numpy.ndarray`
        Input array.
    ndim : int
        Total number of output dimensions.
    pos : int
        Dimension where the array will be placed.

    Returns
    -------
    out : (1, ..., arr, ..., 1) :class:`numpy.ndarray`
        Expanded array.
    """
    shape = (1,)*pos + arr.shape + (1,)*(ndim-pos-1)
    return arr.reshape(shape)


def get_expanded(*arrs) -> list:
    """Orthogonalize the input arrays, in the input order."""
    ndim = len(arrs)
    return [expand_dim(arr, ndim, pos) for pos, arr in enumerate(arrs)]


def get_broadcastable(*arrs) -> list:
    """Helper to get broadcastable arrays.

    Given some input arrays representing different quantities,
    output reshaped arrays which can be broadcast together.

    Examples
    --------

    In the examples following, the variable ``shape`` refers to
    the combined broadcast shape of the output arrays.

    If all arrays are of size 0 or 1, they are not reshaped:

    .. code-block::

        >>> a, b = np.array(0), np.array(1)
        >>> (a, b), shape = get_broadcastable(a, b)
        >>> a, b, shape
        (array(0), array(1), ())

    If the arrays are already broadcastable but **don't** have equal shapes),
    they are broadcast:

    .. code-block::

        >>> a, b = np.arange(0, 2), np.arange(0, 3)
        >>> a, b = a[:, None], b[None, :]
        >>> (a, b), shape = get_broadcastable(a, b)
        >>> a, b, shape
        (array([[0, 0, 0], [1, 1, 1]]), array([[0, 1, 2], [0, 1, 2]]), (2, 3))

    If some arrays have the same shape, their dimensions are expanded
    (as they are meant to represent different quantities):

    .. code-block::

        >>> a, b = np.arange(0, 2), np.arange(1, 3)
        >>> (a, b), shape = get_broadcastable(a, b)
        >>> a, b, shape
        (array([[0], [1]]), array([[1, 2]]), (2, 2))

    The same applies to arrays that cannot be broadcast using NumPy rules:

    .. code-block::

        >>> a, b = np.arange(0, 2), np.arange(0, 3)
        >>> (a, b), shape = get_broadcastable(a, b)
        >>> a, b, shape
        (array([[0], [1]]), array([[0, 1, 2]]), (2, 3))
    """
    # if sum([arr.size < 2 for arr in arrs]) >= len(arrs) - 1:
    #     # Both just numbers. Allow one array to have dimensions.
    #     return arrs

    shapes = [arr.shape for arr in arrs]
    if len(set(shapes)) != len(shapes):
        # Some shapes are equal! While these are broadcastable,
        # we want to orthogonalize them as they are different quantities.
        return get_expanded(*arrs)

    if any([shape == (1,) for shape in shapes]):
        # One of the arrays is just a number, so it is broadcastable.
        # Don't broadcast, orthogonalize instead.
        return get_expanded(*arrs)

    try:
        # Try broadcasting the arrays against each other.
        np.broadcast_shapes(*[arr.shape for arr in arrs])
        return arrs
    except ValueError:
        # Expand the dimensions if numpy broadcasting doesn't work.
        return get_expanded(*arrs)


def squeeze_dispatcher(func=None, *, dims=None):
    """Wrap function to make it squeezable.

    Dimensions are first internally orthogonalized as indicated in ``dims``,
    fed into the original function, and output according to the added
    ``squeeze`` keyword.

    Arguments
    ---------
    func : function
        Function to wrap.
    dims : str
        Names of dimensions in the order they appear in the output separated.
        by whitespace. For example, ``dims=['a b']`` will output an array of
        shape ``(na, nb)``.
    """
    if func is None:
        dims = dims.split()
        return functools.partial(squeeze_dispatcher, dims=dims)

    sign = signature(func)

    @functools.wraps(func)
    def new_func(*args, squeeze=True, **kwargs):
        # Bind input args and kwargs to original function.
        bound = sign.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments

        # Broadcast them as needed.
        axes = [np.atleast_1d(arguments[param]) for param in dims]
        axes = get_broadcastable(*axes)
        arguments.update(dict(zip(dims, axes)))

        # Call original function and squeeze as needed.
        ret = func(**arguments)
        return ret.squeeze()[()] if squeeze else ret

    return new_func



def resample_array():
    pass


def _fftlog_transform():
    pass
