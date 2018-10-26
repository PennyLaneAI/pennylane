import collections
import numpy as np
import numbers


def _flatten(x):
    """Iterate through an arbitrarily nested structure, flattening it in depth-first order.

    See also :func:`_unflatten`.

    Args:
        x (array, Iterable, other): each element of the Iterable may be of the same types as x

    Yields:
        other: elements of x in depth-first order
    """
    if isinstance(x, np.ndarray):
        yield from _flatten(x.flat)  # should we allow object arrays? or just "yield from x.flat"?
    elif isinstance(x, collections.Iterable) and not isinstance(x, (str, bytes)):
        for item in x:
            yield from _flatten(item)
    else:
        yield x


def _unflatten(flat, model):
    """Restores an arbitrary nested structure to a flattened iterable.

    See also :func:`_flatten`.

    Args:
        flat (array): 1D array of items
        model (array, Iterable, Number): model nested structure

    Returns:
        (other, array): first elements of flat arranged into the nested
        structure of model, unused elements of flat
    """
    if isinstance(model, np.ndarray):
        idx = model.size
        res = np.array(flat)[:idx].reshape(model.shape)
        return res, flat[idx:]
    elif isinstance(model, collections.Iterable):
        res = []
        for x in model:
            val, flat = _unflatten(flat, x)
            res.append(val)
        return res, flat
    elif isinstance(model, (numbers.Number,)):
        return flat[0], flat[1:]
    else:
        raise TypeError('Unsupported type in the model: {}'.format(type(model)))
