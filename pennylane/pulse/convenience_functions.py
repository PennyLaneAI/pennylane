# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This file contains convenience functions for pulse programming."""
from typing import Callable, List, Tuple, Union
import numpy as np

has_jax = True
try:
    import jax.numpy as jnp
except ImportError:
    has_jax = False


def constant(windows: List[Tuple[float]] = None):
    """Returns a callable ``f(p, t)`` that returns ``p`` inside the time
    windows defined in ``windows``.

    Args:
        windows (Tuple[float, Tuple[float]]): List of tuples containing time windows where
        ``f(p, t)`` is evaluated. If ``None``, it is always evaluated. Defaults to ``None``.

    **Example**

    The ``constant`` function can be used to create a parametrized hamiltonian

    >>> windows = [(1, 7), (9, 14)]
    >>> H = qml.pulse.constant(windows) * qml.PauliX(0)

    When calling the parametrized hamiltonian, ``constant`` will return the input parameter only
    when the time is inside the given time windows

    >>> params = [5]
    >>> H(params, t=8)  # t is outside the time windows
    0.0*(PauliX(wires=[0]))
    >>> H(params, t=5)  # t is inside the time windows
    5.0*(PauliX(wires=[0]))
    """
    return rect(x=lambda p, _: p, windows=windows)


def rect(x: Union[float, Callable], windows: List[Tuple[float]] = None):
    """Multiplies ``x`` by a rectangular function, returning a callable ``f(p, t)`` that evaluates
    the given function/scalar ``x`` inside the time windows defined in ``windows``.

    .. note::

        If ``x`` is a function, it must accepts two arguments: the trainable parameters and time.

    Args:
        x (Union[float, Callable]): a scalar or a function that accepts two arguments: the trainable
            parameters and time
        windows (Tuple[float, Tuple[float]]): List of tuples containing time windows where x is
            evaluated. If ``None`` it is always evaluated. Defaults to ``None``.

    **Example**

    The ``rect`` function can be used to create a parametrized hamiltonian

    >>> def f1(p, t):
    ...     return jnp.polyval(p, t)
    >>> windows = [(1, 7), (9, 14)]
    >>> H = qml.pulse.rect(f1, windows) * qml.PauliX(0)

    When calling the parametrized hamiltonian, ``rect`` will evaluate the given function only
    inside the time windows

    >>> params = [jnp.ones(4)]
    >>> H(params, t=8)  # t is outside the time windows
    0.0*(PauliX(wires=[0]))
    >>> H(params, t=5)  # t is inside the time windows
    156.0*(PauliX(wires=[0]))

    One can also pass a scalar to the ``rect`` function

    >>> H = qml.pulse.rect(10, windows) * qml.PauliX(0)

    In this case, ``rect`` will return the given scalar only when the time is inside the provided
    time windows

    >>> params = [None]  # the parameter value won't be used!
    >>> H(params, t=8)
    0.0*(PauliX(wires=[0]))
    >>> H(params, t=5)
    10.0*(PauliX(wires=[0]))
    """
    if not has_jax:
        raise ImportError(
            "Module jax is required for any pulse-related convenience function. "
            "You can install jax via: pip install jax"
        )
    if not callable(x):

        def _f(_, __):
            return jnp.array(x, dtype=float)

    else:
        _f = x

    def f(p, t):
        p = jnp.array(p, dtype=float)  # if p is an integer, f(p, t) will be cast to an integer
        if windows is not None:
            return jnp.where(
                jnp.any(jnp.array([(t >= ti) & (t <= tf) for ti, tf in windows])), _f(p, t), 0
            )
        return _f(p, t)

    return f


def pwc(timespan):
    """Creates a function that is piecewise-constant in time.

    Args:
            timespan(Union[float, tuple(float, float)]: The timespan defining the region where the function is non-zero.
              If an integer is provided, the timespan is defined as ``(0, timespan)``.

    Returns:
            func: a function that takes two arguments, an array of trainable parameters and a `float` defining the
            time at which the function is evaluated. When called, the function uses the array of parameters to
            create evenly sized bins within the ``timespan``, with each bin value set by an element of the array.
            It then selects the value of the parameter array corresponding to the specified time, based on the
            assigned binning.

    **Example**

    >>> timespan = (1, 3)
    >>> f1 = pwc(timespan)

    The resulting function ``f1`` has the call signature ``f1(params, t)``. If passed an array of parameters and
    a time, it will assign the array as the constants in the piecewise function, and select the constant corresponding
    to the specified time, based on the time interval defined by ``timespan``.

    >>> params = [10, 11, 12, 13, 14]
    >>> f1(params, 2)
    Array(12, dtype=int32)

    >>> f1(params, 2.1)  # same bin
    Array(12, dtype=int32)

    >>> f1(params, 2.5)  # next bin
    Array(13, dtype=int32)
    """
    if not has_jax:
        raise ImportError(
            "Module jax is required for any pulse-related convenience function. "
            "You can install jax via: pip install jax"
        )

    if isinstance(timespan, tuple):
        t1, t2 = timespan
    else:
        t1 = 0
        t2 = timespan

    def func(params, t):
        num_bins = len(params)
        params = jnp.concatenate([jnp.array(params), jnp.zeros(1)])
        # get idx from timestamp, then set idx=0 if idx is out of bounds for the array
        idx = num_bins / (t2 - t1) * (t - t1)
        idx = jnp.where((idx >= 0) & (idx <= num_bins), jnp.array(idx, dtype=int), -1)

        return params[idx]

    return func


def pwc_from_function(timespan, num_bins):
    """
    Decorator to turn a smooth function into a piecewise constant function.

    Args:
            timespan(Union[float, tuple(float)]): The timespan defining the region where the function is non-zero.
              If an integer is provided, the timespan is defined as ``(0, timespan)``.
            num_bins(int): number of bins for time-binning the function

    Returns:
            a function that takes some smooth function ``f(params, t)`` and converts it to a
            piecewise constant function spanning time ``t`` in `num_bins` bins.

    **Example**

    .. code-block:: python3

        def smooth_function(params, t):
            return params[0] * t + params[1]

        timespan = 10
        num_bins = 10

        binned_function = pwc_from_function(timespan, num_bins)(f0)

    >>> binned_function([2, 4], 3), smooth_function([2, 4], 3)  # t = 3
    (DeviceArray(10.666666, dtype=float32), DeviceArray(10, dtype=int32))

    >>> binned_function([2, 4], 3.2), smooth_function([2, 4], 3.2)  # t = 3.2
    (DeviceArray(10.666666, dtype=float32), DeviceArray(10.4, dtype=float32))

    >>> binned_function([2, 4], 4.5), smooth_function([2, 4], 4.5)  # t = 4.5
    (DeviceArray(12.888889, dtype=float32), DeviceArray(13., dtype=float32))

    The same effect can be achieved by decorating the smooth function:

    >>> @pwc_from_function(timespan, num_bins)
    ... def fn(params, t):
    ...      return params[0] * t + params[1]
    >>> fn([2, 4], 3)
    DeviceArray(10.666666, dtype=float32)

    """
    if not has_jax:
        raise ImportError(
            "Module jax is required for any pulse-related convenience function. "
            "You can install jax via: pip install jax"
        )

    if isinstance(timespan, tuple):
        t1, t2 = timespan
    else:
        t1 = 0
        t2 = timespan

    def inner(fn):
        time_bins = np.linspace(t1, t2, num_bins)

        def wrapper(params, t):
            constants = jnp.array(list(fn(params, time_bins)) + [0])

            idx = num_bins / (t2 - t1) * (t - t1)
            # check interval is within 0 to num_bins, then cast to int, to avoid casting outcomes between -1 and 0 as 0
            idx = jnp.where((idx >= 0) & (idx <= num_bins), jnp.array(idx, dtype=int), -1)

            return constants[idx]

        return wrapper

    return inner
