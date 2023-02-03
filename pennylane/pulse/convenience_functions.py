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


# pylint: disable=unused-argument
def constant(scalar, time):
    """Returns the given ``scalar``, for use in defining a :class:`~.ParametrizedHamiltonian` with a
    trainable coefficient.

    Creates a callable for defining a :class:`~.ParametrizedHamiltonian`.

    Args:
        scalar (float): the scalar to be returned
        time (float): Time. This argument is not used, but is required to match the call
            signature of :class:`~.ParametrizedHamiltonian`.

    This function is mainly used to build a :class:`~.ParametrizedHamiltonian` that can be differentiated
    with respect to its time-independent term. It is an alias for `lambda scalar, t: scalar`.

    **Example**

    The ``constant`` function can be used to create a parametrized Hamiltonian

    >>> H = qml.pulse.constant * qml.PauliX(0)

    When calling the parametrized Hamiltonian, ``constant`` will always return the input parameter

    >>> params = [5]
    >>> H(params, t=8)
    5.0*(PauliX(wires=[0]))
    >>> H(params, t=5)
    5.0*(PauliX(wires=[0]))

    We can differentiate the parametrized hamiltonian with respect to the constant parameter:

    .. code-block:: python

        dev = qml.device("default.qubit", wires=1)
        @qml.qnode(dev, interface="jax")
        def circuit(params):
            qml.evolve(H)(params, t=2)
            return qml.expval(qml.PauliZ(0))


    >>> params = jnp.array([5.0])
    >>> circuit(params)
    Array(0.40808904, dtype=float32)
    >>> jax.grad(circuit)(params)
    Array([-3.6517754], dtype=float32)
    """
    return scalar


def rect(x: Union[float, Callable], windows: List[Tuple[float]] = None):
    """Takes a scalar or a scalar-valued function, x, and applies a rectangular window to it, such that the
    returned function is x inside the window and 0 outside it.

    Creates a callable for defining a :class:`~.ParametrizedHamiltonian`.

    Args:
        x (Union[float, Callable]): a scalar or a function that accepts two arguments: the trainable
            parameters and time
        windows (Tuple[float, Tuple[float]]): List of tuples containing time windows where x is
            evaluated. If ``None`` it is always evaluated. Defaults to ``None``.

    Returns:
        A callable ``f(p, t)`` which evaluates the given function/scalar ``x`` inside the time windows defined in
        ``windows``, and otherwise returns 0.

    .. note::
        If ``x`` is a function, it must accept two arguments: the trainable parameters and time.

    **Example**

    Here we use :func:`~.rect` to create a parametrized coefficient that has a value of ``0`` outside the time interval
    ``t=(1, 7)``, and is defined by ``jnp.polyval(p, t)`` within the interval:

    .. figure:: ../../_static/pulse/rect_example.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    This can be used to create a :class:`~.ParametrizedHamiltonian` in the following way:

    .. code-block:: python

        >>> H = qml.pulse.rect(jnp.polyval, windows=[(1, 7)]) * qml.PauliX(0)

        # inside the window
        >>> H([3], t=2)
        2.7278921604156494*(PauliX(wires=[0]))

        # outside the window
        >>> H([3], t=0.5 )
        0.0*(PauliX(wires=[0]))

    It is also possible to define multiple windows for the same function:

    .. code-block:: python

        windows = [(1, 7), (9, 14)]
        H = qml.pulse.rect(jnp.polyval, windows) * qml.PauliX(0)

    When calling the :class:`.ParametrizedHamiltonian`, ``rect`` will evaluate the given function only
    inside the time windows

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
    """Takes a timespan and returns callable for creating a function that is piecewise-constant in time. The returned
    function takes arguments ``(p, t)``, where ``p`` is an array that defines the bin values for the function.

    Creates a callable for defining a :class:`~.ParametrizedHamiltonian`.

    Args:
            timespan(Union[float, tuple(float, float)]: The timespan defining the region where the function is non-zero.
              If an integer is provided, the timespan is defined as ``(0, timespan)``.

    Returns:
            func: a function that takes two arguments, an array of trainable parameters and a `float` defining the
            time at which the function is evaluated.

    This function can be used to create a parametrized coefficient function that is piecewise constant
    within the interval ``t``, and 0 outside it. When creating the callable, only the timespan is passed. The number
    of bins and values for the parameters are set when ``params`` is passed to the callable. Each bin value is set by
    an element of the ``params`` array. The variable ``t`` is used to select the value of the parameter array
    corresponding to the specified time, based on the assigned binning.

    .. figure:: ../../_static/pulse/pwc_example.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    **Example**

    >>> timespan = (2, 7)
    >>> f1 = qml.pulse.pwc(timespan)

    The resulting function ``f1`` has the call signature ``f1(params, t)``. If passed an array of parameters and
    a time, it will assign the array as the constants in the piecewise function, and select the constant corresponding
    to the specified time, based on the time interval defined by ``timespan``.

    .. code-block:: python

        >>> H = f1 * qml.PauliX(0)

        # passing pwc((2, 7)) an array evenly distributes the array values in the interval t=2 to t=7
        >>> H(params=[[11, 12, 13, 14, 15]], t=2.3)
        11.0*(PauliX(wires=[0]))

        # different time, same bin, same result
        >>> H(params=[[1, 2, 3, 4, 5]], t=2.5)
        11.0*(PauliX(wires=[0]))

        # next bin
        >>> H(params=[[1, 2, 3, 4, 5]], t=3.1)
        12.0*(PauliX(wires=[0]))

        # outside the window - the function is assigned non-zero values
        >>> H(params=[[1, 2, 3, 4, 5]], t=8)
        0.0*(PauliX(wires=[0]))

    .. note::
        The final time in the timespan indicates the index at which the function output switches from params[-1] to 0.
        As such, the final time in ``timespan`` returns 0:

        >>> H(params=[[1, 2, 3, 4, 5]], t=6.999999)
        15.0*(PauliX(wires=[0]))

        >>> H(params=[[1, 2, 3, 4, 5]], t=7)
        0.0*(PauliX(wires=[0]))

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
    Decorates a smooth function, creating a piecewise constant function that approximates it.

    Creates a callable for defining a :class:`~.ParametrizedHamiltonian`.

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

        binned_function = qml.pulse.pwc_from_function(timespan, num_bins)(smooth_function)

    >>> binned_function([2, 4], 3), smooth_function([2, 4], 3)  # t = 3
    (DeviceArray(10.666666, dtype=float32), DeviceArray(10, dtype=int32))

    >>> binned_function([2, 4], 3.2), smooth_function([2, 4], 3.2)  # t = 3.2
    (DeviceArray(10.666666, dtype=float32), DeviceArray(10.4, dtype=float32))

    >>> binned_function([2, 4], 4.5), smooth_function([2, 4], 4.5)  # t = 4.5
    (DeviceArray(12.888889, dtype=float32), DeviceArray(13., dtype=float32))

    The same effect can be achieved by decorating the smooth function:

    .. code-block:: python

        from pennylane.pulse.convenience_functions import pwc_from_function

        @pwc_from_function(timespan, num_bins)
        def fn(params, t):
            return params[0] * t + params[1]

        fn([2, 4], 3)
        >>> DeviceArray(10.666666, dtype=float32)

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
