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

    Args:
        scalar (float): the scalar to be returned
        time (float): Time. This argument is not used, but is required to match the call
            signature of :class:`~.ParametrizedHamiltonian`.
    Returns:
        float: The input ``scalar``.

    This function is mainly used to build a :class:`~.ParametrizedHamiltonian` that can be differentiated
    with respect to its time-independent term. It is an alias for ``lambda scalar, t: scalar``.

    **Example**

    The ``constant`` function can be used to create a parametrized Hamiltonian

    >>> H = qml.pulse.constant * qml.X(0)

    When calling the parametrized Hamiltonian, ``constant`` will always return the input parameter

    >>> params = [5]
    >>> H(params, t=8)
    5 * X(0)

    >>> H(params, t=5)
    5 * X(0)

    We can differentiate the parametrized Hamiltonian with respect to the constant parameter:

    .. code-block:: python

        dev = qml.device("default.qubit.jax", wires=1)
        @qml.qnode(dev, interface="jax")
        def circuit(params):
            qml.evolve(H)(params, t=2)
            return qml.expval(qml.Z(0))


    >>> params = jnp.array([5.0])
    >>> circuit(params)
    Array(0.40808904, dtype=float32)

    >>> jax.grad(circuit)(params)
    Array([-3.6517754], dtype=float32)
    """
    return scalar


def rect(x: Union[float, Callable], windows: Union[Tuple[float], List[Tuple[float]]] = None):
    """Takes a scalar or a scalar-valued function, x, and applies a rectangular window to it, such that the
    returned function is x inside the window and 0 outside it.

    Creates a callable for defining a :class:`~.ParametrizedHamiltonian`.

    Args:
        x (Union[float, Callable]): either a scalar, or a function that accepts two arguments: the trainable
            parameters and time
        windows (Union[Tuple[float], List[Tuple[float]]]): List of tuples containing time windows where ``x`` is
            evaluated. If ``None`` it is always evaluated. Defaults to ``None``.

    Returns:
        callable: A callable ``f(p, t)`` which evaluates the given function/scalar ``x`` inside the time windows defined in
        ``windows``, and otherwise returns 0.

    .. note::
        If ``x`` is a function, it must accept two arguments: the trainable parameters and time. The primary use
        of ``rect`` is for numerical simulations via :class:`ParametrizedEvolution`, which assumes ``t`` to be a single scalar
        argument. If you need to efficiently compute multiple times, you need to broadcast over ``t`` via
        `jax.vmap <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`_ (see examples below).

    **Example**

    Here we use :func:`~.rect` to create a parametrized coefficient that has a value of ``0`` outside the time interval
    ``t=(1, 7)``, and is defined by ``jnp.polyval(p, t)`` within the interval:

    .. code-block:: python3

        def f(p, t):
            return jnp.polyval(p, t)

        p = jnp.array([1, 2, 3])
        time = jnp.linspace(0, 10, 1000)
        windows = [(1, 7)]

        windowed_f = qml.pulse.rect(f, windows=windows)

        y1 = f(p, time)
        y2 = jax.vmap(windowed_f, (None, 0))(p, time)

        plt.plot(time, y1, label=f"polyval(p={p}, t)")
        plt.plot(time, y2, label=f"rect(polyval, windows={windows})(p={p}, t)")
        plt.legend()
        plt.xlabel("t")
        plt.show()

    .. figure:: ../../_static/pulse/rect_example.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    Note that in order to efficiently create ``y2``, we broadcasted ``windowed_f`` over the
    time argument using `jax.vmap <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`_.

    ``rect`` can be used to create a :class:`~.ParametrizedHamiltonian` in the following way:

    >>> H = qml.pulse.rect(jnp.polyval, windows=[(1, 7)]) * qml.X(0)

    The resulting Hamiltonian will be non-zero only inside the window.

    >>> H([[1, 3]], t=2)  # inside the window
    5.0 * X(0)

    >>> H([[1, 3]], t=0.5 )  # outside the window
    0.0 * X(0)

    It is also possible to define multiple windows for the same function:

    .. code-block:: python

        windows = [(1, 7), (9, 14)]
        H = qml.pulse.rect(jnp.polyval, windows) * qml.X(0)

    When calling the :class:`.ParametrizedHamiltonian`, ``rect`` will evaluate the given function only
    inside the time windows, and otherwise return 0.

    One can also pass a scalar to the ``rect`` function

    >>> H = qml.pulse.rect(10, (1, 7)) * qml.X(0)

    In this case, ``rect`` will return the given scalar only when the time is inside the provided
    time windows

    >>> params = [None]  # the parameter value won't be used!
    >>> H(params, t=8)
    0.0 * X(0)

    >>> H(params, t=5)
    10.0 * X(0)
    """
    if not has_jax:
        raise ImportError(
            "Module jax is required for any pulse-related convenience function. "
            "You can install jax via: pip install jax==0.4.10 jaxlib==0.4.10"
        )
    if windows is not None:
        is_nested = any(hasattr(w, "__len__") for w in windows)
        single_window = len(windows) == 2 and not is_nested
        if single_window:
            windows = [windows]
        elif not all(hasattr(w, "__len__") and len(w) == 2 for w in windows):
            raise ValueError("At least one provided window is not a two-element sequence.")

    if not callable(x):

        def _f(_, __):
            return jnp.array(x, dtype=float)

    else:
        _f = x

    def f(p, t):
        p = jnp.array(p, dtype=float)  # if p is an integer, f(p, t) will be cast to an integer
        if windows is not None:
            ti, tf = zip(*windows)
            ti, tf = jnp.array(ti), jnp.array(tf)
            return jnp.where(jnp.any((t >= ti) & (t <= tf)), _f(p, t), 0)
        return _f(p, t)

    return f


def pwc(timespan):
    """Takes a time span and returns a callable for creating a function that is piece-wise constant in time. The returned
    function takes arguments ``(p, t)``, where ``p`` is an array that defines the bin values for the function.

    Creates a callable for defining a :class:`~.ParametrizedHamiltonian`.

    Args:
        timespan(Union[float, tuple(float, float)]): The time span defining the region where the function is non-zero.
            If an integer is provided, the time span is defined as ``(0, timespan)``.

    Returns:
        callable: a function that takes two arguments: an ``array`` of trainable parameters, and a ``float`` defining the
        time at which the function is evaluated.

    The convenience function ``pwc`` essentially implements

    .. code-block:: python3

        def pwc(timespan):
            def wrapped(p, t):
                return p[int(t/len(p))]
            return wrapped

    This function can be used to create a parametrized coefficient function that is piece-wise constant
    within the interval ``t``, and 0 outside it.

    When creating the callable, only the time span is passed. The number
    of bins and values for the parameters are set when ``params`` is passed to the callable. Each bin value is set by
    an element of the ``params`` array. The variable ``t`` is used to select the value of the parameter array
    corresponding to the specified time, based on the assigned binning.

    .. code-block:: python3

        params = jnp.array([1, 2, 3, 4, 5])
        time = jnp.linspace(0, 10, 1000)
        timespan=(2, 7)
        y = qml.pulse.pwc(timespan)(params, time)
        plt.plot(time, y, label=f"params={params}, timespan={timespan}")
        plt.legend()
        plt.show()

    .. figure:: ../../_static/pulse/pwc_example.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    .. warning::
        The final time in the time span indicates the time at which the function output switches from params[-1] to 0.
        As such, the above function returns ``5`` for a time slightly smaller than the final time in ``timespan``,
        but it returns ``0`` for the final time itself:

        >>> qml.pulse.pwc(timespan)(params, 6.999999)
        Array(5., dtype=float32)

        >>> qml.pulse.pwc(timespan)(params, 7.)
        Array(0., dtype=float32)

    **Example**

    >>> timespan = (2, 7)
    >>> f1 = qml.pulse.pwc(timespan)
    >>> H = f1 * qml.X(0)

    The resulting function ``f1`` has the call signature ``f1(params, t)``. If passed an array of parameters and
    a time, it will assign the array as the constants in the piece-wise function, and select the constant corresponding
    to the specified time, based on the time interval defined by ``timespan``.

    In the following example, passing an array to ``pwc((2, 7))`` evenly distributes the array values in the
    interval ``t=2`` to ``t=7``. The time ``t`` is then used to select one of the array values based on this distribution.

    >>> H(params=[[11, 12, 13, 14, 15]], t=2.3)
    11.0 * X(0)

    >>> H(params=[[11, 12, 13, 14, 15]], t=2.5) # different time, same bin, same result
    11.0 * X(0)

    >>> H(params=[[11, 12, 13, 14, 15]], t=3.1) # next bin
    12.0 * X(0)

    >>> H(params=[[11, 12, 13, 14, 15]], t=8) # outside the window returns 0
    0.0 * X(0)

    """
    if not has_jax:
        raise ImportError(
            "Module jax is required for any pulse-related convenience function. "
            "You can install jax via: pip install jax==0.4.3 jaxlib==0.4.3"
        )

    if isinstance(timespan, (tuple, list)):
        t0, t1 = timespan
    else:
        t0 = 0
        t1 = timespan

    def func(params, t):
        num_bins = len(params)
        params = jnp.concatenate([jnp.array(params), jnp.zeros(1)])
        # get idx from timestamp, then set idx=0 if idx is out of bounds for the array
        idx = num_bins / (t1 - t0) * (t - t0)
        idx = jnp.where((idx >= 0) & (idx <= num_bins), jnp.array(idx, dtype=int), -1)

        return params[idx]

    return func


def pwc_from_function(timespan, num_bins):
    """
    Decorates a smooth function, creating a piece-wise constant function that approximates it.

    Creates a callable for defining a :class:`~.ParametrizedHamiltonian`.

    Args:
        timespan(Union[float, tuple(float)]): The time span defining the region where the function is non-zero.
            If a ``float`` is provided, the time span is defined as ``(0, timespan)``.
        num_bins(int): number of bins for time-binning the function

    Returns:
        callable: a function that takes some smooth function ``f(params, t)`` and converts it to a
        piece-wise constant function spanning time ``t`` in ``num_bins`` bins.

    **Example**

    .. code-block:: python3

        def smooth_function(params, t):
            return params[0] * t + params[1]

        timespan = 10
        num_bins = 10

        binned_function = qml.pulse.pwc_from_function(timespan, num_bins)(smooth_function)

    >>> binned_function([2, 4], 3), smooth_function([2, 4], 3)  # t = 3
    (Array(10.666667, dtype=float32), 10)

    >>> binned_function([2, 4], 3.2), smooth_function([2, 4], 3.2)  # t = 3.2
    (Array(10.666667, dtype=float32), 10.4)

    >>> binned_function([2, 4], 4.5), smooth_function([2, 4], 4.5)  # t = 4.5
    (Array(12.888889, dtype=float32), 13.0)

    The same effect can be achieved by decorating the smooth function:

    .. code-block:: python

        from pennylane.pulse.convenience_functions import pwc_from_function

        @pwc_from_function(timespan, num_bins)
        def fn(params, t):
            return params[0] * t + params[1]

    >>> fn([2, 4], 3)
    Array(10.666667, dtype=float32)

    """
    if not has_jax:
        raise ImportError(
            "Module jax is required for any pulse-related convenience function. "
            "You can install jax via: pip install jax==0.4.3 jaxlib==0.4.3"
        )

    if isinstance(timespan, tuple):
        t0, t1 = timespan
    else:
        t0 = 0
        t1 = timespan

    def inner(fn):
        time_bins = np.linspace(t0, t1, num_bins)

        def wrapper(params, t):
            constants = jnp.array(list(fn(params, time_bins)) + [0])

            idx = num_bins / (t1 - t0) * (t - t0)
            # check interval is within 0 to num_bins, then cast to int, to avoid casting outcomes between -1 and 0 as 0
            idx = jnp.where((idx >= 0) & (idx <= num_bins), jnp.array(idx, dtype=int), -1)

            return constants[idx]

        return wrapper

    return inner
