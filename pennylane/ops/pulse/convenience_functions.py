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

import numpy as np

has_jax = True
try:
    import jax.numpy as jnp
except ImportError:
    has_jax = False


def pwc(timespan):
    """Create a function that is piecewise-constant in time, based on the params for a TDHamiltonian.

    Args:
            timespan(Union[float, tuple(float, float)]: The timespan defining the region where the function is non-zero.
              If an integer is provided, the timespan is defined as ``(0, timespan)``.

    Returns:
            func: a function that contains two arguments, one for the trainable parameters(array) and
            one for time(int). When called, the function uses the array of parameters to create evenly sized bins
            within the ``timespan``, with each bin value set by an element of the array. It then selects the value
            the parameter array corresponding to the specified time.

    **Example**

    >>> t1, t2 = 1, 3
    >>> f1 = pwc((t1, t2))

    The resulting function ``f1`` has the call signature ``f1(params, t)``. If passed an array of parameters and
    a time, it will assign the array as the constants in the piecewise function, and select the constant corresponding
    to the specified time, based on the time interval defined by ``timespan``.

    >>> params = [np.linspace(10, 20, 10)]
    >>> f1(params, 2)
    tensor(15.55555556, requires_grad=True)

    >>> f1(params, 2.1)  # same bin
    tensor(15.55555556, requires_grad=True)

    >>> f1(params, 2.5)  # next bin
    tensor(17.77777778, requires_grad=True)
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
        # include 0 as an additional option for function output
        params = jnp.array(list(params) + [0])

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

    >>> def f0(params, t): return params[0] * t + params[1]
    >>> timespan = 10
    >>> num_bins = 10
    >>> f1 = pwc_from_function(timespan, num_bins)(f0)
    >>> f1([2, 4], 3), f0([2, 4], 3)
    (DeviceArray(10.666666, dtype=float32), DeviceArray(10, dtype=int32))

    >>> f1([2, 4], 3.2), f0([2, 4], 3.2)
    (DeviceArray(10.666666, dtype=float32), DeviceArray(10.4, dtype=float32))

    >>> f1([2, 4], 4.5), f0([2, 4], 4.5)
    (DeviceArray(12.888889, dtype=float32), DeviceArray(13., dtype=float32))

    # ToDo: can we include images in the docs for the version rendered for the website? Would be clearest way to illustrate

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
