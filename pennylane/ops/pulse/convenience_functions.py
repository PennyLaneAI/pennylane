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
import pennylane as qml

has_jax = True
try:
    import jax.numpy as jnp
except ImportError:
    has_jax = False


def pwc_from_array(t):
    """Create a function that is piecewise-constant in time, based on the params for a TDHamiltonian.

    Args:
            t(Union[float, tuple(float, float)]: the total duration as a float, or the start and end time as floats.
            index(int): the index at which the relevant parameter array is located in the overall ``params`` variable

    Returns:
            func: a function that can be passed the full ``params`` variable and ``t``, and will return the
                    corresponding constant

    **Example**

    >>> t1, t2 = 1, 3
    >>> index = 0
    >>> f1 = pwc_from_array((t1, t2), index)

    The resulting function ``f1`` has the call signature ``f1(params, t)``. If passed parameters and a time,
    it will assign the array at ``params[index]`` as the constants in the piecewise function, and select
    the constant corresponding to the specified time, based on the time interval defined by ``t``.

    >>> params = [np.linspace(10, 20, 10)]
    >>> f1(params, 2)
    tensor(15.55555556, requires_grad=True)

    >>> f1(params, 2.1)  # same bin
    tensor(15.55555556, requires_grad=True)

    >>> f1(params, 2.5)  # next bin
    tensor(17.77777778, requires_grad=True)
    """

    if isinstance(t, tuple):
        t1, t2 = t
    else:
        t1 = 0
        t2 = t

    def func(params, t):
        num_bins = len(params)
        idx = qml.math.array(num_bins / (t2 - t1) * (t - t1)).astype(int)
        return params[idx]

    return func


def pwc_from_function(t, num_bins):
    """
    Decorator to turn a smooth function into a piecewise constant function.

    Args:
            t(Union[float, tuple(float)]):
            num_bins(int): number of bins for time-binning the function

    Returns:
            a function that takes some smooth function ``f(params, t)`` and converts it to a
            piecewise constant function spanning time ``t`` in `num_bins` bins.

    **Example**

    >>> def f0(params, t): return params[0] * t + params[1]
    >>> t = 10
    >>> num_bins = 10
    >>> f1 = pwc_from_function(t, num_bins)(f0)
    >>> f1([2, 4], 3), f0([2, 4], 3)
    (DeviceArray(10.666666, dtype=float32), DeviceArray(10, dtype=int32))

    >>> f1([2, 4], 3.2), f0([2, 4], 3.2)
    (DeviceArray(10.666666, dtype=float32), DeviceArray(10.4, dtype=float32))

    >>> f1([2, 4], 4.5), f0([2, 4], 4.5)
    (DeviceArray(12.888889, dtype=float32), DeviceArray(13., dtype=float32))

    # ToDo: can we include images in the docs for the version rendered for the website? Would be clearest way to illustrate

    The same effect can be achieved by decorating the smooth function:

    >>> @pwc_from_function(t, num_bins)
    >>> def fn(params, t): return params[0] * t + params[1]
    >>> fn([2, 4], 3)
    DeviceArray(10.666666, dtype=float32)

    """

    if isinstance(t, tuple):
        t1, t2 = t
    else:
        t1 = 0
        t2 = t

    def inner(fn):
        time_bins = np.linspace(t1, t2, num_bins)

        def wrapper(params, t):
            constants = jnp.array(fn(params, time_bins))
            idx = qml.math.array(num_bins / (t2 - t1) * (t - t1)).astype(int)
            return constants[idx]

        return wrapper

    return inner
