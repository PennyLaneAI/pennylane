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
