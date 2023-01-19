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
"""This file contains the ``piecewise`` pulse convenience function."""
from typing import Callable, Tuple, Union

has_jax = True
try:
    import jax.numpy as jnp
except ImportError:
    has_jax = False


def window(x: Union[float, Callable], *cond: Tuple[float, Tuple[float]]):
    """Evaluates the given function/scalar ``x``inside the time windows defined in ``cond``.

    .. note::

        If ``x`` is a callable, it must accepts two arguments: the trainable parameters and time.

    Args:
        x (Union[float, Callable]): a scalar or a function that accepts two arguments: the trainable
            parameters and time
        cond (Tuple[float, Tuple[float]]): tuple of tuples containing time windows where x is evaluated
    """
    if not has_jax:
        raise ImportError(
            "Module jax is required for ``ParametrizedEvolution`` class. "
            "You can install jax via: pip install jax"
        )
    if not callable(x):

        def f(_, __):
            return jnp.array(x, dtype=float)

    else:
        f = x

    def _window(p, t):
        p = jnp.array(p, dtype=float)  # if p is an integer, f(p, t) will be cast to an integer
        return jnp.where(jnp.any(jnp.array([(t >= ti) & (t <= tf) for ti, tf in cond])), f(p, t), 0)

    return _window
