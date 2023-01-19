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
has_jax = True
try:
    import jax.numpy as jnp
except ImportError:
    has_jax = False


def rect(x, ti, tf):
    """Applies a rectangular function to the provided ``x``, such that:

    - ``rect(f, ti, tf) == x / 2`` if ``t == ti``
    - ``rect(f, ti, tf) == x`` if ``ti < t < tf``
    - ``rect(f, ti, tf) == 0`` otherwise

    .. note::

        ``x`` can be a callable that accepts two arguments: trainable parameters and time. In this
        case, the rectangular function is applied to ``x(params, t)``.

    Args:
        x (float | callable): a scalar or a function that accepts two arguments: the trainable
            parameters and time
        ti (float): initial time
        tf (float): final time
    """
    if not has_jax:
        raise ImportError(
            "Module jax is required for ``ParametrizedEvolution`` class. "
            "You can install jax via: pip install jax"
        )
    if not callable(x):

        def f(_, __):
            return jnp.array(x)

    else:
        f = x

    return lambda p, t: jnp.piecewise(p, [ti <= t <= tf], [f], t)
