# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This submodule defines the abstract classes and primitives for capturing mid-circuit measurements.
"""

from functools import lru_cache
from typing import Optional

import pennylane as qml

has_jax = True
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    has_jax = False


@lru_cache
def create_mid_measure_primitive() -> Optional["jax.core.Primitive"]:
    """Create a primitive corresponding to an mid-circuit measurement type.

    Called when using :func:`~pennylane.measure`.

    Returns:
        Optional[jax.core.Primitive]: A new jax primitive corresponding to a mid-circuit
        measurement. ``None`` is returned if jax is not available.

    """
    if not has_jax:
        return None

    primitive = jax.core.Primitive("mid_measure")

    @primitive.def_impl
    def _(wires, reset=False, postselect=None):
        # pylint: disable=protected-access
        return qml.measurements.mid_measure._measure_impl(wires, reset=reset, postselect=postselect)

    @primitive.def_abstract_eval
    def _(*_, **__):
        return jax.core.ShapedArray((), jnp.bool_)

    return primitive
