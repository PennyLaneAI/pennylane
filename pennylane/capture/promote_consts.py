# Copyright 2018-2025 Xanadu Quantum Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=protected-access
"""
This submodule contains the `promote_consts` util.
"""
from typing import TYPE_CHECKING

from pennylane.typing import TensorLike

if TYPE_CHECKING:
    try:
        import jax
    except ImportError:
        pass


def promote_consts(
    jaxpr: "jax.extend.core.ClosedJaxpr", args: list[TensorLike] | tuple[TensorLike, ...]
) -> tuple["jax.extend.core.Jaxpr", tuple[TensorLike]]:
    """Convert a closed jaxpr into a jaxpr without constants and the constants prepended
    to the arguments.

    Args:
        jaxpr (jax.extend.core.ClosedJaxpr): a closed jaxpr with bound constants
        args (tuple[TensorLike]): the arguments we will add the constants to the start of

    Returns:

        (jax.extend.core.Jaxpr, tuple[TensorLike]): A Jaxpr and new arguments with the constants pre-pended.

    >>> import jax
    >>> def f(x):
    ...     return jax.numpy.array([2.0]) + x
    >>> args = (jax.numpy.array([0.5]), )
    >>> jaxpr = jax.make_jaxpr(f)(*args)
    >>> jaxpr
    { lambda a:f32[1]; b:f32[1]. let c:f32[1] = add a b in (c,) }
    >>> qml.capture.promote_consts(jaxpr, args)
    ({ lambda ; a:f32[1] b:f32[1]. let c:f32[1] = add a b in (c,) },
    (array([2.0], dtype=float32), Array([0.5], dtype=float32)))

    """
    new_args = tuple(jaxpr.consts) + tuple(args)
    jaxpr = jaxpr.jaxpr.replace(constvars=(), invars=jaxpr.jaxpr.constvars + jaxpr.jaxpr.invars)
    return jaxpr, new_args
