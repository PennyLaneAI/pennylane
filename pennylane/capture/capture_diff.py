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
This submodule offers differentiation-related primitives and types for
the PennyLane capture module.
"""
from functools import lru_cache

has_jax = True
try:
    import jax
except ImportError:
    has_jax = False


@lru_cache
def _get_grad_prim():
    """Create a primitive for gradient computations.
    This primitive is used when capturing ``qml.grad``.
    """
    if not has_jax:  # pragma: no cover
        return None

    from .custom_primitives import NonInterpPrimitive  # pylint: disable=import-outside-toplevel

    grad_prim = NonInterpPrimitive("grad")
    grad_prim.multiple_results = True  # pylint: disable=attribute-defined-outside-init
    grad_prim.prim_type = "higher_order"

    # pylint: disable=too-many-arguments
    @grad_prim.def_impl
    def _(*args, argnum, jaxpr, n_consts, method, h):
        if method or h:  # pragma: no cover
            raise ValueError(f"Invalid values '{method=}' and '{h=}' without QJIT.")
        consts = args[:n_consts]
        args = args[n_consts:]

        def func(*inner_args):
            return jax.core.eval_jaxpr(jaxpr, consts, *inner_args)[0]

        return jax.grad(func, argnums=argnum)(*args)

    # pylint: disable=unused-argument
    @grad_prim.def_abstract_eval
    def _(*args, argnum, jaxpr, n_consts, method, h):
        if len(jaxpr.outvars) != 1 or jaxpr.outvars[0].aval.shape != ():
            raise TypeError("Grad only applies to scalar-output functions. Try jacobian.")
        return tuple(jaxpr.invars[i].aval for i in argnum)

    return grad_prim


@lru_cache
def _get_jacobian_prim():
    """Create a primitive for Jacobian computations.
    This primitive is used when capturing ``qml.jacobian``.
    """
    if not has_jax:  # pragma: no cover
        return None

    from .custom_primitives import NonInterpPrimitive  # pylint: disable=import-outside-toplevel

    jacobian_prim = NonInterpPrimitive("jacobian")
    jacobian_prim.multiple_results = True  # pylint: disable=attribute-defined-outside-init
    jacobian_prim.prim_type = "higher_order"

    # pylint: disable=too-many-arguments
    @jacobian_prim.def_impl
    def _(*args, argnum, jaxpr, n_consts, method, h):
        if method or h:  # pragma: no cover
            raise ValueError(f"Invalid values '{method=}' and '{h=}' without QJIT.")
        consts = args[:n_consts]
        args = args[n_consts:]

        def func(*inner_args):
            return jax.core.eval_jaxpr(jaxpr, consts, *inner_args)

        return jax.tree_util.tree_leaves(jax.jacobian(func, argnums=argnum)(*args))

    # pylint: disable=unused-argument
    @jacobian_prim.def_abstract_eval
    def _(*args, argnum, jaxpr, n_consts, method, h):
        in_avals = [jaxpr.invars[i].aval for i in argnum]
        out_shapes = (outvar.aval.shape for outvar in jaxpr.outvars)
        return [
            jax.core.ShapedArray(out_shape + in_aval.shape, in_aval.dtype)
            for out_shape in out_shapes
            for in_aval in in_avals
        ]

    return jacobian_prim
