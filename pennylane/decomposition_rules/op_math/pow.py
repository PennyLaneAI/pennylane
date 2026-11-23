# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Built-in decomposition rules for op_math.pow."""

from __future__ import annotations

import pennylane as qp
from pennylane import math
from pennylane.core import Operator2
from pennylane.core.operator import abstractify
from pennylane.decomposition.decomposition_rule import (
    DecompositionRule,
    register_condition,
    register_resources,
)
from pennylane.decomposition.resources import adjoint_resource_rep, pow_resource_rep, resource_rep
from pennylane.decomposition.symbolic_decomposition import is_integer


@register_condition(lambda z, **__: is_integer(z) and z >= 0)
@register_resources(
    lambda base_class, base_params, z: {resource_rep(base_class, **base_params): z},
    name="repeat_base",
)
def _pow_repeat_base(*params, wires, base, z, **__):  # pylint: disable=unused-argument
    """Decompose the power of an operator by repeating the base operator. Assumes z
    is a non-negative integer."""

    @qp.for_loop(0, z)
    def _loop(i):  # pylint: disable=unused-argument
        qp.apply(base)

    _loop()  # pylint: disable=no-value-for-parameter


def _pow_merge_powers_resources(base_class, base_params, z):  # pylint: disable=unused-argument
    return {
        pow_resource_rep(
            base_params["base_class"],
            base_params["base_params"],
            z * base_params["z"],
        ): 1
    }


@register_resources(_pow_merge_powers_resources, name="merge_powers")
def _pow_merge_powers(*params, wires, base, z, **__):  # pylint: disable=unused-argument
    """Decompose nested powers by combining them."""
    qp.pow(base.base, z * base.z)


def _pow_flip_adjoint_resources(base_class, base_params, z):  # pylint: disable=unused-argument
    # base class is adjoint, and the base of the base is the target class
    target_class, target_params = base_params["base_class"], base_params["base_params"]
    return {
        adjoint_resource_rep(
            qp.ops.Pow, {"base_class": target_class, "base_params": target_params, "z": z}
        ): 1
    }


@register_resources(_pow_flip_adjoint_resources, name="flip_adjoint")
def _pow_flip_adjoint(*params, wires, base, z, **__):  # pylint: disable=unused-argument
    """Decompose the power of an adjoint by power to the base of the adjoint and
    then taking the adjoint of the power."""
    qp.adjoint(qp.pow(base.base, z))


def make_pow_decomp_with_period(period) -> DecompositionRule:
    """Make a decomposition rule for the power of an op that has a period."""

    def _condition_fn(base_class, base_params, z):  # pylint: disable=unused-argument
        return math.shape(z) == () and z % period != z

    def _pow_reduce_period_resources(base_class, base_params, z):
        z_mod_period = z % period
        if z_mod_period == 0:
            return {}
        if z_mod_period == 1:
            if issubclass(base_class, Operator2):
                return {abstractify(base_class(**base_params)): 1}
            return {resource_rep(base_class, **base_params): 1}
        return {pow_resource_rep(base_class, base_params, z_mod_period): 1}

    @register_condition(_condition_fn)
    @register_resources(_pow_reduce_period_resources, name="reduce_period")
    def _pow_reduce_period(*params, wires, base, z, **__):  # pylint: disable=unused-argument
        z_mod_period = z % period
        if z_mod_period == 1:
            qp.apply(base)
        elif z_mod_period > 0 and z_mod_period != period:
            qp.pow(base, z_mod_period)

    return _pow_reduce_period


pow_involutory = make_pow_decomp_with_period(2)


def _pow_scale_angle_resources(base_class, base_params, z):  # pylint: disable=unused-argument
    return {resource_rep(base_class, **base_params): 1}


@register_resources(_pow_scale_angle_resources, name="scale_angle")
def _pow_scale_angle(phi, wires, base, z, **__):  # pylint: disable=unused-argument
    """Decompose the power of a general rotation operator by multiplying the power by the angle."""
    qp.ops.functions.bind_new_parameters(base, (phi * z,))
