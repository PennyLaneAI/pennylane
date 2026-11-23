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


"""Built-in decomposition rules for op_math.pow2."""

from pennylane import control_flow, math, ops
from pennylane.core.apply import apply
from pennylane.core.operator import abstractify
from pennylane.decomposition.decomposition_rule import (
    DecompCollection,
    DecompositionRule,
    _get_registered_decomps,
    _list_decomps,
    get_fixed_decomp,
    register_condition,
    register_resources,
)
from pennylane.decomposition.symbolic_decomposition import is_integer
from pennylane.decomposition.utils import to_name
from pennylane.ops.op_math import adjoint
from pennylane.ops.op_math.adjoint import Adjoint
from pennylane.ops.op_math.adjoint2 import Adjoint2
from pennylane.ops.op_math.pow2 import Pow2, _pow_abstract


@register_condition(lambda base, z, **_: is_integer(z) and z >= 0)
@register_resources(lambda base, z: {abstractify(base): z}, name="repeat_base")
def _pow_repeat_base(base, z):
    """Decompose the power of an operator by repeating the base operator. Assumes z
    is a non-negative integer."""

    @control_flow.for_loop(0, z)
    def _loop(_):
        apply(base)

    _loop()  # pylint: disable=no-value-for-parameter


@register_resources(
    lambda base, z: {ops.pow(abstractify(base.base), z * base.z): 1}, name="merge_powers"
)
def _pow_merge_powers(base, z):
    """Decompose nested powers by combining them."""
    ops.pow(base.base, z * base.z)


def _pow_flip_adjoint_resources(base, z):
    # base class is adjoint, and the base of the base is the target class
    return {adjoint(Pow2(base.base, z=z)): 1}


@register_resources(_pow_flip_adjoint_resources, name="flip_adjoint")
def _pow_flip_adjoint(base, z, **__):
    """Decompose the power of an adjoint by power to the base of the adjoint and
    then taking the adjoint of the power."""
    adjoint(ops.pow(base.base, z))


def make_pow_decomp_with_period(period) -> DecompositionRule:
    """Make a decomposition rule for the power of an op that has a period."""

    def _condition_fn(base, z):  # pylint: disable=unused-argument
        return math.shape(z) == () and z % period != z

    def _pow_reduce_period_resources(base, z):
        z_mod_period = z % period
        if z_mod_period == 0:
            return {}
        if z_mod_period == 1:
            return {abstractify(base): 1}
        return {_pow_abstract(base, z_mod_period): 1}

    @register_condition(_condition_fn)
    @register_resources(_pow_reduce_period_resources, name="reduce_period")
    def _pow_reduce_period(base, z, **__):
        z_mod_period = z % period
        if z_mod_period == 1:
            apply(base)
        elif z_mod_period > 0 and z_mod_period != period:
            ops.pow(base, z_mod_period)

    return _pow_reduce_period


pow_involutory = make_pow_decomp_with_period(2)


def _pow_scale_angle_resources(base, z):  # pylint: disable=unused-argument
    return {abstractify(base): 1}


@register_resources(_pow_scale_angle_resources, name="scale_angle")
def _pow_scale_angle(base, z):
    """Decompose the power of a general rotation operator by multiplying the power by the angle."""
    # A rotation should only have 1 dynamic parameter
    assert len(base.dynamic_argnames) == 1
    angle = tuple(base.dynamic_args.values())[0]
    ops.functions.bind_new_parameters(base, (angle * z,))


@_list_decomps.register
def _list_pow_decomps(op: Pow2) -> DecompCollection:
    abs_op = abstractify(op)

    # fixed_decomps would override everything.
    if fixed_rule := get_fixed_decomp(op):
        return DecompCollection([fixed_rule])

    # special case of merging nested powers.
    if isinstance(abs_op.base, Pow2):
        return DecompCollection([_pow_merge_powers])

    # special case that the base is an Adjoint.
    if isinstance(abs_op.base, (Adjoint, Adjoint2)):
        return DecompCollection([_pow_flip_adjoint])

    # Custom decomposition rules registered specifically to the pow operator
    custom_rules = _get_registered_decomps(to_name(abs_op))

    return custom_rules + [_pow_repeat_base] if is_integer(op.z) else custom_rules
