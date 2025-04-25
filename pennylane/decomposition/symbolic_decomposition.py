# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains special logic of decomposing symbolic operations."""

from __future__ import annotations

import numpy as np

import pennylane as qml

from .decomposition_rule import DecompositionRule, register_resources
from .resources import adjoint_resource_rep, pow_resource_rep, resource_rep
from .utils import DecompositionNotApplicable


def make_adjoint_decomp(base_decomposition: DecompositionRule):
    """Create a decomposition rule for the adjoint of a decomposition rule."""

    def _resource_fn(base_class, base_params):  # pylint: disable=unused-argument
        base_resources = base_decomposition.compute_resources(**base_params)
        return {
            adjoint_resource_rep(decomp_op.op_type, decomp_op.params): count
            for decomp_op, count in base_resources.gate_counts.items()
        }

    @register_resources(_resource_fn)
    def _impl(*params, wires, base, **__):
        # pylint: disable=protected-access
        qml.adjoint(base_decomposition._impl)(*params, wires, **base.hyperparameters)

    return _impl


def _cancel_adjoint_resource(*_, base_params, **__):
    # The base of a nested adjoint is an adjoint, so the base of the base is the original operator,
    # and the "base_params" of base_params are the resource params of the original operator.
    base_class, base_params = base_params["base_class"], base_params["base_params"]
    return {resource_rep(base_class, **base_params): 1}


# pylint: disable=protected-access,unused-argument
@register_resources(_cancel_adjoint_resource)
def cancel_adjoint(*params, wires, base):
    """Decompose the adjoint of the adjoint of an operator."""
    _, [_, metadata] = base.base._flatten()
    new_struct = wires, metadata
    base.base._unflatten(params, new_struct)


def _adjoint_rotation(base_class, base_params, **__):
    return {resource_rep(base_class, **base_params): 1}


@register_resources(_adjoint_rotation)
def adjoint_rotation(phi, wires, base, **__):
    """Decompose the adjoint of a rotation operator by negating the angle."""
    _, [_, metadata] = base._flatten()
    new_struct = wires, metadata
    base._unflatten((-phi,), new_struct)


def is_integer(x):
    """Checks if x is an integer."""
    return isinstance(x, int) or np.issubdtype(getattr(x, "dtype", None), np.integer)


def _repeat_pow_base_resource(base_class, base_params, z):
    if (not is_integer(z)) or z < 0:
        raise DecompositionNotApplicable
    return {resource_rep(base_class, **base_params): z}


@register_resources(_repeat_pow_base_resource)
def repeat_pow_base(*params, wires, base, z, **__):
    """Decompose the power of an operator by repeating the base operator. Assumes z
    is a non-negative integer."""

    @qml.for_loop(0, z)
    def _loop(i):
        _, [_, metadata] = base._flatten()
        new_struct = wires, metadata
        base._unflatten(params, new_struct)

    _loop()


def _merge_powers_resource(base_class, base_params, z):  # pylint: disable=unused-argument
    return {
        pow_resource_rep(
            base_params["base_class"],
            base_params["base_params"],
            z * base_params["z"],
        ): 1
    }


@register_resources(_merge_powers_resource)
def merge_powers(*params, wires, base, z, **__):
    """Decompose nested powers by combining them."""
    _, [_, metadata] = base.base._flatten()
    new_struct = wires, metadata
    base_op = base.base._unflatten(params, new_struct)
    qml.pow(base_op, z * base_op.z)


def _flip_pow_adjoint_resource(base_class, base_params, z):  # pylint: disable=unused-argument
    # base class is adjoint, and the base of the base is the target class
    target_class, target_params = base_params["base_class"], base_params["base_params"]
    return {
        adjoint_resource_rep(
            qml.ops.Pow, {"base_class": target_class, "base_params": target_params, "z": z}
        ): 1
    }


@register_resources(_flip_pow_adjoint_resource)
def flip_pow_adjoint(*params, wires, base, z, **__):
    """Decompose the power of an adjoint by power to the base of the adjoint and
    then taking the adjoint of the power."""
    _, [_, metadata] = base.base._flatten()
    new_struct = wires, metadata
    base_op = base.base._unflatten(params, new_struct)
    qml.adjoint(qml.pow(base_op, z))


def _pow_self_adjoint_resource(base_class, base_params, z):  # pylint: disable=unused-argument
    if (not is_integer(z)) or z < 0:
        raise DecompositionNotApplicable
    return {resource_rep(base_class, **base_params): z % 2}


@register_resources(_pow_self_adjoint_resource)
def pow_self_adjoint(*params, wires, base, z, **__):
    """Decompose the power of a self-adjoint operator, assumes z is an integer."""

    def f():
        _, [_, metadata] = base._flatten()
        new_struct = wires, metadata
        base._unflatten(params, new_struct)

    qml.cond(z % 2 == 1, f)()


def _pow_rotation_resource(base_class, base_params, z):  # pylint: disable=unused-argument
    return {resource_rep(base_class, **base_params): 1}


# pylint: disable=protected-access
@register_resources(_pow_rotation_resource)
def pow_rotation(phi, wires, base, z, **__):
    """Decompose the power of a general rotation operator by multiplying the power by the angle."""
    _, [_, metadata] = base._flatten()
    new_struct = wires, metadata
    base._unflatten((phi * z,), new_struct)


def _decompose_to_base_resource(base_class, base_params, **__):
    return {resource_rep(base_class, **base_params): 1}


@register_resources(_decompose_to_base_resource)
def decompose_to_base(*params, wires, base, **__):
    """Decompose a symbolic operator to its base."""
    _, [_, metadata] = base._flatten()
    new_struct = wires, metadata
    base._unflatten(params, new_struct)


self_adjoint: DecompositionRule = decompose_to_base
