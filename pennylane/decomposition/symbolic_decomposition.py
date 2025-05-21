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

from .decomposition_rule import DecompositionRule, register_condition, register_resources
from .resources import adjoint_resource_rep, pow_resource_rep, resource_rep


def make_adjoint_decomp(base_decomposition: DecompositionRule):
    """Create a decomposition rule for the adjoint of a decomposition rule."""

    def _condition_fn(base_class, base_params):  # pylint: disable=unused-argument
        return base_decomposition.is_applicable(**base_params)

    def _resource_fn(base_class, base_params):  # pylint: disable=unused-argument
        base_resources = base_decomposition.compute_resources(**base_params)
        return {
            adjoint_resource_rep(decomp_op.op_type, decomp_op.params): count
            for decomp_op, count in base_resources.gate_counts.items()
        }

    @register_condition(_condition_fn)
    @register_resources(_resource_fn)
    def _impl(*params, wires, base, **__):
        # pylint: disable=protected-access
        qml.adjoint(base_decomposition._impl)(*params, wires=wires, **base.hyperparameters)

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
    base.base._unflatten(*base.base._flatten())


def _adjoint_rotation(base_class, base_params, **__):
    return {resource_rep(base_class, **base_params): 1}


# pylint: disable=protected-access,unused-argument
@register_resources(_adjoint_rotation)
def adjoint_rotation(phi, wires, base, **__):
    """Decompose the adjoint of a rotation operator by inverting the angle."""
    _, struct = base._flatten()
    base._unflatten((-phi,), struct)


def is_integer(x):
    """Checks if x is an integer."""
    return isinstance(x, int) or np.issubdtype(getattr(x, "dtype", None), np.integer)


# pylint: disable=protected-access,unused-argument
@register_condition(lambda z, **__: is_integer(z) and z >= 0)
@register_resources(lambda base_class, base_params, z: {resource_rep(base_class, **base_params): z})
def repeat_pow_base(*params, wires, base, z, **__):
    """Decompose the power of an operator by repeating the base operator. Assumes z
    is a non-negative integer."""

    @qml.for_loop(0, z)
    def _loop(i):
        base._unflatten(*base._flatten())

    _loop()  # pylint: disable=no-value-for-parameter


def _merge_powers_resource(base_class, base_params, z):  # pylint: disable=unused-argument
    return {
        pow_resource_rep(
            base_params["base_class"],
            base_params["base_params"],
            z * base_params["z"],
        ): 1
    }


# pylint: disable=protected-access,unused-argument
@register_resources(_merge_powers_resource)
def merge_powers(*params, wires, base, z, **__):
    """Decompose nested powers by combining them."""
    base_op = base.base._unflatten(*base.base._flatten())
    qml.pow(base_op, z * base.z)


def _flip_pow_adjoint_resource(base_class, base_params, z):  # pylint: disable=unused-argument
    # base class is adjoint, and the base of the base is the target class
    target_class, target_params = base_params["base_class"], base_params["base_params"]
    return {
        adjoint_resource_rep(
            qml.ops.Pow, {"base_class": target_class, "base_params": target_params, "z": z}
        ): 1
    }


# pylint: disable=protected-access,unused-argument
@register_resources(_flip_pow_adjoint_resource)
def flip_pow_adjoint(*params, wires, base, z, **__):
    """Decompose the power of an adjoint by power to the base of the adjoint and
    then taking the adjoint of the power."""
    base_op = base.base._unflatten(*base.base._flatten())
    qml.adjoint(qml.pow(base_op, z))


def make_pow_decomp_with_period(period) -> DecompositionRule:
    """Make a decomposition rule for the power of an op that has a period."""

    def _condition_fn(base_class, base_params, z):  # pylint: disable=unused-argument
        return z % period != z

    def _resource_fn(base_class, base_params, z):
        z_mod_period = z % period
        if z_mod_period == 0:
            return {}
        if z_mod_period == 1:
            return {resource_rep(base_class, **base_params): 1}
        return {pow_resource_rep(base_class, base_params, z_mod_period): 1}

    @register_condition(_condition_fn)
    @register_resources(_resource_fn)
    def _impl(*params, wires, base, z, **__):  # pylint: disable=unused-argument
        z_mod_period = z % period
        if z_mod_period == 1:
            base._unflatten(*base._flatten())
        elif z_mod_period > 0 and z_mod_period != period:
            qml.pow(base, z_mod_period)

    return _impl


pow_involutory = make_pow_decomp_with_period(2)


def _pow_rotation_resource(base_class, base_params, z):  # pylint: disable=unused-argument
    return {resource_rep(base_class, **base_params): 1}


# pylint: disable=protected-access,unused-argument
@register_resources(_pow_rotation_resource)
def pow_rotation(phi, wires, base, z, **__):
    """Decompose the power of a general rotation operator by multiplying the power by the angle."""
    _, struct = base._flatten()
    base._unflatten((phi * z,), struct)


def _decompose_to_base_resource(base_class, base_params, **__):
    return {resource_rep(base_class, **base_params): 1}


# pylint: disable=protected-access,unused-argument
@register_resources(_decompose_to_base_resource)
def decompose_to_base(*params, wires, base, **__):
    """Decompose a symbolic operator to its base."""
    base._unflatten(*base._flatten())


self_adjoint: DecompositionRule = decompose_to_base
