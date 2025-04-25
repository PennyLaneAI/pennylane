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

import pennylane as qml

from .decomposition_rule import DecompositionRule, register_resources
from .resources import adjoint_resource_rep, pow_resource_rep, resource_rep


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


def _self_adjoint_resource(base_class, base_params):
    return {resource_rep(base_class, **base_params): 1}


# pylint: disable=protected-access,unused-argument
@register_resources(_self_adjoint_resource)
def self_adjoint(*params, wires, base):
    """The decomposition rule for an adjoint of an operator that is self-adjoint."""
    _, [_, metadata] = base._flatten()
    new_struct = wires, metadata
    base._unflatten(params, new_struct)


def _pow_resource(base_class, base_params, z):
    """Resources of the power of an operator."""
    if not isinstance(z, int) or z < 0:
        raise NotImplementedError("Non-integer or negative powers are not supported yet.")
    return {resource_rep(base_class, **base_params): z}


@register_resources(_pow_resource)
def pow_decomp(*_, base, z, **__):
    """Decompose the power of an operator."""
    assert isinstance(z, int) and z >= 0
    for _ in range(z):
        base._unflatten(*base._flatten())  # pylint: disable=protected-access


def _pow_pow_resource(base_class, base_params, z):  # pylint: disable=unused-argument
    """Resources of the power of the power of an operator."""
    base_class, base_params, base_z = (
        base_params["base_class"],
        base_params["base_params"],
        base_params["z"],
    )
    return {pow_resource_rep(base_class, base_params, z * base_z): 1}


@register_resources(_pow_pow_resource)
def pow_pow_decomp(*_, base, z, **__):
    """Decompose the power of the power of an operator."""
    qml.pow(base.base, z=z * base.z)
