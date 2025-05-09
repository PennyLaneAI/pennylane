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
from .resources import adjoint_resource_rep, controlled_resource_rep, pow_resource_rep, resource_rep
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
    """Decompose the adjoint of a rotation operator by negating the angle."""
    _, struct = base._flatten()
    base._unflatten((-phi,), struct)


def is_integer(x):
    """Checks if x is an integer."""
    return isinstance(x, int) or np.issubdtype(getattr(x, "dtype", None), np.integer)


def _repeat_pow_base_resource(base_class, base_params, z):
    if (not is_integer(z)) or z < 0:
        raise DecompositionNotApplicable
    return {resource_rep(base_class, **base_params): z}


# pylint: disable=protected-access,unused-argument
@register_resources(_repeat_pow_base_resource)
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


def _pow_self_adjoint_resource(base_class, base_params, z):  # pylint: disable=unused-argument
    if (not is_integer(z)) or z < 0:
        raise DecompositionNotApplicable
    return {resource_rep(base_class, **base_params): z % 2}


# pylint: disable=protected-access,unused-argument
@register_resources(_pow_self_adjoint_resource)
def pow_of_self_adjoint(*params, wires, base, z, **__):
    """Decompose the power of a self-adjoint operator, assumes z is an integer."""

    def f():
        base._unflatten(*base._flatten())

    qml.cond(z % 2 == 1, f)()


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


def make_controlled_decomp(base_decomposition):
    """Create a decomposition rule for the control of a decomposition rule."""

    def _resource_fn(
        *_, base_params, num_control_wires, num_zero_control_values, num_work_wires, **__
    ):
        base_resources = base_decomposition.compute_resources(**base_params)
        gate_counts = {
            _controlled_resource_rep(base_op_rep, num_control_wires, num_work_wires): count
            for base_op_rep, count in base_resources.gate_counts.items()
        }
        # None of the other gates in gate_counts will be X, because they are all
        # controlled operations. So we can safely set the X gate counts here.
        gate_counts[resource_rep(qml.PauliX)] = num_zero_control_values * 2
        return gate_counts

    @register_resources(_resource_fn)
    def _impl(*params, wires, control_wires, control_values, work_wires, base, **__):
        zero_control_wires = [w for w, val in zip(control_wires, control_values) if not val]
        for w in zero_control_wires:
            qml.PauliX(w)
        # We're extracting control wires and base wires from the wires argument instead
        # of directly using control_wires and base.wires, `wires` is properly traced, but
        # `control_wires` and `base.wires` are not.
        qml.ctrl(
            base_decomposition._impl,  # pylint: disable=protected-access
            control=wires[: len(control_wires)],
            work_wires=work_wires,
        )(*params, wires=wires[-len(base.wires) :], **base.hyperparameters)
        for w in zero_control_wires:
            qml.PauliX(w)

    return _impl


def _controlled_resource_rep(base_op_rep, num_control_wires, num_work_wires):
    """Returns the resource rep of a controlled op, dispatches to a custom op if possible.

    The purpose of this function is to replicate the dispatch logic in qml.ctrl. Since in the
    decomposition rule, qml.ctrl is called on the base decomposition, and it may dispatch to
    one of the custom controlled ops, the resource function should also reflect that.

    """

    base_to_custom_ctrl_op = qml.ops.op_math.controlled.base_to_custom_ctrl_op()
    custom_ctrl = base_to_custom_ctrl_op.get((base_op_rep.op_type, num_control_wires))
    if custom_ctrl is not None:
        return resource_rep(custom_ctrl)

    if base_op_rep.op_type in (qml.X, qml.CNOT, qml.Toffoli, qml.MultiControlledX):
        # First call controlled_resource_rep to flatten any nested structures
        rep = controlled_resource_rep(
            base_class=base_op_rep.op_type,
            base_params=base_op_rep.params,
            num_control_wires=num_control_wires,
            num_zero_control_values=0,
            num_work_wires=num_work_wires,
        )
        if rep.params["num_control_wires"] == 1:
            return resource_rep(qml.CNOT)
        if rep.params["num_control_wires"] == 2:
            return resource_rep(qml.Toffoli)
        return resource_rep(
            qml.MultiControlledX,
            num_control_wires=rep.params["num_control_wires"],
            num_zero_control_values=rep.params["num_zero_control_values"],
            num_work_wires=rep.params["num_work_wires"],
        )

    return controlled_resource_rep(
        base_class=base_op_rep.op_type,
        base_params=base_op_rep.params,
        num_control_wires=num_control_wires,
        num_zero_control_values=0,
        num_work_wires=num_work_wires,
    )


def flip_zero_control(inner_decomp):
    """Wraps a decomposition for a controlled operator with X gates to flip zero control wires."""

    def _resource_fn(**resource_params):
        inner_resource = inner_decomp.compute_resources(
            base_class=resource_params["base_class"],
            base_params=resource_params["base_params"],
            num_control_wires=resource_params["num_control_wires"],
            num_zero_control_values=0,  # we will flip them.
            num_work_wires=resource_params["num_work_wires"],
        )
        num_x = resource_params["num_zero_control_values"]
        gate_counts = inner_resource.gate_counts.copy()
        # Add the counts of the flipping X gates to the gate count
        gate_counts[resource_rep(qml.X)] = gate_counts.get(resource_rep(qml.X), 0) + num_x * 2
        return gate_counts

    @register_resources(_resource_fn)
    def _impl(*params, wires, control_wires, control_values, **kwargs):
        zero_control_wires = [w for w, val in zip(control_wires, control_values) if not val]
        for w in zero_control_wires:
            qml.PauliX(w)
        inner_decomp._impl(  # pylint: disable=protected-access
            *params,
            wires=wires,
            control_wires=control_wires,
            control_values=[1] * len(control_wires),  # all control values are 1 now
            **kwargs,
        )
        for w in zero_control_wires:
            qml.PauliX(w)

    return _impl


def _flip_control_adjoint_resource(
    base_class, base_params, num_control_wires, num_zero_control_values, num_work_wires
):  # pylint: disable=unused-argument
    # base class is adjoint, and the base of the base is the target class
    target_class, target_params = base_params["base_class"], base_params["base_params"]
    return {
        adjoint_resource_rep(
            qml.ops.Controlled,
            {
                "base_class": target_class,
                "base_params": target_params,
                "num_control_wires": num_control_wires,
                "num_zero_control_values": num_zero_control_values,
                "num_work_wires": num_work_wires,
            },
        ): 1
    }


@register_resources(_flip_control_adjoint_resource)
def flip_control_adjoint(*params, wires, control_wires, control_values, work_wires, base, **__):
    """Decompose the control of an adjoint by applying control to the base of the adjoint
    and taking the adjoint of the control."""
    _, struct = base.base._flatten()
    base_op = base.base._unflatten(params, struct)
    qml.adjoint(
        qml.ctrl(
            base_op,
            control=wires[: len(control_wires)],
            control_values=control_values,
            work_wires=work_wires,
        )
    )
