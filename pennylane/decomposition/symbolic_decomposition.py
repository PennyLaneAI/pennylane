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

import copy
from textwrap import dedent

import numpy as np

import pennylane as qml
from pennylane import allocation, math

from .decomposition_rule import DecompositionRule, register_condition, register_resources
from .reconstruct import reconstruct
from .resources import adjoint_resource_rep, controlled_resource_rep, pow_resource_rep, resource_rep


def make_adjoint_decomp(base_decomposition: DecompositionRule, use_reconstructor=False):
    """Create a decomposition rule for the adjoint of a decomposition rule."""

    def _condition_fn(base_class, base_params):  # pylint: disable=unused-argument
        return base_decomposition.is_applicable(**base_params)

    def _resource_fn(base_class, base_params):  # pylint: disable=unused-argument
        base_resources = base_decomposition.compute_resources(**base_params)
        return {
            adjoint_resource_rep(decomp_op.op_type, decomp_op.params): count
            for decomp_op, count in base_resources.gate_counts.items()
        }

    base_source = base_decomposition._source

    if not use_reconstructor:

        # pylint: disable=protected-access
        @register_condition(_condition_fn)
        @register_resources(
            _resource_fn,
            work_wires=base_decomposition._work_wire_spec,
            exact=base_decomposition.exact_resources,
            name=f"adjoint({base_decomposition.name})",
        )
        def _impl(*params, wires, base):
            # pylint: disable=protected-access
            qml.adjoint(base_decomposition._impl)(*params, wires=wires, **base.hyperparameters)

        _impl._source = (
            dedent(_impl._source).strip()
            + "\n\nwhere base_decomposition is defined as:\n\n"
            + dedent(base_source).strip()
        )

        return _impl

    # pylint: disable=protected-access
    @register_condition(_condition_fn)
    @register_resources(
        _resource_fn,
        work_wires=base_decomposition._work_wire_spec,
        exact=base_decomposition.exact_resources,
        name=f"adjoint({base_decomposition.name})",
    )
    def _impl_using_reconstructor(*params, wires, base_params, **_):
        # pylint: disable=protected-access
        qml.adjoint(base_decomposition._impl)(*params, wires=wires, **base_params)

    _impl_using_reconstructor._source = (
        dedent(_impl_using_reconstructor._source).strip()
        + "\n\nwhere base_decomposition is defined as:\n\n"
        + dedent(base_source).strip()
    )
    return _impl_using_reconstructor


def _cancel_adjoint_resource(base_class, base_params):  # pylint:disable=unused-argument
    # The base of a nested adjoint is an adjoint, so the base of the base is the original operator,
    # and the "base_params" of base_params are the resource params of the original operator.
    base_class, base_params = base_params["base_class"], base_params["base_params"]
    return {resource_rep(base_class, **base_params): 1}


# pylint: disable=protected-access,unused-argument
@register_resources(_cancel_adjoint_resource)
def cancel_adjoint(*params, wires, base):
    """Decompose the adjoint of the adjoint of an operator."""
    qml.pytrees.unflatten(*qml.pytrees.flatten(base.base))


@register_resources(_cancel_adjoint_resource)
def qjit_compatible_cancel_adjoint(*params, wires, base_class, base_params):
    """A catalyst-compatible decomposition rule that cancels nested adjoints.

    Precondition
    - has_reconstructor(base_class, base_params)

    """
    base_class, base_params = base_params["base_class"], base_params["base_params"]
    reconstruct(params, wires, base_class, base_params)


def _adjoint_rotation_resource(base_class, base_params):
    return {resource_rep(base_class, **base_params): 1}


# pylint: disable=protected-access,unused-argument
@register_resources(_adjoint_rotation_resource)
def adjoint_rotation(phi, wires, base):
    """Decompose the adjoint of a rotation operator by inverting the angle."""
    _, struct = qml.pytrees.flatten(base)
    qml.pytrees.unflatten((-phi,), struct)


# pylint: disable=protected-access,unused-argument
@register_resources(_adjoint_rotation_resource)
def qjit_compatible_adjoint_rotation(phi, wires, base_class, base_params):
    """A catalyst-compatible decomposition rule for single-angle rotations."""
    reconstruct((-phi,), wires, base_class, base_params)


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
        qml.pytrees.unflatten(*qml.pytrees.flatten(base))

    _loop()  # pylint: disable=no-value-for-parameter


# pylint: disable=protected-access,unused-argument
@register_condition(lambda z, **__: is_integer(z) and z >= 0)
@register_resources(lambda base_class, base_params, z: {resource_rep(base_class, **base_params): z})
def qjit_compatible_repeat_pow_base(*params, wires, base_class, base_params, z, **__):
    """Decompose the power of an operator by repeating the base operator, in a qjit compatible way."""

    @qml.for_loop(0, z)
    def _loop(i):
        reconstruct(params, wires, base_class, base_params)

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
    base_op = qml.pytrees.unflatten(*qml.pytrees.flatten(base.base))
    qml.pow(base_op, z * base.z)


@register_resources(_merge_powers_resource)
def qjit_compatible_merge_powers(*params, wires, base_class, base_params, z, **__):
    """Decompose nested powers by combining them in a qjit compatible way."""
    new_params = copy.copy(base_params)
    new_params["z"] = z * base_params["z"]
    return reconstruct(params, wires, base_class, new_params)


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
    base_op = qml.pytrees.unflatten(*qml.pytrees.flatten(base.base))
    qml.adjoint(qml.pow(base_op, z))


@register_resources(_flip_pow_adjoint_resource)
def qjit_compatible_flip_pow_adjoint(*params, wires, base_class, base_params, z, **__):
    """Decompose the power of an adjoint in a qjit compatible way."""
    base = reconstruct(params, wires, base_params["base_class"], base_params["base_params"])
    qml.adjoint(qml.pow(base, z))


def make_pow_decomp_with_period(period, use_reconstructor=False) -> DecompositionRule:
    """Make a decomposition rule for the power of an op that has a period."""

    def _condition_fn(base_class, base_params, z):  # pylint: disable=unused-argument
        return math.shape(z) == () and z % period != z

    def _resource_fn(base_class, base_params, z):
        z_mod_period = z % period
        if z_mod_period == 0:
            return {}
        if z_mod_period == 1:
            return {resource_rep(base_class, **base_params): 1}
        return {pow_resource_rep(base_class, base_params, z_mod_period): 1}

    if not use_reconstructor:

        @register_condition(_condition_fn)
        @register_resources(_resource_fn)
        def _impl(*params, wires, base, z, **__):  # pylint: disable=unused-argument
            z_mod_period = z % period
            if z_mod_period == 1:
                qml.pytrees.unflatten(*qml.pytrees.flatten(base))
            elif z_mod_period > 0 and z_mod_period != period:
                qml.pow(base, z_mod_period)

        return _impl

    @register_condition(_condition_fn)
    @register_resources(_resource_fn)
    def _impl_using_reconstructor(
        *params, wires, base_class, base_params, z, **__
    ):  # pylint: disable=unused-argument
        z_mod_period = z % period
        if z_mod_period == 1:
            reconstruct(params, wires, base_class, base_params)
        elif z_mod_period > 0 and z_mod_period != period:
            qml.pow(reconstruct(params, wires, base_class, base_params), z_mod_period)

    return _impl_using_reconstructor


pow_involutory = make_pow_decomp_with_period(2, True)
pow_involutory_no_reconstructor = make_pow_decomp_with_period(2, False)


def _pow_rotation_resource(base_class, base_params, z):  # pylint: disable=unused-argument
    return {resource_rep(base_class, **base_params): 1}


# pylint: disable=protected-access,unused-argument
@register_resources(_pow_rotation_resource)
def pow_rotation(phi, wires, base, z, **__):
    """Decompose the power of a general rotation operator by multiplying the power by the angle."""
    _, struct = base._flatten()
    base._unflatten((phi * z,), struct)


@register_resources(_pow_rotation_resource)
def qjit_compatible_pow_rotation(phi, wires, base_class, base_params, z, **__):
    """Decompose the power of a general rotation operator by multiplying the power by the angle in a qjit compatible way."""
    reconstruct([phi * z], wires, base_class, base_params)


def _decompose_to_base_resource(base_class, base_params, **__):
    return {resource_rep(base_class, **base_params): 1}


# pylint: disable=protected-access,unused-argument
@register_resources(_decompose_to_base_resource)
def decompose_to_base(*params, wires, base, **__):
    """Decompose a symbolic operator to its base."""
    qml.pytrees.unflatten(*qml.pytrees.flatten(base))


@register_resources(_decompose_to_base_resource)
def qjit_compatible_decompose_to_base(*params, wires, base_class, base_params, **__):
    """Decompose a symbolic operator to its base in a qjit compatible way."""
    reconstruct(params, wires, base_class, base_params)


self_adjoint: DecompositionRule = decompose_to_base
qjit_compatible_self_adjoint: DecompositionRule = qjit_compatible_decompose_to_base


def make_controlled_decomp(base_decomposition: DecompositionRule):
    """Create a decomposition rule for the control of a decomposition rule."""

    def _condition_fn(base_params, **_):
        return base_decomposition.is_applicable(**base_params)

    def _resource_fn(
        base_params, num_control_wires, num_zero_control_values, num_work_wires, work_wire_type, **_
    ):
        base_resources = base_decomposition.compute_resources(**base_params)
        gate_counts = {
            controlled_resource_rep(
                base_class=base_op_rep.op_type,
                base_params=base_op_rep.params,
                num_control_wires=num_control_wires,
                num_zero_control_values=0,
                num_work_wires=num_work_wires,
                work_wire_type=work_wire_type,
            ): count
            for base_op_rep, count in base_resources.gate_counts.items()
        }
        # None of the other gates in gate_counts will be X, because they are all
        # controlled operations. So we can safely set the X gate counts here.
        gate_counts[resource_rep(qml.PauliX)] = num_zero_control_values * 2
        return gate_counts

    # pylint: disable=protected-access,too-many-arguments
    @register_condition(_condition_fn)
    @register_resources(
        _resource_fn,
        work_wires=base_decomposition._work_wire_spec,
        exact=base_decomposition.exact_resources,
        name=f"controlled({base_decomposition.name})",
    )
    def _impl(*params, wires, control_wires, control_values, work_wires, work_wire_type, base, **_):
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
            work_wire_type=work_wire_type,
        )(*params, wires=wires[-len(base.wires) :], **base.hyperparameters)
        for w in zero_control_wires:
            qml.PauliX(w)

    _impl._source = (
        dedent(_impl._source).strip()
        + "\n\nwhere base_decomposition is defined as:\n\n"
        + dedent(base_decomposition._source).strip()
    )
    return _impl


def flip_zero_control(inner_decomp: DecompositionRule) -> DecompositionRule:
    """Wraps a decomposition for a controlled operator with X gates to flip zero control wires."""

    def _condition_fn(**resource_params):
        new_params = resource_params.copy()
        new_params["num_zero_control_values"] = 0
        return inner_decomp.is_applicable(**new_params)

    def _resource_fn(**resource_params):
        new_params = resource_params.copy()
        new_params["num_zero_control_values"] = 0
        inner_resource = inner_decomp.compute_resources(**new_params)
        num_x = resource_params["num_zero_control_values"]
        gate_counts = inner_resource.gate_counts.copy()
        # Add the counts of the flipping X gates to the gate count
        gate_counts[resource_rep(qml.X)] = gate_counts.get(resource_rep(qml.X), 0) + num_x * 2
        return gate_counts

    # pylint: disable=protected-access
    @register_condition(_condition_fn)
    @register_resources(
        _resource_fn,
        work_wires=inner_decomp._work_wire_spec,
        exact=inner_decomp.exact_resources,
        name=f"flip_zero_ctrl_values({inner_decomp.name})",
    )
    def _impl(*params, wires, control_wires, control_values, **kwargs):
        zero_control_wires = [w for w, val in zip(control_wires, control_values) if not val]
        for w in zero_control_wires:
            qml.PauliX(w)
        inner_decomp(
            *params,
            wires=wires,
            control_wires=control_wires,
            control_values=[1] * len(control_wires),  # all control values are 1 now
            **kwargs,
        )
        for w in zero_control_wires:
            qml.PauliX(w)

    base_source = inner_decomp._source
    _impl._source = (
        dedent(_impl._source).strip()
        + "\n\nwhere inner_decomp is defined as:\n\n"
        + dedent(base_source).strip()
    )
    return _impl


def _flip_control_adjoint_resource(
    base_class,
    base_params,
    num_control_wires,
    num_zero_control_values,
    num_work_wires,
    work_wire_type,
):  # pylint: disable=too-many-arguments
    # base class is adjoint, and the base of the base is the target class
    target_class, target_params = base_params["base_class"], base_params["base_params"]
    inner_rep = controlled_resource_rep(
        base_class=target_class,
        base_params=target_params,
        num_control_wires=num_control_wires,
        num_zero_control_values=num_zero_control_values,
        num_work_wires=num_work_wires,
        work_wire_type=work_wire_type,
    )
    return {adjoint_resource_rep(inner_rep.op_type, inner_rep.params): 1}


# pylint: disable=too-many-arguments
@register_resources(_flip_control_adjoint_resource)
def flip_control_adjoint(
    *_, wires, control_wires, control_values, work_wires, work_wire_type, base, **__
):
    """Decompose the control of an adjoint by applying control to the base of the adjoint
    and taking the adjoint of the control."""
    base_op = qml.pytrees.unflatten(*qml.pytrees.flatten(base.base))
    qml.adjoint(
        qml.ctrl(
            base_op,
            control=wires[: len(control_wires)],
            control_values=control_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )
    )


def _ctrl_single_work_wire_resource(base_class, base_params, num_control_wires, **__):
    return {
        controlled_resource_rep(qml.X, {}, num_control_wires): 2,
        controlled_resource_rep(base_class, base_params, 1): 1,
    }


# pylint: disable=protected-access,unused-argument
@register_condition(lambda num_control_wires, **_: num_control_wires > 2)
@register_resources(_ctrl_single_work_wire_resource, work_wires={"zeroed": 1})
def _ctrl_single_work_wire(*params, wires, control_wires, base, **__):
    """Implements Lemma 7.11 from https://arxiv.org/abs/quant-ph/9503016."""
    base_op = qml.pytrees.unflatten(*qml.pytrees.flatten(base))
    with allocation.allocate(1, state="zero", restored=True) as work_wires:
        qml.ctrl(qml.X(work_wires[0]), control=control_wires)
        qml.ctrl(base_op, control=work_wires[0])
        qml.ctrl(qml.X(work_wires[0]), control=control_wires)


ctrl_single_work_wire = flip_zero_control(_ctrl_single_work_wire)


def _to_controlled_qu_condition(base_class, **__):
    return base_class.has_matrix and base_class.num_wires == 1


def _to_controlled_qu_resource(
    num_control_wires, num_zero_control_values, num_work_wires, work_wire_type, **__
):
    return {
        resource_rep(
            qml.ControlledQubitUnitary,
            num_target_wires=1,
            num_control_wires=num_control_wires,
            num_zero_control_values=num_zero_control_values,
            num_work_wires=num_work_wires,
            work_wire_type=work_wire_type,
        ): 1
    }


@register_condition(_to_controlled_qu_condition)
@register_resources(_to_controlled_qu_resource)
def to_controlled_qubit_unitary(*_, wires, control_values, work_wires, work_wire_type, base, **__):
    """Convert a controlled operator to a controlled qubit unitary."""
    matrix = base.matrix()
    qml.ControlledQubitUnitary(
        matrix,
        wires,
        control_values=control_values,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )
