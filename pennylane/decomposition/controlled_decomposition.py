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

"""This module contains special logic of decomposing controlled operations."""

from __future__ import annotations

import functools
from typing import Callable, Type

import pennylane as qml
from pennylane.operation import Operator

from .decomposition_rule import DecompositionRule, register_resources
from .resources import controlled_resource_rep, resource_rep


class CustomControlledDecomposition(DecompositionRule):  # pylint: disable=too-few-public-methods
    """A decomposition rule applicable to an operator with a custom controlled decomposition."""

    def __init__(self, custom_op_type: Type[Operator]):
        self.custom_op_type = custom_op_type
        super().__init__(self._get_impl(), self._get_resource_fn())

    def _get_impl(self):
        """The implementation of a controlled op that decomposes to a custom controlled op."""

        def _impl(*params, wires, control_wires, control_values, **_):
            wires_0_ctrl_vals = [w for w, val in zip(control_wires, control_values) if not val]
            for w in wires_0_ctrl_vals:
                qml.PauliX(w)
            self.custom_op_type(*params, wires=wires)
            for w in wires_0_ctrl_vals:
                qml.PauliX(w)

        return _impl

    def _get_resource_fn(self) -> Callable:
        """The resource function."""

        def _resource_fn(*_, num_zero_control_values, **__):
            return {
                resource_rep(self.custom_op_type): 1,
                resource_rep(qml.PauliX): num_zero_control_values * 2,
            }

        return _resource_fn


def _controlled_resource_rep(base_op_rep, num_control_wires, num_work_wires):
    """Returns the resource rep of a controlled op, dispatches to a custom op if possible."""

    if (base_op_rep.op_type, num_control_wires) in base_to_custom_ctrl_op():
        return resource_rep(base_to_custom_ctrl_op()[(base_op_rep.op_type, num_control_wires)])

    if base_op_rep.op_type in (qml.X, qml.CNOT, qml.Toffoli, qml.MultiControlledX):
        # First call controlled_resource_rep to get flatten any nested structures
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


class GeneralControlledDecomposition(DecompositionRule):  # pylint: disable=too-few-public-methods
    """A decomposition rule for a controlled operation with a decomposition."""

    def __init__(self, base_decomposition: DecompositionRule):
        self._base_decomposition = base_decomposition
        super().__init__(self._get_impl(), self._get_resource_fn())

    def _get_impl(self) -> Callable:
        """The default implementation of a controlled decomposition."""

        def _impl(*_, control_wires, control_values, work_wires, base, **__):
            for w, val in zip(control_wires, control_values):
                if not val:
                    qml.PauliX(w)
            qml.ctrl(
                self._base_decomposition._impl,  # pylint: disable=protected-access
                control=control_wires,
                work_wires=work_wires,
            )(*base.parameters, wires=base.wires, **base.hyperparameters)
            for w, val in zip(control_wires, control_values):
                if not val:
                    qml.PauliX(w)

        return _impl

    def _get_resource_fn(self) -> Callable:
        """The resource function."""

        def _resource_fn(
            *_, base_params, num_control_wires, num_zero_control_values, num_work_wires, **__
        ):
            base_resource_decomp = self._base_decomposition.compute_resources(**base_params)
            gate_counts = {
                _controlled_resource_rep(base_op_rep, num_control_wires, num_work_wires): count
                for base_op_rep, count in base_resource_decomp.gate_counts.items()
            }
            gate_counts[resource_rep(qml.PauliX)] = num_zero_control_values * 2
            return gate_counts

        return _resource_fn


def _controlled_g_phase_resource(
    *_, num_control_wires, num_zero_control_values, num_work_wires, **__
):
    if num_control_wires == 1:
        return {
            resource_rep(qml.PauliX): num_zero_control_values * 2,
            resource_rep(qml.PhaseShift): 1,
        }

    if num_control_wires == 2:
        return {
            resource_rep(qml.PauliX): num_zero_control_values * 2,
            resource_rep(qml.ControlledPhaseShift): 1,
        }

    return {
        resource_rep(qml.PauliX): num_zero_control_values * 2,
        controlled_resource_rep(
            qml.PhaseShift,
            base_params={},
            num_control_wires=num_control_wires - 1,
            num_zero_control_values=0,
            num_work_wires=num_work_wires,
        ): 1,
    }


@register_resources(_controlled_g_phase_resource)
def controlled_global_phase_decomp(*_, control_wires, control_values, work_wires, base, **__):
    """The decomposition rule for a controlled global phase."""

    for w, val in zip(control_wires, control_values):
        if not val:
            qml.PauliX(w)
    if len(control_wires) == 1:
        qml.PhaseShift(-base.data[0], wires=control_wires[-1])
    else:
        qml.ctrl(
            qml.PhaseShift(-base.data[0], wires=control_wires[-1]),
            control=control_wires[:-1],
            work_wires=work_wires,
        )
    for w, val in zip(control_wires, control_values):
        if not val:
            qml.PauliX(w)


def _controlled_x_resource(*_, num_control_wires, num_zero_control_values, num_work_wires, **__):
    if num_control_wires == 1:
        return {qml.CNOT: 1, qml.PauliX: num_zero_control_values * 2}
    if num_control_wires == 2:
        return {qml.Toffoli: 1, qml.PauliX: num_zero_control_values * 2}
    return {
        resource_rep(
            qml.MultiControlledX,
            num_control_wires=num_control_wires,
            num_zero_control_values=num_zero_control_values,
            num_work_wires=num_work_wires,
        ): 1,
    }


@register_resources(_controlled_x_resource)
def controlled_x_decomp(*_, wires, control_wires, control_values, work_wires, **__):
    """The decomposition rule for a controlled PauliX."""

    if len(control_wires) > 2:
        qml.MultiControlledX(wires=wires, control_values=control_values, work_wires=work_wires)
        return

    for w, val in zip(control_wires, control_values):
        if not val:
            qml.PauliX(w)

    if len(control_wires) == 1:
        qml.CNOT(wires=wires)
    else:
        qml.Toffoli(wires=wires)

    for w, val in zip(control_wires, control_values):
        if not val:
            qml.PauliX(w)


@functools.lru_cache()
def base_to_custom_ctrl_op():
    """A dictionary mapping base op types to their custom controlled versions.

    This dictionary is used under the assumption that all custom controlled operations do not
    have resource params (which is why `ControlledQubitUnitary` is not included here).

    """

    ops_with_custom_ctrl_ops = {
        (qml.PauliZ, 1): qml.CZ,
        (qml.PauliZ, 2): qml.CCZ,
        (qml.PauliY, 1): qml.CY,
        (qml.CZ, 1): qml.CCZ,
        (qml.SWAP, 1): qml.CSWAP,
        (qml.Hadamard, 1): qml.CH,
        (qml.RX, 1): qml.CRX,
        (qml.RY, 1): qml.CRY,
        (qml.RZ, 1): qml.CRZ,
        (qml.Rot, 1): qml.CRot,
        (qml.PhaseShift, 1): qml.ControlledPhaseShift,
    }
    return ops_with_custom_ctrl_ops
