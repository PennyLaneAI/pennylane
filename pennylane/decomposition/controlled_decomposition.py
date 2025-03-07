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
from typing import Callable

import pennylane as qml

from .decomposition_rule import DecompositionRule, register_resources
from .resources import Resources, controlled_resource_rep, resource_rep


class CustomControlledDecomposition(DecompositionRule):
    """A decomposition rule applicable to an operator with a custom controlled decomposition."""

    def __init__(self, custom_op_type):
        self.custom_op_type = custom_op_type
        super().__init__(self._get_impl())

    def _get_impl(self):
        """The implementation of a controlled op that decomposes to a custom controlled op."""

        def _impl(*params, wires, control_wires, control_values, **_):
            for w, val in zip(control_wires, control_values):
                if not val:
                    qml.PauliX(w)
            self.custom_op_type(*params, wires=wires)
            for w, val in zip(control_wires, control_values):
                if not val:
                    qml.PauliX(w)

        return _impl

    def compute_resources(
        self, base_params, num_control_wires, num_zero_control_values, num_work_wires
    ) -> Resources:
        return Resources(
            num_zero_control_values * 2 + 1,
            {
                resource_rep(self.custom_op_type): 1,
                resource_rep(qml.X): num_zero_control_values * 2,
            },
        )


class GeneralControlledDecomposition(DecompositionRule):
    """A decomposition rule for a controlled operation with a decomposition."""

    def __init__(self, base_decomposition: DecompositionRule):
        self._base_decomposition = base_decomposition
        super().__init__(self._get_impl())

    def compute_resources(
        self, base_params, num_control_wires, num_zero_control_values, num_work_wires
    ) -> Resources:
        base_resource_decomp = self._base_decomposition.compute_resources(**base_params)
        controlled_resources = {
            controlled_resource_rep(
                base_class=base_op_rep.op_type,
                base_params=base_op_rep.params,
                num_control_wires=num_control_wires,
                num_zero_control_values=0,
                num_work_wires=num_work_wires,
            ): count
            for base_op_rep, count in base_resource_decomp.gate_counts.items()
            if count > 0
        }
        controlled_resources[resource_rep(qml.X)] = num_zero_control_values * 2
        gate_count = sum(controlled_resources.values())
        return Resources(gate_count, controlled_resources)

    def _get_impl(self) -> Callable:
        """The default implementation of a controlled decomposition."""

        def _impl(*_, control_wires, control_values, work_wires, base, **__):
            for w, val in zip(control_wires, control_values):
                if not val:
                    qml.PauliX(w)
            qml.ctrl(
                self._base_decomposition.impl,
                control=control_wires,
                control_values=control_values,
                work_wires=work_wires,
            )(*base.params, wires=base.wires, **base.hyperparameters)
            for w, val in zip(control_wires, control_values):
                if not val:
                    qml.PauliX(w)

        return _impl


def _controlled_g_phase_resource(*_, num_control_wires, num_zero_control_values, num_work_wires):
    if num_control_wires == 1:
        return {
            resource_rep(qml.PauliX): num_zero_control_values * 2,
            resource_rep(qml.PhaseShift): 1,
        }
    else:
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


def _controlled_x_resource(*_, num_control_wires, num_zero_control_values, num_work_wires):
    if num_control_wires == 1 and num_zero_control_values == 0:
        return {resource_rep(qml.CNOT): 1}
    if num_control_wires == 2 and num_zero_control_values == 0:
        return {resource_rep(qml.Toffoli): 1}
    return {
        resource_rep(
            qml.MultiControlledX,
            num_control_wires=num_control_wires,
            num_zero_control_values=num_zero_control_values,
            num_work_wires=num_work_wires,
        ): 1,
    }


@register_resources(_controlled_x_resource)
def controlled_x_decomp(*_, control_wires, control_values, work_wires, base, **__):
    """The decomposition rule for a controlled PauliX."""

    qml.ctrl(
        qml.PauliX(base.wires),
        control=control_wires,
        control_values=control_values,
        work_wires=work_wires,
    )


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
