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

import functools

import pennylane as qml

from .decomposition_rule import DecompositionRule
from .resources import Resources, controlled_resource_rep, resource_rep


class ControlledDecompositionRule(DecompositionRule):
    """A decomposition rule for a controlled operation with a decomposition."""

    def __init__(self, base_decomposition: DecompositionRule):
        self._base_decomposition = base_decomposition
        super().__init__(self._impl)

    def compute_resources(
        self, base_params, num_control_wires, num_zero_control_values, num_work_wires
    ) -> Resources:
        base_resource_decomp = self._base_decomposition.compute_resources(**base_params)
        controlled_resources = {
            controlled_resource_rep(
                base_op_type=base_op_rep.op_type,
                base_op_params=base_op_rep.params,
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

    def _impl(self, base, control_wires, control_values, work_wires, **__):
        """The default implementation of a controlled decomposition."""

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


@functools.lru_cache()
def _special_ctrl_ops():
    """Gets a list of special operations with custom controlled versions."""

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
