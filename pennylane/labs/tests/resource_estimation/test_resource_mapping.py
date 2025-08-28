# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains tests for class needed to map PennyLane operations to their ResourceOperator.
"""
import pytest

import pennylane as qml
import pennylane.labs.resource_estimation.ops as re_ops
import pennylane.templates as qtemps
from pennylane.labs.resource_estimation import map_to_resource_op
from pennylane.operation import Operation

# pylint: disable= no-self-use


class Test_map_to_resource_op:
    """Test the class for mapping PennyLane operations to their ResourceOperators."""

    def test_map_to_resource_op_raises_type_error_if_not_operation(self):
        """Test that a TypeError is raised if the input is not an Operation."""
        with pytest.raises(TypeError, match="The op oper is not a valid operation"):
            map_to_resource_op("oper")

    def test_map_to_resource_op_raises_not_implemented_error(self):
        """Test that a NotImplementedError is raised for a valid Operation."""
        operation = Operation(qml.wires.Wires([4]))
        with pytest.raises(
            NotImplementedError, match="Operation doesn't have a resource equivalent"
        ):
            map_to_resource_op(operation)

    @pytest.mark.parametrize(
        "operator, expected_res_op",
        [
            (qml.Identity(0), re_ops.ResourceIdentity()),
            # Single-Qubit Gates
            (qml.Hadamard(0), re_ops.ResourceHadamard()),
            (qml.S(0), re_ops.ResourceS()),
            (qml.T(0), re_ops.ResourceT()),
            (qml.PauliX(0), re_ops.ResourceX()),
            (qml.PauliY(0), re_ops.ResourceY()),
            (qml.PauliZ(0), re_ops.ResourceZ()),
            (qml.PhaseShift(0.1, wires=0), re_ops.ResourcePhaseShift()),
            (qml.Rot(0.1, 0.2, 0.3, wires=0), re_ops.ResourceRot()),
            (qml.RX(0.1, wires=0), re_ops.ResourceRX()),
            (qml.RY(0.1, wires=0), re_ops.ResourceRY()),
            (qml.RZ(0.1, wires=0), re_ops.ResourceRZ()),
            # Two-Qubit Gates
            (qml.SWAP(wires=(0, 1)), re_ops.ResourceSWAP()),
            (qml.IsingXX(0.1, wires=(0, 1)), re_ops.ResourceIsingXX()),
            (qml.IsingYY(0.1, wires=(0, 1)), re_ops.ResourceIsingYY()),
            (qml.IsingXY(0.1, wires=(0, 1)), re_ops.ResourceIsingXY()),
            (qml.IsingZZ(0.1, wires=(0, 1)), re_ops.ResourceIsingZZ()),
            (qml.PSWAP(0.1, wires=(0, 1)), re_ops.ResourcePSWAP()),
            (qml.SingleExcitation(0.1, wires=(0, 1)), re_ops.ResourceSingleExcitation()),
            (qml.CH(wires=(0, 1)), re_ops.ResourceCH()),
            (qml.CNOT(wires=(0, 1)), re_ops.ResourceCNOT()),
            (qml.ControlledPhaseShift(0.1, wires=(0, 1)), re_ops.ResourceControlledPhaseShift()),
            (qml.CRot(0.1, 0.2, 0.3, wires=(0, 1)), re_ops.ResourceCRot()),
            (qml.CRX(0.1, wires=(0, 1)), re_ops.ResourceCRX()),
            (qml.CRY(0.1, wires=(0, 1)), re_ops.ResourceCRY()),
            (qml.CRZ(0.1, wires=(0, 1)), re_ops.ResourceCRZ()),
            (qml.CY(wires=(0, 1)), re_ops.ResourceCY()),
            (qml.CZ(wires=(0, 1)), re_ops.ResourceCZ()),
            # Three-Qubit Gates
            (qml.CCZ(wires=(0, 1, 2)), re_ops.ResourceCCZ()),
            (qml.CSWAP(wires=(0, 1, 2)), re_ops.ResourceCSWAP()),
            (qml.Toffoli(wires=(0, 1, 2)), re_ops.ResourceToffoli()),
            # Multi-Qubit Gates
            (qml.MultiRZ(0.1, wires=[0, 1, 2]), re_ops.ResourceMultiRZ(num_wires=3)),
            (qml.PauliRot(0.1, "XYZ", wires=[0, 1, 2]), re_ops.ResourcePauliRot("XYZ")),
            (
                qml.MultiControlledX(wires=[0, 1, 2]),
                re_ops.ResourceMultiControlledX(num_ctrl_wires=2, num_ctrl_values=2),
            ),
            # Custom/Template Gates
            (qtemps.TemporaryAND(wires=[0, 1, 2]), re_ops.ResourceTempAND()),
        ],
    )
    def test_map_to_resource_op(self, operator, expected_res_op):
        """Test that map_to_resource_op maps to the appropriate resource operator"""

        assert map_to_resource_op(operator) == expected_res_op
