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
This module contains tests for class needed to map PennyLane operations to their associated resource
operator.
"""
import numpy as np
import pytest

import pennylane as qml
import pennylane.estimator as re_ops
import pennylane.estimator.templates as re_temps
import pennylane.templates as qtemps
from pennylane.estimator.resource_mapping import _map_to_resource_op
from pennylane.operation import Operation

# pylint: disable= no-self-use


class TestMapToResourceOp:
    """Test the class for mapping PennyLane operations to their resource operators."""

    def test_map_to_resource_op_raises_type_error_if_not_operation(self):
        """Test that a TypeError is raised if the input is not an Operation."""
        with pytest.raises(TypeError, match="is not a valid operation"):
            _map_to_resource_op("oper")

    def test_map_to_resource_op_raises_not_implemented_error(self):
        """Test that a NotImplementedError is raised for a valid Operation."""
        operation = Operation(qml.wires.Wires([4]))
        with pytest.raises(
            NotImplementedError, match="Operation doesn't have a resource equivalent"
        ):
            _map_to_resource_op(operation)

    @pytest.mark.parametrize(
        "operator, expected_res_op",
        [
            # Single-Qubit Gates
            (qml.Hadamard(0), re_ops.Hadamard()),
            (qml.S(0), re_ops.S()),
            (qml.T(0), re_ops.T()),
            (qml.PauliX(0), re_ops.X()),
            (qml.PauliY(0), re_ops.Y()),
            (qml.PauliZ(0), re_ops.Z()),
            (qml.PhaseShift(0.1, wires=0), re_ops.PhaseShift()),
            (qml.Rot(0.1, 0.2, 0.3, wires=0), re_ops.Rot()),
            (qml.RX(0.1, wires=0), re_ops.RX()),
            (qml.RY(0.1, wires=0), re_ops.RY()),
            (qml.RZ(0.1, wires=0), re_ops.RZ()),
            # Two-Qubit Gates
            (qml.SWAP(wires=(0, 1)), re_ops.SWAP()),
            (qml.SingleExcitation(0.1, wires=(0, 1)), re_ops.SingleExcitation()),
            (qml.CH(wires=(0, 1)), re_ops.CH()),
            (qml.CNOT(wires=(0, 1)), re_ops.CNOT()),
            (qml.ControlledPhaseShift(0.1, wires=(0, 1)), re_ops.ControlledPhaseShift()),
            (qml.CRot(0.1, 0.2, 0.3, wires=(0, 1)), re_ops.CRot()),
            (qml.CRX(0.1, wires=(0, 1)), re_ops.CRX()),
            (qml.CRY(0.1, wires=(0, 1)), re_ops.CRY()),
            (qml.CRZ(0.1, wires=(0, 1)), re_ops.CRZ()),
            (qml.CY(wires=(0, 1)), re_ops.CY()),
            (qml.CZ(wires=(0, 1)), re_ops.CZ()),
            # Three-Qubit Gates
            (qml.CCZ(wires=(0, 1, 2)), re_ops.CCZ()),
            (qml.CSWAP(wires=(0, 1, 2)), re_ops.CSWAP()),
            (qml.Toffoli(wires=(0, 1, 2)), re_ops.Toffoli()),
            # Multi-Qubit Gates
            (qml.MultiRZ(0.1, wires=[0, 1, 2]), re_ops.MultiRZ(num_wires=3)),
            (qml.PauliRot(0.1, "XYZ", wires=[0, 1, 2]), re_ops.PauliRot("XYZ")),
            (
                qml.MultiControlledX(wires=[0, 1, 2]),
                re_ops.MultiControlledX(num_ctrl_wires=2, num_zero_ctrl=0),
            ),
            # Custom/Template Gates
            (qtemps.TemporaryAND(wires=[0, 1, 2]), re_ops.TemporaryAND()),
        ],
    )
    def test_map_to_resource_op(self, operator, expected_res_op):
        """Test that _map_to_resource_op maps to the appropriate resource operator"""
        assert _map_to_resource_op(operator) == expected_res_op
