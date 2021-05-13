# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import pytest

from pennylane.transforms.light_gate_merging import get_merged_op, get_flipped_control_target_mx 
import pennylane as qml


param = 0.54321
gates_to_flip_ctrl_target = [
    (qml.CNOT(wires=[1,0]), np.array(
                        [
                            [1, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                        ]
                    )),
    (qml.CRY(param, wires=[1,0]), np.array(
                        [
                            [1, 0, 0, 0],
                            [0, np.cos(param / 2), 0, -np.sin(param / 2)],
                            [0, 0, 1, 0],
                            [0, np.sin(param / 2), 0, np.cos(param / 2)],
                        ]
                    )
    )]

class TestGetFlipped:

    @pytest.mark.parametrize("op, expected", gates_to_flip_ctrl_target)
    def test_one_and_two_qubit_gate(self, op, expected):
        mx = op.matrix
        assert np.allclose(get_flipped_control_target_mx(mx), expected)

class TestGetMergedOp:

    wire_zero = [ (qml.CRY, [0, 1], qml.RX, [0]),
                  (qml.CRY, [1, 0], qml.RX, [0])
                ]

    wire_one = [ (qml.CRY, [0, 1], qml.RX, [1]), ]

    @pytest.mark.parametrize("op1, first_wires, op2, second_wires", wire_zero)
    def test_1q_on_wire_0_with_2q(self, op1, first_wires, op2, second_wires):
        """Check that merging a single-qubit gate that acts on wire 0 with a
        two-qubit gate is done correctly."""
        param = 0.5432
        gate_first = op1(param, wires=first_wires)
        gate_second = op2(param, wires=second_wires)
        merged = get_merged_op(gate_first, gate_second)
        expected = gate_first.matrix @ np.kron(np.eye(2), gate_second.matrix)
        assert np.allclose(merged.matrix, expected)

    @pytest.mark.parametrize("op1, first_wires, op2, second_wires", wire_one)
    def test_1q_on_wire_1_with_2q(self, op1, first_wires, op2, second_wires):
        """Check that merging a single-qubit gate that acts on wire 1 with a
        two-qubit gate is done correctly."""
        param = 0.5432
        gate_first = op1(param, wires=first_wires)
        gate_second = op2(param, wires=second_wires)
        merged = get_merged_op(gate_first, gate_second)
        expected = gate_first.matrix @ np.kron(gate_second.matrix, np.eye(2) )
        assert np.allclose(merged.matrix, expected)
