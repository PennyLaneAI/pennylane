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
import pennylane.estimator as re_ops
from pennylane.estimator import map_to_resource_op
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
            (qml.Identity(0), re_ops.Identity()),
            (qml.GlobalPhase(0), re_ops.GlobalPhase()),
            # Single-Qubit Gates
            (qml.Hadamard(0), re_ops.Hadamard()),
            (qml.S(0), re_ops.S()),
            (qml.T(0), re_ops.T()),
            (qml.PauliX(0), re_ops.X()),
            (qml.PauliY(0), re_ops.Y()),
            (qml.PauliZ(0), re_ops.Z()),
            # Two-Qubit Gates
            (qml.SWAP(wires=(0, 1)), re_ops.SWAP()),
        ],
    )
    def test_map_to_resource_op(self, operator, expected_res_op):
        """Test that map_to_resource_op maps to the appropriate resource operator"""
        print(operator)
        assert map_to_resource_op(operator) == expected_res_op
