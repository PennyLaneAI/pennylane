# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the ``decompose`` transform"""

import pytest

import pennylane as qml
from pennylane.transforms.decompose import decompose

# pylint: disable=too-few-public-methods


class InfiniteOp(qml.operation.Operation):
    """An op with an infinite decomposition."""

    num_wires = 1

    def decomposition(self):
        return [InfiniteOp(*self.parameters, self.wires)]


class TestDecompose:
    """Unit tests for decompose function"""

    gate_sets_incorrect_types = [None, "RX", qml.RX, [qml.RX], ["RX"], (qml.RX), ("RX")]

    @pytest.mark.parametrize("gate_set", gate_sets_incorrect_types)
    def test_invalid_gate_set_type(self, gate_set):
        """Tests that invalid gate set types are handled appropriately"""

        tape = qml.tape.QuantumScript([qml.RX(0, wires=[0])])
        decompose(tape, gate_set=gate_set)

    def test_user_warning(self):
        """Tests that user warning is raised if operator does not have a valid decomposition"""
        tape = qml.tape.QuantumScript([qml.RX(0, wires=[0])])
        with pytest.warns(UserWarning, match="has no supported decomposition"):
            decompose(tape, gate_set=lambda op: op.name not in {"RX"})

    def test_infinite_decomposition_loop(self):
        """Test that a device error is raised if decomposition enters an infinite loop."""

        tape = qml.tape.QuantumScript([InfiniteOp(1.23, 0)])
        with pytest.raises(RecursionError, match=r"Reached recursion limit trying to decompose"):
            decompose(tape, lambda obj: obj.has_matrix)
