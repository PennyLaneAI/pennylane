# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Unit tests for the available qubit state preparation operations.
"""
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.exceptions import WireError


def test_QutritBasisState_decomposition():
    """Test the decomposition for QutritBasisState"""

    n = np.array([0, 1, 0])
    wires = (0, 1, 2)
    ops1 = qml.QutritBasisState.compute_decomposition(n, wires)
    ops2 = qml.QutritBasisState(n, wires=wires).decomposition()

    assert len(ops1) == len(ops2) == 1
    assert isinstance(ops1[0], qml.QutritBasisStatePreparation)
    assert isinstance(ops2[0], qml.QutritBasisStatePreparation)


class TestStateVector:
    """Test the state_vector() method of QutritBasisState operation."""

    @pytest.mark.parametrize(
        "num_wires,wire_order,one_position",
        [
            (2, None, (0, 1)),
            (2, [1, 2], (0, 1)),
            (2, [2, 1], (1, 0)),
            (3, [0, 1, 2], (0, 0, 1)),
            (3, ["a", 1, 2], (0, 0, 1)),
            (3, [1, 2, 0], (0, 1, 0)),
            (3, [1, 2, "a"], (0, 1, 0)),
        ],
    )
    def test_QutritBasisState_state_vector(self, num_wires, wire_order, one_position):
        """Tests that QutritBasisState state_vector returns kets as expected."""
        basis_op = qml.QutritBasisState([0, 1], wires=[1, 2])
        ket = basis_op.state_vector(wire_order=wire_order)
        assert qml.math.shape(ket) == (3,) * num_wires
        assert ket[one_position] == 1
        ket[one_position] = 0  # everything else should be zero, as we assert below
        assert np.allclose(np.zeros((3,) * num_wires), ket)

    @pytest.mark.parametrize(
        "state",
        [
            np.array([0, 0]),
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([1, 1]),
        ],
    )
    @pytest.mark.parametrize("device_wires", [3, 4, 5])
    @pytest.mark.parametrize("op_wires", [[0, 1], [1, 0], [2, 0]])
    def test_QutritBasisState_state_vector_computed(self, state, device_wires, op_wires):
        """Test QutritBasisState initialization on a subset of device wires."""
        basis_op = qml.QutritBasisState(state, wires=op_wires)
        basis_state = basis_op.state_vector(wire_order=list(range(device_wires)))

        one_index = [0] * device_wires
        for op_wire, idx_value in zip(op_wires, state):
            if idx_value == 1:
                one_index[op_wire] = 1
        one_index = tuple(one_index)

        assert basis_state[one_index] == 1
        basis_state[one_index] = 0
        assert not np.any(basis_state)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch"])
    def test_QutritBasisState_state_vector_preserves_parameter_type(self, interface):
        """Tests that given an array of some type, the resulting state_vector is also that type."""
        basis_op = qml.QutritBasisState(qml.math.array([0, 1], like=interface), wires=[1, 2])
        assert qml.math.get_interface(basis_op.state_vector()) == interface
        assert qml.math.get_interface(basis_op.state_vector(wire_order=[0, 1, 2])) == interface

    def test_QutritBasisState_state_vector_bad_wire_order(self):
        """Tests that the provided wire_order must contain the wires in the operation."""
        basis_op = qml.QutritBasisState([0, 1], wires=[0, 1])
        with pytest.raises(WireError, match="wire_order must contain all QutritBasisState wires"):
            basis_op.state_vector(wire_order=[1, 2])

    def test_QutritBasisState_explicitly_checks_0_1_2(self):
        """Tests that QutritBasisState gives a clear error if a value other than 0, 1, or 2 is given."""
        op = qml.QutritBasisState([3, 1], wires=[0, 1])
        with pytest.raises(
            ValueError, match="QutritBasisState parameter must consist of 0, 1 or 2 integers."
        ):
            _ = op.state_vector()

    def test_QutritBasisState_wrong_param_size(self):
        """Tests that the parameter must be of length num_wires."""
        op = qml.QutritBasisState([0], wires=[0, 1])
        with pytest.raises(
            ValueError, match="QutritBasisState parameter and wires must be of equal length."
        ):
            _ = op.state_vector()
