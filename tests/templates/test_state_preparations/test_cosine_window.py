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
Unit tests for the CosineWindow template.
"""
# pylint: disable=too-few-public-methods
import pytest
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    def test_correct_gates_single_wire(self):
        """Test that the correct gates are applied."""

        op = qml.CosineWindow(wires=[0])
        queue = op.expand().operations

        assert queue[0].name == "Hadamard"
        assert queue[1].name == "RZ"
        assert queue[2].name == "Adjoint(QFT)"
        assert queue[3].name == "PhaseShift"

        assert np.isclose(queue[3].data[0], np.pi / 2)

    def test_correct_gates_many_wires(self):
        """Test that the correct gates are applied on on two wires."""

        op = qml.CosineWindow(wires=[0, 1, 2, 3, 4])
        queue = op.expand().operations

        assert queue[0].name == "Hadamard"
        assert queue[1].name == "RZ"
        assert queue[2].name == "Adjoint(QFT)"

        for ind, q in enumerate(queue[3:]):
            assert q.name == "PhaseShift"
            assert np.isclose(q.data[0], np.pi / 2 ** (ind + 1))

    def test_custom_wire_labels(self):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""

        dev = qml.device("default.qubit", wires=3)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k"])

        @qml.qnode(dev)
        def circuit():
            qml.CosineWindow(wires=range(3))
            return qml.expval(qml.Identity(0)), qml.state()

        @qml.qnode(dev2)
        def circuit2():
            qml.CosineWindow(wires=["z", "a", "k"])
            return qml.expval(qml.Identity("z")), qml.state()

        res1, state1 = circuit()
        res2, state2 = circuit2()

        assert np.allclose(res1, res2)
        assert np.allclose(state1, state2)


class TestInputs:
    """Test inputs and pre-processing."""

    def test_id(self):
        """Tests that the id attribute can be set."""
        template = qml.CosineWindow(wires=[0, 1, 2], id="a")
        assert template.id == "a"


class TestStateVector:
    """Test the state_vector() method of various CosineWindow operations."""

    @pytest.mark.parametrize(
        "num_wires,wire_order,one_position",
        [
            (2, None),
            (2, [1, 2]),
            (3, [0, 1, 2]),
            (3, [1, 2, 0]),
            (3, [2, 1, 0]),
            (4, [3, 2, 0, 1]),
        ],
    )
    def test_CosineWindow_state_vector(self, num_wires, wire_order):
        """Tests that CosineWindow state_vector returns kets as expected."""

        op = qml.CosineWindow(wires = range(num_wires))

        dev = qml.device("default.qubit", wires = num_wires)
        @qml.qnode(dev)
        def circuit():
            qml.CosineWindow(wires = wire_order)
            return qml.state()
        state_op = circuit()
        ket = np.reshape(op.state_vector(wire_order=wire_order), (-1,))
        assert np.allclose(state_op, ket)


    @pytest.mark.parametrize(
        "num_wires,wire_order,one_positions",
        [
            (2, None, [(0, 1, 0), (1, 0, 1)]),
            (2, [1, 2], [(0, 1, 0), (1, 0, 1)]),
            (3, [0, 1, 2], [(0, 0, 1, 0), (1, 0, 0, 1)]),
            (3, ["a", 1, 2], [(0, 0, 1, 0), (1, 0, 0, 1)]),
            (3, [1, 2, 0], [(0, 1, 0, 0), (1, 0, 1, 0)]),
            (3, [1, 2, "a"], [(0, 1, 0, 0), (1, 0, 1, 0)]),
            (3, [2, 1, 0], [(0, 0, 1, 0), (1, 1, 0, 0)]),
            (4, [3, 2, 0, 1], [(0, 0, 0, 0, 1), (1, 0, 1, 0, 0)]),
        ],
    )
    def test_StatePrep_state_vector_broadcasted(self, num_wires, wire_order, one_positions):
        """Tests that StatePrep state_vector returns kets with broadcasting as expected."""
        qsv_op = qml.StatePrep([[0, 0, 1, 0], [0, 1, 0, 0]], wires=[1, 2])  # |10>, |01>
        ket = qsv_op.state_vector(wire_order=wire_order)
        assert ket[one_positions[0]] == 1 == ket[one_positions[1]]
        ket[one_positions[0]] = ket[one_positions[1]] = 0
        # everything else should be zero, as we assert below
        assert np.allclose(np.zeros((2,) * (num_wires + 1)), ket)

    def test_StatePrep_reordering(self):
        """Tests that wires get re-ordered as expected."""
        qsv_op = qml.StatePrep(np.array([1, -1, 1j, -1j]) / 2, wires=[0, 1])
        ket = qsv_op.state_vector(wire_order=[2, 1, 3, 0])
        expected = np.zeros((2, 2, 2, 2), dtype=np.complex128)
        expected[0, :, 0, :] = np.array([[1, 1j], [-1, -1j]]) / 2
        assert np.array_equal(ket, expected)

    def test_StatePrep_reordering_broadcasted(self):
        """Tests that wires get re-ordered as expected with broadcasting."""
        qsv_op = qml.StatePrep(np.array([[1, -1, 1j, -1j], [1, -1j, -1, 1j]]) / 2, wires=[0, 1])
        ket = qsv_op.state_vector(wire_order=[2, 1, 3, 0])
        expected = np.zeros((2,) * 5, dtype=np.complex128)
        expected[0, 0, :, 0, :] = np.array([[1, 1j], [-1, -1j]]) / 2
        expected[1, 0, :, 0, :] = np.array([[1, -1], [-1j, 1j]]) / 2
        assert np.array_equal(ket, expected)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch", "tensorflow"])
    def test_StatePrep_state_vector_preserves_parameter_type(self, interface):
        """Tests that given an array of some type, the resulting state vector is also that type."""
        qsv_op = qml.StatePrep(qml.math.array([0, 0, 0, 1], like=interface), wires=[1, 2])
        assert qml.math.get_interface(qsv_op.state_vector()) == interface
        assert qml.math.get_interface(qsv_op.state_vector(wire_order=[0, 1, 2])) == interface

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch", "tensorflow"])
    def test_StatePrep_state_vector_preserves_parameter_type_broadcasted(self, interface):
        """Tests that given an array of some type, the resulting state vector is also that type."""
        qsv_op = qml.StatePrep(
            qml.math.array([[0, 0, 0, 1], [1, 0, 0, 0]], like=interface), wires=[1, 2]
        )
        assert qml.math.get_interface(qsv_op.state_vector()) == interface
        assert qml.math.get_interface(qsv_op.state_vector(wire_order=[0, 1, 2])) == interface

    def test_StatePrep_state_vector_bad_wire_order(self):
        """Tests that the provided wire_order must contain the wires in the operation."""
        qsv_op = qml.StatePrep([0, 0, 0, 1], wires=[0, 1])
        with pytest.raises(WireError, match="wire_order must contain all StatePrep wires"):
            qsv_op.state_vector(wire_order=[1, 2])

    @pytest.mark.parametrize("vec", [[0] * 4, [1] * 4])
    def test_StatePrep_state_norm_not_one_fails(self, vec):
        """Tests that the state-vector provided must have norm equal to 1."""
        with pytest.raises(ValueError, match="Sum of amplitudes-squared does not equal one."):
            _ = qml.StatePrep(vec, wires=[0, 1])

    def test_StatePrep_wrong_param_size_fails(self):
        """Tests that the parameter must be of shape (2**num_wires,)."""
        with pytest.raises(ValueError, match="State vector must have shape"):
            _ = qml.StatePrep([0, 1], wires=[0, 1])

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
    def test_BasisState_state_vector(self, num_wires, wire_order, one_position):
        """Tests that BasisState state_vector returns kets as expected."""
        basis_op = qml.BasisState([0, 1], wires=[1, 2])
        ket = basis_op.state_vector(wire_order=wire_order)
        assert qml.math.shape(ket) == (2,) * num_wires
        assert ket[one_position] == 1
        ket[one_position] = 0  # everything else should be zero, as we assert below
        assert np.allclose(np.zeros((2,) * num_wires), ket)

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
    def test_BasisState_state_vector_computed(self, state, device_wires, op_wires):
        """Test BasisState initialization on a subset of device wires."""
        basis_op = qml.BasisState(state, wires=op_wires)
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
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch", "tensorflow"])
    @pytest.mark.parametrize("dtype_like", [0, 0.0])
    def test_BasisState_state_vector_preserves_parameter_type(self, interface, dtype_like):
        """Tests that given an array of some type, the resulting state_vector is also that type."""
        basis_state = qml.math.cast_like(qml.math.asarray([0, 1], like=interface), dtype_like)
        basis_op = qml.BasisState(basis_state, wires=[1, 2])
        assert qml.math.get_interface(basis_op.state_vector()) == interface
        assert qml.math.get_interface(basis_op.state_vector(wire_order=[0, 1, 2])) == interface

    def test_BasisState_state_vector_bad_wire_order(self):
        """Tests that the provided wire_order must contain the wires in the operation."""
        basis_op = qml.BasisState([0, 1], wires=[0, 1])
        with pytest.raises(WireError, match="wire_order must contain all BasisState wires"):
            basis_op.state_vector(wire_order=[1, 2])

    def test_BasisState_explicitly_checks_0_1(self):
        """Tests that BasisState gives a clear error if a value other than 0 or 1 is given."""
        op = qml.BasisState([2, 1], wires=[0, 1])
        with pytest.raises(
            ValueError, match="BasisState parameter must consist of 0 or 1 integers."
        ):
            _ = op.state_vector()

    def test_BasisState_wrong_param_size(self):
        """Tests that the parameter must be of length num_wires."""
        op = qml.BasisState([0], wires=[0, 1])
        with pytest.raises(
            ValueError, match="BasisState parameter and wires must be of equal length."
        ):
            _ = op.state_vector()