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
from pennylane.wires import WireError


densitymat0 = np.array([[1.0, 0.0], [0.0, 0.0]])


@pytest.mark.parametrize(
    "op",
    [
        qml.BasisState(np.array([0, 1]), wires=0),
        qml.QubitStateVector(np.array([1.0, 0.0]), wires=0),
        qml.QubitDensityMatrix(densitymat0, wires=0),
    ],
)
def test_adjoint_error_exception(op):
    with pytest.raises(qml.operation.AdjointUndefinedError):
        op.adjoint()


@pytest.mark.parametrize(
    "op, mat, base",
    [
        (qml.BasisState(np.array([0, 1]), wires=0), [0, 1], "BasisState"),
        (qml.QubitStateVector(np.array([1.0, 0.0]), wires=0), [1.0, 0.0], "QubitStateVector"),
        (qml.QubitDensityMatrix(densitymat0, wires=0), densitymat0, "QubitDensityMatrix"),
    ],
)
def test_labelling_matrix_cache(op, mat, base):
    """Test state prep matrix parameters interact with labelling matrix cache"""

    assert op.label() == base

    cache = {"matrices": []}
    assert op.label(cache=cache) == f"{base}(M0)"
    assert qml.math.allclose(cache["matrices"][0], mat)

    cache = {"matrices": [0, mat, 0]}
    assert op.label(cache=cache) == f"{base}(M1)"
    assert len(cache["matrices"]) == 3


class TestDecomposition:
    def test_BasisState_decomposition(self):
        """Test the decomposition for BasisState"""

        n = np.array([0, 1, 0])
        wires = (0, 1, 2)
        ops1 = qml.BasisState.compute_decomposition(n, wires)
        ops2 = qml.BasisState(n, wires=wires).decomposition()

        assert len(ops1) == len(ops2) == 1
        assert isinstance(ops1[0], qml.BasisStatePreparation)
        assert isinstance(ops2[0], qml.BasisStatePreparation)

    def test_QubitStateVector_decomposition(self):
        """Test the decomposition for QubitStateVector."""

        U = np.array([1, 0, 0, 0])
        wires = (0, 1)

        ops1 = qml.QubitStateVector.compute_decomposition(U, wires)
        ops2 = qml.QubitStateVector(U, wires=wires).decomposition()

        assert len(ops1) == len(ops2) == 1
        assert isinstance(ops1[0], qml.MottonenStatePreparation)
        assert isinstance(ops2[0], qml.MottonenStatePreparation)

    def test_QubitStateVector_broadcasting(self):
        """Test broadcasting for QubitStateVector."""

        U = np.eye(4)[:3]
        wires = (0, 1)

        op = qml.QubitStateVector(U, wires=wires)
        assert op.batch_size == 3


class TestStateVector:
    """Test the state_vector() method of various state-prep operations."""

    @pytest.mark.parametrize(
        "num_wires,wire_order,one_position",
        [
            (2, None, (1, 0)),
            (2, [1, 2], (1, 0)),
            (3, [0, 1, 2], (0, 1, 0)),
            (3, ["a", 1, 2], (0, 1, 0)),
            (3, [1, 2, 0], (1, 0, 0)),
            (3, [1, 2, "a"], (1, 0, 0)),
            (3, [2, 1, 0], (0, 1, 0)),
        ],
    )
    def test_QubitStateVector_state_vector(self, num_wires, wire_order, one_position):
        """Tests that QubitStateVector state_vector returns kets as expected."""
        qsv_op = qml.QubitStateVector([0, 0, 1, 0], wires=[1, 2])  # |10>
        ket = qsv_op.state_vector(wire_order=wire_order)
        assert ket[one_position] == 1
        ket[one_position] = 0  # everything else should be zero, as we assert below
        assert np.allclose(np.zeros((2,) * num_wires), ket)

    def test_QubitStateVector_reordering(self):
        """Tests that wires get re-ordered as expected."""
        qsv_op = qml.QubitStateVector(np.array([1, -1, 1j, -1j]) / 2, wires=[0, 1])
        ket = qsv_op.state_vector(wire_order=[2, 1, 3, 0])
        expected = np.zeros((2, 2, 2, 2), dtype=np.complex128)
        expected[0, :, 0, :] = np.array([[1, 1j], [-1, -1j]]) / 2
        assert np.array_equal(ket, expected)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch", "tensorflow"])
    def test_QubitStateVector_state_vector_preserves_parameter_type(self, interface):
        """Tests that given an array of some type, the resulting state vector is also that type."""
        qsv_op = qml.QubitStateVector(qml.math.array([0, 0, 0, 1], like=interface), wires=[1, 2])
        assert qml.math.get_interface(qsv_op.state_vector()) == interface
        assert qml.math.get_interface(qsv_op.state_vector(wire_order=[0, 1, 2])) == interface

    def test_QubitStateVector_state_vector_bad_wire_order(self):
        """Tests that the provided wire_order must contain the wires in the operation."""
        qsv_op = qml.QubitStateVector([0, 0, 0, 1], wires=[0, 1])
        with pytest.raises(WireError, match="wire_order must contain all QubitStateVector wires"):
            qsv_op.state_vector(wire_order=[1, 2])

    @pytest.mark.parametrize("vec", [[0] * 4, [1] * 4])
    def test_QubitStateVector_state_norm_not_one_fails(self, vec):
        """Tests that the state-vector provided must have norm equal to 1."""
        with pytest.raises(ValueError, match="Sum of amplitudes-squared does not equal one."):
            _ = qml.QubitStateVector(vec, wires=[0, 1])

    def test_QubitStateVector_wrong_param_size_fails(self):
        """Tests that the parameter must be of shape (2**num_wires,)."""
        with pytest.raises(ValueError, match="State vector must have shape"):
            _ = qml.QubitStateVector([0, 1], wires=[0, 1])

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

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch", "tensorflow"])
    def test_BasisState_state_vector_preserves_parameter_type(self, interface):
        """Tests that given an array of some type, the resulting state_vector is also that type."""
        basis_op = qml.BasisState(qml.math.array([0, 1], like=interface), wires=[1, 2])
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
