# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
Unit tests for the composite operator class of qubit operations
"""
from copy import copy
import numpy as np
import pytest

import pennylane as qml
import pennylane.wires
from pennylane import numpy as qnp
from pennylane.operation import DecompositionUndefinedError
from pennylane.ops.op_math import CompositeOp

ops = (
    (qml.PauliX(wires=0), qml.PauliZ(wires=0), qml.Hadamard(wires=0)),
    (qml.CNOT(wires=[0, 1]), qml.RX(1.23, wires=1), qml.Identity(wires=0)),
    (
        qml.IsingXX(4.56, wires=[2, 3]),
        qml.Toffoli(wires=[1, 2, 3]),
        qml.Rot(0.34, 1.0, 0, wires=0),
    ),
)

ops_rep = (
    "PauliX(wires=[0]) # PauliZ(wires=[0]) # Hadamard(wires=[0])",
    "CNOT(wires=[0, 1]) # RX(1.23, wires=[1]) # Identity(wires=[0])",
    "IsingXX(4.56, wires=[2, 3]) # Toffoli(wires=[1, 2, 3]) # Rot(0.34, 1.0, 0, wires=[0])",
)


class ValidOp(CompositeOp):
    _op_symbol = "#"

    @property
    def is_hermitian(self):
        return False

    def matrix(self, wire_order=None):
        return np.eye(2**self.num_wires)

    def eigvals(self):
        return self.eigendecomposition["eigval"]

    def compute_matrix(*args, **kwargs):
        base_wires = args[1]
        return np.eye(2 ** len(base_wires))

    @classmethod
    def _sort(cls, op_list, wire_map: dict = None):
        return op_list


class TestConstruction:
    """Test the construction of composite ops."""

    simple_operands = (qml.S(0), qml.T(1))

    def test_direct_initialization_fails(self):
        """Test directly initializing a CompositeOp fails"""
        with pytest.raises(
            TypeError, match="Can't instantiate abstract class CompositeOp with abstract methods"
        ):
            _ = CompositeOp(*self.simple_operands)

    def test_raise_error_fewer_than_2_operands(self):
        """Test that initializing a composite operator with less than 2 operands raises a ValueError."""
        with pytest.raises(ValueError, match="Require at least two operators to combine;"):
            _ = ValidOp(qml.PauliX(0))

    def test_initialization(self):
        """Test that valid child classes can be initialized without error"""
        op = ValidOp(*self.simple_operands)
        assert op._name == "ValidOp"
        assert op._op_symbol == "#"

    def test_queue_idx(self):
        """Test that queue_idx is None."""
        op = ValidOp(*self.simple_operands)
        assert op.queue_idx is None

    def test_parameters(self):
        """Test that parameters are initialized correctly."""
        op = ValidOp(qml.RX(9.87, wires=0), qml.Rot(1.23, 4.0, 5.67, wires=1))
        assert op.parameters == [[9.87], [1.23, 4.0, 5.67]]

    def test_data(self):
        """Test that data is initialized correctly."""
        op = ValidOp(qml.RX(9.87, wires=0), qml.Rot(1.23, 4.0, 5.67, wires=1))
        assert op.data == [[9.87], [1.23, 4.0, 5.67]]

    def test_data_setter(self):
        """Test the setter method for data"""
        op = ValidOp(qml.RX(9.87, wires=0), qml.Rot(1.23, 4.0, 5.67, wires=1))
        assert op.data == [[9.87], [1.23, 4.0, 5.67]]

        new_data = [[1.23], [0.0, -1.0, -2.0]]
        op.data = new_data
        assert op.data == new_data

        for op, new_entry in zip(op.operands, new_data):
            assert op.data == new_entry

    def test_ndim_params_raises_error(self):
        """Test that calling ndim_params raises a ValueError."""
        op = ValidOp(*self.simple_operands)

        with pytest.raises(AttributeError):
            _ = op.ndim_params

    def test_batch_size_raises_error(self):
        """Test that calling batch_size raises a ValueError."""
        op = ValidOp(*self.simple_operands)

        with pytest.raises(AttributeError):
            _ = op.batch_size

    def test_decomposition_raises_error(self):
        """Test that calling decomposition() raises a ValueError."""
        op = ValidOp(*self.simple_operands)

        with pytest.raises(DecompositionUndefinedError):
            op.decomposition()

    def test_diagonalizing_gates_non_overlapping(self):
        """Test that the diagonalizing gates are correct when wires do not overlap."""
        diag_op = ValidOp(qml.PauliZ(wires=0), qml.Identity(wires=1))
        assert diag_op.diagonalizing_gates() == []

    def test_diagonalizing_gates_overlapping(self):
        """Test that the diagonalizing gates are correct when wires overlap."""
        diag_op = ValidOp(qml.S(0), qml.PauliX(0))
        diagonalizing_gates = diag_op.diagonalizing_gates()

        assert len(diagonalizing_gates) == 1
        diagonalizing_mat = diagonalizing_gates[0].matrix()

        true_mat = np.eye(2)

        assert np.allclose(diagonalizing_mat, true_mat)

    def test_eigen_caching(self):
        """Test that the eigendecomposition is stored in cache."""
        diag_op = ValidOp(*self.simple_operands)
        eig_decomp = diag_op.eigendecomposition

        eig_vecs = eig_decomp["eigvec"]
        eig_vals = eig_decomp["eigval"]

        eigs_cache = diag_op._eigs[diag_op.hash]
        cached_vecs = eigs_cache["eigvec"]
        cached_vals = eigs_cache["eigval"]

        assert np.allclose(eig_vals, cached_vals)
        assert np.allclose(eig_vecs, cached_vecs)


class TestMscMethods:
    """Test dunder and other visualizing methods."""

    simple_operands = (qml.S(0), qml.T(1))

    @pytest.mark.parametrize("ops_lst, ops_rep", tuple((i, j) for i, j in zip(ops, ops_rep)))
    def test_repr(self, ops_lst, ops_rep):
        """Test __repr__ method."""
        op = ValidOp(*ops_lst)
        assert ops_rep == repr(op)

    def test_nested_repr(self):
        """Test nested repr values while other nested features such as equality are not ready"""
        op = ValidOp(qml.PauliX(0), ValidOp(qml.RY(1, wires=1), qml.PauliX(0)))
        assert "PauliX(wires=[0]) # (RY(1, wires=[1]) # PauliX(wires=[0]))" == repr(op)

    def test_label(self):
        """Test label method."""
        op = ValidOp(qml.RY(1, wires=1), qml.PauliX(1))
        assert "RY#X" == op.label()
        with pytest.raises(ValueError):
            op.label(base_label=["only_first"])

        nested_op = ValidOp(qml.PauliX(0), op)
        assert "X#(RY#X)" == nested_op.label()
        assert "X#(RY\n(1.00)#X)" == nested_op.label(decimals=2)
        assert "x0#(ry#x1)" == nested_op.label(base_label=["x0", ["ry", "x1"]])

        U = np.array([[1, 0], [0, -1]])
        cache = {"matrices": []}
        op = ValidOp(qml.PauliX(0), ValidOp(qml.PauliY(1), qml.QubitUnitary(U, wires=0)))
        assert "X#(Y#U(M0))" == op.label(cache=cache)
        assert cache["matrices"] == [U]

    @pytest.mark.parametrize("ops_lst", ops)
    def test_copy(self, ops_lst):
        """Test __copy__ method."""
        op = ValidOp(*ops_lst)
        copied_op = copy(op)

        assert op.id == copied_op.id
        assert op.data == copied_op.data
        assert op.wires == copied_op.wires

        for o1, o2 in zip(op.operands, copied_op.operands):
            assert qml.equal(o1, o2)
            assert o1 is not o2

    @pytest.mark.parametrize("ops_lst", ops)
    def test_len(self, ops_lst):
        """Test __len__ method."""
        op = ValidOp(*ops_lst)
        assert len(op) == len(ops_lst)

    @pytest.mark.parametrize("ops_lst", ops)
    def test_iter(self, ops_lst):
        """Test __iter__ method."""
        op = ValidOp(*ops_lst)
        for i, j in zip(op, ops_lst):
            assert i == j

    @pytest.mark.parametrize("ops_lst", ops)
    def test_getitem(self, ops_lst):
        """Test __getitem__ method."""
        op = ValidOp(*ops_lst)
        for i in range(len(ops_lst)):
            assert op[i] == ops_lst[i]

    def test_get_cached_matrix(self):
        """Test that the matrix gets cached into the instance attribute and obtained"""
        wires = pennylane.wires.Wires([0, 1])
        base_wire_hash = hash(wires)

        mat1 = np.reshape(np.arange(16), (4, 4))
        re_ordered_mat = np.array([[0, 2, 1, 3], [8, 10, 9, 11], [4, 6, 5, 7], [12, 14, 13, 15]])

        op = ValidOp(*self.simple_operands)
        assert np.allclose(op._get_cached_matrix(cache=False), np.eye(4))
        # the cache is empty, should return the result of compute_matrix

        op._mat_cache[base_wire_hash] = mat1  # add mat to cache manually
        assert np.allclose(op._get_cached_matrix(), mat1)  # get mat from cache
        assert np.allclose(
            op._get_cached_matrix(wire_order=[1, 0]), re_ordered_mat
        )  # re-order matrix from cache

    def test_add_matrix_to_cache_get_cached_matrix(self):
        """Test that we can add matrices to cache using the cache kwarg"""
        wires = pennylane.wires.Wires([0, 1])
        base_wire_hash = hash(wires)

        op = ValidOp(*self.simple_operands)
        assert op._mat_cache == {}

        op._get_cached_matrix(cache=True)
        assert base_wire_hash in op._mat_cache
        assert np.allclose(op._mat_cache[base_wire_hash], np.eye(4))


class TestProperties:
    """Test class properties."""

    @pytest.mark.parametrize("ops_lst", ops)
    def test_num_params(self, ops_lst):
        """Test num_params property updates correctly."""
        op = ValidOp(*ops_lst)
        true_num_params = sum(op.num_params for op in ops_lst)

        assert op.num_params == true_num_params

    @pytest.mark.parametrize("ops_lst", ops)
    def test_num_wires(self, ops_lst):
        """Test num_wires property updates correctly."""
        valid_op = ValidOp(*ops_lst)
        true_wires = set()

        for op in ops_lst:
            true_wires = true_wires.union(op.wires.toset())

        assert valid_op.num_wires == len(true_wires)

    def test_depth_property(self):
        """Test depth property."""
        op = ValidOp(qml.RZ(1.32, wires=0), qml.Identity(wires=0), qml.RX(1.9, wires=1))
        assert op.arithmetic_depth == 1

        op = ValidOp(qml.PauliX(0), ValidOp(qml.Identity(wires=0), qml.RX(1.9, wires=1)))
        assert op.arithmetic_depth == 2
