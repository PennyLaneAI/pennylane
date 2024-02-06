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
# pylint:disable=protected-access
from copy import copy

import numpy as np
import pytest

import pennylane as qml
from pennylane.operation import DecompositionUndefinedError
from pennylane.ops.op_math import CompositeOp, Prod, Sum, SProd
from pennylane.wires import Wires

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
    # pylint:disable=unused-argument
    _op_symbol = "#"
    _math_op = None

    def _build_pauli_rep(self):
        return qml.pauli.PauliSentence({})

    @property
    def is_hermitian(self):
        return False

    def matrix(self, wire_order=None):
        return np.eye(2**self.num_wires)

    def eigvals(self):
        return self.eigendecomposition["eigval"]

    @classmethod
    def _sort(cls, op_list, wire_map: dict = None):
        return op_list


class TestConstruction:
    """Test the construction of composite ops."""

    simple_operands = (qml.S(0), qml.T(1))

    def test_direct_initialization_fails(self):
        """Test directly initializing a CompositeOp fails"""
        with pytest.raises(TypeError, match="Can't instantiate abstract class CompositeOp"):
            _ = CompositeOp(*self.simple_operands)  # pylint:disable=abstract-class-instantiated

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
        op = ValidOp(qml.RX(9.87, wires=0), qml.Rot(1.23, 4.0, 5.67, wires=1), qml.PauliX(0))
        assert op.parameters == [9.87, 1.23, 4.0, 5.67]

    def test_data(self):
        """Test that data is initialized correctly."""
        op = ValidOp(qml.RX(9.87, wires=0), qml.Rot(1.23, 4.0, 5.67, wires=1), qml.PauliX(0))
        assert op.data == (9.87, 1.23, 4.0, 5.67)

    def test_data_setter(self):
        """Test the setter method for data"""
        op = ValidOp(qml.RX(9.87, wires=0), qml.Rot(1.23, 4.0, 5.67, wires=1), qml.PauliX(0))
        assert op.data == (9.87, 1.23, 4.0, 5.67)

        new_data = (1.23, 0.0, -1.0, -2.0)
        op.data = new_data  # pylint:disable=attribute-defined-outside-init
        assert op.data == new_data

        for o in op:
            assert o.data == new_data[: o.num_params]
            new_data = new_data[o.num_params :]

    def test_ndim_params_raises_error(self):
        """Test that calling ndim_params raises a ValueError."""
        op = ValidOp(*self.simple_operands)

        with pytest.raises(AttributeError):
            _ = op.ndim_params

    def test_batch_size_None(self):
        """Test that the batch size is none if no operands have batching."""
        prod_op = ValidOp(qml.PauliX(0), qml.RX(1.0, wires=0))
        assert prod_op.batch_size is None

    def test_batch_size_all_batched(self):
        """Test that the batch_size is correct when all operands are batched."""
        base = qml.RX(np.array([1.2, 2.3, 3.4]), 0)
        op = ValidOp(base, base, base)
        assert op.batch_size == 3

    def test_batch_size_not_all_batched(self):
        """Test that the batch_size is correct when some but not all operands are batched."""
        base = qml.RX(np.array([1.2, 2.3, 3.4]), 0)
        op = ValidOp(base, qml.RY(1, 0), qml.RZ(np.array([1, 2, 3]), wires=2))
        assert op.batch_size == 3

    def test_different_batch_sizes_raises_error(self):
        """Test that an error is raised if the operands have different batch sizes."""
        base = qml.RX(np.array([1.2, 2.3, 3.4]), 0)
        op = ValidOp(base, qml.RY(1, 0), qml.RZ(np.array([1, 2, 3, 4]), wires=2))
        with pytest.raises(
            ValueError, match="Broadcasting was attempted but the broadcasted dimensions"
        ):
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

    def test_map_wires(self):
        """Test the map_wires method."""
        diag_op = ValidOp(*self.simple_operands)
        # pylint:disable=attribute-defined-outside-init
        diag_op._pauli_rep = qml.pauli.PauliSentence({qml.pauli.PauliWord({0: "X", 1: "Y"}): 1})

        wire_map = {0: 5, 1: 7, 2: 9, 3: 11}
        mapped_op = diag_op.map_wires(wire_map=wire_map)

        assert mapped_op.wires == Wires([5, 7])
        assert mapped_op[0].wires == Wires(5)
        assert mapped_op[1].wires == Wires(7)
        assert mapped_op.pauli_rep is not diag_op.pauli_rep
        assert mapped_op.pauli_rep == qml.pauli.PauliSentence(
            {qml.pauli.PauliWord({5: "X", 7: "Y"}): 1}
        )

    def test_build_pauli_rep(self):
        """Test the build_pauli_rep"""
        op = ValidOp(*self.simple_operands)
        assert op._build_pauli_rep() == qml.pauli.PauliSentence({})

    def test_tensor_and_hamiltonian_converted(self):
        """Test that Tensor and Hamiltonian instances get converted to Prod and Sum."""
        operands = [
            qml.Hamiltonian(
                [1.1, 2.2], [qml.PauliZ(0), qml.operation.Tensor(qml.PauliX(0), qml.PauliZ(1))]
            ),
            qml.prod(qml.PauliX(0), qml.PauliZ(1)),
            qml.operation.Tensor(qml.PauliX(2), qml.PauliZ(3)),
        ]
        op = qml.sum(*operands)
        assert isinstance(op[0], Sum)
        assert isinstance(op[0][1], SProd)
        assert isinstance(op[0][1].base, Prod)
        assert op[1] is operands[1]
        assert isinstance(op[2], Prod)
        assert op.operands == (
            qml.dot([1.1, 2.2], [qml.PauliZ(0), qml.prod(qml.PauliX(0), qml.PauliZ(1))]),
            operands[1],
            qml.prod(qml.PauliX(2), qml.PauliZ(3)),
        )


class TestMscMethods:
    """Test dunder and other visualizing methods."""

    @pytest.mark.parametrize("ops_lst, op_rep", tuple((i, j) for i, j in zip(ops, ops_rep)))
    def test_repr(self, ops_lst, op_rep):
        """Test __repr__ method."""
        op = ValidOp(*ops_lst)
        assert op_rep == repr(op)

    def test_nested_repr(self):
        """Test nested repr values while other nested features such as equality are not ready"""
        op = ValidOp(qml.PauliX(0), ValidOp(qml.RY(1, wires=1), qml.PauliX(0)))
        assert repr(op) == "PauliX(wires=[0]) # (RY(1, wires=[1]) # PauliX(wires=[0]))"

    def test_label(self):
        """Test label method."""
        op = ValidOp(qml.RY(1, wires=1), qml.PauliX(1))
        assert op.label() == "RY#X"
        with pytest.raises(ValueError):
            op.label(base_label=["only_first"])

        nested_op = ValidOp(qml.PauliX(0), op)
        assert nested_op.label() == "X#(RY#X)"
        assert nested_op.label(decimals=2) == "X#(RY\n(1.00)#X)"
        assert nested_op.label(base_label=["x0", ["ry", "x1"]]) == "x0#(ry#x1)"

        U = np.array([[1, 0], [0, -1]])
        cache = {"matrices": []}
        op = ValidOp(qml.PauliX(0), ValidOp(qml.PauliY(1), qml.QubitUnitary(U, wires=0)))
        assert op.label(cache=cache) == "X#(Y#U(M0))"
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
        for i, operand in enumerate(ops_lst):
            assert op[i] == operand

    @pytest.mark.parametrize("ops_lst", ops)
    def test_flatten_unflatten(self, ops_lst):
        """Test _flatten and _unflatten."""
        op = ValidOp(*ops_lst)
        data, metadata = op._flatten()
        for data_op, input_op in zip(data, ops_lst):
            assert data_op is input_op

        assert metadata == tuple()

        new_op = type(op)._unflatten(*op._flatten())
        assert qml.equal(op, new_op)


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

    def test_overlapping_ops_property(self):
        """Test the overlapping_ops property."""
        valid_op = ValidOp(
            qml.sum(qml.PauliX(0), qml.PauliY(5), qml.PauliZ(10)),
            qml.sum(qml.PauliX(1), qml.PauliY(4), qml.PauliZ(6)),
            qml.prod(qml.PauliX(10), qml.PauliY(2), qml.PauliZ(7)),
            qml.PauliY(7),
            qml.prod(qml.PauliX(4), qml.PauliY(3), qml.PauliZ(8)),
        )
        overlapping_ops = [
            [
                qml.sum(qml.PauliX(0), qml.PauliY(5), qml.PauliZ(10)),
                qml.prod(qml.PauliX(10), qml.PauliY(2), qml.PauliZ(7)),
                qml.PauliY(7),
            ],
            [
                qml.sum(qml.PauliX(1), qml.PauliY(4), qml.PauliZ(6)),
                qml.prod(qml.PauliX(4), qml.PauliY(3), qml.PauliZ(8)),
            ],
        ]

        # TODO: Use qml.equal when supported for nested operators

        for list_op1, list_op2 in zip(overlapping_ops, valid_op.overlapping_ops):
            for op1, op2 in zip(list_op1, list_op2):
                assert op1.name == op2.name
                assert op1.wires == op2.wires
                assert op1.data == op2.data
                assert op1.arithmetic_depth == op2.arithmetic_depth

    def test_overlapping_ops_private_attribute(self):
        """Test that the private `_overlapping_ops` attribute gets updated after a call to
        the `overlapping_ops` property."""
        op = ValidOp(qml.RZ(1.32, wires=0), qml.Identity(wires=0), qml.RX(1.9, wires=1))
        overlapping_ops = op.overlapping_ops
        assert op._overlapping_ops == overlapping_ops
