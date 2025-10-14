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
from collections import defaultdict

# pylint: disable=protected-access
import numpy as np
import pytest
import scipy as sp

import pennylane as qml
from pennylane.exceptions import WireError

densitymat0 = np.array([[1.0, 0.0], [0.0, 0.0]])


def test_basis_state_input_cast_to_int():
    """Test that the input to BasisState is cast to an int."""

    state = np.array([1.0, 0.0], dtype=np.float64)
    op = qml.BasisState(state, wires=(0, 1))
    assert op.data[0].dtype == np.int64


@pytest.mark.jax
def test_assert_valid():
    """Tests that BasisState operators are valid"""

    op = qml.BasisState(np.array([0, 1]), wires=[0, 1])
    qml.ops.functions.assert_valid(op, skip_differentiation=True)

    def abstract_check(state):
        op = qml.BasisState(state, wires=[0, 1])
        op_matrices, decomp_matrices = [], []
        for rule in qml.list_decomps(qml.BasisState):
            resources = rule.compute_resources(**op.resource_params)
            gate_counts = resources.gate_counts

            with qml.queuing.AnnotatedQueue() as q:
                rule(*op.data, wires=op.wires, **op.hyperparameters)
            tape = qml.tape.QuantumScript.from_queue(q)
            actual_gate_counts = defaultdict(int)
            for _op in tape.operations:
                resource_rep = qml.resource_rep(type(_op), **_op.resource_params)
                actual_gate_counts[resource_rep] += 1
            assert all(_op in gate_counts for _op in actual_gate_counts)

            # Tests that the decomposition produces the same matrix
            op_matrix = qml.matrix(op)
            decomp_matrix = qml.matrix(tape, wire_order=op.wires)
            op_matrices.append(op_matrix)
            decomp_matrices.append(decomp_matrix)

        return op_matrices, decomp_matrices

    # pylint: disable=import-outside-toplevel
    import jax

    op_matrices, decomp_matrices = jax.jit(abstract_check)(np.array([0, 1]))
    assert qml.math.allclose(
        op_matrices, decomp_matrices
    ), "decomposition must produce the same matrix as the operator."


@pytest.mark.parametrize(
    "op",
    [
        qml.BasisState(np.array([0, 1]), wires=[0, 1]),
        qml.StatePrep(np.array([1.0, 0.0]), wires=0),
        qml.QubitDensityMatrix(densitymat0, wires=0),
    ],
)
def test_adjoint_error_exception(op):
    with pytest.raises(qml.operation.AdjointUndefinedError):
        op.adjoint()


@pytest.mark.parametrize(
    "op, mat, base",
    [
        (qml.QubitDensityMatrix(densitymat0, wires=0), densitymat0, "QubitDensityMatrix"),
    ],
)
def test_labelling_matrix_cache(op, mat, base):
    """Test state prep matrix parameters interact with labelling matrix cache"""

    assert op.label() == base

    cache = {"matrices": []}
    assert op.label(cache=cache) == f"{base}\n(M0)"
    assert qml.math.allclose(cache["matrices"][0], mat)

    cache = {"matrices": [0, mat, 0]}
    assert op.label(cache=cache) == f"{base}\n(M1)"
    assert len(cache["matrices"]) == 3


class TestDecomposition:
    def test_BasisState_decomposition(self):
        """Test the decomposition for BasisState"""

        n = np.array([0, 1, 0])
        wires = (0, 1, 2)
        ops1 = qml.BasisState.compute_decomposition(n, wires)
        ops2 = qml.BasisState(n, wires=wires).decomposition()

        assert len(ops1) == len(ops2) == 1
        assert isinstance(ops1[0], qml.X)
        assert isinstance(ops2[0], qml.X)

    def test_StatePrep_decomposition(self):
        """Test the decomposition for StatePrep."""

        U = np.array([1, 0, 0, 0])
        wires = (0, 1)

        ops1 = qml.StatePrep.compute_decomposition(U, wires)
        ops2 = qml.StatePrep(U, wires=wires).decomposition()

        assert len(ops1) == len(ops2) == 1
        assert isinstance(ops1[0], qml.MottonenStatePreparation)
        assert isinstance(ops2[0], qml.MottonenStatePreparation)

    def test_stateprep_resources(self):
        """Test the resources for StatePrep"""

        assert qml.StatePrep.resource_keys == frozenset({"num_wires"})

        op = qml.StatePrep([0, 0, 0, 1], wires=(0, 1))
        assert op.resource_params == {"num_wires": 2}

    def test_decomposition_rule_stateprep(self):
        """Test that stateprep has a correct decomposition rule registered."""

        decomp = qml.list_decomps(qml.StatePrep)[0]

        resource_obj = decomp.compute_resources(num_wires=2)
        assert resource_obj.num_gates == 1
        assert resource_obj.gate_counts == {
            qml.resource_rep(qml.MottonenStatePreparation, num_wires=2): 1
        }

        with qml.queuing.AnnotatedQueue() as q:
            decomp(np.array([0, 0, 0, 1]), wires=(0, 1))

        qml.assert_equal(q.queue[0], qml.MottonenStatePreparation(np.array([0, 0, 0, 1]), (0, 1)))


class TestStatePrepIntegration:

    @pytest.mark.parametrize(
        "state, pad_with, expected",
        [
            (np.array([1, 0]), 0, np.array([1, 0, 0, 0])),
            (np.array([1j, 1]) / np.sqrt(2), 0, np.array([1j, 1, 0, 0]) / np.sqrt(2)),
            (np.array([1, 1]) / 2, 0.5, np.array([1, 1, 1, 1]) / 2),
            (np.array([1, 1]) / 2, 0.5j, np.array([1, 1, 1j, 1j]) / 2),
        ],
    )
    def test_StatePrep_padding(self, state, pad_with, expected):
        """Test that StatePrep pads the input state correctly."""

        wires = (0, 1)

        @qml.qnode(qml.device("default.qubit", wires=2))
        def circuit():
            qml.StatePrep(state, pad_with=pad_with, wires=wires)
            return qml.state()

        assert np.allclose(circuit(), expected)

    @pytest.mark.parametrize(
        "state",
        [
            (np.array([1, 1, 1, 1])),
            (np.array([1, 1j, 1j, 1])),
        ],
    )
    @pytest.mark.parametrize("validate_norm", [True, False])
    def test_StatePrep_normalize(self, state, validate_norm):
        """Test that StatePrep normalizes the input state correctly."""

        wires = (0, 1)

        @qml.qnode(qml.device("default.qubit", wires=2))
        def circuit():
            qml.StatePrep(state, normalize=True, wires=wires, validate_norm=validate_norm)
            return qml.state()

        assert np.allclose(circuit(), state / 2)

    def test_StatePrep_broadcasting(self):
        """Test broadcasting for StatePrep."""

        U = np.eye(4)[:3]
        wires = (0, 1)

        op = qml.StatePrep(U, wires=wires)
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
            (4, [3, 2, 0, 1], (0, 0, 0, 1)),
        ],
    )
    def test_StatePrep_state_vector(self, num_wires, wire_order, one_position):
        """Tests that StatePrep state_vector returns kets as expected."""
        qsv_op = qml.StatePrep([0, 0, 1, 0], wires=[1, 2])  # |10>
        ket = qsv_op.state_vector(wire_order=wire_order)
        assert ket[one_position] == 1
        ket[one_position] = 0  # everything else should be zero, as we assert below
        assert np.allclose(np.zeros((2,) * num_wires), ket)

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
    @pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
    def test_StatePrep_state_vector_preserves_parameter_type(self, interface):
        """Tests that given an array of some type, the resulting state vector is also that type."""
        qsv_op = qml.StatePrep(qml.math.array([0, 0, 0, 1], like=interface), wires=[1, 2])
        assert qml.math.get_interface(qsv_op.state_vector()) == interface
        assert qml.math.get_interface(qsv_op.state_vector(wire_order=[0, 1, 2])) == interface

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
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

        with pytest.raises(ValueError, match="The state must be a vector of norm 1"):
            _ = qml.StatePrep(vec, wires=[0, 1], validate_norm=True)

    def test_StatePrep_wrong_param_size_fails(self):
        """Tests that the parameter must be of shape (2**num_wires,)."""
        with pytest.raises(ValueError, match="State must be of length"):
            _ = qml.StatePrep([0, 1], wires=[0, 1])

    @pytest.mark.torch
    def test_StatePrep_torch_differentiable(self):
        """Test that StatePrep works with torch."""
        import torch

        def QuantumLayer():
            @qml.qnode(qml.device("default.qubit"), interface="torch")
            def qlayer(inputs, weights):
                qml.StatePrep(inputs, wires=[1, 2, 3])
                qml.RY(phi=weights, wires=[0])
                return qml.expval(qml.PauliZ(wires=0))

            weight_shapes = {"weights": (1)}
            return qml.qnn.TorchLayer(qlayer, weight_shapes)

        class SimpleQuantumModel(torch.nn.Module):  # pylint:disable=too-few-public-methods
            def __init__(self):
                super().__init__()
                self.quantum_layer = QuantumLayer()

            def forward(self, x):
                return self.quantum_layer(x)

        model = SimpleQuantumModel()
        features = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            requires_grad=True,
        )
        result = model(features)
        assert qml.math.get_interface(result) == "torch"
        assert qml.math.shape(result) == (2,)

    def test_StatePrep_backprop_autograd(self):
        """Test backprop with autograd"""

        @qml.qnode(qml.device("default.qubit"), diff_method="backprop")
        def circuit(state):
            qml.StatePrep(state, wires=(0,))
            qml.S(1)
            return qml.expval(qml.PauliZ(0))

        state = qml.numpy.array([1.0, 0.0])
        grad = qml.jacobian(circuit)(state)
        assert np.array_equal(grad, [2.0, 0.0])

    @pytest.mark.torch
    def test_StatePrep_backprop_torch(self):
        """Test backprop with torch, getting state.grad"""
        import torch

        @qml.qnode(qml.device("default.qubit"), diff_method="backprop")
        def circuit(state):
            qml.StatePrep(state, wires=(0,))
            qml.S(1)
            return qml.expval(qml.PauliZ(0))

        state = torch.tensor([1.0, 0.0], requires_grad=True)
        res = circuit(state)
        res.backward()
        grad = state.grad
        assert qml.math.get_interface(grad) == "torch"
        assert np.array_equal(grad, [2.0, 0.0])

    @pytest.mark.jax
    def test_StatePrep_backprop_jax(self):
        """Test backprop with jax"""
        import jax

        @qml.qnode(qml.device("default.qubit"), diff_method="backprop")
        def circuit(state):
            qml.StatePrep(state, wires=(0,))
            qml.S(1)
            return qml.expval(qml.PauliZ(0))

        state = jax.numpy.array([1.0, 0.0])
        grad = jax.jacobian(circuit)(state)
        assert qml.math.get_interface(grad) == "jax"
        assert np.array_equal(grad, [2.0, 0.0])

    @pytest.mark.tf
    def test_StatePrep_backprop_tf(self):
        """Test backprop with tf"""
        import tensorflow as tf

        @qml.qnode(qml.device("default.qubit"), diff_method="backprop")
        def circuit(state):
            qml.StatePrep(state, wires=(0,))
            qml.S(1)
            return qml.expval(qml.PauliZ(0))

        state = tf.Variable([1.0, 0.0])
        with tf.GradientTape() as tape:
            res = circuit(state)

        grad = tape.jacobian(res, state)
        assert qml.math.get_interface(grad) == "tensorflow"
        assert np.array_equal(grad, [2.0, 0.0])

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
    @pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
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

    def test_BasisState_wrong_param_size(self):
        """Tests that the parameter must be of length num_wires."""
        with pytest.raises(
            ValueError, match=r"State must be of length 2; got length 1 \(state=\[0\]\)."
        ):
            _ = qml.BasisState([0], wires=[0, 1])


class TestSparseStateVector:
    """Test the sparse_state_vector() method of various state-prep operations."""

    def test_sparse_state_convert_to_csr(self):
        """Test that the sparse_state_vector() method returns a csr_matrix."""
        sp_vec = sp.sparse.coo_matrix([0, 0, 1, 0])
        qsv_op = qml.StatePrep(sp_vec, wires=[0, 1])
        assert qsv_op.batch_size is None
        ket = qsv_op.state_vector()
        assert sp.sparse.issparse(ket), "Output is not sparse type"

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
            (4, [3, 2, 0, 1], (0, 0, 0, 1)),
        ],
    )
    def test_StatePrep_sparse_state_vector(self, num_wires, wire_order, one_position):
        """Tests that StatePrep sparse_state_vector returns kets as expected."""
        init_state = sp.sparse.csr_matrix([0, 0, 1, 0])
        qsv_op = qml.StatePrep(init_state, wires=[1, 2])
        assert qsv_op.batch_size is None
        ket = qsv_op.state_vector(wire_order=wire_order)
        # Convert one position from binary to integer
        one_position = int("".join([str(i) for i in one_position]), 2)
        assert ket.shape == (1, 2**num_wires)
        assert ket[0, one_position] == 1
        ket[0, one_position] = 0
        assert ket.count_nonzero() == 0

    def test_sparse_vector(self):
        """Test that state prep operations can be created with a 1D sparse array."""

        state = sp.sparse.csr_array([1, 0, 0, 0])
        op = qml.StatePrep(state, wires=(0, 1))
        assert op.batch_size is None
        expected = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        assert qml.math.allclose(op.state_vector([0, 1, 2]).todense(), expected)

    def test_preprocess_nonzero_padding_unsupported(self):
        """Test that sparse_state_vector does not support padding with nonzero values."""
        init_state = sp.sparse.csr_matrix([0, 0, 1, 0])
        with pytest.raises(ValueError, match="Non-zero Padding is not supported"):
            qml.StatePrep(
                init_state, wires=[1, 2], pad_with=1, normalize=False, validate_norm=False
            )

    def test_preprocess_one_dimensional_tensor(self):
        """Test that the state tensor is one-dimensional."""
        init_state = sp.sparse.csr_matrix([[0, 0], [1, 0]])
        with pytest.raises(
            NotImplementedError, match="does not yet support parameter broadcasting"
        ):
            qml.StatePrep(
                init_state, wires=[1, 2], pad_with=None, normalize=False, validate_norm=False
            )

    def test_preprocess_length_of_tensor(self):
        """Test that the state tensor is one-dimensional."""
        init_state = sp.sparse.csr_matrix([0, 0, 2, 0, 1])
        with pytest.raises(ValueError, match="State must be of length"):
            qml.StatePrep(
                init_state, wires=[1, 2], pad_with=None, normalize=False, validate_norm=False
            )

    def test_preprocess_auto_padding_tensor(self):
        """Test that the state tensor is one-dimensional."""
        init_state = sp.sparse.csr_matrix([0, 0, 2])
        with pytest.warns(UserWarning, match="Automatically padding with zeros"):
            state = qml.StatePrep._preprocess_csr(
                init_state, wires=[1, 2], pad_with=None, normalize=False, validate_norm=False
            )
            assert state.shape == (1, 4), f"Expected shape (1, 4), got {state.shape}"

    def test_preprocess_normalize_false(self):
        """Test that the state tensor is normalized to one if normalize is False."""
        init_state = sp.sparse.csr_matrix([0, 0, 2, 0])
        with pytest.raises(ValueError, match="The state must be a vector of norm 1.0; got norm"):
            qml.StatePrep(
                init_state, wires=[1, 2], pad_with=None, normalize=False, validate_norm=True
            )

    def test_preprocess_normalize_true(self):
        """Test that the state tensor is normalized if normalize is True."""
        init_state = sp.sparse.csr_matrix([0, 0, 2, 0])
        processed_state = qml.StatePrep._preprocess_csr(
            init_state, wires=[1, 2], pad_with=None, normalize=True, validate_norm=True
        )
        norm = sp.sparse.linalg.norm(processed_state)
        assert qml.math.allclose(norm, 1.0)

    def test_StatePrep_sparse_state_vector_bad_wire_order(self):
        """Tests that the provided wire_order must contain the wires in the operation."""
        qsv_op = qml.StatePrep(sp.sparse.csr_matrix([0, 0, 0, 1]), wires=[0, 1])
        with pytest.raises(WireError, match="wire_order must contain all wires"):
            qsv_op.state_vector(wire_order=[1, 2])
