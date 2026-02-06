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
Unit tests for the get_unitary_matrix transform
"""
# pylint: disable=too-few-public-methods,too-many-function-args
from functools import partial, reduce
from warnings import catch_warnings

import pytest
import scipy as sp
from gate_data import CNOT, H, I
from gate_data import Rotx as RX
from gate_data import Roty as RY
from gate_data import S, X, Y, Z

import pennylane as qp
from pennylane import numpy as np
from pennylane.pauli import PauliSentence, PauliWord
from pennylane.transforms import TransformError

one_qubit_no_parameter = [
    qp.PauliX,
    qp.PauliY,
    qp.PauliZ,
    qp.Hadamard,
    qp.S,
    qp.T,
    qp.SX,
]


I_CNOT = np.kron(I, CNOT)
S_H = np.kron(S, H)
I_S_H = np.kron(I, S_H)
X_S_H = np.kron(X, S_H)

one_qubit_one_parameter = [qp.RX, qp.RY, qp.RZ, qp.PhaseShift]


class TestSingleOperation:

    def test_unsupported_operator(self):
        """Test that an error is raised when the operator is not supported, that means,
        it is an operator without matrix, sparse matrix or decomposition defined."""

        class CustomOp(qp.operation.Operation):
            has_matrix = False
            has_sparse_matrix = False
            has_decomposition = False

            num_params = 1
            num_wires = 1

            def __init__(self, param, wires):
                super().__init__(param, wires=wires)

        dummy_op = CustomOp(0.5, wires=0)
        with pytest.raises(
            qp.operation.MatrixUndefinedError,
            match="Operator must define a matrix, sparse matrix, or decomposition",
        ):
            qp.matrix(dummy_op)

    @pytest.mark.parametrize("op_class", [qp.QubitUnitary])
    @pytest.mark.parametrize("n_wires", [1, 2, 3])
    def test_sparse_operators_supported(self, op_class, n_wires):
        """Test that sparse operators are supported and directly output as dense."""
        matrix = X
        for _ in range(n_wires - 1):
            matrix = np.kron(matrix, X)
        X_csr = sp.sparse.csr_matrix(matrix)
        op = op_class(X_csr, wires=range(n_wires))
        res = qp.matrix(op)
        assert np.allclose(res, matrix, atol=0, rtol=0)

    @pytest.mark.parametrize("op_class", one_qubit_no_parameter)
    def test_non_parametric_instantiated(self, op_class):
        """Verify that the matrices of non-parametric one qubit gates is correct
        when provided as an instantiated operation"""
        op = op_class(wires=0)
        res = qp.matrix(op)
        expected = op.matrix()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("op_class", one_qubit_no_parameter)
    def test_non_parametric_gates_qfunc(self, op_class):
        """Verify that the matrices of non-parametric one qubit gates is correct
        when provided as a qfunc"""
        res = qp.matrix(op_class, wire_order=[0])(wires=0)
        expected = op_class(wires=0).matrix()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("op_class", one_qubit_no_parameter)
    def test_non_parametric_gates_qnode(self, op_class):
        """Verify that the matrices of non-parametric one qubit gates is correct
        when provided as a QNode"""
        dev = qp.device("default.qubit", wires=1)
        qnode = qp.QNode(lambda: op_class(wires=0) and qp.probs(wires=0), dev)
        res = qp.matrix(qnode)()
        expected = op_class(wires=0).matrix()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("op_class", one_qubit_no_parameter)
    def test_matrix_expansion(self, op_class):
        """Verify that matrices are correctly expanded when a wire order is provided"""
        res = qp.matrix(op_class, wire_order=[1, 0, 2])(wires=0)
        expected = np.kron(np.eye(2), np.kron(op_class(wires=0).matrix(), np.eye(2)))
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("op_class", one_qubit_one_parameter)
    def test_parametric_gates_instantiated(self, op_class):
        """Verify that the matrix of non-parametric one qubit gates is correct
        when provided as an instantiated operation"""
        op = op_class(0.54, wires=0)
        res = qp.matrix(op)
        expected = op.matrix()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("op_class", one_qubit_one_parameter)
    def test_parametric_gates_qfunc(self, op_class):
        """Verify that the matrices of non-parametric one qubit gates is correct
        when provided as a qfunc"""
        res = qp.matrix(op_class, wire_order=[0])(0.54, wires=0)
        expected = op_class(0.54, wires=0).matrix()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("op_class", one_qubit_one_parameter)
    def test_parametric_gates_qnode(self, op_class):
        """Verify that the matrices of non-parametric one qubit gates is correct
        when provided as a QNode"""
        dev = qp.device("default.qubit", wires=1)
        qnode = qp.QNode(lambda x: op_class(x, wires=0) and qp.probs(wires=0), dev)
        res = qp.matrix(qnode)(0.54)
        expected = op_class(0.54, wires=0).matrix()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("op_class", one_qubit_one_parameter)
    def test_adjoint(self, op_class):
        """Test that the adjoint is correctly taken into account"""
        res = qp.matrix(qp.adjoint(op_class), wire_order=[0])(0.54, wires=0)
        expected = op_class(-0.54, wires=0).matrix()
        assert np.allclose(res, expected)

    def test_ctrl(self):
        """Test that the ctrl is correctly taken into account"""
        res = qp.matrix(qp.ctrl(qp.PauliX, 0), wire_order=[0, 1])(wires=1)
        expected = CNOT
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("target_wire", [0, 2, 3, 4])
    def test_CNOT_permutations(self, target_wire):
        """Test CNOT: 2-qubit gate with different target wires, some non-adjacent."""
        res = qp.matrix(qp.CNOT, wire_order=[0, 1, 2, 3, 4])(wires=[1, target_wire])

        # compute the expected matrix
        perm = np.swapaxes(
            np.swapaxes(np.arange(2**5).reshape([2] * 5), 0, 1), 0, target_wire
        ).flatten()
        expected = np.kron(CNOT, np.eye(8))[:, perm][perm]
        assert np.allclose(res, expected)

    def test_hamiltonian(self):
        """Test that the matrix of a Hamiltonian is correctly returned"""
        ham = qp.PauliZ(0) @ qp.PauliY(1) - 0.5 * qp.PauliX(1)
        mat = qp.matrix(ham, wire_order=[1, 0, 2])
        expected = reduce(np.kron, [Y, Z, I]) - 0.5 * np.kron(X, np.eye(4))
        assert qp.math.allclose(mat, expected)

    @pytest.mark.xfail(
        reason="This test will fail because Hamiltonians are not queued to tapes yet!"
    )
    def test_hamiltonian_qfunc(self):
        """Test that the matrix of a Hamiltonian is correctly returned"""

        def ansatz(x):
            return qp.PauliZ(0) @ qp.PauliY(1) - x * qp.PauliX(1)

        x = 0.5
        mat = qp.matrix(ansatz, wire_order=[1, 0, 2])(x)
        expected = reduce(np.kron, [Y, Z, I]) - x * np.kron(X, np.eye(4))
        assert qp.math.allclose(mat, expected)

    def test_qutrits(self):
        """Test that the function works with qutrits"""

        dev = qp.device("default.qutrit", wires=2)

        @qp.qnode(dev)
        def circuit():
            qp.TAdd(wires=[0, 1])
            return qp.state()

        mat = qp.matrix(circuit)()
        expected = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
            ]
        )

        assert np.allclose(mat, expected)

    def test_empty_decomposition(self):
        """Test the matrix of a single operation that has an empty list as decomposition."""

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev)
        def node():
            qp.Snapshot("tag")
            return qp.expval(qp.Z(0))

        mats = [
            qp.matrix(qp.Snapshot(), wire_order=[0, 1]),  # op without wires
            qp.matrix(qp.Barrier(1), wire_order=[0, 1]),  # op with wires w/ wire_order
            qp.matrix(qp.Barrier([0, 1])),  # op with wires w/o wire_order
            qp.matrix((lambda *_: qp.Barrier(1) and None), wire_order=[0, 1])(),  # qfunc
            qp.matrix(node)(),  # qnode
        ]
        assert all(np.allclose(mat, np.eye(4)) for mat in mats)

    def test_matrix_dequeues_operation(self):
        """Tests that the operator is dequeued."""

        with qp.queuing.AnnotatedQueue() as q:
            mat = qp.matrix(qp.X(0))
            qp.QubitUnitary(mat, wires=[0])

        assert len(q.queue) == 1
        assert isinstance(q.queue[0], qp.QubitUnitary)


class TestMultipleOperations:
    def test_multiple_operations_tape(self):
        """Check the total matrix for a tape containing multiple gates"""
        wire_order = ["a", "b", "c"]

        with qp.queuing.AnnotatedQueue() as q:
            qp.PauliX(wires="a")
            qp.S(wires="b")
            qp.Hadamard(wires="c")
            qp.CNOT(wires=["b", "c"])

        tape = qp.tape.QuantumScript.from_queue(q)
        matrix = qp.matrix(tape, wire_order)
        expected_matrix = I_CNOT @ X_S_H
        assert np.allclose(matrix, expected_matrix)

        qs = qp.tape.QuantumScript(tape.operations)
        qs_matrix = qp.matrix(qs, wire_order)

        assert np.allclose(qs_matrix, expected_matrix)

    def test_multiple_operations_qfunc(self):
        """Check the total matrix for a qfunc containing multiple gates"""
        wire_order = ["a", "b", "c"]

        def testcircuit():
            qp.PauliX(wires="a")
            qp.S(wires="b")
            qp.Hadamard(wires="c")
            qp.CNOT(wires=["b", "c"])

        matrix = qp.matrix(testcircuit, wire_order)()
        expected_matrix = I_CNOT @ X_S_H
        assert np.allclose(matrix, expected_matrix)

    def test_qfunc_arguments_dequeued(self):
        """Tests that operators passed as arguments to the qfunc are dequeued"""

        def func(op, op1=None):
            qp.apply(op)
            if op1:
                qp.apply(op1)

        with qp.queuing.AnnotatedQueue() as q:
            mat = qp.matrix(func, wire_order=[0])(qp.X(0), op1=qp.Z(0))
            qp.QubitUnitary(mat, wires=[0])

        assert len(q.queue) == 1
        assert isinstance(q.queue[0], qp.QubitUnitary)

    def test_multiple_operations_qnode(self):
        """Check the total matrix for a QNode containing multiple gates"""
        dev = qp.device("default.qubit", wires=["a", "b", "c"])

        @qp.qnode(dev)
        def testcircuit():
            qp.PauliX(wires="a")
            qp.adjoint(qp.S)(wires="b")
            qp.Hadamard(wires="c")
            qp.CNOT(wires=["b", "c"])
            return qp.expval(qp.PauliZ("a"))

        matrix = qp.matrix(testcircuit)()
        expected_matrix = I_CNOT @ (X_S_H.conj().T)
        assert np.allclose(matrix, expected_matrix)


class TestWithParameterBroadcasting:
    def test_multiple_operations_tape_single_broadcasted_op(self):
        """Check the total matrix for a tape containing multiple gates
        and a single broadcasted gate."""
        wire_order = ["a", "b", "c"]

        angles = np.array([0.0, np.pi, 0.7])
        with qp.queuing.AnnotatedQueue() as q:
            qp.S(wires="b")
            qp.RX(angles, wires="a")
            qp.Hadamard(wires="c")
            qp.CNOT(wires=["b", "c"])

        tape = qp.tape.QuantumScript.from_queue(q)
        matrix = qp.matrix(tape, wire_order)
        expected_matrix = [I_CNOT @ I_S_H, -1j * I_CNOT @ X_S_H, I_CNOT @ np.kron(RX(0.7), S_H)]
        assert np.allclose(matrix, expected_matrix)

        qs = qp.tape.QuantumScript(tape.operations)
        qs_matrix = qp.matrix(qs, wire_order)
        assert np.allclose(qs_matrix, expected_matrix)

    def test_multiple_operations_tape_leading_broadcasted_op(self):
        """Check the total matrix for a tape containing multiple gates
        and a leading single broadcasted gate."""
        wire_order = ["a", "b", "c"]

        angles = np.array([0.0, np.pi, 0.7])
        with qp.queuing.AnnotatedQueue() as q:
            qp.RX(angles, wires="a")
            qp.S(wires="b")
            qp.Hadamard(wires="c")
            qp.CNOT(wires=["b", "c"])

        tape = qp.tape.QuantumScript.from_queue(q)
        matrix = qp.matrix(tape, wire_order)
        expected_matrix = [I_CNOT @ I_S_H, -1j * I_CNOT @ X_S_H, I_CNOT @ np.kron(RX(0.7), S_H)]
        assert np.allclose(matrix, expected_matrix)

    def test_multiple_operations_tape_multi_broadcasted_op(self):
        """Check the total matrix for a tape containing multiple gates
        and a multiple broadcasted gate."""
        wire_order = ["a", "b", "c"]

        angles1 = np.array([0.0, np.pi, 0.0, np.pi])
        angles2 = np.array([0.0, 0.0, np.pi, np.pi])
        with qp.queuing.AnnotatedQueue() as q:
            qp.S(wires="b")
            qp.RX(angles1, wires="a")
            qp.Hadamard(wires="c")
            qp.CNOT(wires=["b", "c"])
            qp.RX(angles2, wires="c")

        tape = qp.tape.QuantumScript.from_queue(q)
        matrix = qp.matrix(tape, wire_order)
        I_I_X = np.kron(np.eye(4), X)
        expected_matrix = [
            I_CNOT @ I_S_H,
            -1j * I_CNOT @ X_S_H,
            -1j * I_I_X @ I_CNOT @ I_S_H,
            -I_I_X @ I_CNOT @ X_S_H,
        ]
        assert np.allclose(matrix, expected_matrix)

    def test_multiple_operations_tape_bcasting_matches_Hilbert_dim(self):
        """Check the total matrix for a tape containing multiple gates
        and a multiple broadcasted gate."""
        wire_order = ["a", "b"]

        angles1 = np.array([0.0, np.pi, 0.0, np.pi])
        angles2 = np.array([0.0, 0.0, np.pi, np.pi])
        with qp.queuing.AnnotatedQueue() as q:
            qp.S(wires="b")
            qp.RX(angles1, wires="a")
            qp.Hadamard(wires="b")
            qp.CNOT(wires=["a", "b"])
            qp.RX(angles2, wires="b")

        tape = qp.tape.QuantumScript.from_queue(q)
        matrix = qp.matrix(tape, wire_order)
        I_HS = np.kron(I, H @ S)
        X_HS = np.kron(X, H @ S)
        I_X = np.kron(I, X)
        expected_matrix = [
            CNOT @ I_HS,
            -1j * CNOT @ X_HS,
            -1j * I_X @ CNOT @ I_HS,
            -I_X @ CNOT @ X_HS,
        ]
        assert np.allclose(matrix, expected_matrix)


class TestCustomWireOrdering:
    def test_tensor_wire_oder(self):
        """Test wire order of a tensor product"""
        ham = qp.PauliZ(0) @ qp.PauliX(1)
        res = qp.matrix(ham, wire_order=[0, 2, 1])
        expected = np.kron(Z, np.kron(I, X))
        assert np.allclose(res, expected)

    def test_tape_wireorder(self):
        """Test changing the wire order when using a tape"""

        with qp.queuing.AnnotatedQueue() as q:
            qp.PauliX(wires=0)
            qp.PauliY(wires=1)
            qp.PauliZ(wires=2)

        tape = qp.tape.QuantumScript.from_queue(q)
        matrix = qp.matrix(tape, wire_order=tape.wires)
        expected_matrix = np.kron(X, np.kron(Y, Z))
        assert np.allclose(matrix, expected_matrix)

        matrix = qp.matrix(tape, wire_order=[1, 0, 2])
        expected_matrix = np.kron(Y, np.kron(X, Z))
        assert np.allclose(matrix, expected_matrix)

    def test_qfunc_wireorder(self):
        """Test changing the wire order when using a qfunc"""

        def testcircuit():
            qp.PauliX(wires=0)
            qp.PauliY(wires=1)
            qp.PauliZ(wires=2)

        matrix = qp.matrix(testcircuit, wire_order=[0, 1, 2])()
        expected_matrix = np.kron(X, np.kron(Y, Z))
        assert np.allclose(matrix, expected_matrix)

        matrix = qp.matrix(testcircuit, wire_order=[1, 0, 2])()
        expected_matrix = np.kron(Y, np.kron(X, Z))
        assert np.allclose(matrix, expected_matrix)

    def test_qnode_wireorder(self):
        """Test changing the wire order when using a QNode"""
        dev = qp.device("default.qubit", wires=[1, 0, 2, 3])

        @qp.matrix
        @qp.qnode(dev)
        def testcircuit1(x):
            qp.PauliX(wires=0)
            qp.RY(x, wires=1)
            qp.PauliZ(wires=2)
            return qp.expval(qp.PauliZ(0))

        x = 0.5

        # default wire ordering will come from the device
        expected_matrix = np.kron(RY(x), np.kron(X, np.kron(Z, I)))
        assert np.allclose(testcircuit1(x), expected_matrix)

        @partial(qp.matrix, wire_order=[1, 0, 2])
        @qp.qnode(dev)
        def testcircuit2(x):
            qp.PauliX(wires=0)
            qp.RY(x, wires=1)
            qp.PauliZ(wires=2)
            return qp.expval(qp.PauliZ(0))

        expected_matrix = np.kron(RY(x), np.kron(X, Z))
        assert np.allclose(testcircuit2(x), expected_matrix)

    @pytest.mark.parametrize("wire_order", [[0, 2, 1], [1, 0, 2], [2, 1, 0]])
    def test_pauli_word_wireorder(self, wire_order):
        """Test changing the wire order when using PauliWord"""
        pw = PauliWord({0: "X", 1: "Y", 2: "Z"})
        op = qp.PauliX(0) @ qp.PauliY(1) @ qp.PauliZ(2)
        assert qp.math.allclose(qp.matrix(pw, wire_order), qp.matrix(op, wire_order))

    @pytest.mark.parametrize("wire_order", [[0, 2, 1], [1, 0, 2], [2, 1, 0]])
    def test_pauli_sentence_wireorder(self, wire_order):
        """Test changing the wire order when using PauliWord"""
        pauli1 = PauliWord({0: "X", 1: "Y", 2: "Z"})
        pauli2 = PauliWord({0: "Y", 1: "Z", 2: "X"})
        ps = PauliSentence({pauli1: 1.5, pauli2: 0.5})

        op1 = qp.PauliX(0) @ qp.PauliY(1) @ qp.PauliZ(2)
        op2 = qp.PauliY(0) @ qp.PauliZ(1) @ qp.PauliX(2)
        op = 1.5 * op1 + 0.5 * op2
        assert qp.math.allclose(qp.matrix(ps, wire_order), qp.matrix(op, wire_order))

    def test_empty_tape_with_wire_order(self):
        """Test that an empty tape's matrix can be computed if wire_order is specified."""
        qs = qp.tape.QuantumScript()
        assert np.allclose(qp.matrix(qs, wire_order=[0, 2]), np.eye(4))


pw1 = PauliWord({0: "X", 1: "Z"})
pw2 = PauliWord({0: "Y", 1: "Z"})
pw3 = PauliWord({"a": "Y", "b": "Z"})
op_pairs = (
    (pw1, qp.prod(qp.PauliX(0), qp.PauliZ(1))),
    (pw2, qp.prod(qp.PauliY(0), qp.PauliZ(1))),
    # (pw3, qp.prod(qp.PauliY("a"), qp.PauliZ("b"))), # uncomment after fix 5041
)


class TestPauliWordPauliSentence:
    @pytest.mark.parametrize("pw, op", op_pairs)
    def test_PauliWord_matrix(self, pw, op):
        """Test that a PauliWord is correctly transformed using qp.matrix"""
        res = qp.matrix(pw, wire_order=[0, 1])
        true_res = qp.matrix(op)
        assert qp.math.allclose(res, true_res)

    @pytest.mark.parametrize("pw, op", op_pairs)
    def test_PauliSentence_matrix(self, pw, op):
        """Test that a PauliWord is correctly transformed using qp.matrix"""
        res = qp.matrix(PauliSentence({pw: 0.5}), wire_order=[0, 1])
        true_res = qp.matrix(qp.s_prod(0.5, op))
        assert qp.math.allclose(res, true_res)

    @pytest.mark.xfail
    def test_PauliSentence_matrix_xfail(self, pw, op):
        """Test that a PauliWord is correctly transformed using qp.matrix"""
        # needs fix https://github.com/PennyLaneAI/pennylane/pull/5041
        # once fixed, uncomment the last line in op_pairs above, i.e. add
        # (pw3, qp.prod(qp.PauliY("a"), qp.PauliZ("b"))),
        pw, op = pw3, qp.prod(qp.PauliY("a"), qp.PauliZ("b"))
        res = qp.matrix(PauliSentence({pw: 0.5}), wire_order=["a", "b"])
        true_res = qp.matrix(qp.s_prod(0.5, op))
        assert qp.math.allclose(res, true_res)


class TestTemplates:
    """These tests are useful as they test operators that might not have
    matrix forms defined, requiring decomposition."""

    def test_instantiated(self):
        """Test an instantiated template"""
        weights = np.array([[[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]])
        op = qp.StronglyEntanglingLayers(weights, wires=[0, 1])
        res = qp.matrix(op)

        with qp.queuing.AnnotatedQueue() as q:
            op.decomposition()

        tape = qp.tape.QuantumScript.from_queue(q)
        expected = qp.matrix(tape, wire_order=tape.wires)
        np.allclose(res, expected)

    def test_qfunc(self):
        """Test a template used within a qfunc"""

        def circuit(weights, x):
            qp.StronglyEntanglingLayers(weights, wires=[0, 1])
            qp.RX(x, wires=0)

        weights = np.array([[[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]])
        x = 0.54
        res = qp.matrix(circuit, wire_order=[0, 1])(weights, x)

        op = qp.StronglyEntanglingLayers(weights, wires=[0, 1])

        with qp.queuing.AnnotatedQueue() as q:
            op.decomposition()
            qp.RX(x, wires=0)

        tape = qp.tape.QuantumScript.from_queue(q)
        expected = qp.matrix(tape, wire_order=[0, 1])
        np.allclose(res, expected)

    def test_nested_instantiated(self):
        """Test an operation that must be decomposed twice"""

        class CustomOp(qp.operation.Operation):
            num_params = 1
            num_wires = 2

            @staticmethod
            def compute_decomposition(weights, wires):
                return [qp.StronglyEntanglingLayers(weights, wires=wires)]

        weights = np.array([[[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]])
        op = CustomOp(weights, wires=[0, 1])
        with qp.queuing.AnnotatedQueue() as q_test:
            res = qp.matrix(op)
        assert len(q_test) == 0  # Test that no operators were leaked

        op = qp.StronglyEntanglingLayers(weights, wires=[0, 1])
        with qp.queuing.AnnotatedQueue() as q:
            op.decomposition()

        tape = qp.tape.QuantumScript.from_queue(q)
        expected = qp.matrix(tape, wire_order=[0, 1])
        np.allclose(res, expected)

    def test_nested_qfunc(self):
        """Test an operation that must be decomposed twice"""

        class CustomOp(qp.operation.Operation):
            num_params = 1
            num_wires = 2

            @staticmethod
            def compute_decomposition(weights, wires):
                return [qp.StronglyEntanglingLayers(weights, wires=wires)]

        def circuit(weights, x):
            CustomOp(weights, wires=[0, 1])
            qp.RX(x, wires=0)

        weights = np.array([[[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]])
        x = 0.54
        with qp.queuing.AnnotatedQueue() as q_test:
            res = qp.matrix(circuit, wire_order=[0, 1])(weights, x)

        assert len(q_test) == 0  # Test that no operators were leaked

        op = qp.StronglyEntanglingLayers(weights, wires=[0, 1])

        with qp.queuing.AnnotatedQueue() as q:
            op.decomposition()
            qp.RX(x, wires=0)

        tape = qp.tape.QuantumScript.from_queue(q)
        expected = qp.matrix(tape, wire_order=[0, 1])
        np.allclose(res, expected)


class TestValidation:
    def test_invalid_argument(self):
        """Assert error raised when input is neither a tape, QNode, nor quantum function"""
        with pytest.raises(
            TransformError,
            match="Input is not an Operator, tape, QNode, or quantum function",
        ):
            _ = qp.matrix(None)

    def test_inconsistent_wires(self):
        """Assert error raised when wire labels in wire_order and circuit are inconsistent"""

        def circuit():
            qp.PauliX(wires=1)
            qp.PauliZ(wires=0)

        wires = [0, "b"]

        with pytest.raises(
            TransformError,
            match=r"Wires in circuit \[1, 0\] are inconsistent with those in wire_order \[0, 'b'\]",
        ):
            qp.matrix(circuit, wire_order=wires)()

        with pytest.raises(
            TransformError,
            match=r"Wires in circuit \[0\] are inconsistent with those in wire_order \[1\]",
        ):
            qp.matrix(qp.PauliX(0), wire_order=[1])


class TestInterfaces:
    @pytest.mark.tf
    def test_tf(self):
        """Test with tensorflow interface"""
        import tensorflow as tf

        @partial(qp.matrix, wire_order=[0, 1, 2])
        def circuit(beta, theta):
            qp.RZ(beta, wires=0)
            qp.RZ(theta[0], wires=1)
            qp.CRY(theta[1], wires=[1, 2])

        beta = 0.1
        # input tensorflow parameters
        theta = tf.Variable([0.2, 0.3])
        matrix = circuit(beta, theta)

        # expected matrix
        theta_np = theta.numpy()
        matrix1 = np.kron(
            qp.RZ(beta, wires=0).matrix(),
            np.kron(qp.RZ(theta_np[0], wires=1).matrix(), I),
        )
        matrix2 = np.kron(I, qp.CRY(theta_np[1], wires=[1, 2]).matrix())
        expected_matrix = matrix2 @ matrix1

        assert np.allclose(matrix, expected_matrix)

    @pytest.mark.torch
    def test_torch(self):
        """Test with torch interface"""

        import torch

        dev = qp.device("default.qubit", wires=3)

        @qp.matrix
        @qp.qnode(dev)
        def circuit(theta):
            qp.RZ(theta[0], wires=0)
            qp.RZ(theta[1], wires=1)
            qp.CRY(theta[2], wires=[1, 2])
            return qp.expval(qp.PauliZ(1))

        # input torch parameters
        theta = torch.tensor([0.1, 0.2, 0.3])
        matrix = circuit(theta)

        # expected matrix
        matrix1 = np.kron(
            qp.RZ(theta[0], wires=0).matrix(),
            np.kron(qp.RZ(theta[1], wires=1).matrix(), I),
        )
        matrix2 = np.kron(I, qp.CRY(theta[2], wires=[1, 2]).matrix())
        expected_matrix = matrix2 @ matrix1

        assert np.allclose(matrix, expected_matrix)

    @pytest.mark.autograd
    def test_autograd(self):
        """Test with autograd interface"""

        @partial(qp.matrix, wire_order=[0, 1, 2])
        def circuit(theta):
            qp.RZ(theta[0], wires=0)
            qp.RZ(theta[1], wires=1)
            qp.CRY(theta[2], wires=[1, 2])

        # set input parameters
        theta = np.array([0.1, 0.2, 0.3], requires_grad=True)
        matrix = circuit(theta)

        # expected matrix
        matrix1 = np.kron(
            qp.RZ(theta[0], wires=0).matrix(),
            np.kron(qp.RZ(theta[1], wires=1).matrix(), I),
        )
        matrix2 = np.kron(I, qp.CRY(theta[2], wires=[1, 2]).matrix())
        expected_matrix = matrix2 @ matrix1

        assert np.allclose(matrix, expected_matrix)

    @pytest.mark.catalyst
    @pytest.mark.external
    def test_catalyst(self):
        """Test with Catalyst interface"""

        catalyst = pytest.importorskip("catalyst")

        dev = qp.device("lightning.qubit", wires=1)

        # create a plain QNode
        @qp.qnode(dev)
        def f():
            qp.PauliX(0)
            return qp.state()

        # create a qjit-compiled QNode by decorating a function
        @catalyst.qjit
        @qp.qnode(dev)
        def g():
            qp.PauliX(0)
            return qp.state()

        # create a qjit-compiled QNode by passing in the plain QNode directly
        h = catalyst.qjit(f)

        assert np.allclose(f(), g(), h())
        assert np.allclose(qp.matrix(f)(), qp.matrix(g)(), qp.matrix(h)())

    @pytest.mark.jax
    def test_get_unitary_matrix_interface_jax(self):
        """Test with JAX interface"""

        from jax import numpy as jnp

        @partial(qp.matrix, wire_order=[0, 1, 2])
        def circuit(theta):
            qp.RZ(theta[0], wires=0)
            qp.RZ(theta[1], wires=1)
            qp.CRY(theta[2], wires=[1, 2])

        # input jax parameters
        theta = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float64)

        matrix = circuit(theta)

        # expected matrix
        matrix1 = np.kron(
            qp.RZ(theta[0], wires=0).matrix(),
            np.kron(qp.RZ(theta[1], wires=1).matrix(), I),
        )
        matrix2 = np.kron(I, qp.CRY(theta[2], wires=[1, 2]).matrix())
        expected_matrix = matrix2 @ matrix1

        assert np.allclose(matrix, expected_matrix)


class TestDifferentiation:
    @pytest.mark.jax
    @pytest.mark.parametrize("v", np.linspace(0.2, 1.6, 8))
    def test_jax(self, v):
        import jax

        def circuit(theta):
            qp.RX(theta, wires=0)
            qp.PauliZ(wires=0)
            qp.CNOT(wires=[0, 1])

        def loss(theta):
            U = qp.matrix(circuit, wire_order=[0, 1])(theta)
            return qp.math.real(qp.math.trace(U))

        x = jax.numpy.array(v)

        l = loss(x)
        dl = jax.grad(loss)(x)
        matrix = qp.matrix(circuit, wire_order=[0, 1])(x)

        assert isinstance(matrix, jax.numpy.ndarray)
        assert np.allclose(l, 2 * np.cos(v / 2))
        assert np.allclose(dl, -np.sin(v / 2))

    @pytest.mark.torch
    @pytest.mark.parametrize("v", np.linspace(0.2, 1.6, 8))
    def test_torch(self, v):
        import torch

        def circuit(theta):
            qp.RX(theta, wires=0)
            qp.PauliZ(wires=0)
            qp.CNOT(wires=[0, 1])

        def loss(theta):
            U = qp.matrix(circuit, wire_order=[0, 1])(theta)
            return qp.math.real(qp.math.trace(U))

        x = torch.tensor(v, requires_grad=True)
        l = loss(x)
        l.backward()
        dl = x.grad
        matrix = qp.matrix(circuit, wire_order=[0, 1])(x)

        assert isinstance(matrix, torch.Tensor)
        assert np.allclose(l.detach(), 2 * np.cos(v / 2))
        assert np.allclose(dl.detach(), -np.sin(v / 2))

    @pytest.mark.tf
    @pytest.mark.parametrize("v", np.linspace(0.2, 1.6, 8))
    def test_tensorflow(self, v):
        import tensorflow as tf

        def circuit(theta):
            qp.RX(theta, wires=0)
            qp.PauliZ(wires=0)
            qp.CNOT(wires=[0, 1])

        def loss(theta):
            U = qp.matrix(circuit, wire_order=[0, 1])(theta)
            return qp.math.real(qp.math.trace(U))

        x = tf.Variable(v)
        with tf.GradientTape() as tape:
            l = loss(x)
        dl = tape.gradient(l, x)
        matrix = qp.matrix(circuit, wire_order=[0, 1])(x)

        assert isinstance(matrix, tf.Tensor)
        assert np.allclose(l, 2 * np.cos(v / 2))
        assert np.allclose(dl, -np.sin(v / 2))

    @pytest.mark.parametrize("v", np.linspace(0.2, 1.6, 8))
    def test_get_unitary_matrix_autograd_differentiable(self, v):
        def circuit(theta):
            qp.RX(theta, wires=0)
            qp.PauliZ(wires=0)
            qp.CNOT(wires=[0, 1])

        def loss(theta):
            U = qp.matrix(circuit, wire_order=[0, 1])(theta)
            return qp.math.real(qp.math.trace(U))

        x = np.array(v, requires_grad=True)
        l = loss(x)
        dl = qp.grad(loss)(x)
        matrix = qp.matrix(circuit, wire_order=[0, 1])(x)

        assert isinstance(matrix, qp.numpy.tensor)
        assert np.allclose(l, 2 * np.cos(v / 2))
        assert np.allclose(dl, -np.sin(v / 2))


class TestMeasurements:
    @pytest.mark.parametrize(
        "measurements,N",
        [
            ([qp.expval(qp.PauliX(0))], 2),
            ([qp.probs(op=qp.PauliX(0)), qp.probs(op=qp.PauliZ(1))], 4),
            ([qp.probs(wires=[0, 1])], 4),
            ([qp.counts(wires=[0, 1, 2])], 8),
        ],
    )
    def test_all_measurement_matrices_are_identity(self, measurements, N):
        """Test that the matrix of a script with only observables is Identity."""
        qscript = qp.tape.QuantumScript(measurements=measurements)
        assert np.array_equal(qp.matrix(qscript, wire_order=qscript.wires), np.eye(N))


class TestWireOrderErrors:
    """Test that wire_order=None raises an error in qp.matrix transform."""

    def test_error_pauli_word(self):
        """Test that an error is raised when calling qp.matrix without wire_order on a PauliWord"""
        pw = PauliWord({0: "X", 1: "X"})
        with pytest.raises(ValueError, match=r"wire_order is required"):
            _ = qp.matrix(pw)

    def test_error_pauli_sentence(self):
        """Test that an error is raised when calling qp.matrix without wire_order on a PauliSentence"""
        ps = PauliSentence({PauliWord({0: "X", 1: "X"}): 1.0})
        with pytest.raises(ValueError, match=r"wire_order is required"):
            _ = qp.matrix(ps)

    @pytest.mark.parametrize("ops", [[qp.PauliX(1), qp.PauliX(0)], []])
    def test_error_tape_multiple_wires(self, ops):
        """Test that an error is raised when calling qp.matrix without wire_order on a tape."""
        qs = qp.tape.QuantumScript(ops)
        with pytest.raises(ValueError, match=r"wire_order is required"):
            _ = qp.matrix(qs)

    def test_error_qnode(self):
        """Test that an error is raised when calling qp.matrix without wire_order on a QNode."""

        @qp.qnode(qp.device("default.qubit"))  # devices does not provide wire_order
        def circuit():
            qp.PauliX(0)
            return qp.state()

        with pytest.raises(ValueError, match=r"wire_order is required"):
            _ = qp.matrix(circuit)

    def test_error_qfunc(self):
        """Test that an error is raised when calling qp.matrix without wire_order on a qfunc."""

        def circuit():
            qp.PauliX(0)

        with pytest.raises(ValueError, match=r"wire_order is required"):
            _ = qp.matrix(circuit)

    def test_op_class(self):
        """Tests that an error is raised when calling qp.matrix without wire_order
        on an operator class with multiple wires."""

        with pytest.raises(ValueError, match=r"wire_order is required"):
            _ = qp.matrix(qp.CNOT)(wires=[0, 1])

        # No error should be raised if the operator class has only one wire.
        _ = qp.matrix(qp.Hadamard)(wires=0)

    def test_no_error_cases(self):
        """Test that an error is not raised when calling qp.matrix on an operator, a
        single-wire tape, or a QNode with a device that provides wires."""

        @qp.qnode(qp.device("default.qubit", wires=2))  # device provides wire_order
        def circuit():
            qp.PauliX(0)
            return qp.state()

        with catch_warnings(record=True) as record:
            qp.matrix(qp.PauliX(0))
            qp.matrix(qp.tape.QuantumScript([qp.PauliX(1)]))
            qp.matrix(circuit)

        assert len(record) == 0


@pytest.mark.jax
def test_jitting_matrix():
    """Test that qp.matrix is jittable with jax."""
    import jax

    op = qp.adjoint(qp.Rot(1.2, 2.3, 3.4, wires=0))

    jit_mat = jax.jit(qp.matrix)(op)
    normal_mat = qp.matrix(op)

    assert qp.math.allclose(normal_mat, jit_mat)
