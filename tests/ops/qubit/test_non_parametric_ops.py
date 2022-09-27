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
Unit tests for the available non-parametric qubit operations
"""
import itertools
import pytest
import copy
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import unitary_group

import pennylane as qml
from pennylane import math
from pennylane.wires import Wires

from gate_data import I, X, Y, Z, H, CNOT, SWAP, ISWAP, SISWAP, CZ, S, T, CSWAP, Toffoli, ECR


# Non-parametrized operations and their matrix representation
NON_PARAMETRIZED_OPERATIONS = [
    (qml.Identity, I),
    (qml.CNOT, CNOT),
    (qml.SWAP, SWAP),
    (qml.ISWAP, ISWAP),
    (qml.SISWAP, SISWAP),
    (qml.CZ, CZ),
    (qml.S, S),
    (qml.T, T),
    (qml.CSWAP, CSWAP),
    (qml.Toffoli, Toffoli),
    (qml.ECR, ECR),
]

SPARSE_MATRIX_SUPPORTED_OPERATIONS = (
    (qml.Identity(wires=0), I),
    (qml.Hadamard(wires=0), H),
    (qml.PauliZ(wires=0), Z),
    (qml.PauliX(wires=0), X),
    (qml.PauliY(wires=0), Y),
)


class TestOperations:
    @pytest.mark.parametrize("op_cls, mat", NON_PARAMETRIZED_OPERATIONS)
    def test_nonparametrized_op_copy(self, op_cls, mat, tol):
        """Tests that copied nonparametrized ops function as expected"""
        op = op_cls(wires=range(op_cls.num_wires))
        copied_op = copy.copy(op)
        np.testing.assert_allclose(op.matrix(), copied_op.matrix(), atol=tol)

        op._inverse = True
        copied_op2 = copy.copy(op)
        np.testing.assert_allclose(op.matrix(), copied_op2.matrix(), atol=tol)

    @pytest.mark.parametrize("ops, mat", NON_PARAMETRIZED_OPERATIONS)
    def test_matrices(self, ops, mat, tol):
        """Test matrices of non-parametrized operations are correct"""
        op = ops(wires=range(ops.num_wires))
        res_static = op.compute_matrix()
        res_dynamic = op.matrix()
        assert np.allclose(res_static, mat, atol=tol, rtol=0)
        assert np.allclose(res_dynamic, mat, atol=tol, rtol=0)


class TestDecompositions:
    def test_x_decomposition(self, tol):
        """Tests that the decomposition of the PauliX is correct"""
        op = qml.PauliX(wires=0)
        res = op.decomposition()

        assert len(res) == 3

        assert res[0].name == "PhaseShift"

        assert res[0].wires == Wires([0])
        assert res[0].data[0] == np.pi / 2

        assert res[1].name == "RX"
        assert res[1].wires == Wires([0])
        assert res[1].data[0] == np.pi

        assert res[2].name == "PhaseShift"
        assert res[2].wires == Wires([0])
        assert res[2].data[0] == np.pi / 2

        decomposed_matrix = np.linalg.multi_dot([i.matrix() for i in reversed(res)])
        assert np.allclose(decomposed_matrix, op.matrix(), atol=tol, rtol=0)

    def test_y_decomposition(self, tol):
        """Tests that the decomposition of the PauliY is correct"""
        op = qml.PauliY(wires=0)
        res = op.decomposition()

        assert len(res) == 3

        assert res[0].name == "PhaseShift"

        assert res[0].wires == Wires([0])
        assert res[0].data[0] == np.pi / 2

        assert res[1].name == "RY"
        assert res[1].wires == Wires([0])
        assert res[1].data[0] == np.pi

        assert res[2].name == "PhaseShift"
        assert res[2].wires == Wires([0])
        assert res[2].data[0] == np.pi / 2

        decomposed_matrix = np.linalg.multi_dot([i.matrix() for i in reversed(res)])
        assert np.allclose(decomposed_matrix, op.matrix(), atol=tol, rtol=0)

    def test_z_decomposition(self, tol):
        """Tests that the decomposition of the PauliZ is correct"""
        op = qml.PauliZ(wires=0)
        res = op.decomposition()

        assert len(res) == 1

        assert res[0].name == "PhaseShift"

        assert res[0].wires == Wires([0])
        assert res[0].data[0] == np.pi

        decomposed_matrix = res[0].matrix()
        assert np.allclose(decomposed_matrix, op.matrix(), atol=tol, rtol=0)

    def test_s_decomposition(self, tol):
        """Tests that the decomposition of the S gate is correct"""
        op = qml.S(wires=0)
        res = op.decomposition()

        assert len(res) == 1

        assert res[0].name == "PhaseShift"

        assert res[0].wires == Wires([0])
        assert res[0].data[0] == np.pi / 2

        decomposed_matrix = res[0].matrix()
        assert np.allclose(decomposed_matrix, op.matrix(), atol=tol, rtol=0)

    def test_t_decomposition(self, tol):
        """Tests that the decomposition of the T gate is correct"""
        op = qml.T(wires=0)
        res = op.decomposition()

        assert len(res) == 1

        assert res[0].name == "PhaseShift"

        assert res[0].wires == Wires([0])
        assert res[0].data[0] == np.pi / 4

        decomposed_matrix = res[0].matrix()
        assert np.allclose(decomposed_matrix, op.matrix(), atol=tol, rtol=0)

    def test_sx_decomposition(self, tol):
        """Tests that the decomposition of the SX gate is correct"""
        op = qml.SX(wires=0)
        res = op.decomposition()

        assert len(res) == 4

        assert all([res[i].wires == Wires([0]) for i in range(4)])

        assert res[0].name == "RZ"
        assert res[1].name == "RY"
        assert res[2].name == "RZ"
        assert res[3].name == "PhaseShift"

        assert res[0].data[0] == np.pi / 2
        assert res[1].data[0] == np.pi / 2
        assert res[2].data[0] == -np.pi
        assert res[3].data[0] == np.pi / 2

        decomposed_matrix = np.linalg.multi_dot([i.matrix() for i in reversed(res)])
        assert np.allclose(decomposed_matrix, op.matrix(), atol=tol, rtol=0)

    def test_hadamard_decomposition(self, tol):
        """Tests that the decomposition of the Hadamard gate is correct"""
        op = qml.Hadamard(wires=0)
        res = op.decomposition()

        assert len(res) == 3

        assert res[0].name == "PhaseShift"

        assert res[0].wires == Wires([0])
        assert res[0].data[0] == np.pi / 2

        assert res[1].name == "RX"
        assert res[1].wires == Wires([0])
        assert res[0].data[0] == np.pi / 2

        assert res[2].name == "PhaseShift"
        assert res[2].wires == Wires([0])
        assert res[0].data[0] == np.pi / 2

        decomposed_matrix = np.linalg.multi_dot([i.matrix() for i in reversed(res)])
        assert np.allclose(decomposed_matrix, op.matrix(), atol=tol, rtol=0)

    def test_CY_decomposition(self, tol):
        """Tests that the decomposition of the CY gate is correct"""
        op = qml.CY(wires=[0, 1])
        res = op.decomposition()

        mats = []
        for i in reversed(res):
            if len(i.wires) == 1:
                mats.append(np.kron(i.matrix(), np.eye(2)))
            else:
                mats.append(i.matrix())

        decomposed_matrix = np.linalg.multi_dot(mats)
        assert np.allclose(decomposed_matrix, op.matrix(), atol=tol, rtol=0)

    def test_ISWAP_decomposition(self, tol):
        """Tests that the decomposition of the ISWAP gate is correct"""
        op = qml.ISWAP(wires=[0, 1])
        res = op.decomposition()

        assert len(res) == 6

        assert res[0].wires == Wires([0])
        assert res[1].wires == Wires([1])
        assert res[2].wires == Wires([0])
        assert res[3].wires == Wires([0, 1])
        assert res[4].wires == Wires([1, 0])
        assert res[5].wires == Wires([1])

        assert res[0].name == "S"
        assert res[1].name == "S"
        assert res[2].name == "Hadamard"
        assert res[3].name == "CNOT"
        assert res[4].name == "CNOT"
        assert res[5].name == "Hadamard"
        mats = []
        for i in reversed(res):
            if i.wires == Wires([1]):
                mats.append(np.kron(np.eye(2), i.matrix()))
            elif i.wires == Wires([0]):
                mats.append(np.kron(i.matrix(), np.eye(2)))
            elif i.wires == Wires([1, 0]) and i.name == "CNOT":
                mats.append(np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]))
            else:
                mats.append(i.matrix())

        decomposed_matrix = np.linalg.multi_dot(mats)

        assert np.allclose(decomposed_matrix, op.matrix(), atol=tol, rtol=0)

    def test_ECR_decomposition(self, tol):
        """Tests that the decomposition of the ECR gate is correct"""
        op = qml.ECR(wires=[0, 1])
        res = op.decomposition()

        assert len(res) == 6

        assert res[0].wires == Wires([0])
        assert res[1].wires == Wires([0, 1])
        assert res[2].wires == Wires([1])
        assert res[3].wires == Wires([0])
        assert res[4].wires == Wires([0])
        assert res[5].wires == Wires([0])

        assert res[0].name == "PauliZ"
        assert res[1].name == "CNOT"
        assert res[2].name == "SX"
        assert res[3].name == "RX"
        assert res[4].name == "RY"
        assert res[5].name == "RX"

        mats = []
        for i in reversed(res):
            if i.wires == Wires([1]):
                mats.append(np.kron(np.eye(2), i.matrix()))
            elif i.wires == Wires([0]):
                mats.append(np.kron(i.matrix(), np.eye(2)))
            elif i.wires == Wires([1, 0]) and i.name == "CNOT":
                mats.append(np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]))
            else:
                mats.append(i.matrix())

        decomposed_matrix = np.linalg.multi_dot(mats)

        assert np.allclose(decomposed_matrix, op.matrix(), atol=tol, rtol=0)

    @pytest.mark.parametrize("siswap_op", [qml.SISWAP, qml.SQISW])
    def test_SISWAP_decomposition(self, siswap_op, tol):
        """Tests that the decomposition of the SISWAP gate and its SQISW alias gate is correct"""
        op = siswap_op(wires=[0, 1])
        res = op.decomposition()

        assert len(res) == 12

        assert res[0].wires == Wires([0])
        assert res[1].wires == Wires([0])
        assert res[2].wires == Wires([0, 1])
        assert res[3].wires == Wires([0])
        assert res[4].wires == Wires([0])
        assert res[5].wires == Wires([0])
        assert res[6].wires == Wires([0])
        assert res[7].wires == Wires([1])
        assert res[8].wires == Wires([1])
        assert res[9].wires == Wires([0, 1])
        assert res[10].wires == Wires([0])
        assert res[11].wires == Wires([1])

        assert res[0].name == "SX"
        assert res[1].name == "RZ"
        assert res[2].name == "CNOT"
        assert res[3].name == "SX"
        assert res[4].name == "RZ"
        assert res[5].name == "SX"
        assert res[6].name == "RZ"
        assert res[7].name == "SX"
        assert res[8].name == "RZ"
        assert res[9].name == "CNOT"
        assert res[10].name == "SX"
        assert res[11].name == "SX"

        mats = []
        for i in reversed(res):
            if i.wires == Wires([1]):
                mats.append(np.kron(np.eye(2), i.matrix()))
            elif i.wires == Wires([0]):
                mats.append(np.kron(i.matrix(), np.eye(2)))
            elif i.wires == Wires([1, 0]) and i.name == "CNOT":
                mats.append(np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]))
            else:
                mats.append(i.matrix())

        decomposed_matrix = np.linalg.multi_dot(mats)

        assert np.allclose(decomposed_matrix, op.matrix(), atol=tol, rtol=0)

    def test_toffoli_decomposition(self, tol):
        """Tests that the decomposition of the Toffoli gate is correct"""
        op = qml.Toffoli(wires=[0, 1, 2])
        res = op.decomposition()

        assert len(res) == 15

        mats = []

        for i in reversed(res):
            if i.wires == Wires([2]):
                mats.append(np.kron(np.eye(4), i.matrix()))
            elif i.wires == Wires([1]):
                mats.append(np.kron(np.eye(2), np.kron(i.matrix(), np.eye(2))))
            elif i.wires == Wires([0]):
                mats.append(np.kron(i.matrix(), np.eye(4)))
            elif i.wires == Wires([0, 1]) and i.name == "CNOT":
                mats.append(np.kron(i.matrix(), np.eye(2)))
            elif i.wires == Wires([1, 2]) and i.name == "CNOT":
                mats.append(np.kron(np.eye(2), i.matrix()))
            elif i.wires == Wires([0, 2]) and i.name == "CNOT":
                mats.append(
                    np.array(
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 1, 0],
                        ]
                    )
                )

        decomposed_matrix = np.linalg.multi_dot(mats)

        assert np.allclose(decomposed_matrix, op.matrix(), atol=tol, rtol=0)

    def test_CSWAP_decomposition(self, tol):
        """Tests that the decomposition of the CSWAP gate is correct"""
        op = qml.CSWAP(wires=[0, 1, 2])
        res = op.decomposition()

        assert len(res) == 3

        mats = []

        for i in reversed(res):  # only use 3 toffoli gates
            if i.wires == Wires([0, 2, 1]) and i.name == "Toffoli":
                mats.append(
                    np.array(
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                        ]
                    )
                )
            elif i.wires == Wires([0, 1, 2]) and i.name == "Toffoli":
                mats.append(
                    np.array(
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 1, 0],
                        ]
                    )
                )

        decomposed_matrix = np.linalg.multi_dot(mats)

        assert np.allclose(decomposed_matrix, op.matrix(), atol=tol, rtol=0)

    def test_swap_decomposition(self):
        """Tests the swap operator produces the correct output"""
        opr = qml.SWAP(wires=[0, 1])
        decomp = opr.decomposition()

        mat = []
        for op in reversed(decomp):
            if isinstance(op, qml.CNOT) and op.wires.tolist() == [0, 1]:
                mat.append(CNOT)
            elif isinstance(op, qml.CNOT) and op.wires.tolist() == [1, 0]:
                mat.append(np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]))

        decomposed_matrix = np.linalg.multi_dot(mat)

        assert np.allclose(decomposed_matrix, opr.matrix())


class TestEigenval:
    def test_iswap_eigenval(self):
        """Tests that the ISWAP eigenvalue matches the numpy eigenvalues of the ISWAP matrix"""
        op = qml.ISWAP(wires=[0, 1])
        exp = np.linalg.eigvals(op.matrix())
        res = op.eigvals()
        assert np.allclose(res, exp)

    def test_ECR_eigenval(self):
        """Tests that the ECR eigenvalue matches the numpy eigenvalues of the ECR matrix"""
        op = qml.ECR(wires=[0, 1])
        exp = np.linalg.eigvals(op.matrix())
        res = op.eigvals()
        assert np.allclose(res, exp)

    @pytest.mark.parametrize("siswap_op", [qml.SISWAP, qml.SQISW])
    def test_siswap_eigenval(self, siswap_op):
        """Tests that the ISWAP eigenvalue matches the numpy eigenvalues of the ISWAP matrix"""
        op = siswap_op(wires=[0, 1])
        exp = np.linalg.eigvals(op.matrix())
        res = op.eigvals()
        assert np.allclose(res, exp)

    def test_sx_eigenvals(self):
        """Tests that the SX eigenvalues are correct."""
        evals = qml.SX(wires=0).eigvals()
        expected = np.linalg.eigvals(qml.SX(wires=0).matrix())
        assert np.allclose(evals, expected)


class TestBarrier:
    """Tests that the Barrier gate is correct."""

    def test_use_barrier(self):
        r"""Test that the barrier influences compilation."""

        def qfunc():
            qml.Hadamard(wires=0)
            qml.Barrier(wires=0)
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=3)
        qnode = qml.QNode(qfunc, dev)
        gates = qml.specs(qnode)()["gate_sizes"][1]

        assert gates == 3

        optimized_qfunc = qml.compile()(qfunc)
        optimized_qnode = qml.QNode(optimized_qfunc, dev)
        optimized_gates = qml.specs(optimized_qnode)()["gate_sizes"][1]

        assert optimized_gates == 2

    def test_barrier_only_visual(self):
        r"""Test that the barrier doesn't influence compilation when the only_visual parameter is True."""

        def qfunc():
            qml.Hadamard(wires=0)
            qml.Barrier(only_visual=True, wires=0)
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=3)
        optimized_qfunc = qml.compile()(qfunc)
        optimized_qnode = qml.QNode(optimized_qfunc, dev)
        optimized_gates = qml.specs(optimized_qnode)()["gate_sizes"][1]

        assert optimized_gates == 0

    def test_barrier_edge_cases(self):
        r"""Test that the barrier works in edge cases."""

        def qfunc():
            qml.Barrier(wires=0)
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            qml.Barrier(wires=0)
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=3)
        qnode = qml.QNode(qfunc, dev)
        gates = qml.specs(qnode)()["gate_sizes"][1]

        assert gates == 4

        optimized_qfunc = qml.compile()(qfunc)
        optimized_qnode = qml.QNode(optimized_qfunc, dev)
        optimized_gates = qml.specs(optimized_qnode)()["gate_sizes"][1]

        assert optimized_gates == 0

        def qfunc():

            qml.Hadamard(wires=0)
            qml.Barrier(wires=0)
            qml.Barrier(wires=0)
            qml.Hadamard(wires=0)

            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=3)
        qnode = qml.QNode(qfunc, dev)
        gates = qml.specs(qnode)()["gate_sizes"][1]

        assert gates == 4

        def qfunc():
            qml.Hadamard(wires=0)
            qml.Barrier(only_visual=True, wires=0)
            qml.Barrier(wires=0)
            qml.Hadamard(wires=0)

            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=3)
        optimized_qfunc = qml.compile()(qfunc)
        optimized_qnode = qml.QNode(optimized_qfunc, dev)
        optimized_gates = qml.specs(optimized_qnode)()["gate_sizes"][1]

        assert optimized_gates == 2

    def test_barrier_adjoint(self):
        """Test if adjoint of a Barrier is decomposed correctly."""

        base = qml.Barrier(wires=(0, 1))
        adj = qml.ops.op_math.Adjoint(base)

        assert adj.decomposition()[0].name == "Barrier"

    def test_barrier_control(self):
        """Test if Barrier is correctly included in queue when controlling"""
        dev = qml.device("default.qubit", wires=3)

        def barrier():
            qml.PauliX(wires=0)
            qml.Barrier(wires=[0, 1])
            qml.CNOT(wires=[0, 1])

        @qml.qnode(dev)
        def circuit():
            barrier()
            qml.ctrl(barrier, 2)()
            return qml.state()

        circuit()
        tape = circuit.tape.expand(stop_at=lambda op: op.name in ["Barrier", "PauliX", "CNOT"])

        assert tape.operations[1].name == "Barrier"
        assert tape.operations[4].name == "Barrier"

    def test_barrier_empty_wire_list_no_error(self):
        """Test that barrier does not raise an error when instantiated with wires=[]."""
        barrier = qml.Barrier(wires=[])
        assert isinstance(barrier, qml.Barrier)

    def test_simplify_only_visual_one_wire(self):
        """Test that if `only_visual=True`, the operation simplifies to the identity."""
        op = qml.Barrier(wires="a", only_visual=True)
        simplified = op.simplify()
        assert qml.equal(simplified, qml.Identity("a"))

    def test_simplify_only_visual_multiple_wires(self):
        """Test that if `only_visual=True`, the operation simplifies to a product of identities."""
        op = qml.Barrier(wires=(0, 1, 2), only_visual=True)
        simplified = op.simplify()
        assert isinstance(simplified, qml.ops.op_math.Prod)
        for i, op in enumerate(simplified.operands):
            assert qml.equal(op, qml.Identity(i))

    def test_simplify_only_visual_False(self):
        """Test that no simplification occurs if only_visual is False."""
        op = qml.Barrier(wires=(0, 1, 2, 3), only_visual=False)
        assert op.simplify() is op


class TestWireCut:
    """Tests for the WireCut operator"""

    def test_behaves_as_identity(self):
        """Tests that the WireCut operator behaves as the Identity in the
        absence of cutting"""

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def with_wirecut():
            qml.PauliX(wires=0)
            qml.WireCut(wires=0)
            return qml.state()

        @qml.qnode(dev)
        def without_wirecut():
            qml.PauliX(wires=0)
            return qml.state()

        assert np.allclose(with_wirecut(), without_wirecut())

    def test_wires_empty_list_raises_error(self):
        """Test that the WireCut operator raises an error when instantiated with an empty list."""
        with pytest.raises(
            ValueError, match="WireCut: wrong number of wires. At least one wire has to be given."
        ):
            qml.WireCut(wires=[])


class TestMultiControlledX:
    """Tests for the MultiControlledX"""

    X = np.array([[0, 1], [1, 0]])

    @pytest.mark.parametrize(
        "control_wires,wires,control_values,expected_warning_message",
        [
            ([0], 1, "0", "The control_wires keyword will be removed soon."),
        ],
    )
    def test_warning_depractation_controlwires(
        self, control_wires, wires, control_values, expected_warning_message
    ):
        target_wires = wires
        with pytest.warns(UserWarning, match=expected_warning_message):
            qml.MultiControlledX(
                control_wires=control_wires, wires=target_wires, control_values=control_values
            )

    @pytest.mark.parametrize(
        "control_wires,wires,control_values,expected_error_message",
        [
            (None, None, "10", "Must specify the wires where the operation acts on"),
            (None, [0, 1, 2], "ab", "String of control values can contain only '0' or '1'."),
            (
                None,
                [0, 1, 2],
                "011",
                "Length of control bit string must equal number of control wires.",
            ),
            (
                None,
                [0, 1, 2],
                [0, 1],
                "Control values must be passed as a string.",
            ),
            (
                None,
                [1],
                "1",
                r"MultiControlledX: wrong number of wires. 1 wire\(s\) given. Need at least 2.",
            ),
            ([0], None, "", "Must specify the wires where the operation acts on"),
            ([0, 1], 2, "ab", "String of control values can contain only '0' or '1'."),
            ([0, 1], 2, "011", "Length of control bit string must equal number of control wires."),
            ([0, 1], 2, [0, 1], "Control values must be passed as a string."),
            ([0, 1], [2, 3], "10", "MultiControlledX accepts a single target wire."),
        ],
    )
    @pytest.mark.filterwarnings("ignore:The control_wires keyword will be removed soon.")
    def test_invalid_mixed_polarity_controls(
        self, control_wires, wires, control_values, expected_error_message
    ):
        """Test if MultiControlledX properly handles invalid mixed-polarity
        control values."""
        target_wires = wires

        with pytest.raises(ValueError, match=expected_error_message):
            qml.MultiControlledX(
                control_wires=control_wires, wires=target_wires, control_values=control_values
            ).matrix()

    @pytest.mark.parametrize(
        "control_wires,wires,control_values",
        [
            ([0], 1, "0"),
            ([0, 1], 2, "00"),
            ([0, 1], 2, "10"),
            ([1, 0], 2, "10"),
            ([0, 1], 2, "11"),
            ([0, 2], 1, "10"),
            ([1, 2, 0], 3, "100"),
            ([1, 0, 2, 4], 3, "1001"),
            ([0, 1, 2, 5, 3, 6], 4, "100001"),
        ],
    )
    @pytest.mark.filterwarnings("ignore:The control_wires keyword will be removed soon.")
    def test_mixed_polarity_controls_old(self, control_wires, wires, control_values):
        """Test if MultiControlledX properly applies mixed-polarity
        control values with old version of the arguments."""

        target_wires = Wires(wires)
        dev = qml.device("default.qubit", wires=len(control_wires + target_wires))

        # Pick random starting state for the control and target qubits
        control_state_weights = np.random.normal(size=(2 ** (len(control_wires) + 1) - 2))
        target_state_weights = np.random.normal(size=(2 ** (len(target_wires) + 1) - 2))

        @qml.qnode(dev)
        def circuit_mpmct():
            qml.templates.ArbitraryStatePreparation(control_state_weights, wires=control_wires)
            qml.templates.ArbitraryStatePreparation(target_state_weights, wires=target_wires)

            qml.MultiControlledX(
                control_wires=control_wires, wires=target_wires, control_values=control_values
            )
            return qml.state()

        # The result of applying the mixed-polarity gate should be the same as
        # if we conjugated the specified control wires with Pauli X and applied the
        # "regular" ControlledQubitUnitary in between.

        x_locations = [x for x in range(len(control_values)) if control_values[x] == "0"]

        @qml.qnode(dev)
        def circuit_pauli_x():
            qml.templates.ArbitraryStatePreparation(control_state_weights, wires=control_wires)
            qml.templates.ArbitraryStatePreparation(target_state_weights, wires=target_wires)

            for wire in x_locations:
                qml.PauliX(wires=control_wires[wire])

            qml.ControlledQubitUnitary(X, control_wires=control_wires, wires=target_wires)

            for wire in x_locations:
                qml.PauliX(wires=control_wires[wire])

            return qml.state()

        mpmct_state = circuit_mpmct()
        pauli_x_state = circuit_pauli_x()

        assert np.allclose(mpmct_state, pauli_x_state)

    def test_decomposition_no_control_values(self):
        """Test decomposition has default control values of all ones."""
        decomp1 = qml.MultiControlledX.compute_decomposition((0, 1, 2))
        decomp2 = qml.MultiControlledX.compute_decomposition((0, 1, 2), control_values="111")

        assert len(decomp1) == len(decomp2)

        for op1, op2 in zip(decomp1, decomp2):
            assert op1.__class__ == op2.__class__

    @pytest.mark.parametrize(
        "wires,control_values",
        [
            ([0, 1], "0"),
            ([0, 1, 2], "00"),
            ([0, 1, 2], "10"),
            ([1, 0, 2], "10"),
            ([0, 1, 2], "11"),
            ([0, 2, 1], "10"),
            ([1, 2, 0, 3], "100"),
            ([1, 0, 2, 4, 3], "1001"),
            ([0, 1, 2, 5, 3, 6, 4], "100001"),
        ],
    )
    def test_mixed_polarity_controls(self, wires, control_values):
        """Test if MultiControlledX properly applies mixed-polarity
        control values."""

        control_wires = Wires(wires[:-1])
        target_wires = Wires(wires[-1])

        dev = qml.device("default.qubit", wires=len(control_wires + target_wires))

        # Pick random starting state for the control and target qubits
        control_state_weights = np.random.normal(size=(2 ** (len(control_wires) + 1) - 2))
        target_state_weights = np.random.normal(size=(2 ** (len(target_wires) + 1) - 2))

        @qml.qnode(dev)
        def circuit_mpmct():
            qml.templates.ArbitraryStatePreparation(control_state_weights, wires=control_wires)
            qml.templates.ArbitraryStatePreparation(target_state_weights, wires=target_wires)

            qml.MultiControlledX(wires=Wires(wires), control_values=control_values)
            return qml.state()

        # The result of applying the mixed-polarity gate should be the same as
        # if we conjugated the specified control wires with Pauli X and applied the
        # "regular" ControlledQubitUnitary in between.

        x_locations = [x for x in range(len(control_values)) if control_values[x] == "0"]

        @qml.qnode(dev)
        def circuit_pauli_x():
            qml.templates.ArbitraryStatePreparation(control_state_weights, wires=control_wires)
            qml.templates.ArbitraryStatePreparation(target_state_weights, wires=target_wires)

            for wire in x_locations:
                qml.PauliX(wires=control_wires[wire])

            qml.ControlledQubitUnitary(X, control_wires=control_wires, wires=target_wires)

            for wire in x_locations:
                qml.PauliX(wires=control_wires[wire])

            return qml.state()

        mpmct_state = circuit_mpmct()
        pauli_x_state = circuit_pauli_x()

        assert np.allclose(mpmct_state, pauli_x_state)

    @pytest.mark.parametrize("n_ctrl_wires", range(3, 6))
    def test_decomposition_with_many_workers(self, n_ctrl_wires):
        """Test that the decomposed MultiControlledX gate performs the same unitary as the
        matrix-based version by checking if U^dagger U applies the identity to each basis
        state. This test focuses on the case where there are many work wires."""
        control_wires = range(n_ctrl_wires)
        target_wire = n_ctrl_wires
        work_wires = range(n_ctrl_wires + 1, 2 * n_ctrl_wires + 1)

        dev = qml.device("default.qubit", wires=2 * n_ctrl_wires + 1)

        with qml.tape.QuantumTape() as tape:
            qml.MultiControlledX._decomposition_with_many_workers(
                control_wires, target_wire, work_wires
            )
        assert all(isinstance(op, qml.Toffoli) for op in tape.operations)

        @qml.qnode(dev)
        def f(bitstring):
            qml.BasisState(bitstring, wires=range(n_ctrl_wires + 1))
            qml.MultiControlledX(wires=list(control_wires) + [target_wire]).inv()
            for op in tape.operations:
                op.queue()
            return qml.probs(wires=range(n_ctrl_wires + 1))

        u = np.array(
            [f(np.array(b)) for b in itertools.product(range(2), repeat=n_ctrl_wires + 1)]
        ).T
        assert np.allclose(u, np.eye(2 ** (n_ctrl_wires + 1)))

    @pytest.mark.parametrize("n_ctrl_wires", range(3, 6))
    def test_decomposition_with_one_worker(self, n_ctrl_wires):
        """Test that the decomposed MultiControlledX gate performs the same unitary as the
        matrix-based version by checking if U^dagger U applies the identity to each basis
        state. This test focuses on the case where there is one work wire."""
        control_wires = Wires(range(n_ctrl_wires))
        target_wire = n_ctrl_wires
        work_wires = n_ctrl_wires + 1

        dev = qml.device("default.qubit", wires=n_ctrl_wires + 2)

        with qml.tape.QuantumTape() as tape:
            qml.MultiControlledX._decomposition_with_one_worker(
                control_wires, target_wire, work_wires
            )
        tape = tape.expand(depth=1)
        assert all(
            isinstance(op, qml.Toffoli) or isinstance(op, qml.CNOT) for op in tape.operations
        )

        @qml.qnode(dev)
        def f(bitstring):
            qml.BasisState(bitstring, wires=range(n_ctrl_wires + 1))
            qml.MultiControlledX(wires=list(control_wires) + [target_wire]).inv()
            for op in tape.operations:
                op.queue()
            return qml.probs(wires=range(n_ctrl_wires + 1))

        u = np.array(
            [f(np.array(b)) for b in itertools.product(range(2), repeat=n_ctrl_wires + 1)]
        ).T
        assert np.allclose(u, np.eye(2 ** (n_ctrl_wires + 1)))

    def test_not_enough_workers(self):
        """Test that a ValueError is raised when more than 2 control wires are to be decomposed with
        no work wires supplied"""
        control_target_wires = range(4)
        op = qml.MultiControlledX(wires=control_target_wires)

        match = f"At least one work wire is required to decompose operation: MultiControlledX"
        with pytest.raises(ValueError, match=match):
            op.decomposition()

    def test_not_unique_wires(self):
        """Test that a ValueError is raised when work_wires is not complementary to control_wires"""
        control_target_wires = range(4)
        work_wires = range(2)
        with pytest.raises(
            ValueError, match="The work wires must be different from the control and target wires"
        ):
            qml.MultiControlledX(wires=control_target_wires, work_wires=work_wires)

    @pytest.mark.parametrize("control_val", ["0", "1"])
    @pytest.mark.parametrize("n_ctrl_wires", range(1, 6))
    def test_decomposition_with_flips(self, n_ctrl_wires, control_val, mocker):
        """Test that the decomposed MultiControlledX gate performs the same unitary as the
        matrix-based version by checking if U^dagger U applies the identity to each basis
        state. This test focuses on varying the control values."""
        control_values = control_val * n_ctrl_wires
        control_target_wires = list(range(n_ctrl_wires)) + [n_ctrl_wires]
        work_wires = range(n_ctrl_wires + 1, 2 * n_ctrl_wires + 1)

        spy = mocker.spy(qml.MultiControlledX, "decomposition")
        dev = qml.device("default.qubit", wires=2 * n_ctrl_wires + 1)

        with qml.tape.QuantumTape() as tape:
            qml.MultiControlledX(
                wires=control_target_wires,
                work_wires=work_wires,
                control_values=control_values,
            )
        tape = tape.expand(depth=1)
        assert all(not isinstance(op, qml.MultiControlledX) for op in tape.operations)

        @qml.qnode(dev)
        def f(bitstring):
            qml.BasisState(bitstring, wires=range(n_ctrl_wires + 1))
            qml.MultiControlledX(wires=control_target_wires, control_values=control_values).inv()
            for op in tape.operations:
                op.queue()
            return qml.probs(wires=range(n_ctrl_wires + 1))

        u = np.array(
            [f(np.array(b)) for b in itertools.product(range(2), repeat=n_ctrl_wires + 1)]
        ).T
        spy.assert_called()
        assert np.allclose(u, np.eye(2 ** (n_ctrl_wires + 1)))

    def test_decomposition_with_custom_wire_labels(self, mocker):
        """Test that the decomposed MultiControlledX gate performs the same unitary as the
        matrix-based version by checking if U^dagger U applies the identity to each basis
        state. This test focuses on using custom wire labels."""
        n_ctrl_wires = 4
        control_target_wires = [-1, "alice", 42, 3.14, "bob"]
        work_wires = ["charlie"]
        all_wires = control_target_wires + work_wires

        spy = mocker.spy(qml.MultiControlledX, "decomposition")
        dev = qml.device("default.qubit", wires=all_wires)

        with qml.tape.QuantumTape() as tape:
            qml.MultiControlledX(wires=control_target_wires, work_wires=work_wires)
        tape = tape.expand(depth=2)
        assert all(not isinstance(op, qml.MultiControlledX) for op in tape.operations)

        @qml.qnode(dev)
        def f(bitstring):
            qml.BasisState(bitstring, wires=control_target_wires)
            qml.MultiControlledX(wires=control_target_wires).inv()
            for op in tape.operations:
                op.queue()
            return qml.probs(wires=control_target_wires)

        u = np.array(
            [f(np.array(b)) for b in itertools.product(range(2), repeat=n_ctrl_wires + 1)]
        ).T
        spy.assert_called()
        assert np.allclose(u, np.eye(2 ** (n_ctrl_wires + 1)))

    def test_worker_state_unperturbed(self, mocker):
        """Test that the state of the worker wires is unperturbed after the decomposition has used
        them. To do this, a random state over all the qubits (control, target and workers) is
        loaded and U^dagger U(decomposed) is applied. If the workers are uncomputed, the output
        state will be the same as the input."""
        control_target_wires = range(5)
        worker_wires = [5, 6]
        n_all_wires = 7

        rnd_state = unitary_group.rvs(2**n_all_wires, random_state=1)[0]
        spy = mocker.spy(qml.MultiControlledX, "decomposition")
        dev = qml.device("default.qubit", wires=n_all_wires)

        with qml.tape.QuantumTape() as tape:
            qml.MultiControlledX(wires=control_target_wires, work_wires=worker_wires)
        tape = tape.expand(depth=1)
        assert all(not isinstance(op, qml.MultiControlledX) for op in tape.operations)

        @qml.qnode(dev)
        def f():
            qml.QubitStateVector(rnd_state, wires=range(n_all_wires))
            qml.MultiControlledX(wires=control_target_wires).inv()
            for op in tape.operations:
                op.queue()
            return qml.state()

        assert np.allclose(f(), rnd_state)
        spy.assert_called()

    def test_compute_matrix_no_control_values(self):
        """Test compute_matrix assumes all control on "1" if no
        `control_values` provided"""
        mat1 = qml.MultiControlledX.compute_matrix([0, 1])
        mat2 = qml.MultiControlledX.compute_matrix([0, 1], control_values="11")
        assert np.allclose(mat1, mat2)


class TestIntegerComparator:
    """Tests for the IntegerComparator"""

    X = np.array([[0, 1], [1, 0]])

    @pytest.mark.parametrize(
        "value,geq,control_wires,wires,expected_warning_message",
        [
            (2, True, [0, 1], [2], "The control_wires keyword will be removed soon."),
        ],
    )
    def test_warning_depractation_controlwires(
        self, value, geq, control_wires, wires, expected_warning_message
    ):
        target_wires = wires
        with pytest.warns(UserWarning, match=expected_warning_message):
            qml.IntegerComparator(
                value=value, geq=geq, control_wires=control_wires, wires=target_wires
            )

    @pytest.mark.parametrize(
        "value,geq,control_wires,wires,expected_warning_message",
        [
            (None, True, None, None, "Must specify the integer value to compare against."),
            (4.20, False, None, [0,1,2], "The comparable value must be an integer."),
            (2, True, None, None, "Must specify the target wire where the operation acts on."),
            (2, True, None, [1], r"IntegerComparator: wrong number of wires. 1 wire\(s\) given. Need at least 2." ),
            ()
        ],
    )

    def test_compute_matrix_geq_True(self):
        """Test compute_matrix for geq=True"""
        mat1 = qml.IntegerComparator.compute_matrix(2, [0,1], geq=True)
        mat2 = np.zeros((8,8))

        mat2[0,0] = 1
        mat2[1,1] = 1
        mat2[2,2] = 1
        mat2[3,3] = 1
        mat2[4,5] = 1
        mat2[5,4] = 1
        mat2[6,7] = 1
        mat2[7,6] = 1

        assert np.allclose(mat1, mat2)

    def test_compute_matrix_geq_False(self):
        """Test compute_matrix for geq=False"""
        mat1 = qml.IntegerComparator.compute_matrix(2, [0,1], geq=False)
        mat2 = np.zeros((8,8))

        mat2[0,1] = 1
        mat2[1,0] = 1
        mat2[2,3] = 1
        mat2[3,2] = 1
        mat2[4,4] = 1
        mat2[5,5] = 1
        mat2[6,6] = 1
        mat2[7,7] = 1

        assert np.allclose(mat1, mat2)

period_two_ops = (
    qml.PauliX(0),
    qml.PauliY(0),
    qml.PauliZ(0),
    qml.Hadamard("a"),
    qml.CNOT(wires=(0, 1)),
    qml.CZ(wires=(0, 1)),
    qml.CY(wires=(0, 1)),
    qml.SWAP(wires=(0, 1)),
    qml.ISWAP(wires=(0, 1)),
    qml.ECR(wires=(0, 1)),
    qml.CSWAP(wires=(0, 1, 2)),
    qml.Toffoli(wires=(0, 1, 2)),
    qml.MultiControlledX(wires=(0, 1, 2, 3)),
    qml.IntegerComparator(wires=(0, 1, 2, 3)),
)


class TestPowMethod:
    @pytest.mark.parametrize("op", period_two_ops)
    @pytest.mark.parametrize("n", (1, 5, -1, -5))
    def test_period_two_pow_odd(self, op, n):
        """Test that ops with a period of 2 raised to an odd power are the same as the original op."""
        assert op.pow(n)[0].__class__ is op.__class__

    @pytest.mark.parametrize("op", period_two_ops)
    @pytest.mark.parametrize("n", (2, 6, 0, -2))
    def test_period_two_pow_even(self, op, n):
        """Test that ops with a period of 2 raised to an even power are empty lists."""
        assert len(op.pow(n)) == 0

    @pytest.mark.parametrize("op", period_two_ops)
    def test_period_two_noninteger_power(self, op):
        """Test that ops with a period of 2 raised to a non-integer power raise an error."""
        if op.__class__ is qml.PauliZ:
            pytest.skip("PauliZ can be raised to any power.")
        with pytest.raises(qml.operation.PowUndefinedError):
            op.pow(1.234)

    @pytest.mark.parametrize("n", (0.5, 2.5, -1.5))
    def test_paulix_squareroot(self, n):
        """Check that the square root of PauliX is SX"""
        op = qml.PauliX(0)

        pow_ops = op.pow(n)
        assert len(pow_ops) == 1
        assert pow_ops[0].__class__ is qml.SX

        sqrt_mat = qml.matrix(op.pow)(n)
        sqrt_mat_squared = qml.math.linalg.matrix_power(sqrt_mat, 2)

        assert qml.math.allclose(sqrt_mat_squared, qml.matrix(op))

    @pytest.mark.parametrize("n", (0.5, 2.5, -1.5))
    def test_pauliz_squareroot(self, n):
        """Check that the square root of PauliZ is S"""
        assert qml.PauliZ(0).pow(n)[0].__class__ is qml.S

        op = qml.PauliZ(0)
        sqrt_mat = qml.matrix(op.pow)(n)
        sqrt_mat_squared = qml.math.linalg.matrix_power(sqrt_mat, 2)

        assert qml.math.allclose(sqrt_mat_squared, qml.matrix(op))

    @pytest.mark.parametrize("n", (0.25, 2.25, -1.75))
    def test_pauliz_fourthroot(self, n):
        """Check that the fourth root of PauliZ is T."""
        assert qml.PauliZ(0).pow(n)[0].__class__ is qml.T

        op = qml.PauliZ(0)
        quad_mat = qml.matrix(op.pow)(n)
        quad_mat_pow = qml.math.linalg.matrix_power(quad_mat, 4)

        assert qml.math.allclose(quad_mat_pow, qml.matrix(op))

    @pytest.mark.parametrize("n", (0.12, -3.462, 3.693))
    def test_pauliz_general_power(self, n):
        """Check that PauliZ raised to an non-integer power that's not the square root
        results in a PhaseShift."""
        op_pow = qml.PauliZ(0).pow(n)

        assert len(op_pow) == 1
        assert op_pow[0].__class__ is qml.PhaseShift
        assert qml.math.allclose(op_pow[0].data[0], np.pi * (n % 2))

    @pytest.mark.parametrize("n", (0.5, 2.5, -1.5))
    def test_ISWAP_sqaure_root(self, n):
        """Test that SISWAP is the square root of ISWAP."""
        op = qml.ISWAP(wires=(0, 1))

        assert op.pow(n)[0].__class__ is qml.SISWAP

        sqrt_mat = qml.matrix(op.pow)(n)
        sqrt_mat_squared = qml.math.linalg.matrix_power(sqrt_mat, 2)
        assert qml.math.allclose(sqrt_mat_squared, qml.matrix(op))

    @pytest.mark.parametrize("offset", (0, 4, -4))
    def test_S_pow(self, offset):
        op = qml.S("a")

        assert len(op.pow(0 + offset)) == 0

        assert op.pow(0.5 + offset)[0].__class__ is qml.T
        assert op.pow(1 + offset)[0].__class__ is qml.S
        assert op.pow(2 + offset)[0].__class__ is qml.PauliZ

        n = 1.234
        op_pow = op.pow(n + offset)
        assert op_pow[0].__class__ is qml.PhaseShift
        assert qml.math.allclose(op_pow[0].data[0], np.pi * n / 2)

    @pytest.mark.parametrize("offset", (0, 8, -8))
    def test_T_pow(self, offset):
        """Test the powers of the T gate."""
        op = qml.T("b")

        assert len(op.pow(0 + offset)) == 0
        assert op.pow(1 + offset)[0].__class__ is qml.T
        assert op.pow(2 + offset)[0].__class__ is qml.S
        assert op.pow(4 + offset)[0].__class__ is qml.PauliZ

        n = 1.234
        op_pow = op.pow(n + offset)
        assert op_pow[0].__class__ is qml.PhaseShift
        assert qml.math.allclose(op_pow[0].data[0], np.pi * n / 4)

    @pytest.mark.parametrize("offset", (0, 4, -4))
    def test_SX_pow(self, offset):
        op = qml.SX("d")

        assert len(op.pow(0 + offset)) == 0

        assert op.pow(1 + offset)[0].__class__ is qml.SX
        assert op.pow(2 + offset)[0].__class__ is qml.PauliX

        with pytest.raises(qml.operation.PowUndefinedError):
            op.pow(2.43 + offset)

    @pytest.mark.parametrize("offset", (0, 4, -4))
    def test_SISWAP_pow(self, offset):
        """Test powers of the SISWAP operator"""
        op = qml.SISWAP(wires=("b", "c"))

        assert len(op.pow(0 + offset)) == 0
        assert op.pow(1 + offset)[0].__class__ is qml.SISWAP
        assert op.pow(2 + offset)[0].__class__ is qml.ISWAP

        with pytest.raises(qml.operation.PowUndefinedError):
            op.pow(2.34 + offset)

    @pytest.mark.parametrize("op", (qml.WireCut(0), qml.Barrier(0)))
    @pytest.mark.parametrize("n", (2, 0.123, -2.3))
    def test_pow_independent_ops(self, op, n):
        """Assert that the pow-independent ops WireCut and Barrier can be raised
        to any power and just return a copy."""
        assert op.pow(n)[0].__class__ is op.__class__


class TestControlledMethod:
    """Tests for the _controlled method of non-parametric operations."""

    def test_PauliX(self):
        """Test the PauliX _controlled method."""
        out = qml.PauliX(0)._controlled("a")
        assert qml.equal(out, qml.CNOT(("a", 0)))

    def test_PauliY(self):
        """Test the PauliY _controlled method."""
        out = qml.PauliY(0)._controlled("a")
        assert qml.equal(out, qml.CY(("a", 0)))

    def test_PauliZ(self):
        """Test the PauliZ _controlled method."""
        out = qml.PauliZ(0)._controlled("a")
        assert qml.equal(out, qml.CZ(("a", 0)))

    def test_CNOT(self):
        """Test the CNOT _controlled method"""
        out = qml.CNOT((0, 1))._controlled("a")
        assert qml.equal(out, qml.Toffoli(("a", 0, 1)))

    def test_SWAP(self):
        """Test the SWAP _controlled method."""
        out = qml.SWAP((0, 1))._controlled("a")
        assert qml.equal(out, qml.CSWAP(("a", 0, 1)))

    def test_Barrier(self):
        """Tests the _controlled behavior of Barrier."""
        original = qml.Barrier((0, 1, 2), only_visual=True)
        out = original._controlled("a")
        assert qml.equal(original, out)


class TestSparseMatrix:
    @pytest.mark.parametrize("op, mat", SPARSE_MATRIX_SUPPORTED_OPERATIONS)
    def test_sparse_matrix(self, op, mat):
        expected_sparse_mat = csr_matrix(mat)
        sparse_mat = op.sparse_matrix()

        assert type(sparse_mat) == type(expected_sparse_mat)
        assert all(sparse_mat.data == expected_sparse_mat.data)
        assert all(sparse_mat.indices == expected_sparse_mat.indices)


label_data = [
    (qml.Identity(0), "I", "I"),
    (qml.Hadamard(0), "H", "H"),
    (qml.PauliX(0), "X", "X"),
    (qml.PauliY(0), "Y", "Y"),
    (qml.PauliZ(0), "Z", "Z"),
    (qml.S(wires=0), "S", "S⁻¹"),
    (qml.T(wires=0), "T", "T⁻¹"),
    (qml.SX(wires=0), "SX", "SX⁻¹"),
    (qml.CNOT(wires=(0, 1)), "X", "X"),
    (qml.CZ(wires=(0, 1)), "Z", "Z"),
    (qml.CY(wires=(0, 1)), "Y", "Y"),
    (qml.SWAP(wires=(0, 1)), "SWAP", "SWAP⁻¹"),
    (qml.ISWAP(wires=(0, 1)), "ISWAP", "ISWAP⁻¹"),
    (qml.ECR(wires=(0, 1)), "ECR", "ECR⁻¹"),
    (qml.SISWAP(wires=(0, 1)), "SISWAP", "SISWAP⁻¹"),
    (qml.SQISW(wires=(0, 1)), "SISWAP", "SISWAP⁻¹"),
    (qml.CSWAP(wires=(0, 1, 2)), "SWAP", "SWAP"),
    (qml.Toffoli(wires=(0, 1, 2)), "X", "X"),
    (qml.MultiControlledX(wires=(0, 1, 2, 3)), "X", "X"),
    (qml.IntegerComparator(wires=(0, 1, 2, 3)), "X", "X"),
    (qml.Barrier(0), "||", "||"),
    (qml.WireCut(wires=0), "//", "//"),
]


@pytest.mark.parametrize("op, label1, label2", label_data)
def test_label_method(op, label1, label2):
    assert op.label() == label1
    assert op.label(decimals=2) == label1

    op.inv()
    assert op.label() == label2


control_data = [
    (qml.Identity(0), Wires([])),
    (qml.Hadamard(0), Wires([])),
    (qml.PauliX(0), Wires([])),
    (qml.PauliY(0), Wires([])),
    (qml.S(wires=0), Wires([])),
    (qml.T(wires=0), Wires([])),
    (qml.SX(wires=0), Wires([])),
    (qml.SWAP(wires=(0, 1)), Wires([])),
    (qml.ISWAP(wires=(0, 1)), Wires([])),
    (qml.SISWAP(wires=(0, 1)), Wires([])),
    (qml.ECR(wires=(0, 1)), Wires([])),
    (qml.CNOT(wires=(0, 1)), Wires(0)),
    (qml.CZ(wires=(0, 1)), Wires(0)),
    (qml.CY(wires=(0, 1)), Wires(0)),
    (qml.CSWAP(wires=(0, 1, 2)), Wires([0])),
    (qml.Toffoli(wires=(0, 1, 2)), Wires([0, 1])),
    (qm)l.MultiControlledX(wires=[0, 1, 2, 3, 4], Wires([0, 1, 2, 3])),
    (qml.IntegerComparator(wires=[0, 1, 2, 3, 4]), Wires([0, 1, 2, 3])),
]


@pytest.mark.parametrize("op, control_wires", control_data)
def test_control_wires(op, control_wires):
    """Test ``control_wires`` attribute for non-parametrized operations."""

    assert op.control_wires == control_wires


involution_ops = [  # ops who are their own inverses
    qml.Identity(0),
    qml.Hadamard(0),
    qml.PauliX(0),
    qml.PauliY(0),
    qml.PauliZ(0),
    qml.CNOT((0, 1)),
    qml.CZ((0, 1)),
    qml.CY((0, 1)),
    qml.SWAP((0, 1)),
    qml.ECR((0, 1)),
    qml.CSWAP((0, 1, 2)),
    qml.Toffoli((0, 1, 2)),
    qml.MultiControlledX(wires=(0, 1, 2, 3)),
    qml.IntegerComparator(wires=(0, 1, 2, 3)),
    qml.Barrier(0),
    qml.WireCut(0),
]


@pytest.mark.parametrize("op", involution_ops)
def test_adjoint_method(op, tol):
    adj_op = copy.copy(op)
    for _ in range(4):
        adj_op = adj_op.adjoint()

        assert adj_op.name == op.name
