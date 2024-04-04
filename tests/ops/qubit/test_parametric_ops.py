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
Unit tests for the available built-in parametric qubit operations.
"""
# pylint: disable=too-few-public-methods,too-many-public-methods
import copy
from functools import reduce

import numpy as np
import pytest
from gate_data import CPhaseShift00, CPhaseShift01, CPhaseShift10, Z
from scipy import sparse

import pennylane as qml
from pennylane import numpy as npp
from pennylane.ops.qubit import (
    RX as old_loc_RX,
    MultiRZ as old_loc_MultiRZ,
)

from pennylane.wires import Wires

PARAMETRIZED_OPERATIONS = [
    qml.RX(0.123, wires=0),
    qml.RY(1.434, wires=0),
    qml.RZ(2.774, wires=0),
    qml.PauliRot(0.123, "Y", wires=0),
    qml.IsingXX(0.123, wires=[0, 1]),
    qml.IsingYY(0.123, wires=[0, 1]),
    qml.IsingZZ(0.123, wires=[0, 1]),
    qml.IsingXY(0.123, wires=[0, 1]),
    qml.Rot(0.123, 0.456, 0.789, wires=0),
    qml.PhaseShift(2.133, wires=0),
    qml.PCPhase(1.23, dim=2, wires=[0, 1]),
    qml.ControlledPhaseShift(1.777, wires=[0, 2]),
    qml.CPhase(1.777, wires=[0, 2]),
    qml.CPhaseShift00(1.777, wires=[0, 2]),
    qml.CPhaseShift01(1.777, wires=[0, 2]),
    qml.CPhaseShift10(1.777, wires=[0, 2]),
    qml.MultiRZ(0.112, wires=[1, 2, 3]),
    qml.CRX(0.836, wires=[2, 3]),
    qml.CRY(0.721, wires=[2, 3]),
    qml.CRZ(0.554, wires=[2, 3]),
    qml.U1(0.123, wires=0),
    qml.U2(3.556, 2.134, wires=0),
    qml.U3(2.009, 1.894, 0.7789, wires=0),
    qml.CRot(0.123, 0.456, 0.789, wires=[0, 1]),
    qml.QubitUnitary(np.eye(2) * 1j, wires=0),
    qml.DiagonalQubitUnitary(np.array([1.0, 1.0j]), wires=1),
    qml.ControlledQubitUnitary(np.eye(2) * 1j, wires=[0], control_wires=[2]),
    qml.SingleExcitation(0.123, wires=[0, 3]),
    qml.SingleExcitationPlus(0.123, wires=[0, 3]),
    qml.SingleExcitationMinus(0.123, wires=[0, 3]),
    qml.DoubleExcitation(0.123, wires=[0, 1, 2, 3]),
    qml.DoubleExcitationPlus(0.123, wires=[0, 1, 2, 3]),
    qml.DoubleExcitationMinus(0.123, wires=[0, 1, 2, 3]),
    qml.PSWAP(0.123, wires=[0, 1]),
    qml.GlobalPhase(0.123, wires=[0]),
    qml.GlobalPhase(0.123),
]

BROADCASTED_OPERATIONS = [
    qml.RX(np.array([0.142, -0.61, 2.3]), wires=0),
    qml.RY(np.array([1.291, -0.10, 5.2]), wires=0),
    qml.RZ(np.array([4.239, -3.21, 1.1]), wires=0),
    qml.PauliRot(np.array([0.142, -0.61, 2.3]), "Y", wires=0),
    qml.IsingXX(np.array([0.142, -0.61, 2.3]), wires=[0, 1]),
    qml.IsingYY(np.array([0.142, -0.61, 2.3]), wires=[0, 1]),
    qml.IsingZZ(np.array([0.142, -0.61, 2.3]), wires=[0, 1]),
    qml.Rot(np.array([0.142, -0.61, 2.3]), 0.456, 0.789, wires=0),
    qml.PhaseShift(np.array([2.12, 0.21, -6.2]), wires=0),
    qml.PCPhase(np.array([1.23, 4.56, -7]), dim=3, wires=[0, 1]),
    qml.ControlledPhaseShift(np.array([1.777, -0.1, 5.29]), wires=[0, 2]),
    qml.CPhase(np.array([1.777, -0.1, 5.29]), wires=[0, 2]),
    qml.CPhaseShift00(np.array([1.777, -0.1, 5.29]), wires=[0, 2]),
    qml.CPhaseShift01(np.array([1.777, -0.1, 5.29]), wires=[0, 2]),
    qml.CPhaseShift10(np.array([1.777, -0.1, 5.29]), wires=[0, 2]),
    qml.MultiRZ(np.array([1.124, -2.31, 0.112]), wires=[1, 2, 3]),
    qml.CRX(np.array([0.836, 0.21, -3.57]), wires=[2, 3]),
    qml.CRY(np.array([0.721, 2.31, 0.983]), wires=[2, 3]),
    qml.CRZ(np.array([0.554, 1.11, 2.2]), wires=[2, 3]),
    qml.U1(np.array([0.142, -0.61, 2.3]), wires=0),
    qml.U2(np.array([9.23, 1.33, 3.556]), np.array([2.134, 1.2, 0.2]), wires=0),
    qml.U3(
        np.array([2.009, 1.33, 3.556]),
        np.array([2.134, 1.2, 0.2]),
        np.array([0.78, 0.48, 0.83]),
        wires=0,
    ),
    qml.CRot(
        np.array([0.142, -0.61, 2.3]),
        np.array([9.82, 0.2, 0.53]),
        np.array([0.12, 2.21, 0.789]),
        wires=[0, 1],
    ),
    qml.QubitUnitary(1j * np.array([[[1, 0], [0, -1]], [[0, 1], [1, 0]]]), wires=0),
    qml.DiagonalQubitUnitary(np.array([[1.0, 1.0j], [1.0j, 1.0j]]), wires=1),
]

NON_PARAMETRIZED_OPERATIONS = [
    qml.Identity(0),
    qml.S(wires=0),
    qml.SX(wires=0),
    qml.T(wires=0),
    qml.CNOT(wires=[0, 1]),
    qml.CZ(wires=[0, 1]),
    qml.CY(wires=[0, 1]),
    qml.SWAP(wires=[0, 1]),
    qml.ISWAP(wires=[0, 1]),
    qml.SISWAP(wires=[0, 1]),
    qml.SQISW(wires=[0, 1]),
    qml.CSWAP(wires=[0, 1, 2]),
    qml.Toffoli(wires=[0, 1, 2]),
    qml.Hadamard(wires=0),
    qml.PauliX(wires=0),
    qml.PauliZ(wires=0),
    qml.PauliY(wires=0),
    qml.MultiControlledX(wires=[0, 1, 2], control_values=[0, 1]),
    qml.QubitSum(wires=[0, 1, 2]),
]

ALL_OPERATIONS = NON_PARAMETRIZED_OPERATIONS + PARAMETRIZED_OPERATIONS


def dot_broadcasted(a, b):
    return np.einsum("...ij,...jk->...ik", a, b)


def multi_dot_broadcasted(matrices):
    return reduce(dot_broadcasted, matrices)


class TestOperations:
    @pytest.mark.parametrize("op", ALL_OPERATIONS + BROADCASTED_OPERATIONS)
    def test_parametrized_op_copy(self, op, tol):
        """Tests that copied parametrized ops function as expected"""
        copied_op = copy.copy(op)
        np.testing.assert_allclose(op.matrix(), copied_op.matrix(), atol=tol)

    # pylint: disable=protected-access
    @pytest.mark.parametrize("op", ALL_OPERATIONS + BROADCASTED_OPERATIONS)
    def test_flatten_unflatten(self, op):
        """Test that the flatten and unflatten methods work as expected."""
        _, metadata = op._flatten()
        assert hash(metadata)

        new_op = type(op)._unflatten(*op._flatten())
        assert qml.equal(op, new_op)

    @pytest.mark.jax
    @pytest.mark.parametrize("op", ALL_OPERATIONS + BROADCASTED_OPERATIONS)
    def test_jax_pytrees(self, op):
        import jax

        leaves = jax.tree_util.tree_leaves(op)
        for d1, d2 in zip(leaves, op.data):
            assert d1 is d2

        leaves, tree_def = jax.tree_util.tree_flatten(op)
        op_unflattened = jax.tree_util.tree_unflatten(tree_def, leaves)
        assert qml.equal(op_unflattened, op)

        new_op = jax.tree_util.tree_map(lambda x: x + 1.0, op)
        for d1, d2 in zip(new_op.data, op.data):
            assert qml.math.allclose(d1, d2 + 1.0)

    @pytest.mark.parametrize("op", PARAMETRIZED_OPERATIONS)
    def test_adjoint_unitaries(self, op, tol):
        op_d = op.adjoint()
        res1 = np.dot(op.matrix(), op_d.matrix())
        res2 = np.dot(op_d.matrix(), op.matrix())
        np.testing.assert_allclose(res1, np.eye(2 ** len(op.wires)), atol=tol)
        np.testing.assert_allclose(res2, np.eye(2 ** len(op.wires)), atol=tol)
        assert op.wires == op_d.wires

    @pytest.mark.parametrize("op", BROADCASTED_OPERATIONS)
    def test_adjoint_unitaries_broadcasted(self, op, tol):
        op_d = op.adjoint()
        res1 = dot_broadcasted(op.matrix(), op_d.matrix())
        res2 = dot_broadcasted(op_d.matrix(), op.matrix())
        I = [np.eye(2 ** len(op.wires))] * op.batch_size
        np.testing.assert_allclose(res1, I, atol=tol)
        np.testing.assert_allclose(res2, I, atol=tol)
        assert op.wires == op_d.wires

    @pytest.mark.parametrize("d", [1, 2, 3, 4])
    @pytest.mark.parametrize("theta", [-np.pi, np.pi / 2, -0.5, 0, 0.5, np.pi / 2, np.pi])
    def test_pcphase_integration(self, theta, d):
        """Test that the PCPhase gate applied to a circuit produces the
        correct final state."""
        wires = [0, 1]
        dev = qml.device("default.qubit", wires=wires)

        @qml.qnode(dev)
        def circuit(phase, dim):
            for wire in wires:  # Construct equal superposition over all states
                qml.Hadamard(wire)

            qml.PCPhase(phase, dim=dim, wires=wires)  # Apply phase to the first dim entries
            return qml.state()

        def _get_expected_state(phase, dim, size):
            return (
                np.array(
                    [np.exp(1j * phase) if i < dim else np.exp(-1j * phase) for i in range(size)]
                )
                * 1
                / 2
            )

        assert np.allclose(circuit(theta, d), _get_expected_state(theta, d, 4))

    def test_pcphase_raises_error(self):
        """Test that the PCPhase operator raises an error when dim is incorrect."""
        phi, dim = (1.23, 3)  # dimension too big
        with pytest.raises(ValueError, match=f"The projected dimension {dim} "):
            _ = qml.PCPhase(phi, dim=dim, wires=0)

        phi, dim = (1.23, 1.5)  # non integer dimension
        with pytest.raises(ValueError, match=f"The projected dimension {dim} "):
            _ = qml.PCPhase(phi, dim=dim, wires=0)


class TestParameterFrequencies:
    @pytest.mark.usefixtures("use_legacy_and_new_opmath")
    @pytest.mark.parametrize("op", PARAMETRIZED_OPERATIONS)
    def test_parameter_frequencies_match_generator(self, op, tol):
        if not qml.operation.has_gen(op):
            pytest.skip(f"Operation {op.name} does not have a generator defined to test against.")

        gen = op.generator()

        try:
            mat = gen.matrix()
        except (AttributeError, qml.operation.MatrixUndefinedError):
            if isinstance(gen, (qml.Hamiltonian, qml.SparseHamiltonian)):
                mat = gen.sparse_matrix().toarray()
            else:
                pytest.skip(f"Operation {op.name}'s generator does not define a matrix.")

        gen_eigvals = np.round(np.linalg.eigvalsh(mat), 8)
        freqs_from_gen = qml.gradients.eigvals_to_frequencies(tuple(gen_eigvals))

        freqs = op.parameter_frequencies
        assert np.allclose(freqs, freqs_from_gen, atol=tol)


class TestDecompositions:
    @pytest.mark.parametrize("phi", [0.3, np.array([0.4, 2.1, 0.2])])
    def test_phase_decomposition(self, phi, tol):
        """Tests that the decomposition of the Phase gate is correct"""
        op = qml.PhaseShift(phi, wires=0)
        res = op.decomposition()

        assert len(res) == 2

        assert res[0].name == "RZ"

        assert res[0].wires == Wires([0])
        assert np.allclose(res[0].data[0], phi)

        decomposed_matrix = res[0].matrix()
        global_phase = np.exp(-1j * phi / 2)[..., np.newaxis, np.newaxis]

        assert res[1].name == "GlobalPhase"
        assert np.allclose(qml.matrix(res[1]), np.exp(1j * phi / 2))

        assert np.allclose(decomposed_matrix, global_phase * op.matrix(), atol=tol, rtol=0)
        if qml.math.shape(phi) == ():  # GlobalPhase matrix doesn't support batching
            assert np.allclose(op.matrix(), qml.prod(*res[::-1]).matrix())

    def test_Rot_decomposition(self):
        """Test the decomposition of Rot."""
        phi = 0.432
        theta = 0.654
        omega = -5.43

        ops1 = qml.Rot.compute_decomposition(phi, theta, omega, wires=0)
        ops2 = qml.Rot(phi, theta, omega, wires=0).decomposition()

        assert len(ops1) == len(ops2) == 3

        classes = [qml.RZ, qml.RY, qml.RZ]
        params = [[phi], [theta], [omega]]

        for ops in [ops1, ops2]:
            for c, p, op in zip(classes, params, ops):
                assert isinstance(op, c)
                assert op.parameters == p

    def test_Rot_decomposition_broadcasted(self):
        """Test the decomposition of broadcasted Rot."""
        phi = np.array([0.1, 2.1])
        theta = np.array([0.4, -0.2])
        omega = np.array([1.1, 0.2])

        ops1 = qml.Rot.compute_decomposition(phi, theta, omega, wires=0)
        ops2 = qml.Rot(phi, theta, omega, wires=0).decomposition()

        assert len(ops1) == len(ops2) == 3

        classes = [qml.RZ, qml.RY, qml.RZ]
        params = [[phi], [theta], [omega]]

        for ops in [ops1, ops2]:
            for c, p, op in zip(classes, params, ops):
                assert isinstance(op, c)
                assert op.parameters == p

    def test_U1_decomposition(self):
        """Test the decomposition for U1."""
        phi = 0.432
        res = qml.U1(phi, wires=0).decomposition()
        res2 = qml.U1.compute_decomposition(phi, wires=0)

        assert len(res) == len(res2) == 1
        assert res[0].name == res2[0].name == "PhaseShift"
        assert res[0].parameters == res2[0].parameters == [phi]

    def test_U1_decomposition_broadcasted(self):
        """Test the decomposition for broadcasted U1."""
        phi = np.array([0.6, 1.2, 9.5])
        res = qml.U1(phi, wires=0).decomposition()
        res2 = qml.U1.compute_decomposition(phi, wires=0)

        assert len(res) == len(res2) == 1
        assert res[0].name == res2[0].name == "PhaseShift"
        assert qml.math.allclose(res[0].parameters[0], phi)
        assert qml.math.allclose(res2[0].parameters[0], phi)

    def test_U2_decomposition(self):
        """Test the decomposition for U2."""
        phi = 0.432
        lam = 0.654

        ops1 = qml.U2.compute_decomposition(phi, lam, wires=0)
        ops2 = qml.U2(phi, lam, wires=0).decomposition()

        classes = [qml.Rot, qml.PhaseShift, qml.PhaseShift]
        params = [[lam, np.ones_like(lam) * np.pi / 2, -lam], [lam], [phi]]

        for ops in [ops1, ops2]:
            for op, c, p in zip(ops, classes, params):
                assert isinstance(op, c)
                assert op.parameters == p

    def test_U2_decomposition_broadcasted(self):
        """Test the decomposition for broadcasted U2."""
        phi = np.array([0.1, 2.1])
        lam = np.array([1.2, 4.9])

        ops1 = qml.U2.compute_decomposition(phi, lam, wires=0)
        ops2 = qml.U2(phi, lam, wires=0).decomposition()

        classes = [qml.Rot, qml.PhaseShift, qml.PhaseShift]
        params = [[lam, np.ones_like(lam) * np.pi / 2, -lam], [lam], [phi]]

        for ops in [ops1, ops2]:
            for op, c, p in zip(ops, classes, params):
                assert isinstance(op, c)
                assert np.allclose(op.parameters, p)

    def test_U3_decomposition(self):
        """Test the decomposition for U3."""
        theta = 0.654
        phi = 0.432
        lam = 0.654

        ops1 = qml.U3.compute_decomposition(theta, phi, lam, wires=0)
        ops2 = qml.U3(theta, phi, lam, wires=0).decomposition()

        classes = [qml.Rot, qml.PhaseShift, qml.PhaseShift]
        params = [[lam, theta, -lam], [lam], [phi]]

        for ops in [ops1, ops2]:
            for op, c, p in zip(ops, classes, params):
                assert isinstance(op, c)
                assert op.parameters == p

    def test_U3_decomposition_broadcasted(self):
        """Test the decomposition for broadcasted U3."""
        theta = np.array([0.1, 2.1])
        phi = np.array([1.2, 4.9])
        lam = np.array([-1.7, 3.2])

        ops1 = qml.U3.compute_decomposition(theta, phi, lam, wires=0)
        ops2 = qml.U3(theta, phi, lam, wires=0).decomposition()

        classes = [qml.Rot, qml.PhaseShift, qml.PhaseShift]
        params = [[lam, theta, -lam], [lam], [phi]]

        for ops in [ops1, ops2]:
            for op, c, p in zip(ops, classes, params):
                assert isinstance(op, c)
                assert np.allclose(op.parameters, p)

    def test_pswap_decomposition(self, tol):
        """Tests that the decomposition of the PSWAP gate is correct"""
        param = 0.1234
        op = qml.PSWAP(param, wires=[0, 1])
        res = op.decomposition()

        assert len(res) == 4

        assert res[0].wires == Wires([0, 1])
        assert res[1].wires == Wires([0, 1])
        assert res[2].wires == Wires([1])
        assert res[3].wires == Wires([0, 1])

        assert res[0].name == "SWAP"
        assert res[1].name == "CNOT"
        assert res[2].name == "PhaseShift"
        assert res[3].name == "CNOT"

        mats = []
        for i in reversed(res):
            if i.wires == Wires([1]):
                # PhaseShift gate
                mats.append(np.kron(np.eye(2), i.matrix()))
            else:
                mats.append(i.matrix())

        decomposed_matrix = np.linalg.multi_dot(mats)

        assert np.allclose(decomposed_matrix, op.matrix(), atol=tol, rtol=0)

    def test_isingxx_decomposition(self, tol):
        """Tests that the decomposition of the IsingXX gate is correct"""
        param = 0.1234
        op = qml.IsingXX(param, wires=[3, 2])
        res = op.decomposition()

        assert len(res) == 3

        assert res[0].wires == Wires([3, 2])
        assert res[1].wires == Wires([3])
        assert res[2].wires == Wires([3, 2])

        assert res[0].name == "CNOT"
        assert res[1].name == "RX"
        assert res[2].name == "CNOT"

        mats = []
        for i in reversed(res):
            if i.wires == Wires([3]):
                # RX gate
                mats.append(np.kron(i.matrix(), np.eye(2)))
            else:
                mats.append(i.matrix())

        decomposed_matrix = np.linalg.multi_dot(mats)

        assert np.allclose(decomposed_matrix, op.matrix(), atol=tol, rtol=0)

    def test_isingxy_decomposition(self, tol):
        """Tests that the decomposition of the IsingXY gate is correct"""
        param = 0.1234
        op = qml.IsingXY(param, wires=[3, 2])
        res = op.decomposition()

        assert len(res) == 6

        assert res[0].wires == Wires([3])
        assert res[1].wires == Wires([3, 2])
        assert res[2].wires == Wires([3])
        assert res[3].wires == Wires([2])
        assert res[4].wires == Wires([3, 2])
        assert res[5].wires == Wires([3])

        assert res[0].name == "Hadamard"
        assert res[1].name == "CY"
        assert res[2].name == "RY"
        assert res[3].name == "RX"
        assert res[4].name == "CY"
        assert res[5].name == "Hadamard"

        mats = []
        for i in reversed(res):
            if i.wires == Wires([3]):
                # RY and Hadamard gate
                mats.append(np.kron(i.matrix(), np.eye(2)))
            elif i.wires == Wires([2]):
                # RX gate
                mats.append(np.kron(np.eye(2), i.matrix()))
            else:
                mats.append(i.matrix())

        decomposed_matrix = np.linalg.multi_dot(mats)

        assert np.allclose(decomposed_matrix, op.matrix(), atol=tol, rtol=0)

    def test_isingxx_decomposition_broadcasted(self, tol):
        """Tests that the decomposition of the broadcasted IsingXX gate is correct"""
        param = np.array([-0.1, 0.2, 0.5])
        op = qml.IsingXX(param, wires=[3, 2])
        res = op.decomposition()

        assert len(res) == 3

        assert res[0].wires == Wires([3, 2])
        assert res[1].wires == Wires([3])
        assert res[2].wires == Wires([3, 2])

        assert res[0].name == "CNOT"
        assert res[1].name == "RX"
        assert res[2].name == "CNOT"

        mats = []
        for i in reversed(res):
            mat = i.matrix()
            if i.wires == Wires([3]):
                # RX gate
                I = np.eye(2)[np.newaxis] if len(mat.shape) == 3 else np.eye(2)
                mats.append(np.kron(mat, I))
            else:
                mats.append(mat)

        decomposed_matrix = multi_dot_broadcasted(mats)

        assert np.allclose(decomposed_matrix, op.matrix(), atol=tol, rtol=0)

    def test_isingyy_decomposition(self, tol):
        """Tests that the decomposition of the IsingYY gate is correct"""
        param = 0.1234
        op = qml.IsingYY(param, wires=[3, 2])
        res = op.decomposition()

        assert len(res) == 3

        assert res[0].wires == Wires([3, 2])
        assert res[1].wires == Wires([3])
        assert res[2].wires == Wires([3, 2])

        assert res[0].name == "CY"
        assert res[1].name == "RY"
        assert res[2].name == "CY"

        mats = []
        for i in reversed(res):
            if i.wires == Wires([3]):
                # RY gate
                mats.append(np.kron(i.matrix(), np.eye(2)))
            else:
                mats.append(i.matrix())

        decomposed_matrix = np.linalg.multi_dot(mats)

        assert np.allclose(decomposed_matrix, op.matrix(), atol=tol, rtol=0)

    def test_isingyy_decomposition_broadcasted(self, tol):
        """Tests that the decomposition of the broadcasted IsingYY gate is correct"""
        param = np.array([-0.1, 0.2, 0.5])
        op = qml.IsingYY(param, wires=[3, 2])
        res = op.decomposition()

        assert len(res) == 3

        assert res[0].wires == Wires([3, 2])
        assert res[1].wires == Wires([3])
        assert res[2].wires == Wires([3, 2])

        assert res[0].name == "CY"
        assert res[1].name == "RY"
        assert res[2].name == "CY"

        mats = []
        for i in reversed(res):
            mat = i.matrix()
            if i.wires == Wires([3]):
                # RY gate
                I = np.eye(2)[np.newaxis] if len(mat.shape) == 3 else np.eye(2)
                mats.append(np.kron(mat, I))
            else:
                mats.append(mat)

        decomposed_matrix = multi_dot_broadcasted(mats)

        assert np.allclose(decomposed_matrix, op.matrix(), atol=tol, rtol=0)

    def test_isingzz_decomposition(self, tol):
        """Tests that the decomposition of the IsingZZ gate is correct"""
        param = 0.1234
        op = qml.IsingZZ(param, wires=[3, 2])
        res = op.decomposition()

        assert len(res) == 3

        assert res[0].wires == Wires([3, 2])
        assert res[1].wires == Wires([2])
        assert res[2].wires == Wires([3, 2])

        assert res[0].name == "CNOT"
        assert res[1].name == "RZ"
        assert res[2].name == "CNOT"

        mats = []
        for i in reversed(res):
            if i.wires == Wires([2]):
                # RZ gate
                mats.append(np.kron(np.eye(2), i.matrix()))
            else:
                mats.append(i.matrix())

        decomposed_matrix = np.linalg.multi_dot(mats)

        assert np.allclose(decomposed_matrix, op.matrix(), atol=tol, rtol=0)

    def test_isingzz_decomposition_broadcasted(self, tol):
        """Tests that the decomposition of the broadcasted IsingZZ gate is correct"""
        param = np.array([-0.1, 0.2, 0.5])
        op = qml.IsingZZ(param, wires=[3, 2])
        res = op.decomposition()

        assert len(res) == 3

        assert res[0].wires == Wires([3, 2])
        assert res[1].wires == Wires([2])
        assert res[2].wires == Wires([3, 2])

        assert res[0].name == "CNOT"
        assert res[1].name == "RZ"
        assert res[2].name == "CNOT"

        mats = []
        for i in reversed(res):
            mat = i.matrix()
            if i.wires == Wires([2]):
                # RX gate
                I = np.eye(2)[np.newaxis] if len(mat.shape) == 3 else np.eye(2)
                mats.append(np.kron(I, mat))
            else:
                mats.append(mat)

        decomposed_matrix = multi_dot_broadcasted(mats)

        assert np.allclose(decomposed_matrix, op.matrix(), atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "op", (qml.PCPhase(1.23, dim=1, wires=[0]), qml.PCPhase(1.23, dim=5, wires=[0, 1, 2]))
    )
    def test_pc_phase_decomposition(self, op):
        """Test that the PCPhase decomposition produces the same unitary"""
        decomp_ops = op.decomposition()
        decomp_op = qml.prod(*decomp_ops) if len(decomp_ops) > 1 else decomp_ops[0]

        expected_mat = qml.matrix(op)
        decomp_mat = qml.matrix(decomp_op)
        assert np.allclose(expected_mat, decomp_mat)

    def test_pc_phase_decomposition_broadcasted(self):
        """Test that the broadcasted PCPhase decomposition produces the same unitary"""
        op = qml.PCPhase([1.23, 4.56, 7.89], dim=5, wires=[0, 1, 2])
        decomp_ops = op.decomposition()
        decomp_op = qml.prod(*decomp_ops) if len(decomp_ops) >= 1 else decomp_ops[0]

        expected_mats = qml.matrix(op)
        decomp_mats = qml.matrix(decomp_op)

        for expected_mat, decomp_mat in zip(expected_mats, decomp_mats):
            assert np.allclose(expected_mat, decomp_mat)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, 0.5])
    @pytest.mark.parametrize(
        "cphase_op,lam_pos",
        [
            (qml.CPhaseShift00, 0),
            (qml.CPhaseShift01, 1),
            (qml.CPhaseShift10, 2),
        ],
    )
    def test_c_phase_shift_decomp(self, phi, cphase_op, lam_pos):
        """Tests that the CPhaseShift operations
        calculate the correct decomposition"""
        op = cphase_op(phi, wires=[0, 2])
        decomposed_matrix = qml.matrix(op.decomposition, wire_order=op.wires)()
        lam = np.exp(1j * phi)
        exp = np.eye(4, dtype=complex)
        exp[..., lam_pos, lam_pos] = lam

        assert np.allclose(decomposed_matrix, exp)


class TestMatrix:
    def test_phase_shift(self, tol):
        """Test phase shift is correct"""

        # test identity for theta=0
        assert np.allclose(qml.PhaseShift.compute_matrix(0), np.identity(2), atol=tol, rtol=0)
        assert np.allclose(qml.PhaseShift(0, wires=0).matrix(), np.identity(2), atol=tol, rtol=0)
        assert np.allclose(qml.U1.compute_matrix(0), np.identity(2), atol=tol, rtol=0)

        # test arbitrary phase shift
        phi = 0.5432
        expected = np.array([[1, 0], [0, np.exp(1j * phi)]])
        assert np.allclose(qml.PhaseShift.compute_matrix(phi), expected, atol=tol, rtol=0)
        assert np.allclose(qml.U1.compute_matrix(phi), expected, atol=tol, rtol=0)

        # test arbitrary broadcasted phase shift
        phi = np.array([0.5, 0.4, 0.3])
        expected = np.array([[[1, 0], [0, np.exp(1j * p)]] for p in phi])
        assert np.allclose(qml.PhaseShift.compute_matrix(phi), expected, atol=tol, rtol=0)
        assert np.allclose(qml.U1.compute_matrix(phi), expected, atol=tol, rtol=0)

    def test_global_phase(self, tol):
        """Test GlobalPhase matrix is correct"""

        # test identity for theta=0
        assert np.allclose(qml.GlobalPhase.compute_matrix(0), np.identity(2), atol=tol, rtol=0)
        assert np.allclose(
            qml.GlobalPhase(0).matrix(wire_order=[0]), np.identity(2), atol=tol, rtol=0
        )

        # test arbitrary phase shift
        phi = 0.5432
        expected = np.array([[qml.math.exp(-1j * phi), 0], [0, qml.math.exp(-1j * phi)]])
        assert np.allclose(qml.GlobalPhase.compute_matrix(phi), expected, atol=tol, rtol=0)
        assert np.allclose(qml.GlobalPhase(phi).matrix(wire_order=[0]), expected, atol=tol, rtol=0)

    def test_identity(self, tol):
        """Test Identity matrix is correct with no wires"""

        # test Identity().compute_matrix()
        assert np.allclose(qml.Identity().compute_matrix(1), np.identity(2), atol=tol, rtol=0)
        assert np.allclose(qml.Identity().compute_matrix(2), np.identity(4), atol=tol, rtol=0)

        # test Identity().matrix()
        assert np.allclose(qml.Identity().matrix(), np.identity(1), atol=tol, rtol=0)
        assert np.allclose(qml.Identity().matrix(wire_order=[0]), np.identity(2), atol=tol, rtol=0)
        assert np.allclose(
            qml.Identity().matrix(wire_order=[0, "a"]), np.identity(4), atol=tol, rtol=0
        )

    def test_rx(self, tol):
        """Test x rotation is correct"""

        # test identity for theta=0
        assert np.allclose(qml.RX.compute_matrix(0), np.identity(2), atol=tol, rtol=0)
        assert np.allclose(qml.RX(0, wires=0).matrix(), np.identity(2), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected = np.array([[1, -1j], [-1j, 1]]) / np.sqrt(2)
        assert np.allclose(qml.RX.compute_matrix(np.pi / 2), expected, atol=tol, rtol=0)
        assert np.allclose(qml.RX(np.pi / 2, wires=0).matrix(), expected, atol=tol, rtol=0)

        # test identity for broadcasted theta=pi/2
        expected = np.tensordot([1, 1], np.array([[1, -1j], [-1j, 1]]) / np.sqrt(2), axes=0)
        pi_half = np.array([np.pi / 2, np.pi / 2])
        assert np.allclose(qml.RX.compute_matrix(pi_half), expected, atol=tol, rtol=0)
        assert np.allclose(qml.RX(pi_half, wires=0).matrix(), expected, atol=tol, rtol=0)

        # test identity for theta=pi
        expected = -1j * np.array([[0, 1], [1, 0]])
        assert np.allclose(qml.RX.compute_matrix(np.pi), expected, atol=tol, rtol=0)
        assert np.allclose(qml.RX(np.pi, wires=0).matrix(), expected, atol=tol, rtol=0)

    def test_ry(self, tol):
        """Test y rotation is correct"""

        # test identity for theta=0
        assert np.allclose(qml.RY.compute_matrix(0), np.identity(2), atol=tol, rtol=0)
        assert np.allclose(qml.RY(0, wires=0).matrix(), np.identity(2), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
        assert np.allclose(qml.RY.compute_matrix(np.pi / 2), expected, atol=tol, rtol=0)
        assert np.allclose(qml.RY(np.pi / 2, wires=0).matrix(), expected, atol=tol, rtol=0)

        # test identity for broadcasted theta=pi/2
        expected = np.tensordot([1, 1], np.array([[1, -1], [1, 1]]) / np.sqrt(2), axes=0)
        pi_half = np.array([np.pi / 2, np.pi / 2])
        assert np.allclose(qml.RY.compute_matrix(pi_half), expected, atol=tol, rtol=0)
        assert np.allclose(qml.RY(pi_half, wires=0).matrix(), expected, atol=tol, rtol=0)

        # test identity for theta=pi
        expected = np.array([[0, -1], [1, 0]])
        assert np.allclose(qml.RY.compute_matrix(np.pi), expected, atol=tol, rtol=0)
        assert np.allclose(qml.RY(np.pi, wires=0).matrix(), expected, atol=tol, rtol=0)

    def test_rz(self, tol):
        """Test z rotation is correct"""

        # test identity for theta=0
        assert np.allclose(qml.RZ.compute_matrix(0), np.identity(2), atol=tol, rtol=0)
        assert np.allclose(qml.RZ(0, wires=0).matrix(), np.identity(2), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected = np.diag(np.exp([-1j * np.pi / 4, 1j * np.pi / 4]))
        assert np.allclose(qml.RZ.compute_matrix(np.pi / 2), expected, atol=tol, rtol=0)
        assert np.allclose(qml.RZ(np.pi / 2, wires=0).matrix(), expected, atol=tol, rtol=0)

        # test identity for broadcasted theta=pi/2
        expected = np.tensordot([1, 1], np.diag(np.exp([-1j * np.pi / 4, 1j * np.pi / 4])), axes=0)
        pi_half = np.array([np.pi / 2, np.pi / 2])
        assert np.allclose(qml.RZ.compute_matrix(pi_half), expected, atol=tol, rtol=0)
        assert np.allclose(qml.RZ(pi_half, wires=0).matrix(), expected, atol=tol, rtol=0)

        # test identity for theta=pi
        assert np.allclose(qml.RZ.compute_matrix(np.pi), -1j * Z, atol=tol, rtol=0)
        assert np.allclose(qml.RZ(np.pi, wires=0).matrix(), -1j * Z, atol=tol, rtol=0)

    @pytest.mark.parametrize("dim", range(3))
    @pytest.mark.parametrize("wires", (range(2), range(3)))
    @pytest.mark.parametrize("phi", np.linspace(-np.pi, np.pi, 10))
    def test_pcphase(self, phi, dim, wires):
        """Test that the PCPhase operator matrix is correct."""
        num_wires = len(wires)
        op = qml.PCPhase(phi, dim=dim, wires=wires)

        mat1 = qml.matrix(op)
        mat2 = op.compute_matrix(*op.parameters, **op.hyperparameters)

        expected_mat = np.diag(
            [np.exp(1j * phi) if i < dim else np.exp(-1j * phi) for i in range(2**num_wires)]
        )
        assert np.allclose(mat1, expected_mat)
        assert np.allclose(mat2, expected_mat)
        assert qml.math.get_interface(mat1) == "numpy"

    @pytest.mark.tf
    @pytest.mark.parametrize("dim", range(3))
    @pytest.mark.parametrize("wires", (range(2), range(3)))
    @pytest.mark.parametrize("phi", np.linspace(-np.pi, np.pi, 10))
    def test_pcphase_tf(self, phi, dim, wires):
        """Test that the PCPhase operator matrix is correct for tf."""
        import tensorflow as tf

        num_wires = len(wires)
        op = qml.PCPhase(tf.Variable(phi), dim=dim, wires=wires)

        mat1 = qml.matrix(op)
        mat2 = op.compute_matrix(*op.parameters, **op.hyperparameters)

        expected_mat = tf.Variable(
            np.diag(
                [np.exp(1j * phi) if i < dim else np.exp(-1j * phi) for i in range(2**num_wires)]
            )
        )

        assert np.allclose(mat1, expected_mat)
        assert np.allclose(mat2, expected_mat)
        assert qml.math.get_interface(mat1) == "tensorflow"

    @pytest.mark.torch
    @pytest.mark.parametrize("dim", range(3))
    @pytest.mark.parametrize("wires", (range(2), range(3)))
    @pytest.mark.parametrize("phi", np.linspace(-np.pi, np.pi, 10))
    def test_pcphase_torch(self, phi, dim, wires):
        import torch

        num_wires = len(wires)
        op = qml.PCPhase(torch.tensor(phi), dim=dim, wires=wires)

        mat1 = qml.matrix(op)
        mat2 = op.compute_matrix(*op.parameters, **op.hyperparameters)

        expected_mat = torch.tensor(
            np.diag(
                [np.exp(1j * phi) if i < dim else np.exp(-1j * phi) for i in range(2**num_wires)]
            )
        )

        assert np.allclose(mat1, expected_mat)
        assert np.allclose(mat2, expected_mat)
        assert qml.math.get_interface(mat1) == "torch"

    @pytest.mark.jax
    @pytest.mark.parametrize("dim", range(3))
    @pytest.mark.parametrize("wires", (range(2), range(3)))
    @pytest.mark.parametrize("phi", np.linspace(-np.pi, np.pi, 10))
    def test_pcphase_jax(self, phi, dim, wires):
        import jax.numpy as jnp

        phi = jnp.array(phi)

        num_wires = len(wires)
        op = qml.PCPhase(phi, dim=dim, wires=wires)

        mat1 = qml.matrix(op)
        mat2 = op.compute_matrix(*op.parameters, **op.hyperparameters)

        expected_mat = jnp.diag(
            jnp.array(
                [jnp.exp(1j * phi) if i < dim else np.exp(-1j * phi) for i in range(2**num_wires)]
            )
        )

        assert np.allclose(mat1, expected_mat)
        assert np.allclose(mat2, expected_mat)
        assert qml.math.get_interface(mat1) == "jax"

    def test_pcphase_broadcasted(self):
        """Test that the PCPhase matrix works with broadcasted parameters"""
        dim = 2
        size = 4
        broadcasted_phi = [1.23, 4.56, -0.7]

        op = qml.PCPhase(broadcasted_phi, dim=dim, wires=[0, 1])

        mat1 = qml.matrix(op)
        mat2 = op.compute_matrix(*op.parameters, **op.hyperparameters)

        mats = [
            np.diag([np.exp(1j * phi) if i < dim else np.exp(-1j * phi) for i in range(size)])
            for phi in broadcasted_phi
        ]
        expected_mat = np.array(mats)
        assert np.allclose(mat1, expected_mat)
        assert np.allclose(mat2, expected_mat)

    def test_isingxx(self, tol):
        """Test that the IsingXX operation is correct"""
        assert np.allclose(qml.IsingXX.compute_matrix(0), np.identity(4), atol=tol, rtol=0)
        assert np.allclose(qml.IsingXX(0, wires=[0, 1]).matrix(), np.identity(4), atol=tol, rtol=0)

        def get_expected(theta):
            expected = np.array(np.diag([np.cos(theta / 2)] * 4), dtype=np.complex128)
            sin_coeff = -1j * np.sin(theta / 2)
            expected[3, 0] = sin_coeff
            expected[2, 1] = sin_coeff
            expected[1, 2] = sin_coeff
            expected[0, 3] = sin_coeff
            return expected

        param = np.pi / 2
        assert np.allclose(qml.IsingXX.compute_matrix(param), get_expected(param), atol=tol, rtol=0)
        assert np.allclose(
            qml.IsingXX(param, wires=[0, 1]).matrix(), get_expected(param), atol=tol, rtol=0
        )

        param = np.pi
        assert np.allclose(qml.IsingXX.compute_matrix(param), get_expected(param), atol=tol, rtol=0)
        assert np.allclose(
            qml.IsingXX(param, wires=[0, 1]).matrix(), get_expected(param), atol=tol, rtol=0
        )

    def test_pswap(self, tol):
        """Test that the PSWAP operation is correct"""
        assert np.allclose(
            qml.PSWAP.compute_matrix(0), np.diag([1, 1, 1, 1])[[0, 2, 1, 3]], atol=tol, rtol=0
        )
        assert np.allclose(
            qml.PSWAP(0, wires=[0, 1]).matrix(),
            np.diag([1, 1, 1, 1])[[0, 2, 1, 3]],
            atol=tol,
            rtol=0,
        )

        def get_expected(theta):
            return np.diag([1, np.exp(1j * theta), np.exp(1j * theta), 1])[[0, 2, 1, 3]]

        param = np.pi / 2
        assert np.allclose(qml.PSWAP.compute_matrix(param), get_expected(param), atol=tol, rtol=0)
        assert np.allclose(
            qml.PSWAP(param, wires=[0, 1]).matrix(), get_expected(param), atol=tol, rtol=0
        )

        param = np.pi
        assert np.allclose(qml.PSWAP.compute_matrix(param), get_expected(param), atol=tol, rtol=0)
        assert np.allclose(
            qml.PSWAP(param, wires=[0, 1]).matrix(), get_expected(param), atol=tol, rtol=0
        )

    @pytest.mark.parametrize("phi", np.linspace(-np.pi, np.pi, 10))
    def test_pswap_eigvals(self, phi):
        """Test eigenvalues computation for PSWAP"""
        evs = qml.PSWAP.compute_eigvals(phi)
        evs_expected = [1, 1, -qml.math.exp(1j * phi), qml.math.exp(1j * phi)]
        assert qml.math.allclose(evs, evs_expected)

    @pytest.mark.tf
    @pytest.mark.parametrize("phi", np.linspace(-np.pi, np.pi, 10))
    def test_pswap_eigvals_tf(self, phi):
        """Test eigenvalues computation for PSWAP using Tensorflow interface"""
        import tensorflow as tf

        param_tf = tf.Variable(phi)
        evs = qml.PSWAP.compute_eigvals(param_tf)
        evs_expected = [1, 1, -qml.math.exp(1j * phi), qml.math.exp(1j * phi)]
        assert qml.math.allclose(evs, evs_expected)

    @pytest.mark.torch
    @pytest.mark.parametrize("phi", np.linspace(-np.pi, np.pi, 10))
    def test_pswap_eigvals_torch(self, phi):
        """Test eigenvalues computation for PSWAP using Torch interface"""
        import torch

        param_torch = torch.tensor(phi)
        evs = qml.PSWAP.compute_eigvals(param_torch)
        evs_expected = [1, 1, -qml.math.exp(1j * phi), qml.math.exp(1j * phi)]
        assert qml.math.allclose(evs, evs_expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("phi", np.linspace(-np.pi, np.pi, 10))
    def test_pswap_eigvals_jax(self, phi):
        """Test eigenvalues computation for PSWAP using JAX interface"""
        import jax

        param_jax = jax.numpy.array(phi)
        evs = qml.PSWAP.compute_eigvals(param_jax)
        evs_expected = [1, 1, -qml.math.exp(1j * phi), qml.math.exp(1j * phi)]
        assert qml.math.allclose(evs, evs_expected)

    @pytest.mark.tf
    @pytest.mark.parametrize("phi", np.linspace(-np.pi, np.pi, 10))
    def test_pcphase_eigvals_tf(self, phi):
        """Test eigenvalues computation for PCPhase using Tensorflow interface"""
        import tensorflow as tf

        param_tf = tf.Variable(phi)

        op = qml.PCPhase(param_tf, dim=2, wires=[0, 1])
        evs = qml.PCPhase.compute_eigvals(*op.parameters, **op.hyperparameters)
        evs_expected = np.array(
            [np.exp(1j * phi), np.exp(1j * phi), np.exp(-1j * phi), np.exp(-1j * phi)]
        )

        assert qml.math.allclose(evs, evs_expected)

    def test_isingxy(self, tol):
        """Test that the IsingXY operation is correct"""
        assert np.allclose(qml.IsingXY.compute_matrix(0), np.identity(4), atol=tol, rtol=0)
        assert np.allclose(qml.IsingXY(0, wires=[0, 1]).matrix(), np.identity(4), atol=tol, rtol=0)

        def get_expected(theta):
            expected = np.eye(4, dtype=np.complex128)
            expected[1][1] = np.cos(theta / 2)
            expected[2][2] = np.cos(theta / 2)
            expected[1][2] = 1j * np.sin(theta / 2)
            expected[2][1] = 1j * np.sin(theta / 2)
            return expected

        param = np.pi / 2
        assert np.allclose(qml.IsingXY.compute_matrix(param), get_expected(param), atol=tol, rtol=0)
        assert np.allclose(
            qml.IsingXY(param, wires=[0, 1]).matrix(), get_expected(param), atol=tol, rtol=0
        )

        param = np.pi
        assert np.allclose(qml.IsingXY.compute_matrix(param), get_expected(param), atol=tol, rtol=0)
        assert np.allclose(
            qml.IsingXY(param, wires=[0, 1]).matrix(), get_expected(param), atol=tol, rtol=0
        )

    def test_isingxy_broadcasted(self, tol):
        """Test that the broadcasted IsingXY operation is correct"""
        z = np.zeros(3)
        assert np.allclose(qml.IsingXY.compute_matrix(z), np.identity(4), atol=tol, rtol=0)
        assert np.allclose(qml.IsingXY(z, wires=[0, 1]).matrix(), np.identity(4), atol=tol, rtol=0)

        def get_expected(theta):
            expected = np.array([np.eye(4) for i in theta], dtype=complex)
            expected[:, 1, 1] = np.cos(theta / 2)
            expected[:, 2, 2] = np.cos(theta / 2)
            expected[:, 1, 2] = 1j * np.sin(theta / 2)
            expected[:, 2, 1] = 1j * np.sin(theta / 2)
            return expected

        param = np.array([np.pi / 2, np.pi])
        assert np.allclose(qml.IsingXY.compute_matrix(param), get_expected(param), atol=tol, rtol=0)
        assert np.allclose(
            qml.IsingXY(param, wires=[0, 1]).matrix(), get_expected(param), atol=tol, rtol=0
        )

        param = np.array([2.152, np.pi / 2, 0.213])
        assert np.allclose(qml.IsingXY.compute_matrix(param), get_expected(param), atol=tol, rtol=0)
        assert np.allclose(
            qml.IsingXY(param, wires=[0, 1]).matrix(), get_expected(param), atol=tol, rtol=0
        )

    @pytest.mark.parametrize("phi", np.linspace(-np.pi, np.pi, 10))
    def test_isingxy_eigvals(self, phi):
        """Test eigenvalues computation for IsingXY"""
        evs = qml.IsingXY.compute_eigvals(phi)
        evs_expected = [
            qml.math.cos(phi / 2) + 1j * qml.math.sin(phi / 2),
            qml.math.cos(phi / 2) - 1j * qml.math.sin(phi / 2),
            1,
            1,
        ]
        assert qml.math.allclose(evs, evs_expected)

    def test_isingxy_eigvals_broadcasted(self):
        """Test broadcasted eigenvalues computation for IsingXY"""
        phi = np.linspace(-np.pi, np.pi, 10)
        evs = qml.IsingXY.compute_eigvals(phi)
        evs_expected = np.array(
            [[qml.math.exp(1j * _phi / 2), qml.math.exp(-1j * _phi / 2), 1, 1] for _phi in phi]
        )
        assert qml.math.allclose(evs, evs_expected)

    @pytest.mark.tf
    @pytest.mark.parametrize("phi", np.linspace(-np.pi, np.pi, 10))
    def test_isingxy_eigvals_tf(self, phi):
        """Test eigenvalues computation for IsingXY using Tensorflow interface"""
        import tensorflow as tf

        param_tf = tf.Variable(phi)
        evs = qml.IsingXY.compute_eigvals(param_tf)
        evs_expected = [
            qml.math.cos(phi / 2) + 1j * qml.math.sin(phi / 2),
            qml.math.cos(phi / 2) - 1j * qml.math.sin(phi / 2),
            1,
            1,
        ]
        assert qml.math.allclose(evs, evs_expected)

    @pytest.mark.tf
    def test_isingxy_eigvals_tf_broadcasted(self):
        """Test broadcasted eigenvalues computation for IsingXY on TF"""
        import tensorflow as tf

        phi = np.linspace(-np.pi, np.pi, 10)
        evs = qml.IsingXY.compute_eigvals(tf.Variable(phi))
        c = np.cos(phi / 2)
        s = np.sin(phi / 2)
        ones = np.ones_like(c)
        expected = np.stack([c + 1j * s, c - 1j * s, ones, ones], axis=-1)
        assert qml.math.allclose(evs, expected)

    @pytest.mark.torch
    @pytest.mark.parametrize("phi", np.linspace(-np.pi, np.pi, 10))
    def test_isingxy_eigvals_torch(self, phi):
        """Test eigenvalues computation for IsingXY using Torch interface"""
        import torch

        param_torch = torch.tensor(phi)
        evs = qml.IsingXY.compute_eigvals(param_torch)
        evs_expected = [
            qml.math.cos(phi / 2) + 1j * qml.math.sin(phi / 2),
            qml.math.cos(phi / 2) - 1j * qml.math.sin(phi / 2),
            1,
            1,
        ]
        assert qml.math.allclose(evs, evs_expected)

    @pytest.mark.torch
    def test_isingxy_eigvals_torch_broadcasted(self):
        """Test broadcasted eigenvalues computation for IsingXY with torch"""
        import torch

        phi = np.linspace(-np.pi, np.pi, 10)
        evs = qml.IsingXY.compute_eigvals(torch.tensor(phi, requires_grad=True))
        c = np.cos(phi / 2)
        s = np.sin(phi / 2)
        ones = np.ones_like(c)
        expected = np.stack([c + 1j * s, c - 1j * s, ones, ones], axis=-1)
        assert qml.math.allclose(evs.detach().numpy(), expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("phi", np.linspace(-np.pi, np.pi, 10))
    def test_isingxy_eigvals_jax(self, phi):
        """Test eigenvalues computation for IsingXY using JAX interface"""
        import jax

        param_jax = jax.numpy.array(phi)
        evs = qml.IsingXY.compute_eigvals(param_jax)
        evs_expected = [
            qml.math.cos(phi / 2) + 1j * qml.math.sin(phi / 2),
            qml.math.cos(phi / 2) - 1j * qml.math.sin(phi / 2),
            1,
            1,
        ]
        assert qml.math.allclose(evs, evs_expected)

    @pytest.mark.jax
    def test_isingxy_eigvals_jax_broadcasted(self):
        """Test broadcasted eigenvalues computation for IsingXY with jax"""
        import jax

        phi = np.linspace(-np.pi, np.pi, 10)
        evs = qml.IsingXY.compute_eigvals(jax.numpy.array(phi))
        c = np.cos(phi / 2)
        s = np.sin(phi / 2)
        ones = np.ones_like(c)
        expected = np.stack([c + 1j * s, c - 1j * s, ones, ones], axis=-1)
        assert qml.math.allclose(evs, expected)

    def test_isingxx_broadcasted(self, tol):
        """Test that the broadcasted IsingXX operation is correct"""
        z = np.zeros(3)
        assert np.allclose(qml.IsingXX.compute_matrix(z), np.identity(4), atol=tol, rtol=0)
        assert np.allclose(qml.IsingXX(z, wires=[0, 1]).matrix(), np.identity(4), atol=tol, rtol=0)

        def get_expected(theta):
            expected = np.array([np.diag([np.cos(t / 2)] * 4) for t in theta], dtype=np.complex128)
            sin_coeff = -1j * np.sin(theta / 2)
            expected[:, 3, 0] = sin_coeff
            expected[:, 2, 1] = sin_coeff
            expected[:, 1, 2] = sin_coeff
            expected[:, 0, 3] = sin_coeff
            return expected

        param = np.array([np.pi / 2, np.pi])
        assert np.allclose(qml.IsingXX.compute_matrix(param), get_expected(param), atol=tol, rtol=0)
        assert np.allclose(
            qml.IsingXX(param, wires=[0, 1]).matrix(), get_expected(param), atol=tol, rtol=0
        )

        param = np.array([2.152, np.pi / 2, 0.213])
        assert np.allclose(qml.IsingXX.compute_matrix(param), get_expected(param), atol=tol, rtol=0)
        assert np.allclose(
            qml.IsingXX(param, wires=[0, 1]).matrix(), get_expected(param), atol=tol, rtol=0
        )

    def test_isingyy(self, tol):
        """Test that the IsingYY operation is correct"""
        assert np.allclose(qml.IsingYY.compute_matrix(0), np.identity(4), atol=tol, rtol=0)
        assert np.allclose(qml.IsingYY(0, wires=[0, 1]).matrix(), np.identity(4), atol=tol, rtol=0)

        def get_expected(theta):
            expected = np.array(np.diag([np.cos(theta / 2)] * 4), dtype=np.complex128)
            sin_coeff = 1j * np.sin(theta / 2)
            expected[3, 0] = sin_coeff
            expected[2, 1] = -sin_coeff
            expected[1, 2] = -sin_coeff
            expected[0, 3] = sin_coeff
            return expected

        param = np.pi / 2
        assert np.allclose(qml.IsingYY.compute_matrix(param), get_expected(param), atol=tol, rtol=0)
        assert np.allclose(
            qml.IsingYY(param, wires=[0, 1]).matrix(), get_expected(param), atol=tol, rtol=0
        )

        param = np.pi
        assert np.allclose(qml.IsingYY.compute_matrix(param), get_expected(param), atol=tol, rtol=0)
        assert np.allclose(
            qml.IsingYY(param, wires=[0, 1]).matrix(), get_expected(param), atol=tol, rtol=0
        )

    def test_isingyy_broadcasted(self, tol):
        """Test that the broadcasted IsingYY operation is correct"""
        z = np.zeros(3)
        assert np.allclose(qml.IsingYY.compute_matrix(z), np.identity(4), atol=tol, rtol=0)
        assert np.allclose(qml.IsingYY(z, wires=[0, 1]).matrix(), np.identity(4), atol=tol, rtol=0)

        def get_expected(theta):
            expected = np.array([np.diag([np.cos(t / 2)] * 4) for t in theta], dtype=np.complex128)
            sin_coeff = 1j * np.sin(theta / 2)
            expected[:, 3, 0] = sin_coeff
            expected[:, 2, 1] = -sin_coeff
            expected[:, 1, 2] = -sin_coeff
            expected[:, 0, 3] = sin_coeff
            return expected

        param = np.array([np.pi / 2, np.pi])
        assert np.allclose(qml.IsingYY.compute_matrix(param), get_expected(param), atol=tol, rtol=0)
        assert np.allclose(
            qml.IsingYY(param, wires=[0, 1]).matrix(), get_expected(param), atol=tol, rtol=0
        )

        param = np.array([2.152, np.pi / 2, 0.213])
        assert np.allclose(qml.IsingYY.compute_matrix(param), get_expected(param), atol=tol, rtol=0)
        assert np.allclose(
            qml.IsingYY(param, wires=[0, 1]).matrix(), get_expected(param), atol=tol, rtol=0
        )

    def test_isingzz(self, tol):
        """Test that the IsingZZ operation is correct"""
        assert np.allclose(qml.IsingZZ.compute_matrix(0), np.identity(4), atol=tol, rtol=0)
        assert np.allclose(qml.IsingZZ(0, wires=[0, 1]).matrix(), np.identity(4), atol=tol, rtol=0)
        assert np.allclose(
            qml.IsingZZ.compute_eigvals(0), np.diagonal(np.identity(4)), atol=tol, rtol=0
        )

        def get_expected(theta):
            neg_imag = np.exp(-1j * theta / 2)
            plus_imag = np.exp(1j * theta / 2)
            expected = np.array(
                np.diag([neg_imag, plus_imag, plus_imag, neg_imag]), dtype=np.complex128
            )
            return expected

        param = np.pi / 2
        assert np.allclose(qml.IsingZZ.compute_matrix(param), get_expected(param), atol=tol, rtol=0)
        assert np.allclose(
            qml.IsingZZ(param, wires=[0, 1]).matrix(), get_expected(param), atol=tol, rtol=0
        )
        assert np.allclose(
            qml.IsingZZ.compute_eigvals(param), np.diagonal(get_expected(param)), atol=tol, rtol=0
        )

        param = np.pi
        assert np.allclose(qml.IsingZZ.compute_matrix(param), get_expected(param), atol=tol, rtol=0)
        assert np.allclose(
            qml.IsingZZ(param, wires=[0, 1]).matrix(), get_expected(param), atol=tol, rtol=0
        )
        assert np.allclose(
            qml.IsingZZ.compute_eigvals(param), np.diagonal(get_expected(param)), atol=tol, rtol=0
        )

    @pytest.mark.tf
    @pytest.mark.parametrize("phi", np.linspace(-np.pi, np.pi, 10))
    def test_isingzz_eigvals_tf(self, phi):
        """Test eigenvalues computation for IsingXY using Tensorflow interface"""
        import tensorflow as tf

        param_tf = tf.Variable(phi)
        evs = qml.IsingZZ.compute_eigvals(param_tf)

        def get_expected(theta):
            neg_imag = np.exp(-1j * theta / 2)
            plus_imag = np.exp(1j * theta / 2)
            expected = np.array(
                np.diag([neg_imag, plus_imag, plus_imag, neg_imag]), dtype=np.complex128
            )
            return expected

        assert qml.math.allclose(evs, np.diagonal(get_expected(phi)))

    def test_isingzz_broadcasted(self, tol):
        """Test that the broadcasted IsingZZ operation is correct"""
        z = np.zeros(3)
        assert np.allclose(qml.IsingZZ.compute_matrix(z), np.identity(4), atol=tol, rtol=0)
        assert np.allclose(qml.IsingZZ(z, wires=[0, 1]).matrix(), np.identity(4), atol=tol, rtol=0)
        assert np.allclose(
            qml.IsingZZ.compute_eigvals(z), np.diagonal(np.identity(4)), atol=tol, rtol=0
        )

        def get_expected(theta):
            neg_imag = np.exp(-1j * theta / 2)
            plus_imag = np.exp(1j * theta / 2)
            expected = np.array([np.diag([n, p, p, n]) for n, p in zip(neg_imag, plus_imag)])
            return expected

        param = np.array([np.pi / 2, np.pi])
        assert np.allclose(qml.IsingZZ.compute_matrix(param), get_expected(param), atol=tol, rtol=0)
        assert np.allclose(
            qml.IsingZZ(param, wires=[0, 1]).matrix(), get_expected(param), atol=tol, rtol=0
        )
        expected_eigvals = np.array([np.diag(m) for m in get_expected(param)])
        assert np.allclose(qml.IsingZZ.compute_eigvals(param), expected_eigvals, atol=tol, rtol=0)

        param = np.array([0.5, 1.2, np.pi / 8])
        assert np.allclose(qml.IsingZZ.compute_matrix(param), get_expected(param), atol=tol, rtol=0)
        assert np.allclose(
            qml.IsingZZ(param, wires=[0, 1]).matrix(), get_expected(param), atol=tol, rtol=0
        )
        expected_eigvals = np.array([np.diag(m) for m in get_expected(param)])
        assert np.allclose(qml.IsingZZ.compute_eigvals(param), expected_eigvals, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_isingzz_matrix_tf(self, tol):
        """Tests the matrix representation for IsingZZ for tensorflow, since the method contains
        different logic for this framework"""
        import tensorflow as tf

        def get_expected(theta):
            neg_imag = np.exp(-1j * theta / 2)
            plus_imag = np.exp(1j * theta / 2)
            expected = np.array(
                np.diag([neg_imag, plus_imag, plus_imag, neg_imag]), dtype=np.complex128
            )
            return expected

        param = tf.Variable(np.pi)
        assert np.allclose(qml.IsingZZ.compute_matrix(param), get_expected(np.pi), atol=tol, rtol=0)

    @pytest.mark.tf
    def test_isingzz_matrix_tf_broadcasted(self, tol):
        """Tests the matrix representation for broadcasted IsingZZ for tensorflow,
        since the method contains different logic for this framework"""
        import tensorflow as tf

        def get_expected(theta):
            neg_imag = np.exp(-1j * theta / 2)
            plus_imag = np.exp(1j * theta / 2)
            expected = np.array([np.diag([n, p, p, n]) for n, p in zip(neg_imag, plus_imag)])
            return expected

        param = np.array([np.pi, 0.1242])
        param_tf = tf.Variable(param)
        assert np.allclose(
            qml.IsingZZ.compute_matrix(param_tf), get_expected(param), atol=tol, rtol=0
        )

    def test_Rot(self, tol):
        """Test arbitrary single qubit rotation is correct"""

        # test identity for phi,theta,omega=0
        assert np.allclose(qml.Rot.compute_matrix(0, 0, 0), np.identity(2), atol=tol, rtol=0)
        assert np.allclose(qml.Rot(0, 0, 0, wires=0).matrix(), np.identity(2), atol=tol, rtol=0)

        # expected result
        def arbitrary_rotation(x, y, z):
            """arbitrary single qubit rotation"""
            c = np.cos(y / 2)
            s = np.sin(y / 2)
            return np.array(
                [
                    [np.exp(-0.5j * (x + z)) * c, -np.exp(0.5j * (x - z)) * s],
                    [np.exp(-0.5j * (x - z)) * s, np.exp(0.5j * (x + z)) * c],
                ]
            )

        a, b, c = 0.432, -0.152, 0.9234
        assert np.allclose(
            qml.Rot.compute_matrix(a, b, c), arbitrary_rotation(a, b, c), atol=tol, rtol=0
        )
        assert np.allclose(
            qml.Rot(a, b, c, wires=0).matrix(), arbitrary_rotation(a, b, c), atol=tol, rtol=0
        )

    def test_Rot_broadcasted(self, tol):
        """Test broadcasted arbitrary single qubit rotation is correct"""

        # test identity for phi,theta,omega=0
        z = np.zeros(5)
        assert np.allclose(qml.Rot.compute_matrix(z, z, z), np.identity(2), atol=tol, rtol=0)
        assert np.allclose(qml.Rot(z, z, z, wires=0).matrix(), np.identity(2), atol=tol, rtol=0)

        # expected result
        def arbitrary_rotation(x, y, z):
            """arbitrary single qubit rotation"""
            c = np.cos(y / 2)
            s = np.sin(y / 2)
            return np.array(
                [
                    [
                        [np.exp(-0.5j * (_x + _z)) * _c, -np.exp(0.5j * (_x - _z)) * _s],
                        [np.exp(-0.5j * (_x - _z)) * _s, np.exp(0.5j * (_x + _z)) * _c],
                    ]
                    for _x, _z, _c, _s in zip(x, z, c, s)
                ]
            )

        a, b, c = np.array([0.432, -0.124]), np.array([-0.152, 2.912]), np.array([0.9234, -9.2])
        assert np.allclose(
            qml.Rot.compute_matrix(a, b, c), arbitrary_rotation(a, b, c), atol=tol, rtol=0
        )
        assert np.allclose(
            qml.Rot(a, b, c, wires=0).matrix(), arbitrary_rotation(a, b, c), atol=tol, rtol=0
        )

    def test_U2_gate(self, tol):
        """Test U2 gate matrix matches the documentation"""
        phi = 0.432
        lam = -0.12
        expected = np.array(
            [[1, -np.exp(1j * lam)], [np.exp(1j * phi), np.exp(1j * (phi + lam))]]
        ) / np.sqrt(2)
        assert np.allclose(qml.U2.compute_matrix(phi, lam), expected, atol=tol, rtol=0)
        assert np.allclose(qml.U2(phi, lam, wires=[0]).matrix(), expected, atol=tol, rtol=0)

    def test_U2_gate_broadcasted(self, tol):
        """Test U2 gate matrix matches the documentation"""

        def get_expected(phi, lam):
            one = np.ones_like(phi) * np.ones_like(lam)
            expected = np.array(
                [[one, -np.exp(1j * lam * one)], [np.exp(1j * phi * one), np.exp(1j * (phi + lam))]]
            ) / np.sqrt(2)
            return np.transpose(expected, (2, 0, 1))

        phi = np.array([0.1, 2.1, -0.6])
        lam = np.array([1.2, 4.9, 0.7])
        expected = get_expected(phi, lam)
        assert np.allclose(qml.U2.compute_matrix(phi, lam), expected, atol=tol, rtol=0)
        assert np.allclose(qml.U2(phi, lam, wires=[0]).matrix(), expected, atol=tol, rtol=0)

        phi = 0.432
        lam = np.array([1.2, 4.9, 0.7])
        expected = get_expected(phi, lam)
        assert np.allclose(qml.U2.compute_matrix(phi, lam), expected, atol=tol, rtol=0)
        assert np.allclose(qml.U2(phi, lam, wires=[0]).matrix(), expected, atol=tol, rtol=0)

        phi = np.array([0.1, 2.1, -0.6])
        lam = -0.12
        expected = get_expected(phi, lam)
        assert np.allclose(qml.U2.compute_matrix(phi, lam), expected, atol=tol, rtol=0)
        assert np.allclose(qml.U2(phi, lam, wires=[0]).matrix(), expected, atol=tol, rtol=0)

    def test_U3_gate(self, tol):
        """Test U3 gate matrix matches the documentation"""
        theta = 0.65
        phi = 0.432
        lam = -0.12

        expected = np.array(
            [
                [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
                [
                    np.exp(1j * phi) * np.sin(theta / 2),
                    np.exp(1j * (phi + lam)) * np.cos(theta / 2),
                ],
            ]
        )

        assert np.allclose(qml.U3.compute_matrix(theta, phi, lam), expected, atol=tol, rtol=0)
        assert np.allclose(qml.U3(theta, phi, lam, wires=[0]).matrix(), expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", [0.432, np.array([0.1, 2.1, -0.6])])
    @pytest.mark.parametrize("phi", [0.654, np.array([1.2, 4.9, 0.7])])
    @pytest.mark.parametrize("lam", [0.218, np.array([-1.7, 3.2, 1.9])])
    def test_U3_gate_broadcasted(self, tol, theta, phi, lam):
        """Test broadcasted U3 gate matrix matches the documentation"""
        if np.ndim(theta) == np.ndim(phi) == np.ndim(lam) == 0:
            pytest.skip("The scalars-only case is covered in a separate test.")
        one = np.ones_like(phi) * np.ones_like(lam) * np.ones_like(theta)
        expected = np.array(
            [
                [one * np.cos(theta / 2), one * -np.exp(1j * lam) * np.sin(theta / 2)],
                [
                    one * np.exp(1j * phi) * np.sin(theta / 2),
                    np.exp(1j * (phi + lam)) * np.cos(theta / 2),
                ],
            ]
        )
        expected = np.transpose(expected, (2, 0, 1))
        assert np.allclose(qml.U3.compute_matrix(theta, phi, lam), expected, atol=tol, rtol=0)
        assert np.allclose(qml.U3(theta, phi, lam, wires=[0]).matrix(), expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, 0.5])
    @pytest.mark.parametrize(
        "cphase_op,gate_data_mat",
        [
            (qml.CPhaseShift00, CPhaseShift00),
            (qml.CPhaseShift01, CPhaseShift01),
            (qml.CPhaseShift10, CPhaseShift10),
        ],
    )
    def test_c_phase_shift_matrix_and_eigvals(self, phi, cphase_op, gate_data_mat):
        """Tests that the CPhaseShift operations calculate the correct
        matrix and eigenvalues"""
        op = cphase_op(phi, wires=[0, 1])
        res = op.matrix()
        exp = gate_data_mat(phi)
        assert np.allclose(res, exp)

        res = op.eigvals()
        assert np.allclose(res, np.diag(exp))

    @pytest.mark.tf
    @pytest.mark.parametrize("phi", np.linspace(-np.pi, np.pi, 10))
    @pytest.mark.parametrize(
        "cphase_op,gate_data_mat",
        [
            (qml.CPhaseShift00, CPhaseShift00),
            (qml.CPhaseShift01, CPhaseShift01),
            (qml.CPhaseShift10, CPhaseShift10),
        ],
    )
    def test_c_phase_shift_matrix_and_eigvals_tf(self, phi, cphase_op, gate_data_mat):
        """Test matrix and eigenvalues computation for CPhaseShift using Tensorflow interface"""
        import tensorflow as tf

        param_tf = tf.Variable(phi)
        op = cphase_op(param_tf, wires=[0, 1])
        res = op.matrix()
        exp = gate_data_mat(phi)
        assert np.allclose(res, exp)

        res = op.eigvals()
        assert np.allclose(res, np.diag(exp))

    @pytest.mark.torch
    @pytest.mark.parametrize("phi", np.linspace(-np.pi, np.pi, 10))
    @pytest.mark.parametrize(
        "cphase_op,gate_data_mat",
        [
            (qml.CPhaseShift00, CPhaseShift00),
            (qml.CPhaseShift01, CPhaseShift01),
            (qml.CPhaseShift10, CPhaseShift10),
        ],
    )
    def test_c_phase_shift_matrix_and_eigvals_torch(self, phi, cphase_op, gate_data_mat):
        """Test matrix and eigenvalues computation for CPhaseShift using Torch interface"""
        import torch

        param_torch = torch.tensor(phi)
        op = cphase_op(param_torch, wires=[0, 1])
        res = op.matrix()
        exp = gate_data_mat(phi)
        assert np.allclose(res, exp)

        res = op.eigvals()
        assert np.allclose(res, np.diag(exp))

    @pytest.mark.jax
    @pytest.mark.parametrize("phi", np.linspace(-np.pi, np.pi, 10))
    @pytest.mark.parametrize(
        "cphase_op,gate_data_mat",
        [
            (qml.CPhaseShift00, CPhaseShift00),
            (qml.CPhaseShift01, CPhaseShift01),
            (qml.CPhaseShift10, CPhaseShift10),
        ],
    )
    def test_c_phase_shift_matrix_and_eigvals_jax(self, phi, cphase_op, gate_data_mat):
        """Test matrix and eigenvalues computation for CPhaseShift using JAX interface"""
        import jax

        param_jax = jax.numpy.array(phi)
        op = cphase_op(param_jax, wires=[0, 1])
        res = op.matrix()
        exp = gate_data_mat(phi)
        assert np.allclose(res, exp)

        res = op.eigvals()
        assert np.allclose(res, np.diag(exp))

    @pytest.mark.parametrize(
        "cphase_op,shift_pos",
        [
            (qml.CPhaseShift00, 0),
            (qml.CPhaseShift01, 1),
            (qml.CPhaseShift10, 2),
        ],
    )
    def test_c_phase_shift_matrix_and_eigvals_broadcasted(self, cphase_op, shift_pos):
        """Tests that the CPhaseShift operations calculate the
        correct matrix and eigenvalues for broadcasted parameters"""
        phi = np.array([0.2, np.pi / 2, -0.1])
        op = cphase_op(phi, wires=[0, 1])
        res = op.matrix()
        expected = np.array([np.eye(4, dtype=complex)] * 3)
        expected[..., shift_pos, shift_pos] = np.exp(1j * phi)
        assert np.allclose(res, expected)

        res = op.eigvals()
        exp_eigvals = np.ones((3, 4), dtype=complex)
        exp_eigvals[..., shift_pos] = np.exp(1j * phi)
        assert np.allclose(res, exp_eigvals)


class TestEigvals:
    """Test eigvals of parametrized operations."""

    def test_rz_eigvals(self, tol):
        """Test eigvals of z rotation are correct"""

        # test identity for theta=0
        assert np.allclose(qml.RZ.compute_eigvals(0), np.ones(2), atol=tol, rtol=0)
        assert np.allclose(qml.RZ(0, wires=0).eigvals(), np.ones(2), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected = np.exp([-1j * np.pi / 4, 1j * np.pi / 4])
        assert np.allclose(qml.RZ.compute_eigvals(np.pi / 2), expected, atol=tol, rtol=0)
        assert np.allclose(qml.RZ(np.pi / 2, wires=0).eigvals(), expected, atol=tol, rtol=0)

        # test identity for broadcasted theta=pi/2
        expected = np.tensordot([1, 1], np.exp([-1j * np.pi / 4, 1j * np.pi / 4]), axes=0)
        pi_half = np.array([np.pi / 2, np.pi / 2])
        assert np.allclose(qml.RZ.compute_eigvals(pi_half), expected, atol=tol, rtol=0)
        assert np.allclose(qml.RZ(pi_half, wires=0).eigvals(), expected, atol=tol, rtol=0)

        # test identity for theta=pi
        assert np.allclose(qml.RZ.compute_eigvals(np.pi), np.diag(-1j * Z), atol=tol, rtol=0)
        assert np.allclose(qml.RZ(np.pi, wires=0).eigvals(), np.diag(-1j * Z), atol=tol, rtol=0)

    def test_phase_shift_eigvals(self, tol):
        """Test phase shift eigvals are correct"""

        # test identity for theta=0
        assert np.allclose(qml.PhaseShift.compute_eigvals(0), np.ones(2), atol=tol, rtol=0)
        assert np.allclose(qml.PhaseShift(0, wires=0).eigvals(), np.ones(2), atol=tol, rtol=0)

        # test arbitrary phase shift
        phi = 0.5432
        expected = np.array([1, np.exp(1j * phi)])
        assert np.allclose(qml.PhaseShift.compute_eigvals(phi), expected, atol=tol, rtol=0)

        # test arbitrary broadcasted phase shift
        phi = np.array([0.5, 0.4, 0.3])
        expected = np.array([[1, np.exp(1j * p)] for p in phi])
        assert np.allclose(qml.PhaseShift.compute_eigvals(phi), expected, atol=tol, rtol=0)

    def test_pcphase_eigvals(self):
        """Test pcphase eigenvalues are correct"""

        # test identity for theta=0
        op = qml.PCPhase(0.0, dim=2, wires=[0, 1])
        assert np.allclose(op.compute_eigvals(*op.parameters, **op.hyperparameters), np.ones(4))
        assert np.allclose(op.eigvals(), np.ones(4))

        # test arbitrary phase shift
        phi = 0.5432
        op = qml.PCPhase(phi, dim=2, wires=[0, 1])
        expected = np.array(
            [np.exp(1j * phi), np.exp(1j * phi), np.exp(-1j * phi), np.exp(-1j * phi)]
        )
        assert np.allclose(op.eigvals(), expected)

        # test arbitrary broadcasted phase shift
        phi = np.array([0.5, 0.4, 0.3])
        op = qml.PCPhase(phi, dim=2, wires=[0, 1])
        expected = np.array(
            [[np.exp(1j * p), np.exp(1j * p), np.exp(-1j * p), np.exp(-1j * p)] for p in phi]
        )
        assert np.allclose(op.eigvals(), expected)

    def test_global_phase_eigvals(self):
        """Test GlobalPhase eigenvalues are correct"""

        # test identity for theta=0
        op = qml.GlobalPhase(0.0)
        assert np.allclose(op.compute_eigvals(*op.parameters, **op.hyperparameters), np.ones(2))
        assert np.allclose(op.eigvals(), np.ones(2))

        # test arbitrary phase shift
        phi = 0.5432
        op = qml.GlobalPhase(phi)
        expected = np.array([np.exp(-1j * phi), np.exp(-1j * phi)])
        assert np.allclose(op.compute_eigvals(*op.parameters, **op.hyperparameters), expected)
        assert np.allclose(op.eigvals(), expected)


class TestGrad:
    device_methods = [
        ["default.qubit", "finite-diff"],
        ["default.qubit", "parameter-shift"],
        ["default.qubit", "backprop"],
        ["default.qubit", "adjoint"],
    ]

    phis = [0.1, 0.2, 0.3]

    configuration = []

    for phi in phis:
        for device, method in device_methods:
            configuration.append([device, method, npp.array(phi, requires_grad=True)])

    @pytest.mark.autograd
    @pytest.mark.parametrize("dev_name,diff_method,phi", configuration)
    def test_pswap_autograd_grad(self, tol, dev_name, diff_method, phi):
        """Test the gradient with Autograd for the gate PSWAP."""

        if diff_method in {"adjoint"}:
            # PSWAP does not have a generator defined
            pytest.skip("PSWAP does not support adjoint")

        dev = qml.device(dev_name, wires=2)

        psi_0 = 0.1
        psi_1 = 0.2
        psi_2 = 0.3
        psi_3 = 0.4

        init_state = npp.array([psi_0, psi_1, psi_2, psi_3], requires_grad=False)
        norm = np.linalg.norm(init_state)
        init_state /= norm

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(phi):
            qml.StatePrep(init_state, wires=[0, 1])
            qml.PSWAP(phi, wires=[0, 1])
            return qml.expval(qml.PauliY(0))

        expected = 2 * np.cos(phi) * (psi_0 * psi_1 - psi_3 * psi_2) / norm**2

        res = qml.grad(circuit)(phi)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.torch
    @pytest.mark.parametrize("dev_name,diff_method,phi", configuration)
    def test_pswap_torch_grad(self, tol, dev_name, diff_method, phi):
        """Test the gradient with Torch for the gate PSWAP."""

        if diff_method in {"adjoint"}:
            # PSWAP does not have a generator defined
            pytest.skip("PSWAP does not support adjoint")

        import torch

        dev = qml.device(dev_name, wires=2)

        psi_0 = torch.tensor(0.1)
        psi_1 = torch.tensor(0.2)
        psi_2 = torch.tensor(0.3)
        psi_3 = torch.tensor(0.4)

        init_state = torch.tensor([psi_0, psi_1, psi_2, psi_3], requires_grad=False)
        norm = torch.norm(init_state)
        init_state /= norm

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(phi):
            qml.StatePrep(init_state, wires=[0, 1])
            qml.PSWAP(phi, wires=[0, 1])
            return qml.expval(qml.PauliY(0))

        phi = torch.tensor(phi, requires_grad=True)

        expected = 2 * torch.cos(phi) * (psi_0 * psi_1 - psi_3 * psi_2) / norm**2

        result = circuit(phi)
        result.backward()
        res = phi.grad
        assert np.allclose(res, expected.detach(), atol=tol, rtol=0)

    @pytest.mark.jax
    @pytest.mark.parametrize("dev_name,diff_method,phi", configuration)
    def test_pswap_jax_grad(self, tol, dev_name, diff_method, phi):
        """Test the gradient with JAX for the gate PSWAP."""

        if diff_method in {"adjoint"}:
            # PSWAP does not have a generator defined
            pytest.skip("PSWAP does not support adjoint")

        import jax
        import jax.numpy as jnp

        dev = qml.device(dev_name, wires=2)

        psi_0 = 0.1
        psi_1 = 0.2
        psi_2 = 0.3
        psi_3 = 0.4

        init_state = jnp.array([psi_0, psi_1, psi_2, psi_3])
        norm = jnp.linalg.norm(init_state)
        init_state = init_state / norm

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(phi):
            qml.StatePrep(init_state, wires=[0, 1])
            qml.PSWAP(phi, wires=[0, 1])
            return qml.expval(qml.PauliY(0))

        phi = jnp.array(phi)

        expected = 2 * np.cos(phi) * (psi_0 * psi_1 - psi_3 * psi_2) / norm**2

        res = jax.grad(circuit, argnums=0)(phi)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.tf
    @pytest.mark.parametrize("dev_name,diff_method,phi", configuration)
    def test_pswap_tf_grad(self, tol, dev_name, diff_method, phi):
        """Test the gradient with Tensorflow for the gate PSWAP."""

        if diff_method in {"adjoint"}:
            # PSWAP does not have a generator defined
            pytest.skip("PSWAP does not support adjoint")

        import tensorflow as tf

        dev = qml.device(dev_name, wires=2)

        psi_0 = tf.Variable(0.1, dtype=tf.complex128)
        psi_1 = tf.Variable(0.2, dtype=tf.complex128)
        psi_2 = tf.Variable(0.3, dtype=tf.complex128)
        psi_3 = tf.Variable(0.4, dtype=tf.complex128)

        init_state = tf.Variable([psi_0, psi_1, psi_2, psi_3], dtype=tf.complex128)
        norm = tf.norm(init_state)
        init_state = init_state / norm

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(phi):
            qml.StatePrep(init_state, wires=[0, 1])
            qml.PSWAP(phi, wires=[0, 1])
            return qml.expval(qml.PauliY(0))

        phi = tf.Variable(phi, dtype=tf.complex128)

        expected = 2 * tf.cos(phi) * (psi_0 * psi_1 - psi_3 * psi_2) / norm**2

        with tf.GradientTape() as tape:
            result = circuit(phi)

        res = tape.gradient(result, phi)
        if diff_method == "backprop":
            # Check #2872 https://github.com/PennyLaneAI/pennylane/issues/2872
            assert np.allclose(np.real(res), expected, atol=tol, rtol=0)
        else:
            assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.autograd
    @pytest.mark.parametrize("dev_name,diff_method,phi", configuration)
    def test_isingxx_autograd_grad(self, tol, dev_name, diff_method, phi):
        """Test the gradient for the gate IsingXX."""
        dev = qml.device(dev_name, wires=2)

        psi_0 = 0.1
        psi_1 = 0.2
        psi_2 = 0.3
        psi_3 = 0.4

        init_state = npp.array([psi_0, psi_1, psi_2, psi_3], requires_grad=False)
        norm = np.linalg.norm(init_state)
        init_state /= norm

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(phi):
            qml.StatePrep(init_state, wires=[0, 1])
            qml.IsingXX(phi, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        expected = (
            0.5
            * (1 / norm**2)
            * (
                -np.sin(phi) * (psi_0**2 + psi_1**2 - psi_2**2 - psi_3**2)
                + 2
                * np.sin(phi / 2)
                * np.cos(phi / 2)
                * (-(psi_0**2) - psi_1**2 + psi_2**2 + psi_3**2)
            )
        )

        res = qml.grad(circuit)(phi)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.autograd
    @pytest.mark.parametrize("dev_name,diff_method,phi", configuration)
    def test_isingyy_autograd_grad(self, tol, dev_name, diff_method, phi):
        """Test the gradient for the gate IsingYY."""
        dev = qml.device(dev_name, wires=2)

        psi_0 = 0.1
        psi_1 = 0.2
        psi_2 = 0.3
        psi_3 = 0.4

        init_state = npp.array([psi_0, psi_1, psi_2, psi_3], requires_grad=False)
        norm = np.linalg.norm(init_state)
        init_state /= norm

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(phi):
            qml.StatePrep(init_state, wires=[0, 1])
            qml.IsingYY(phi, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        expected = (
            0.5
            * (1 / norm**2)
            * (
                -np.sin(phi) * (psi_0**2 + psi_1**2 - psi_2**2 - psi_3**2)
                + 2
                * np.sin(phi / 2)
                * np.cos(phi / 2)
                * (-(psi_0**2) - psi_1**2 + psi_2**2 + psi_3**2)
            )
        )

        res = qml.grad(circuit)(phi)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.autograd
    @pytest.mark.parametrize("dev_name,diff_method,phi", configuration)
    def test_isingzz_autograd_grad(self, tol, dev_name, diff_method, phi):
        """Test the gradient for the gate IsingZZ."""
        dev = qml.device(dev_name, wires=2)

        psi_0 = 0.1
        psi_1 = 0.2
        psi_2 = 0.3
        psi_3 = 0.4

        init_state = npp.array([psi_0, psi_1, psi_2, psi_3], requires_grad=False)
        norm = np.linalg.norm(init_state)
        init_state /= norm

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(phi):
            qml.StatePrep(init_state, wires=[0, 1])
            qml.IsingZZ(phi, wires=[0, 1])
            return qml.expval(qml.PauliX(0))

        phi = npp.array(phi, requires_grad=True)

        expected = (1 / norm**2) * (-2 * (psi_0 * psi_2 + psi_1 * psi_3) * np.sin(phi))

        res = qml.grad(circuit)(phi)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.autograd
    @pytest.mark.parametrize("dev_name,diff_method,phi", configuration)
    def test_isingxy_autograd_grad(self, tol, dev_name, diff_method, phi):
        """Test the gradient with Autograd for the gate IsingXY."""
        dev = qml.device(dev_name, wires=2)

        psi_0 = 0.1
        psi_1 = 0.2
        psi_2 = 0.3
        psi_3 = 0.4

        init_state = npp.array([psi_0, psi_1, psi_2, psi_3], requires_grad=False)
        norm = np.linalg.norm(init_state)
        init_state /= norm

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(phi):
            qml.StatePrep(init_state, wires=[0, 1])
            qml.IsingXY(phi, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        phi = npp.array(phi, requires_grad=True)

        expected = (1 / norm**2) * (psi_2**2 - psi_1**2) * np.sin(phi)

        res = qml.grad(circuit)(phi)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.autograd
    @pytest.mark.parametrize("dev_name,diff_method", device_methods)
    @pytest.mark.parametrize("wires", [(0, 1), (1, 0)])
    def test_globalphase_autograd_grad(self, tol, dev_name, diff_method, wires):
        """Test the gradient with Autograd for a controlled GlobalPhase."""

        dev = qml.device(dev_name, wires=2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(x):
            qml.Identity(wires[0])
            qml.Hadamard(wires[1])
            qml.ctrl(qml.GlobalPhase(x), control=wires[1])
            qml.Hadamard(wires[1])
            return qml.expval(qml.PauliZ(wires[1]))

        phi = npp.array(2.1, requires_grad=True)

        expected = [-0.8632093]

        res = qml.grad(circuit)(phi)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.jax
    @pytest.mark.parametrize("dev_name,diff_method,phi", configuration)
    def test_isingxy_jax_grad(self, tol, dev_name, diff_method, phi):
        """Test the gradient with JAX for the gate IsingXY."""

        if diff_method in {"finite-diff"}:
            pytest.skip("Test does not support finite-diff")

        if diff_method in {"parameter-shift"}:
            pytest.skip("Test does not support parameter-shift")

        import jax
        import jax.numpy as jnp

        dev = qml.device(dev_name, wires=2)

        psi_0 = 0.1
        psi_1 = 0.2
        psi_2 = 0.3
        psi_3 = 0.4

        init_state = jnp.array([psi_0, psi_1, psi_2, psi_3])
        norm = jnp.linalg.norm(init_state)
        init_state = init_state / norm

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(phi):
            qml.StatePrep(init_state, wires=[0, 1])
            qml.IsingXY(phi, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        phi = jnp.array(phi)

        expected = (1 / norm**2) * (psi_2**2 - psi_1**2) * np.sin(phi)

        res = jax.grad(circuit, argnums=0)(phi)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.jax
    @pytest.mark.parametrize("dev_name,diff_method,phi", configuration)
    def test_isingxx_jax_grad(self, tol, dev_name, diff_method, phi):
        """Test the gradient for the gate IsingXX."""

        if diff_method in {"finite-diff"}:
            pytest.skip("Test does not support finite-diff")

        if diff_method in {"parameter-shift"}:
            pytest.skip("Test does not support parameter-shift")

        import jax
        import jax.numpy as jnp

        dev = qml.device(dev_name, wires=2)

        psi_0 = 0.1
        psi_1 = 0.2
        psi_2 = 0.3
        psi_3 = 0.4

        init_state = jnp.array([psi_0, psi_1, psi_2, psi_3])
        norm = jnp.linalg.norm(init_state)
        init_state = init_state / norm

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(phi):
            qml.StatePrep(init_state, wires=[0, 1])
            qml.IsingXX(phi, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        phi = jnp.array(phi)

        expected = (
            0.5
            * (1 / norm**2)
            * (
                -np.sin(phi) * (psi_0**2 + psi_1**2 - psi_2**2 - psi_3**2)
                + 2
                * np.sin(phi / 2)
                * np.cos(phi / 2)
                * (-(psi_0**2) - psi_1**2 + psi_2**2 + psi_3**2)
            )
        )

        res = jax.grad(circuit, argnums=0)(phi)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.jax
    @pytest.mark.parametrize("dev_name,diff_method,phi", configuration)
    def test_isingyy_jax_grad(self, tol, dev_name, diff_method, phi):
        """Test the gradient for the gate IsingYY."""

        if diff_method in {"finite-diff"}:
            pytest.skip("Test does not support finite-diff")

        if diff_method in {"parameter-shift"}:
            pytest.skip("Test does not support parameter-shift")

        import jax
        import jax.numpy as jnp

        dev = qml.device(dev_name, wires=2)

        psi_0 = 0.1
        psi_1 = 0.2
        psi_2 = 0.3
        psi_3 = 0.4

        init_state = jnp.array([psi_0, psi_1, psi_2, psi_3])
        norm = jnp.linalg.norm(init_state)
        init_state = init_state / norm

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(phi):
            qml.StatePrep(init_state, wires=[0, 1])
            qml.IsingYY(phi, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        phi = jnp.array(phi)

        expected = (
            0.5
            * (1 / norm**2)
            * (
                -np.sin(phi) * (psi_0**2 + psi_1**2 - psi_2**2 - psi_3**2)
                + 2
                * np.sin(phi / 2)
                * np.cos(phi / 2)
                * (-(psi_0**2) - psi_1**2 + psi_2**2 + psi_3**2)
            )
        )

        res = jax.grad(circuit, argnums=0)(phi)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.jax
    @pytest.mark.parametrize("dev_name,diff_method,phi", configuration)
    def test_isingzz_jax_grad(self, tol, dev_name, diff_method, phi):
        """Test the gradient for the gate IsingZZ."""

        if diff_method in {"finite-diff"}:
            pytest.skip("Test does not support finite-diff")

        if diff_method in {"parameter-shift"}:
            pytest.skip("Test does not support parameter-shift")

        import jax
        import jax.numpy as jnp

        dev = qml.device(dev_name, wires=2)

        psi_0 = 0.1
        psi_1 = 0.2
        psi_2 = 0.3
        psi_3 = 0.4

        init_state = jnp.array([psi_0, psi_1, psi_2, psi_3])
        norm = jnp.linalg.norm(init_state)
        init_state = init_state / norm

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(phi):
            qml.StatePrep(init_state, wires=[0, 1])
            qml.IsingZZ(phi, wires=[0, 1])
            return qml.expval(qml.PauliX(0))

        phi = jnp.array(phi)

        expected = (1 / norm**2) * (-2 * (psi_0 * psi_2 + psi_1 * psi_3) * np.sin(phi))

        res = jax.grad(circuit, argnums=0)(phi)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.tf
    @pytest.mark.parametrize("dev_name,diff_method,phi", configuration)
    def test_isingxy_tf_grad(self, tol, dev_name, diff_method, phi):
        """Test the gradient with Tensorflow for the gate IsingXY."""
        import tensorflow as tf

        dev = qml.device(dev_name, wires=2)

        psi_0 = tf.Variable(0.1, dtype=tf.complex128)
        psi_1 = tf.Variable(0.2, dtype=tf.complex128)
        psi_2 = tf.Variable(0.3, dtype=tf.complex128)
        psi_3 = tf.Variable(0.4, dtype=tf.complex128)

        init_state = tf.Variable([psi_0, psi_1, psi_2, psi_3], dtype=tf.complex128)
        norm = tf.norm(init_state)
        init_state = init_state / norm

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(phi):
            qml.StatePrep(init_state, wires=[0, 1])
            qml.IsingXY(phi, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        phi = tf.Variable(phi, dtype=tf.complex128)

        expected = (1 / norm**2) * (psi_2**2 - psi_1**2) * tf.sin(phi)

        with tf.GradientTape() as tape:
            result = circuit(phi)
        res = tape.gradient(result, phi)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.tf
    @pytest.mark.parametrize("dev_name,diff_method,phi", configuration)
    def test_isingxx_tf_grad(self, tol, dev_name, diff_method, phi):
        """Test the gradient for the gate IsingXX."""
        import tensorflow as tf

        dev = qml.device(dev_name, wires=2)

        psi_0 = tf.Variable(0.1, dtype=tf.complex128)
        psi_1 = tf.Variable(0.2, dtype=tf.complex128)
        psi_2 = tf.Variable(0.3, dtype=tf.complex128)
        psi_3 = tf.Variable(0.4, dtype=tf.complex128)

        init_state = tf.Variable([psi_0, psi_1, psi_2, psi_3], dtype=tf.complex128)
        norm = tf.norm(init_state)
        init_state = init_state / norm

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(phi):
            qml.StatePrep(init_state, wires=[0, 1])
            qml.IsingXX(phi, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        phi = tf.Variable(phi, dtype=tf.complex128)

        # pylint:disable=invalid-unary-operand-type
        expected = (
            0.5
            * (1 / norm**2)
            * (
                -1 * tf.sin(phi) * (psi_0**2 + psi_1**2 - psi_2**2 - psi_3**2)
                + 2
                * tf.sin(phi / 2)
                * tf.cos(phi / 2)
                * (-(psi_0**2) - psi_1**2 + psi_2**2 + psi_3**2)
            )
        )

        with tf.GradientTape() as tape:
            result = circuit(phi)
        res = tape.gradient(result, phi)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.tf
    @pytest.mark.parametrize("dev_name,diff_method,phi", configuration)
    def test_isingyy_tf_grad(self, tol, dev_name, diff_method, phi):
        """Test the gradient for the gate IsingYY."""
        import tensorflow as tf

        dev = qml.device(dev_name, wires=2)

        psi_0 = tf.Variable(0.1, dtype=tf.complex128)
        psi_1 = tf.Variable(0.2, dtype=tf.complex128)
        psi_2 = tf.Variable(0.3, dtype=tf.complex128)
        psi_3 = tf.Variable(0.4, dtype=tf.complex128)

        init_state = tf.Variable([psi_0, psi_1, psi_2, psi_3], dtype=tf.complex128)
        norm = tf.norm(init_state)
        init_state = init_state / norm

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(phi):
            qml.StatePrep(init_state, wires=[0, 1])
            qml.IsingYY(phi, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        phi = tf.Variable(phi, dtype=tf.complex128)

        # pylint:disable=invalid-unary-operand-type
        expected = (
            0.5
            * (1 / norm**2)
            * (
                -1 * tf.sin(phi) * (psi_0**2 + psi_1**2 - psi_2**2 - psi_3**2)
                + 2
                * tf.sin(phi / 2)
                * tf.cos(phi / 2)
                * (-(psi_0**2) - psi_1**2 + psi_2**2 + psi_3**2)
            )
        )

        with tf.GradientTape() as tape:
            result = circuit(phi)
        res = tape.gradient(result, phi)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.tf
    @pytest.mark.parametrize("dev_name,diff_method,phi", configuration)
    def test_isingzz_tf_grad(self, tol, dev_name, diff_method, phi):
        """Test the gradient for the gate IsingZZ."""
        import tensorflow as tf

        dev = qml.device(dev_name, wires=2)

        psi_0 = tf.Variable(0.1, dtype=tf.complex128)
        psi_1 = tf.Variable(0.2, dtype=tf.complex128)
        psi_2 = tf.Variable(0.3, dtype=tf.complex128)
        psi_3 = tf.Variable(0.4, dtype=tf.complex128)

        init_state = tf.Variable([psi_0, psi_1, psi_2, psi_3], dtype=tf.complex128)
        norm = tf.norm(init_state)
        init_state = init_state / norm

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(phi):
            qml.StatePrep(init_state, wires=[0, 1])
            qml.IsingZZ(phi, wires=[0, 1])
            return qml.expval(qml.PauliX(0))

        phi = tf.Variable(phi, dtype=tf.float64)

        expected = (1 / norm**2) * (-2 * (psi_0 * psi_2 + psi_1 * psi_3) * np.sin(phi))

        with tf.GradientTape() as tape:
            result = circuit(phi)
        res = tape.gradient(result, phi)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.tf
    @pytest.mark.parametrize("dev_name,diff_method", device_methods)
    @pytest.mark.parametrize("wires", [(0, 1), (1, 0)])
    def test_globalphase_tf_grad(self, tol, dev_name, diff_method, wires):
        """Test the gradient with Tensorflow for a controlled GlobalPhase."""

        import tensorflow as tf

        dev = qml.device(dev_name, wires=2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(x):
            qml.Identity(wires[0])
            qml.Hadamard(wires[1])
            qml.ctrl(qml.GlobalPhase(x), control=wires[1])
            qml.Hadamard(wires[1])
            return qml.expval(qml.PauliZ(wires[1]))

        phi = tf.Variable(2.1, dtype=tf.complex128)

        expected = [-0.8632093]

        with tf.GradientTape() as tape:
            result = circuit(phi)
        res = tape.gradient(result, phi)
        assert np.allclose(np.real(res), expected, atol=tol, rtol=0)

    @pytest.mark.jax
    @pytest.mark.parametrize("par", np.linspace(0, 2 * np.pi, 3))
    def test_qnode_with_rx_and_state_jacobian_jax(self, par, tol):
        """Test the jacobian of a complex valued QNode that contains a rotation
        using the JAX interface."""
        import jax

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, diff_method="backprop")
        def test(x):
            qml.RX(x, wires=[0])
            return qml.state()

        res = jax.jacobian(test, holomorphic=True)(par + 0j)
        expected = -1 / 2 * np.sin(par / 2), -1 / 2 * 1j * np.cos(par / 2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("dev_name,diff_method,phi", configuration)
    def test_pcphase_grad(self, dev_name, diff_method, phi):
        """Test pcphase operator gradient"""
        if diff_method in {"adjoint"}:
            pytest.skip("PCPhase does not support adjoint diff")

        dev = qml.device(dev_name, wires=[0, 1])
        expected_grad = -4 * npp.cos(phi) * npp.sin(phi)  # computed by hand

        @qml.qnode(dev, diff_method=diff_method)
        def circ(phi):
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)

            qml.PCPhase(phi, dim=2, wires=[0, 1])

            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            return qml.expval(qml.PauliZ(wires=0))

        phi = npp.array(phi, requires_grad=True)
        computed_grad = qml.grad(circ)(phi)
        assert np.isclose(computed_grad, expected_grad)

    @pytest.mark.tf
    @pytest.mark.parametrize("dev_name,diff_method,phi", configuration)
    def test_pcphase_grad_tf(self, dev_name, diff_method, phi):
        """Test pcphase operator gradient"""
        if diff_method in {"adjoint"}:
            pytest.skip("PCPhase does not support adjoint diff")

        import tensorflow as tf

        dev = qml.device(dev_name, wires=[0, 1])
        expected_grad = tf.Variable(-4 * npp.cos(phi) * npp.sin(phi))  # computed by hand
        phi = tf.Variable(phi)

        @qml.qnode(dev, diff_method=diff_method)
        def circ(phi):
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)

            qml.PCPhase(phi, dim=2, wires=[0, 1])

            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            return qml.expval(qml.PauliZ(wires=0))

        with tf.GradientTape() as tape:
            result = circ(phi)

        computed_grad = tape.gradient(result, phi)
        assert np.isclose(computed_grad, expected_grad)

    @pytest.mark.torch
    @pytest.mark.parametrize("dev_name,diff_method,phi", configuration)
    def test_pcphase_grad_torch(self, dev_name, diff_method, phi):
        """Test pcphase operator gradient"""
        if diff_method in {"adjoint"}:
            pytest.skip("PCPHase does not support adjoint diff")

        import torch

        dev = qml.device(dev_name, wires=[0, 1])
        expected_grad = torch.tensor(-4 * npp.cos(phi) * npp.sin(phi))  # computed by hand
        phi = torch.tensor(phi, requires_grad=True)

        @qml.qnode(dev, diff_method=diff_method)
        def circ(phi):
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)

            qml.PCPhase(phi, dim=2, wires=[0, 1])

            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            return qml.expval(qml.PauliZ(wires=0))

        res = circ(phi)
        res.backward()
        assert np.isclose(phi.grad, expected_grad)

    @pytest.mark.jax
    @pytest.mark.parametrize("dev_name,diff_method,phi", configuration)
    def test_pcphase_grad_jax(self, dev_name, diff_method, phi):
        """Test pcphase operator gradient"""
        if diff_method in {"adjoint"}:
            pytest.skip("PCPHase does not support adjoint diff")

        import jax
        import jax.numpy as jnp

        dev = qml.device(dev_name, wires=[0, 1])
        expected_grad = jnp.array(-4 * npp.cos(phi) * npp.sin(phi))  # computed by hand
        phi = jnp.array(phi)

        @qml.qnode(dev, diff_method=diff_method)
        def circ(phi):
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)

            qml.PCPhase(phi, dim=2, wires=[0, 1])

            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            return qml.expval(qml.PauliZ(wires=0))

        computed_grad = jax.grad(circ, argnums=0)(phi)
        assert np.isclose(computed_grad, expected_grad)

    @pytest.mark.jax
    @pytest.mark.parametrize("dev_name,diff_method", device_methods)
    @pytest.mark.parametrize("wires", [(1, 0), (0, 1)])
    def test_globalphase_jax_grad(self, tol, dev_name, diff_method, wires):
        """Test the gradient with JAX for a controlled GlobalPhase."""

        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)

        dev = qml.device(dev_name, wires=2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(x):
            qml.Identity(wires[0])
            qml.Hadamard(wires[1])
            qml.ctrl(qml.GlobalPhase(x), control=wires[1])
            qml.Hadamard(wires[1])
            return qml.expval(qml.PauliZ(wires[1]))

        phi = jnp.array(2.1)

        expected = [-0.8632093]

        res = jax.grad(circuit, argnums=0)(phi)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.torch
    @pytest.mark.parametrize("dev_name,diff_method", device_methods)
    @pytest.mark.parametrize("wires", [(1, 0), (0, 1)])
    def test_globalphase_torch_grad(self, tol, dev_name, diff_method, wires):
        """Test the gradient with Torch for a controlled GlobalPhase."""

        import torch

        dev = qml.device(dev_name, wires=2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(x):
            qml.Identity(wires[0])
            qml.Hadamard(wires[1])
            qml.ctrl(qml.GlobalPhase(x), control=wires[1])
            qml.Hadamard(wires[1])
            return qml.expval(qml.PauliZ(wires[1]))

        phi = torch.tensor(2.1, requires_grad=True, dtype=torch.float64)

        expected = [-0.8632093]

        result = circuit(phi)
        result.backward()
        res = phi.grad
        assert np.allclose(res, expected, atol=tol, rtol=0)


class TestGenerator:
    @pytest.mark.parametrize(
        "cphase_op,expected",
        [
            (qml.CPhaseShift00(1.234, wires=(0, 1)), qml.Projector(np.array([0, 0]), wires=(0, 1))),
            (qml.CPhaseShift01(1.234, wires=(0, 1)), qml.Projector(np.array([0, 1]), wires=(0, 1))),
            (qml.CPhaseShift10(1.234, wires=(0, 1)), qml.Projector(np.array([1, 0]), wires=(0, 1))),
        ],
    )
    def test_c_phase_shift_generator(self, cphase_op, expected):
        """Test that the generator of the CPhaseShift operations
        is correctly returned."""
        gen, coeff = qml.generator(cphase_op)

        assert coeff == 1.0
        assert gen.name == expected.name
        assert gen.wires == expected.wires

    def test_pcphase_generator(self):
        """Test the pcphase generator is the projector onto the subspace
        we are applying the phase shift to."""
        phi = 1.23
        op = qml.PCPhase(phi, dim=2, wires=[0, 1])

        expected_mat = np.diag([1.0, 1.0, -1.0, -1.0])

        gen, coeff = qml.generator(op)
        assert np.allclose(qml.matrix(gen), expected_mat)
        assert np.isclose(coeff, 1.0)


PAULI_ROT_PARAMETRIC_MATRIX_TEST_DATA = [
    (
        "XY",
        lambda theta: np.array(
            [
                [np.cos(theta / 2), 0, 0, -np.sin(theta / 2)],
                [0, np.cos(theta / 2), np.sin(theta / 2), 0],
                [0, -np.sin(theta / 2), np.cos(theta / 2), 0],
                [np.sin(theta / 2), 0, 0, np.cos(theta / 2)],
            ],
            dtype=complex,
        ),
    ),
    (
        "ZZ",
        lambda theta: np.diag(
            [
                np.exp(-1j * theta / 2),
                np.exp(1j * theta / 2),
                np.exp(1j * theta / 2),
                np.exp(-1j * theta / 2),
            ],
        ),
    ),
    (
        "XI",
        lambda theta: np.array(
            [
                [np.cos(theta / 2), 0, -1j * np.sin(theta / 2), 0],
                [0, np.cos(theta / 2), 0, -1j * np.sin(theta / 2)],
                [-1j * np.sin(theta / 2), 0, np.cos(theta / 2), 0],
                [0, -1j * np.sin(theta / 2), 0, np.cos(theta / 2)],
            ],
        ),
    ),
    ("X", qml.RX.compute_matrix),
    ("Y", qml.RY.compute_matrix),
    ("Z", qml.RZ.compute_matrix),
]

PAULI_ROT_MATRIX_TEST_DATA = [
    (
        np.pi,
        "XIZ",
        np.array(
            [
                [0, 0, 0, 0, -1j, 0, 0, 0],
                [0, 0, 0, 0, 0, 1j, 0, 0],
                [0, 0, 0, 0, 0, 0, -1j, 0],
                [0, 0, 0, 0, 0, 0, 0, 1j],
                [-1j, 0, 0, 0, 0, 0, 0, 0],
                [0, 1j, 0, 0, 0, 0, 0, 0],
                [0, 0, -1j, 0, 0, 0, 0, 0],
                [0, 0, 0, 1j, 0, 0, 0, 0],
            ]
        ),
    ),
    (
        np.pi / 3,
        "XYZ",
        np.array(
            [
                [np.sqrt(3) / 2, 0, 0, 0, 0, 0, -(1 / 2), 0],
                [0, np.sqrt(3) / 2, 0, 0, 0, 0, 0, 1 / 2],
                [0, 0, np.sqrt(3) / 2, 0, 1 / 2, 0, 0, 0],
                [0, 0, 0, np.sqrt(3) / 2, 0, -(1 / 2), 0, 0],
                [0, 0, -(1 / 2), 0, np.sqrt(3) / 2, 0, 0, 0],
                [0, 0, 0, 1 / 2, 0, np.sqrt(3) / 2, 0, 0],
                [1 / 2, 0, 0, 0, 0, 0, np.sqrt(3) / 2, 0],
                [0, -(1 / 2), 0, 0, 0, 0, 0, np.sqrt(3) / 2],
            ]
        ),
    ),
]


class TestPauliRot:
    """Test the PauliRot operation."""

    def test_paulirot_repr(self):
        op = qml.PauliRot(1.234, "XYX", wires=(0, 1, 2))
        assert repr(op) == "PauliRot(1.234, XYX, wires=[0, 1, 2])"

    @pytest.mark.parametrize("theta", np.linspace(0, 2 * np.pi, 7))
    @pytest.mark.parametrize(
        "pauli_word,expected_matrix",
        PAULI_ROT_PARAMETRIC_MATRIX_TEST_DATA,
    )
    def test_PauliRot_matrix_parametric(self, theta, pauli_word, expected_matrix, tol):
        """Test parametrically that the PauliRot matrix is correct."""

        res = qml.PauliRot.compute_matrix(theta, pauli_word)
        expected = expected_matrix(theta)

        assert np.allclose(res, expected, atol=tol, rtol=0)

        # Test broadcasted matrix
        res = qml.PauliRot.compute_matrix(np.ones(3) * theta, pauli_word)
        expected = [expected_matrix(theta)] * 3

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "theta,pauli_word,expected_matrix",
        PAULI_ROT_MATRIX_TEST_DATA,
    )
    def test_PauliRot_matrix(self, theta, pauli_word, expected_matrix, tol):
        """Test non-parametrically that the PauliRot matrix is correct."""

        res = qml.PauliRot.compute_matrix(theta, pauli_word)
        expected = expected_matrix

        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = qml.PauliRot.compute_matrix(np.ones(5) * theta, pauli_word)
        expected = [expected_matrix] * 5

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "theta,pauli_word,compressed_pauli_word,wires,compressed_wires",
        [
            (np.pi, "XIZ", "XZ", [0, 1, 2], [0, 2]),
            (np.pi / 3, "XIYIZI", "XYZ", [0, 1, 2, 3, 4, 5], [0, 2, 4]),
            (np.pi / 7, "IXI", "X", [0, 1, 2], [1]),
            (np.pi / 9, "IIIIIZI", "Z", [0, 1, 2, 3, 4, 5, 6], [5]),
            (np.pi / 11, "XYZIII", "XYZ", [0, 1, 2, 3, 4, 5], [0, 1, 2]),
            (np.pi / 11, "IIIXYZ", "XYZ", [0, 1, 2, 3, 4, 5], [3, 4, 5]),
        ],
    )
    def test_PauliRot_matrix_identity(
        self, theta, pauli_word, compressed_pauli_word, wires, compressed_wires, tol
    ):
        """Test PauliRot matrix correctly accounts for identities."""
        # pylint: disable=too-many-arguments

        res = qml.PauliRot.compute_matrix(theta, pauli_word)
        expected = qml.math.expand_matrix(
            qml.PauliRot.compute_matrix(theta, compressed_pauli_word), compressed_wires, wires
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)

        batch = np.ones(3) * theta
        res = qml.PauliRot.compute_matrix(batch, pauli_word)
        expected = qml.math.expand_matrix(
            qml.PauliRot.compute_matrix(batch, compressed_pauli_word), compressed_wires, wires
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_PauliRot_wire_as_int(self):
        """Test that passing a single wire as an integer works."""

        theta = 0.4
        op = qml.PauliRot(theta, "Z", wires=0)
        decomp_ops = qml.PauliRot.compute_decomposition(theta, wires=0, pauli_word="Z")

        assert np.allclose(
            op.eigvals(), np.array([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])
        )
        assert np.allclose(op.matrix(), np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)]))

        assert len(decomp_ops) == 1

        assert decomp_ops[0].name == "MultiRZ"

        assert decomp_ops[0].wires == Wires([0])
        assert decomp_ops[0].data[0] == theta

    def test_PauliRot_all_Identity(self):
        """Test handling of the all-identity Pauli."""

        theta = 0.4
        op = qml.PauliRot(theta, "II", wires=[0, 1])
        decomp_ops = op.decomposition()

        assert np.allclose(op.eigvals(), np.exp(-1j * theta / 2) * np.ones(4))
        assert np.allclose(op.matrix() / op.matrix()[0, 0], np.eye(4))

        assert len(decomp_ops) == 0

    def test_PauliRot_all_Identity_broadcasted(self):
        """Test handling of the broadcasted all-identity Pauli."""

        theta = np.array([0.4, 0.9, 1.2])
        op = qml.PauliRot(theta, "II", wires=[0, 1])
        decomp_ops = op.decomposition()

        phases = np.exp(-1j * theta / 2)
        assert np.allclose(op.eigvals(), np.outer(phases, np.ones(4)))
        mat = op.matrix()
        for phase, sub_mat in zip(phases, mat):
            assert np.allclose(sub_mat, phase * np.eye(4))

        assert len(decomp_ops) == 0

    @pytest.mark.parametrize("theta", [0.4, np.array([np.pi / 3, 0.1, -0.9])])
    def test_PauliRot_decomposition_ZZ(self, theta):
        """Test that the decomposition for a ZZ rotation is correct."""
        op = qml.PauliRot(theta, "ZZ", wires=[0, 1])
        decomp_ops = op.decomposition()

        assert len(decomp_ops) == 1

        assert decomp_ops[0].name == "MultiRZ"

        assert decomp_ops[0].wires == Wires([0, 1])
        assert np.allclose(decomp_ops[0].data[0], theta)

    @pytest.mark.parametrize("theta", [0.4, np.array([np.pi / 3, 0.1, -0.9])])
    def test_PauliRot_decomposition_XY(self, theta):
        """Test that the decomposition for a XY rotation is correct."""

        op = qml.PauliRot(theta, "XY", wires=[0, 1])
        decomp_ops = op.decomposition()

        assert len(decomp_ops) == 5

        assert decomp_ops[0].name == "Hadamard"
        assert decomp_ops[0].wires == Wires([0])

        assert decomp_ops[1].name == "RX"
        assert decomp_ops[1].wires == Wires([1])
        assert decomp_ops[1].data[0] == np.pi / 2

        assert decomp_ops[2].name == "MultiRZ"
        assert decomp_ops[2].wires == Wires([0, 1])
        assert np.allclose(decomp_ops[2].data[0], theta)

        assert decomp_ops[3].name == "Hadamard"
        assert decomp_ops[3].wires == Wires([0])

        assert decomp_ops[4].name == "RX"
        assert decomp_ops[4].wires == Wires([1])
        assert decomp_ops[4].data[0] == -np.pi / 2

    @pytest.mark.parametrize("theta", [0.4, np.array([np.pi / 3, 0.1, -0.9])])
    def test_PauliRot_decomposition_XIYZ(self, theta):
        """Test that the decomposition for a XIYZ rotation is correct."""

        op = qml.PauliRot(theta, "XIYZ", wires=[0, 1, 2, 3])
        decomp_ops = op.decomposition()

        assert len(decomp_ops) == 5

        assert decomp_ops[0].name == "Hadamard"
        assert decomp_ops[0].wires == Wires([0])

        assert decomp_ops[1].name == "RX"

        assert decomp_ops[1].wires == Wires([2])
        assert decomp_ops[1].data[0] == np.pi / 2

        assert decomp_ops[2].name == "MultiRZ"
        assert decomp_ops[2].wires == Wires([0, 2, 3])
        assert np.allclose(decomp_ops[2].data[0], theta)

        assert decomp_ops[3].name == "Hadamard"
        assert decomp_ops[3].wires == Wires([0])

        assert decomp_ops[4].name == "RX"

        assert decomp_ops[4].wires == Wires([2])
        assert decomp_ops[4].data[0] == -np.pi / 2

    @pytest.mark.parametrize("angle", npp.linspace(0, 2 * np.pi, 7, requires_grad=True))
    @pytest.mark.parametrize("pauli_word", ["XX", "YY", "ZZ"])
    def test_differentiability(self, angle, pauli_word, tol):
        """Test that differentiation of PauliRot works."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(theta):
            qml.PauliRot(theta, pauli_word, wires=[0, 1])

            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        gradient = np.squeeze(qml.grad(circuit)(angle))

        assert gradient == pytest.approx(
            0.5 * (circuit(angle + np.pi / 2) - circuit(angle - np.pi / 2)), abs=tol
        )

    @pytest.mark.parametrize("pauli_word", ["XX", "YY", "ZZ"])
    def test_differentiability_broadcasted(self, pauli_word, tol):
        """Test that differentiation of PauliRot works with broadcasted parameters."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(theta):
            qml.PauliRot(theta, pauli_word, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        angle = npp.linspace(0, 2 * np.pi, 7, requires_grad=True)
        jac = qml.jacobian(circuit)(angle)

        assert np.allclose(
            jac,
            0.5 * (circuit(angle + np.pi / 2) - circuit(angle - np.pi / 2)),
            atol=tol,
        )

    @pytest.mark.parametrize("angle", npp.linspace(0, 2 * np.pi, 7, requires_grad=True))
    def test_decomposition_integration(self, angle):
        """Test that the decompositon of PauliRot yields the same results."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(theta):
            qml.PauliRot(theta, "XX", wires=[0, 1])

            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev)
        def decomp_circuit(theta):
            qml.PauliRot.compute_decomposition(theta, wires=[0, 1], pauli_word="XX")
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(circuit(angle), decomp_circuit(angle))
        assert np.allclose(qml.grad(circuit)(angle), qml.grad(decomp_circuit)(angle))

    def test_matrix_incorrect_pauli_word_error(self):
        """Test that _matrix throws an error if a wrong Pauli word is supplied."""

        with pytest.raises(
            ValueError,
            match='The given Pauli word ".*" contains characters that are not allowed.'
            " Allowed characters are I, X, Y and Z",
        ):
            qml.PauliRot.compute_matrix(0.3, "IXYZV")

    def test_init_incorrect_pauli_word_error(self):
        """Test that __init__ throws an error if a wrong Pauli word is supplied."""

        with pytest.raises(
            ValueError,
            match='The given Pauli word ".*" contains characters that are not allowed.'
            " Allowed characters are I, X, Y and Z",
        ):
            qml.PauliRot(0.3, "IXYZV", wires=[0, 1, 2, 3, 4])

    def test_empty_wire_list_error_paulirot(self):
        """Test that PauliRot operator raises an error when instantiated with wires=[]."""

        with pytest.raises(ValueError, match="wrong number of wires"):
            qml.PauliRot(0.5, "X", wires=[])

    @pytest.mark.parametrize(
        "pauli_word,wires",
        [
            ("XYZ", [0, 1]),
            ("XYZ", [0, 1, 2, 3]),
        ],
    )
    def test_init_incorrect_pauli_word_length_error(self, pauli_word, wires):
        """Test that __init__ throws an error if a Pauli word of wrong length is supplied."""

        with pytest.raises(
            ValueError,
            match="The given Pauli word has length .*, length .* was expected for wires .*",
        ):
            qml.PauliRot(0.3, pauli_word, wires=wires)

    @pytest.mark.parametrize(
        "pauli_word",
        [
            ("XIZ"),
            ("IIII"),
            ("XIYIZI"),
            ("IXI"),
            ("IIIIIZI"),
            ("XYZIII"),
            ("IIIXYZ"),
        ],
    )
    @pytest.mark.usefixtures("use_legacy_and_new_opmath")
    def test_multirz_generator(self, pauli_word):
        """Test that the generator of the MultiRZ gate is correct."""
        op = qml.PauliRot(0.3, pauli_word, wires=range(len(pauli_word)))
        gen = op.generator()

        assert isinstance(gen, qml.Hamiltonian)

        if pauli_word[0] == "I":
            # this is the identity
            expected_gen = qml.Identity(wires=0)
        else:
            expected_gen = getattr(qml, f"Pauli{pauli_word[0]}")(wires=0)

        for i, pauli in enumerate(pauli_word[1:]):
            i += 1
            if pauli == "I":
                expected_gen = expected_gen @ qml.Identity(wires=i)
            else:
                expected_gen = expected_gen @ getattr(qml, f"Pauli{pauli}")(wires=i)

        assert qml.equal(gen, qml.Hamiltonian([-0.5], [expected_gen]))

    @pytest.mark.torch
    @pytest.mark.gpu
    @pytest.mark.parametrize("theta", np.linspace(0, 2 * np.pi, 7))
    @pytest.mark.parametrize("torch_device", [None, "cuda"])
    def test_pauli_rot_identity_torch(self, torch_device, theta):
        """Test that the PauliRot operation returns the correct matrix when
        providing a gate parameter on the GPU and only specifying the identity
        operation."""
        import torch

        if torch_device == "cuda" and not torch.cuda.is_available():
            pytest.skip("No GPU available")

        x = torch.tensor(theta, device=torch_device)
        mat = qml.PauliRot(x, "I", wires=[0]).matrix()

        val = np.cos(-theta / 2) + 1j * np.sin(-theta / 2)
        exp = torch.tensor(np.diag([val, val]), device=torch_device)
        assert qml.math.allclose(mat, exp)

    @pytest.mark.usefixtures("use_legacy_opmath")
    def test_pauli_rot_generator_legacy_opmath(self):
        """Test that the generator of the PauliRot operation
        is correctly returned."""
        op = qml.PauliRot(0.65, "ZY", wires=["a", 7])
        gen, coeff = qml.generator(op)
        expected = qml.PauliZ("a") @ qml.PauliY(7)

        assert coeff == -0.5
        assert gen.operands[0].name == expected.obs[0].name
        assert gen.operands[1].wires == expected.obs[1].wires

    def test_pauli_rot_generator(self):
        """Test that the generator of the PauliRot operation
        is correctly returned."""
        op = qml.PauliRot(0.65, "ZY", wires=["a", 7])
        gen, coeff = qml.generator(op)
        expected = qml.PauliZ("a") @ qml.PauliY(7)

        assert coeff == -0.5
        assert gen == expected


class TestMultiRZ:
    """Test the MultiRZ operation."""

    @pytest.mark.parametrize("theta", np.linspace(0, 2 * np.pi, 7))
    @pytest.mark.parametrize(
        "wires,expected_matrix",
        [
            ([0], qml.RZ.compute_matrix),
            (
                [0, 1],
                lambda theta: np.diag(
                    np.exp(1j * np.array([-1, 1, 1, -1]) * theta / 2),
                ),
            ),
            (
                [0, 1, 2],
                lambda theta: np.diag(
                    np.exp(1j * np.array([-1, 1, 1, -1, 1, -1, -1, 1]) * theta / 2),
                ),
            ),
        ],
    )
    def test_MultiRZ_matrix_parametric(self, theta, wires, expected_matrix, tol):
        """Test parametrically that the MultiRZ matrix is correct."""

        res_static = qml.MultiRZ.compute_matrix(theta, len(wires))
        res_dynamic = qml.MultiRZ(theta, wires=wires).matrix()
        expected = expected_matrix(theta)

        assert np.allclose(res_static, expected, atol=tol, rtol=0)
        assert np.allclose(res_dynamic, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("num_wires", [1, 2, 3])
    def test_MultiRZ_matrix_broadcasted(self, num_wires, tol):
        """Test that the MultiRZ matrix is correct for broadcasted parameters."""

        theta = np.linspace(0, 2 * np.pi, 7)[:3]
        res_static = qml.MultiRZ.compute_matrix(theta, num_wires)
        res_dynamic = qml.MultiRZ(theta, wires=list(range(num_wires))).matrix()
        signs = reduce(np.kron, [np.array([1, -1])] * num_wires) / 2
        expected = [np.diag(np.exp(-1j * signs * p)) for p in theta]

        assert np.allclose(res_static, expected, atol=tol, rtol=0)
        assert np.allclose(res_dynamic, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", [0.4, np.array([np.pi / 3, 0.1, -0.9])])
    def test_MultiRZ_decomposition_ZZ(self, theta):
        """Test that the decomposition for a ZZ rotation is correct."""

        op = qml.MultiRZ(theta, wires=[0, 1])
        decomp_ops = op.decomposition()

        assert decomp_ops[0].name == "CNOT"
        assert decomp_ops[0].wires == Wires([1, 0])

        assert decomp_ops[1].name == "RZ"

        assert decomp_ops[1].wires == Wires([0])
        assert np.allclose(decomp_ops[1].data[0], theta)

        assert decomp_ops[2].name == "CNOT"
        assert decomp_ops[2].wires == Wires([1, 0])

    @pytest.mark.parametrize("theta", [0.4, np.array([np.pi / 3, 0.1, -0.9])])
    def test_MultiRZ_decomposition_ZZZ(self, theta):
        """Test that the decomposition for a ZZZ rotation is correct."""

        op = qml.MultiRZ(theta, wires=[0, 2, 3])
        decomp_ops = op.decomposition()

        assert decomp_ops[0].name == "CNOT"
        assert decomp_ops[0].wires == Wires([3, 2])

        assert decomp_ops[1].name == "CNOT"
        assert decomp_ops[1].wires == Wires([2, 0])

        assert decomp_ops[2].name == "RZ"

        assert decomp_ops[2].wires == Wires([0])
        assert np.allclose(decomp_ops[2].data[0], theta)

        assert decomp_ops[3].name == "CNOT"
        assert decomp_ops[3].wires == Wires([2, 0])

        assert decomp_ops[4].name == "CNOT"
        assert decomp_ops[4].wires == Wires([3, 2])

    @pytest.mark.parametrize("angle", npp.linspace(0, 2 * np.pi, 7, requires_grad=True))
    def test_differentiability(self, angle, tol):
        """Test that differentiation of MultiRZ works."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(theta):
            qml.Hadamard(0)
            qml.MultiRZ(theta, wires=[0, 1])

            return qml.expval(qml.PauliX(0))

        gradient = np.squeeze(qml.grad(circuit)(angle))

        assert gradient == pytest.approx(
            0.5 * (circuit(angle + np.pi / 2) - circuit(angle - np.pi / 2)), abs=tol
        )

    def test_differentiability_broadcasted(self, tol):
        """Test that differentiation of MultiRZ works."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(theta):
            qml.Hadamard(0)
            qml.Hadamard(1)
            qml.MultiRZ(theta, wires=[0, 1])

            return qml.expval(qml.PauliX(0) @ qml.PauliX(1))

        angle = npp.linspace(0, 2 * np.pi, 7, requires_grad=True)
        jac = qml.jacobian(circuit)(angle)

        assert np.allclose(
            jac, 0.5 * (circuit(angle + np.pi / 2) - circuit(angle - np.pi / 2)), atol=tol
        )

    @pytest.mark.parametrize("angle", npp.linspace(0, 2 * np.pi, 7, requires_grad=True))
    def test_decomposition_integration(self, angle):
        """Test that the decompositon of MultiRZ yields the same results."""
        angle = qml.numpy.array(angle)
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(theta):
            qml.Hadamard(0)
            qml.MultiRZ(theta, wires=[0, 1])

            return qml.expval(qml.PauliX(0))

        @qml.qnode(dev)
        def decomp_circuit(theta):
            qml.Hadamard(0)
            qml.MultiRZ.compute_decomposition(theta, wires=[0, 1])

            return qml.expval(qml.PauliX(0))

        assert np.allclose(circuit(angle), decomp_circuit(angle))
        assert np.allclose(qml.jacobian(circuit)(angle), qml.jacobian(decomp_circuit)(angle))

    @pytest.mark.parametrize("qubits", range(3, 6))
    @pytest.mark.usefixtures("use_legacy_and_new_opmath")
    def test_multirz_generator(self, qubits, mocker):
        """Test that the generator of the MultiRZ gate is correct."""
        op = qml.MultiRZ(0.3, wires=range(qubits))
        gen = op.generator()

        assert isinstance(gen, qml.Hamiltonian)

        expected_gen = qml.PauliZ(wires=0)
        for i in range(1, qubits):
            expected_gen = expected_gen @ qml.PauliZ(wires=i)

        assert qml.equal(gen, qml.Hamiltonian([-0.5], [expected_gen]))

        spy = mocker.spy(qml.utils, "pauli_eigs")

        op.generator()
        spy.assert_not_called()

    @pytest.mark.parametrize("theta", [0.4, np.array([np.pi / 3, 0.1, -0.9])])
    def test_multirz_eigvals(self, theta):
        """Test that the eigenvalues of the MultiRZ gate are correct."""
        op = qml.MultiRZ(theta, wires=range(3))

        pos_phase = np.exp(1j * theta / 2)
        neg_phase = np.exp(-1j * theta / 2)
        expected = np.array(
            [
                neg_phase,
                pos_phase,
                pos_phase,
                neg_phase,
                pos_phase,
                neg_phase,
                neg_phase,
                pos_phase,
            ]
        ).T
        eigvals = op.eigvals()
        assert np.allclose(eigvals, expected)

    def test_empty_wire_list_error_multirz(self):
        """Test that MultiRZ operator raises an error when instantiated with wires=[]."""

        with pytest.raises(ValueError, match="wrong number of wires"):
            qml.MultiRZ(0.5, wires=[])


rotations = [
    qml.RX,
    qml.RY,
    qml.RZ,
    qml.PhaseShift,
    qml.PCPhase,
    qml.ControlledPhaseShift,
    qml.Rot,
    qml.MultiRZ,
    qml.CRX,
    qml.CRY,
    qml.CRZ,
    qml.CRot,
    qml.U1,
    qml.U2,
    qml.U3,
    qml.IsingXX,
    qml.IsingYY,
    qml.IsingZZ,
    qml.IsingXY,
    qml.PSWAP,
]


class TestSimplify:
    """Test rotation simplification methods."""

    @staticmethod
    def get_unsimplified_op(op_class):
        # construct the parameters of the op
        if op_class.num_params == 1:
            params = npp.array([[-50.0, 3.0, 50.0]])
        elif op_class.num_params == 2:
            params = npp.array([[-50.0, 3.0, 50.0], [3.0, 50.0, -50.0]])
        else:
            params = npp.array([[-50.0, 3.0, 50.0], [3.0, 50.0, -50.0], [50.0, -50.0, 3.0]])

        # construct the wires
        if op_class.num_wires == 1:
            wires = 0
        else:
            wires = [0, 1]

        if op_class == qml.PCPhase:
            return op_class(*params, dim=2, wires=wires)
        return op_class(*params, wires)

    @staticmethod
    def _get_params_wires(op):
        return op.data, op.wires

    @pytest.mark.parametrize("op", rotations)
    def test_simplify_rotations(self, op):
        """Test that the matrices and wires are the same after simplification"""

        unsimplified_op = self.get_unsimplified_op(op)
        simplified_op = qml.simplify(unsimplified_op)

        assert qml.math.allclose(qml.matrix(unsimplified_op), qml.matrix(simplified_op))
        assert all((p >= 0).all() and (p < 4 * np.pi).all() for p in simplified_op.data)
        assert unsimplified_op.wires == simplified_op.wires

    @pytest.mark.autograd
    @pytest.mark.parametrize("op", rotations)
    def test_simplify_rotations_grad_autograd(self, op):
        """Test the gradient of an op after simplication for the autograd interface"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(simplify, wires, *params, **hyperparams):
            if simplify:
                qml.simplify(op(*params, wires=wires, **hyperparams))
            else:
                op(*params, wires=wires, **hyperparams)

            return qml.expval(qml.PauliZ(0))

        unsimplified_op = self.get_unsimplified_op(op)
        params, wires = self._get_params_wires(unsimplified_op)
        hyperparams = {"dim": 2} if unsimplified_op.name == "PCPhase" else {}

        for i in range(params[0].shape[0]):
            parameters = [p[i] for p in params]

            unsimplified_res = circuit(False, wires, *parameters, **hyperparams)
            simplified_res = circuit(True, wires, *parameters, **hyperparams)

            unsimplified_grad = qml.grad(circuit, argnum=list(range(2, 2 + len(parameters))))(
                False,
                wires,
                *parameters,
                **hyperparams,
            )
            simplified_grad = qml.grad(circuit, argnum=list(range(2, 2 + len(parameters))))(
                True,
                wires,
                *parameters,
                **hyperparams,
            )

            assert qml.math.allclose(unsimplified_res, simplified_res)
            assert qml.math.allclose(unsimplified_grad, simplified_grad)

    @pytest.mark.tf
    @pytest.mark.parametrize("op", rotations)
    def test_simplify_rotations_grad_tensorflow(self, op):
        """Test the gradient of an op after simplication for the tensorflow interface"""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(simplify, wires, *params, **hyperparams):
            if simplify:
                qml.simplify(op(*params, wires=wires, **hyperparams))
            else:
                op(*params, wires=wires, **hyperparams)

            return qml.expval(qml.PauliZ(0))

        unsimplified_op = self.get_unsimplified_op(op)
        params, wires = self._get_params_wires(unsimplified_op)
        hyperparams = {"dim": 2} if unsimplified_op.name == "PCPhase" else {}

        for i in range(params[0].shape[0]):
            parameters = [tf.Variable(p[i]) for p in params]

            with tf.GradientTape() as unsimplified_tape:
                unsimplified_res = circuit(False, wires, *parameters, **hyperparams)

            unsimplified_grad = unsimplified_tape.gradient(unsimplified_res, parameters)

            with tf.GradientTape() as simplified_tape:
                simplified_res = circuit(False, wires, *parameters, **hyperparams)

            simplified_grad = simplified_tape.gradient(simplified_res, parameters)

            assert qml.math.allclose(unsimplified_res, simplified_res)
            assert qml.math.allclose(unsimplified_grad, simplified_grad)

    @pytest.mark.tf
    def test_simplify_rotations_grad_tf_function(self):
        """Test the gradient of an op after simplication for the tensorflow interface with
        tf.function"""
        import tensorflow as tf

        op = qml.U2
        wires = list(range(op.num_wires))

        dev = qml.device("default.qubit", wires=2)

        @tf.function
        @qml.qnode(dev)
        def circuit(simplify, *params, **hyperparams):
            if simplify:
                qml.simplify(op(*params, wires=wires, **hyperparams))
            else:
                op(*params, wires=wires, **hyperparams)

            return qml.expval(qml.PauliZ(0))

        unsimplified_op = self.get_unsimplified_op(op)
        params, _ = self._get_params_wires(unsimplified_op)
        hyperparams = {"dim": 2} if unsimplified_op.name == "PCPhase" else {}

        for i in range(params[0].shape[0]):
            parameters = [tf.Variable(p[i]) for p in params]

            with tf.GradientTape() as unsimplified_tape:
                unsimplified_res = circuit(False, *parameters, **hyperparams)

            unsimplified_grad = unsimplified_tape.gradient(unsimplified_res, parameters)

            with tf.GradientTape() as simplified_tape:
                simplified_res = circuit(True, *parameters, **hyperparams)

            simplified_grad = simplified_tape.gradient(simplified_res, parameters)

            assert qml.math.allclose(unsimplified_res, simplified_res)
            assert qml.math.allclose(unsimplified_grad, simplified_grad)

    @pytest.mark.torch
    @pytest.mark.parametrize("op", rotations)
    def test_simplify_rotations_grad_torch(self, op):
        """Test the gradient of an op after simplication for the torch interface"""
        import torch

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(simplify, wires, *params, **hyperparams):
            if simplify:
                qml.simplify(op(*params, wires=wires, **hyperparams))
            else:
                op(*params, wires=wires, **hyperparams)

            return qml.expval(qml.PauliZ(0))

        unsimplified_op = self.get_unsimplified_op(op)
        params, wires = self._get_params_wires(unsimplified_op)
        hyperparams = {"dim": 2} if unsimplified_op.name == "PCPhase" else {}

        for i in range(params[0].shape[0]):
            parameters = [torch.tensor(p[i], requires_grad=True) for p in params]

            unsimplified_res = circuit(False, wires, *parameters, **hyperparams)
            unsimplified_res.backward()
            unsimplified_grad = [p.grad for p in parameters]

            simplified_res = circuit(True, wires, *parameters, **hyperparams)
            simplified_res.backward()
            simplified_grad = [p.grad for p in parameters]

            assert qml.math.allclose(unsimplified_res, simplified_res)
            assert qml.math.allclose(unsimplified_grad, simplified_grad)

    @pytest.mark.jax
    @pytest.mark.parametrize("op", rotations)
    def test_simplify_rotations_grad_jax(self, op):
        """Test the gradient of an op after simplication for the JAX interface"""
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit.jax", wires=2)

        @qml.qnode(dev)
        def circuit(simplify, wires, *params, **hyperparams):
            if simplify:
                qml.simplify(op(*params, wires=wires, **hyperparams))
            else:
                op(*params, wires=wires, **hyperparams)

            return qml.expval(qml.PauliZ(0))

        unsimplified_op = self.get_unsimplified_op(op)
        params, wires = self._get_params_wires(unsimplified_op)
        hyperparams = {"dim": 2} if unsimplified_op.name == "PCPhase" else {}

        for i in range(params[0].shape[0]):
            parameters = [jnp.array(p[i]) for p in params]

            unsimplified_res = circuit(False, wires, *parameters, **hyperparams)
            simplified_res = circuit(True, wires, *parameters, **hyperparams)

            unsimplified_grad = jax.grad(circuit, argnums=list(range(2, 2 + len(parameters))))(
                False, wires, *parameters, **hyperparams
            )
            simplified_grad = jax.grad(circuit, argnums=list(range(2, 2 + len(parameters))))(
                True, wires, *parameters, **hyperparams
            )

            assert qml.math.allclose(unsimplified_res, simplified_res, atol=1e-6)
            assert qml.math.allclose(unsimplified_grad, simplified_grad, atol=1e-6)

    @pytest.mark.jax
    def test_simplify_rotations_grad_jax_jit(self):
        """Test the gradient of an op after simplication for the JAX interface with jitting"""
        import jax
        import jax.numpy as jnp

        op = qml.U2

        dev = qml.device("default.qubit", wires=2)

        wires = 0 if op.num_wires == 1 else [0, 1]

        @jax.jit
        @qml.qnode(dev)
        def simplified_circuit(*params):
            qml.simplify(op(*params, wires=wires))
            return qml.expval(qml.PauliZ(0))

        @jax.jit
        @qml.qnode(dev)
        def unsimplified_circuit(*params):
            op(*params, wires=wires)
            return qml.expval(qml.PauliZ(0))

        unsimplified_op = self.get_unsimplified_op(op)
        params = unsimplified_op.data

        for i in range(params[0].shape[0]):
            parameters = [jnp.array(p[i]) for p in params]

            unsimplified_res = unsimplified_circuit(*parameters)
            simplified_res = simplified_circuit(*parameters)

            unsimplified_grad = jax.grad(
                unsimplified_circuit, argnums=list(range(len(parameters)))
            )(*parameters)
            simplified_grad = jax.grad(simplified_circuit, argnums=list(range(len(parameters))))(
                *parameters
            )

            assert qml.math.allclose(unsimplified_res, simplified_res, atol=1e-6)
            assert qml.math.allclose(unsimplified_grad, simplified_grad, atol=1e-6)

    @pytest.mark.parametrize("op", rotations)
    def test_simplify_to_identity(self, op):
        """Test that the operator correctly simplifies to the identity when the rotation is 0"""
        if op == qml.U2:
            pytest.skip("U2 gate does not simplify to Identity")

        num_wires = op.num_wires if op.num_wires is not qml.operation.AnyWires else 2

        if op == qml.PCPhase:
            unsimplified_op = op(*([0] * op.num_params), dim=2, wires=range(num_wires))
        else:
            unsimplified_op = op(*([0] * op.num_params), wires=range(num_wires))

        simplified_op = qml.simplify(unsimplified_op)

        if op != qml.PSWAP:
            assert isinstance(simplified_op, qml.Identity)
        else:
            # PSWAP reduces to SWAP when the angle is 0
            assert qml.equal(simplified_op, qml.SWAP(wires=[0, 1]))

    def test_simplify_rot(self):
        """Simplify rot operations with different parameters."""

        rot_x = qml.Rot(np.pi / 2, 0.1, -np.pi / 2, wires=0)
        simplify_rot_x = rot_x.simplify()

        assert simplify_rot_x.name == "RX"
        assert simplify_rot_x.data == (0.1,)
        assert np.allclose(simplify_rot_x.matrix(), rot_x.matrix())

        rot_y = qml.Rot(0, 0.1, 0, wires=0)
        simplify_rot_y = rot_y.simplify()

        assert simplify_rot_y.name == "RY"
        assert simplify_rot_y.data == (0.1,)
        assert np.allclose(simplify_rot_y.matrix(), rot_y.matrix())

        rot_z = qml.Rot(0.1, 0, 0.2, wires=0)
        simplify_rot_z = rot_z.simplify()

        assert simplify_rot_z.name == "RZ"
        assert np.allclose(simplify_rot_z.data, [0.3])
        assert np.allclose(simplify_rot_z.matrix(), rot_z.matrix())

        rot_h = qml.Rot(np.pi, np.pi / 2, 0, wires=0)
        simplify_rot_h = rot_h.simplify()

        assert simplify_rot_h.name == "Hadamard"
        assert np.allclose(simplify_rot_h.matrix(), 1.0j * rot_h.matrix())

        rot = qml.Rot(0.1, 0.2, 0.3, wires=0)
        not_simplified_rot = rot.simplify()

        assert not_simplified_rot.name == "Rot"
        assert np.allclose(not_simplified_rot.matrix(), rot.matrix())

    def test_simplify_u2(self):
        """Simplify u2 operations with different parameters."""

        u2_x = qml.U2(-np.pi / 2, np.pi / 2, wires=0)
        simplify_u2_x = u2_x.simplify()

        assert simplify_u2_x.name == "RX"
        assert simplify_u2_x.data == (np.pi / 2,)
        assert np.allclose(simplify_u2_x.matrix(), u2_x.matrix())

        u2_y = qml.U2(-2 * np.pi, 2 * np.pi, wires=0)
        simplify_u2_y = u2_y.simplify()

        assert simplify_u2_y.name == "RY"
        assert simplify_u2_y.data == (np.pi / 2,)
        assert np.allclose(simplify_u2_y.matrix(), u2_y.matrix())

        u2 = qml.U2(0.1, 0.2, wires=0)
        u2_not_simplified = u2.simplify()

        assert u2_not_simplified.name == "U2"
        assert u2_not_simplified.data == (0.1, 0.2)
        assert np.allclose(u2_not_simplified.matrix(), u2.matrix())

    def test_simplify_u3(self):
        """Simplify u3 operations with different parameters."""

        u3_x = qml.U3(0.1, -np.pi / 2, np.pi / 2, wires=0)
        simplify_u3_x = u3_x.simplify()

        assert simplify_u3_x.name == "RX"
        assert simplify_u3_x.data == (0.1,)
        assert np.allclose(simplify_u3_x.matrix(), u3_x.matrix())

        u3_y = qml.U3(0.1, 0.0, 0.0, wires=0)
        simplify_u3_y = u3_y.simplify()

        assert simplify_u3_y.name == "RY"
        assert simplify_u3_y.data == (0.1,)
        assert np.allclose(simplify_u3_y.matrix(), u3_y.matrix())

        u3_z = qml.U3(0.0, 0.1, 0.0, wires=0)
        simplify_u3_z = u3_z.simplify()

        assert simplify_u3_z.name == "PhaseShift"
        assert simplify_u3_z.data == (0.1,)
        assert np.allclose(simplify_u3_z.matrix(), u3_z.matrix())

        u3 = qml.U3(0.1, 0.2, 0.3, wires=0)
        u3_not_simplified = u3.simplify()

        assert u3_not_simplified.name == "U3"
        assert u3_not_simplified.data == (0.1, 0.2, 0.3)
        assert np.allclose(u3_not_simplified.matrix(), u3.matrix())


label_data = [
    (
        qml.Rot(1.23456, 2.3456, 3.45678, wires=0),
        "Rot",
        "Rot\n(1.23,\n2.35,\n3.46)",
        "Rot\n(1,\n2,\n3)",
    ),
    (qml.RX(1.23456, wires=0), "RX", "RX\n(1.23)", "RX\n(1)"),
    (qml.RY(1.23456, wires=0), "RY", "RY\n(1.23)", "RY\n(1)"),
    (qml.RZ(1.23456, wires=0), "RZ", "RZ\n(1.23)", "RZ\n(1)"),
    (qml.MultiRZ(1.23456, wires=0), "MultiRZ", "MultiRZ\n(1.23)", "MultiRZ\n(1)"),
    (
        qml.PauliRot(1.2345, "XYZ", wires=(0, 1, 2)),
        "RXYZ",
        "RXYZ\n(1.23)",
        "RXYZ\n(1)",
    ),
    (
        qml.PhaseShift(1.2345, wires=0),
        "Rϕ",
        "Rϕ\n(1.23)",
        "Rϕ\n(1)",
    ),
    (
        qml.PCPhase(1.23, dim=2, wires=[0, 1]),
        "∏_ϕ",
        "∏_ϕ\n(1.23)",
        "∏_ϕ\n(1)",
    ),
    (
        qml.CPhaseShift00(1.2345, wires=(0, 1)),
        "Rϕ(00)",
        "Rϕ(00)\n(1.23)",
        "Rϕ(00)\n(1)",
    ),
    (
        qml.CPhaseShift01(1.2345, wires=(0, 1)),
        "Rϕ(01)",
        "Rϕ(01)\n(1.23)",
        "Rϕ(01)\n(1)",
    ),
    (
        qml.CPhaseShift10(1.2345, wires=(0, 1)),
        "Rϕ(10)",
        "Rϕ(10)\n(1.23)",
        "Rϕ(10)\n(1)",
    ),
    (qml.U1(1.2345, wires=0), "U1", "U1\n(1.23)", "U1\n(1)"),
    (qml.U2(1.2345, 2.3456, wires=0), "U2", "U2\n(1.23,\n2.35)", "U2\n(1,\n2)"),
    (
        qml.U3(1.2345, 2.345, 3.4567, wires=0),
        "U3",
        "U3\n(1.23,\n2.35,\n3.46)",
        "U3\n(1,\n2,\n3)",
    ),
    (
        qml.IsingXX(1.2345, wires=(0, 1)),
        "IsingXX",
        "IsingXX\n(1.23)",
        "IsingXX\n(1)",
    ),
    (
        qml.IsingYY(1.2345, wires=(0, 1)),
        "IsingYY",
        "IsingYY\n(1.23)",
        "IsingYY\n(1)",
    ),
    (
        qml.IsingZZ(1.2345, wires=(0, 1)),
        "IsingZZ",
        "IsingZZ\n(1.23)",
        "IsingZZ\n(1)",
    ),
    # Controlled operations
    (qml.CRX(1.234, wires=(0, 1)), "RX", "RX\n(1.23)", "RX\n(1)"),
    (qml.CRY(1.234, wires=(0, 1)), "RY", "RY\n(1.23)", "RY\n(1)"),
    (qml.CRZ(1.234, wires=(0, 1)), "RZ", "RZ\n(1.23)", "RZ\n(1)"),
    (
        qml.CRot(1.234, 2.3456, 3.456, wires=(0, 1)),
        "Rot",
        "Rot\n(1.23,\n2.35,\n3.46)",
        "Rot\n(1,\n2,\n3)",
    ),
    (
        qml.ControlledPhaseShift(1.2345, wires=(0, 1)),
        "Rϕ",
        "Rϕ\n(1.23)",
        "Rϕ\n(1)",
    ),
]

# labels with broadcasted parameters are not implemented properly yet, the parameters are truncated
label_data_broadcasted = [
    (qml.RX(np.array([1.23, 4.56]), wires=0), "RX", "RX", "RX"),
    (qml.PauliRot(np.array([1.23, 4.5]), "XYZ", wires=(0, 1, 2)), "RXYZ", "RXYZ", "RXYZ"),
    (qml.PCPhase(np.array([1.23, 4.56]), dim=2, wires=[0, 1]), "∏_ϕ", "∏_ϕ", "∏_ϕ"),
    (
        qml.U3(np.array([0.1, 0.2]), np.array([-0.1, -0.2]), np.array([1.2, -0.1]), wires=0),
        "U3",
        "U3",
        "U3",
    ),
]


class TestLabel:
    """Test the label method on parametric ops"""

    @pytest.mark.parametrize("op, label1, label2, label3", label_data)
    def test_label_method(self, op, label1, label2, label3):
        """Test label method with plain scalers."""

        assert op.label() == label1
        assert op.label(decimals=2) == label2
        assert op.label(decimals=0) == label3

    @pytest.mark.parametrize("op, label1, label2, label3", label_data_broadcasted)
    def test_label_method_broadcasted(self, op, label1, label2, label3):
        """Test label method with plain scalers."""

        assert op.label() == label1
        assert op.label(decimals=2) == label2
        assert op.label(decimals=0) == label3

    @pytest.mark.tf
    def test_label_tf(self):
        """Test label methods work with tensorflow variables"""
        import tensorflow as tf

        op1 = qml.RX(tf.Variable(0.123456), wires=0)
        assert op1.label(decimals=2) == "RX\n(0.12)"

        op2 = qml.CRX(tf.Variable(0.12345), wires=(0, 1))
        assert op2.label(decimals=2) == "RX\n(0.12)"

        op3 = qml.Rot(tf.Variable(0.1), tf.Variable(0.2), tf.Variable(0.3), wires=0)
        assert op3.label(decimals=2) == "Rot\n(0.10,\n0.20,\n0.30)"

    @pytest.mark.torch
    def test_label_torch(self):
        """Test label methods work with torch tensors"""
        import torch

        op1 = qml.RX(torch.tensor(1.23456), wires=0)
        assert op1.label(decimals=2) == "RX\n(1.23)"

        op2 = qml.CRX(torch.tensor(1.23456), wires=(0, 1))
        assert op2.label(decimals=2) == "RX\n(1.23)"

        op3 = qml.Rot(torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.3), wires=0)
        assert op3.label(decimals=2) == "Rot\n(0.10,\n0.20,\n0.30)"

    @pytest.mark.jax
    def test_label_jax(self):
        """Test the label method works with jax"""
        import jax

        op1 = qml.RX(jax.numpy.array(1.23456), wires=0)
        assert op1.label(decimals=2) == "RX\n(1.23)"

        op2 = qml.CRX(jax.numpy.array(1.23456), wires=(0, 1))
        assert op2.label(decimals=2) == "RX\n(1.23)"

        op3 = qml.Rot(jax.numpy.array(0.1), jax.numpy.array(0.2), jax.numpy.array(0.3), wires=0)
        assert op3.label(decimals=2) == "Rot\n(0.10,\n0.20,\n0.30)"

    def test_string_parameter(self):
        """Test labelling works if variable is a string instead of a float."""

        op1 = qml.RX("x", wires=0)
        assert op1.label() == "RX"
        assert op1.label(decimals=0) == "RX\n(x)"

        op2 = qml.CRX("y", wires=(0, 1))
        assert op2.label(decimals=0) == "RX\n(y)"

        op3 = qml.Rot("x", "y", "z", wires=0)
        assert op3.label(decimals=0) == "Rot\n(x,\ny,\nz)"

    def test_string_parameter_broadcasted(self):
        """Test labelling works (i.e. does not raise an Error) if variable is a
        string instead of a float."""

        x = np.array(["x0", "x1", "x2"])
        y = np.array(["y0", "y1", "y2"])
        z = np.array(["z0", "z1", "z2"])

        op1 = qml.RX(x, wires=0)
        assert op1.label() == "RX"
        assert op1.label(decimals=0) == "RX"

        op2 = qml.CRX(y, wires=(0, 1))
        assert op2.label(decimals=0) == "RX"

        op3 = qml.Rot(x, y, z, wires=0)
        assert op3.label(decimals=0) == "Rot"


pow_parametric_ops = (
    qml.RX(1.234, wires=0),
    qml.RY(2.345, wires=0),
    qml.RZ(3.456, wires=0),
    qml.PhaseShift(6.78, wires=0),
    qml.ControlledPhaseShift(0.234, wires=(0, 1)),
    qml.PCPhase(6.78, dim=2, wires=[0, 1]),
    qml.MultiRZ(-0.4432, wires=(0, 1, 2)),
    qml.PauliRot(0.5, "X", wires=0),
    qml.CRX(-6.5432, wires=(0, 1)),
    qml.CRY(-0.543, wires=(0, 1)),
    qml.CRZ(1.234, wires=(0, 1)),
    qml.U1(1.23, wires=0),
    qml.IsingXX(-2.345, wires=(0, 1)),
    qml.IsingYY(3.1652, wires=(0, 1)),
    qml.IsingXY(-1.234, wires=(0, 1)),
    qml.IsingZZ(1.789, wires=("a", "b")),
    qml.GlobalPhase(0.123),
    # broadcasted ops
    qml.RX(np.array([1.234, 4.129]), wires=0),
    qml.RY(np.array([2.345, 6.789]), wires=0),
    qml.RZ(np.array([3.456]), wires=0),
    qml.PhaseShift(np.array([6.0, 7.0, 8.0]), wires=0),
    qml.ControlledPhaseShift(np.array([0.234]), wires=(0, 1)),
    qml.PCPhase(np.array([6.0, 7.0, 8.0]), dim=2, wires=[0, 1]),
    qml.CPhaseShift00(np.array([0.234]), wires=(0, 1)),
    qml.CPhaseShift01(np.array([0.234]), wires=(0, 1)),
    qml.CPhaseShift10(np.array([0.234]), wires=(0, 1)),
    qml.MultiRZ(np.array([-0.4432, -0.231, 0.251]), wires=(0, 1, 2)),
    qml.PauliRot(np.array([0.5, 0.9]), "X", wires=0),
    qml.CRX(np.array([-6.5432, 0.7653]), wires=(0, 1)),
    qml.CRY(np.array([-0.543, 0.21]), wires=(0, 1)),
    qml.CRZ(np.array([1.234, 5.678]), wires=(0, 1)),
    qml.U1(np.array([1.23, 0.241]), wires=0),
    qml.IsingXX(np.array([9.32, -2.345]), wires=(0, 1)),
    qml.IsingYY(np.array([3.1652]), wires=(0, 1)),
    qml.IsingZZ(np.array([1.789, 2.52, 0.211]), wires=("a", "b")),
)


class TestParametricPow:
    @pytest.mark.parametrize("op", pow_parametric_ops)
    @pytest.mark.parametrize("n", (2, -1, 0.2631, -0.987))
    def test_pow_method_parametric_ops(self, op, n):
        """Assert that a matrix raised to a power is the same as
        multiplying the data by n for relevant ops."""
        pow_op = op.pow(n)

        assert len(pow_op) == 1
        assert pow_op[0].__class__ is op.__class__
        assert all((qml.math.allclose(d1, d2 * n) for d1, d2 in zip(pow_op[0].data, op.data)))

    @pytest.mark.parametrize("op", pow_parametric_ops)
    @pytest.mark.parametrize("n", (3, -2))
    def test_pow_matrix(self, op, n):
        """Test that the matrix of an op first raised to a power is the same as the
        matrix raised to the power.  This test only can work for integer powers."""
        op_mat = qml.matrix(op)
        pow_mat = qml.matrix(op.pow(n)[0])

        assert qml.math.allclose(qml.math.linalg.matrix_power(op_mat, n), pow_mat)


# pylint:disable = use-implicit-booleaness-not-comparison
def test_diagonalization_static_global_phase():
    """Test the static compute_diagonalizing_gates method for the GlobalPhase operation."""
    assert qml.GlobalPhase.compute_diagonalizing_gates(0.123, wires=1) == []


@pytest.mark.parametrize("phi", [0.123, np.pi / 4, 0])
@pytest.mark.parametrize("n_wires", [0, 1, 2])
def test_global_phase_compute_sparse_matrix(phi, n_wires):
    """Test that compute_sparse_matrix"""

    sparse_matrix = qml.GlobalPhase.compute_sparse_matrix(phi, n_wires=n_wires)
    expected = np.exp(-1j * phi) * sparse.eye(int(2**n_wires), format="csr")

    assert np.allclose(sparse_matrix.todense(), expected.todense())


def test_decomposition():
    """Test the decomposition of the GlobalPhase operation."""

    assert qml.GlobalPhase.compute_decomposition(1.23) == []
    assert qml.GlobalPhase(1.23).decomposition() == []


control_data = [
    (qml.Rot(1, 2, 3, wires=0), Wires([])),
    (qml.RX(1.23, wires=0), Wires([])),
    (qml.RY(1.23, wires=0), Wires([])),
    (qml.MultiRZ(1.234, wires=(0, 1, 2)), Wires([])),
    (qml.PauliRot(1.234, "IXY", wires=(0, 1, 2)), Wires([])),
    (qml.PhaseShift(1.234, wires=0), Wires([])),
    (qml.U1(1.234, wires=0), Wires([])),
    (qml.U2(1.234, 2.345, wires=0), Wires([])),
    (qml.U3(1.234, 2.345, 3.456, wires=0), Wires([])),
    (qml.IsingXX(1.234, wires=(0, 1)), Wires([])),
    (qml.IsingYY(1.234, wires=(0, 1)), Wires([])),
    (qml.IsingXY(1.234, wires=(0, 1)), Wires([])),
    (qml.IsingYY(np.array([-5.1, 0.219]), wires=(0, 1)), Wires([])),
    (qml.IsingZZ(1.234, wires=(0, 1)), Wires([])),
    (qml.PSWAP(1.234, wires=(0, 1)), Wires([])),
    # Controlled Ops
    (qml.ControlledPhaseShift(1.234, wires=(0, 1)), Wires(0)),
    (qml.CPhaseShift00(1.234, wires=(0, 1)), Wires(0)),
    (qml.CPhaseShift01(1.234, wires=(0, 1)), Wires(0)),
    (qml.CPhaseShift10(1.234, wires=(0, 1)), Wires(0)),
    (qml.CPhase(1.234, wires=(0, 1)), Wires(0)),
    (qml.CRX(1.234, wires=(0, 1)), Wires(0)),
    (qml.CRY(1.234, wires=(0, 1)), Wires(0)),
    (qml.CRZ(np.array([1.234, 0.219]), wires=(0, 1)), Wires(0)),
    (qml.CRot(1.234, 2.2345, 3.456, wires=(0, 1)), Wires(0)),
]


@pytest.mark.parametrize("op, control_wires", control_data)
def test_control_wires(op, control_wires):
    """Test the ``control_wires`` attribute for parametrized operations."""
    assert op.control_wires == control_wires


control_value_data = [
    (qml.CPhaseShift00(1.234, wires=(0, 1)), "0"),
    (qml.CPhaseShift01(1.234, wires=(0, 1)), "0"),
]


@pytest.mark.parametrize("op, control_values", control_value_data)
def test_control_values(op, control_values):
    """Test the ``control_values`` attribute for parametrized operations."""
    assert op.control_values == control_values


def test_op_aliases_are_valid():
    """Tests that ops in new files can still be accessed from the old parametric_ops module."""
    assert qml.ops.qubit.parametric_ops_multi_qubit.MultiRZ is old_loc_MultiRZ
    assert qml.ops.qubit.parametric_ops_single_qubit.RX is old_loc_RX
