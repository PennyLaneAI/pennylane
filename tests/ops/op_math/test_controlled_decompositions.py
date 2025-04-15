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
Tests for the controlled decompositions.
"""

import itertools

import numpy as np
import pytest

import pennylane as qml
from pennylane import math
from pennylane.ops import ctrl_decomp_zyz
from pennylane.ops.op_math.controlled import Controlled, _is_single_qubit_special_unitary
from pennylane.ops.op_math.controlled_decompositions import (
    _bisect_compute_a,
    _bisect_compute_b,
    _convert_to_su2,
    _ctrl_decomp_bisect_general,
    _ctrl_decomp_bisect_md,
    _ctrl_decomp_bisect_od,
    _decompose_mcx_with_many_workers,
    _decompose_mcx_with_one_worker_b95,
    _decompose_mcx_with_one_worker_kg24,
    _decompose_mcx_with_two_workers,
    _decompose_multicontrolled_unitary,
    _decompose_recursive,
    ctrl_decomp_bisect,
    decompose_mcx,
)
from pennylane.wires import Wires

cw5 = tuple(list(range(1, 1 + n)) for n in range(2, 6))


def _matrix_adjoint(matrix: np.ndarray):
    return math.transpose(math.conj(matrix))


def record_from_list(func):
    """
    Decorates a function to
    - not record any operators instantiated during the function
    - record the operators returned by the function, in order
    - not return the return value

    Not returning the return value is intentional, as the contexts where this is used
    (qnode, matrix) only care about recorded operators and not the return value.
    """

    def irecord_from_list(*args, **kwargs):
        with qml.QueuingManager.stop_recording():
            decomposition = func(*args, **kwargs)

        if qml.QueuingManager.recording():
            for iop in decomposition:
                qml.apply(iop)

    return irecord_from_list


def equal_list(lhs, rhs):
    if not isinstance(lhs, list):
        lhs = [lhs]
    if not isinstance(rhs, list):
        rhs = [rhs]
    return len(lhs) == len(rhs) and all(qml.equal(l, r) for l, r in zip(lhs, rhs))


class TestControlledDecompositionZYZ:
    """tests for qml.ops.ctrl_decomp_zyz"""

    def test_invalid_op_error(self):
        """Tests that an error is raised when an invalid operation is passed"""
        with pytest.raises(
            ValueError, match="The target operation must be a single-qubit operation"
        ):
            _ = ctrl_decomp_zyz(qml.CNOT([0, 1]), [2])

    su2_ops = [
        qml.RX(0.123, wires=0),
        qml.RY(0.123, wires=0),
        qml.RZ(0.123, wires=0),
        qml.Rot(0.123, 0.456, 0.789, wires=0),
    ]

    general_unitary_ops = [
        qml.QubitUnitary(
            np.array(
                [
                    [-0.28829348 - 0.78829734j, 0.30364367 + 0.45085995j],
                    [0.53396245 - 0.10177564j, 0.76279558 - 0.35024096j],
                ]
            ),
            wires=0,
        ),
        qml.DiagonalQubitUnitary(np.array([1, -1]), wires=0),
        qml.Hadamard(0),
        qml.PauliZ(0),
        qml.S(0),
        qml.PhaseShift(1.5, wires=0),
    ]

    @pytest.mark.parametrize("op", su2_ops + general_unitary_ops)
    @pytest.mark.parametrize("control_wires", ([1], [2], [3]))
    def test_decomposition_circuit_general_ops(self, op, control_wires, tol):
        """Tests that the controlled decomposition of a single-qubit operation
        behaves as expected in a quantum circuit for general_unitary_ops"""
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def decomp_circuit():
            for wire in control_wires:
                qml.Hadamard(wire)
            ctrl_decomp_zyz(op, Wires(control_wires))
            return qml.probs()

        @qml.qnode(dev)
        def expected_circuit():
            for wire in control_wires:
                qml.Hadamard(wire)
            qml.ctrl(op, control_wires)
            return qml.probs()

        res = decomp_circuit()
        expected = expected_circuit()

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("op", general_unitary_ops)
    @pytest.mark.parametrize("control_wires", ([1, 2], [1, 2, 3]))
    def test_decomposition_circuit_general_ops_error(self, op, control_wires):
        """Tests that the controlled decomposition of a single-qubit operation
        with multiple controlled wires raises a ValueError for general_unitary_ops"""
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def decomp_circuit():
            for wire in control_wires:
                qml.Hadamard(wire)
            ctrl_decomp_zyz(op, Wires(control_wires))
            return qml.probs()

        with pytest.raises(
            ValueError,
            match="The global_phase should be zero",
        ):
            decomp_circuit()

    @pytest.mark.parametrize("control_wires", ([1], [1, 2], [1, 2, 3]))
    def test_decomposition_circuit_gradient(self, control_wires, tol, seed):
        """Tests that the controlled decomposition of a single-qubit operation
        behaves as expected in a quantum circuit"""
        n_qubits = 4
        rng = np.random.default_rng(seed)

        dev = qml.device("default.qubit", wires=n_qubits)
        init_state = rng.random(2**n_qubits) + 1.0j * rng.random(2**n_qubits)
        init_state /= np.sqrt(np.dot(np.conj(init_state), init_state))
        init_state = np.array(init_state)
        target_wires = [0]
        control_values = [True] * len(control_wires)

        def circuit(p):
            qml.StatePrep(init_state, wires=range(n_qubits))
            qml.ctrl(
                qml.Rot(*p, wires=target_wires),
                control_wires,
                control_values=control_values,
            )
            return qml.probs(wires=target_wires)

        circ_ad = qml.QNode(circuit, dev, diff_method="adjoint")
        circ_bp = qml.QNode(circuit, dev, diff_method="backprop")
        par = qml.numpy.array([0.1234, 0.235, 0.5678])
        jac_ad = qml.jacobian(circ_ad)(par)
        jac_bp = qml.jacobian(circ_bp)(par)

        # different methods must agree
        assert jac_ad.size == 2 * 3
        assert np.allclose(jac_ad.shape, [2, 3])
        assert np.allclose(jac_ad.shape, jac_bp.shape)
        assert np.allclose(jac_ad, jac_bp, atol=tol, rtol=0)

    @pytest.mark.parametrize("op", su2_ops)
    @pytest.mark.parametrize("control_wires", ([1], [2], [3]))
    def test_decomposition_matrix(self, op, control_wires, tol):
        """Tests that the matrix representation of the controlled ZYZ decomposition
        of a single-qubit operation is correct"""
        expected_op = qml.ctrl(op, control_wires)
        res = qml.matrix(ctrl_decomp_zyz, wire_order=control_wires + [0])(op, control_wires)
        expected = expected_op.matrix()

        assert np.allclose(expected, res, atol=tol, rtol=0)

    def test_correct_decomp(self):
        """Test that the operations in the decomposition are correct."""
        phi, theta, omega = 0.123, 0.456, 0.789
        op = qml.Rot(phi, theta, omega, wires=0)
        control_wires = [1, 2, 3]
        decomps = ctrl_decomp_zyz(op, Wires(control_wires))

        expected_ops = [
            qml.CRZ(0.123, wires=[3, 0]),
            qml.CRY(0.456 / 2, wires=[3, 0]),
            qml.Toffoli(wires=control_wires[:-1] + [0]),
            qml.CRY(-0.456 / 2, wires=[3, 0]),
            qml.CRZ(-(0.123 + 0.789) / 2, wires=[3, 0]),
            qml.Toffoli(wires=control_wires[:-1] + [0]),
            qml.CRZ((0.789 - 0.123) / 2, wires=[3, 0]),
        ]
        for decomp_op, expected_op in zip(decomps, expected_ops):
            qml.assert_equal(decomp_op, expected_op)
        assert len(decomps) == 7

    @pytest.mark.parametrize("op", su2_ops + general_unitary_ops)
    @pytest.mark.parametrize("control_wires", ([1], [2], [3]))
    def test_decomp_queues_correctly(self, op, control_wires, tol):
        """Test that any incorrect operations aren't queued when using
        ``ctrl_decomp_zyz``."""
        decomp = ctrl_decomp_zyz(op, control_wires=Wires(control_wires))
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def queue_from_list():
            for wire in control_wires:
                qml.Hadamard(wire)
            for o in decomp:
                qml.apply(o)
            return qml.state()

        @qml.qnode(dev)
        def queue_from_qnode():
            for wire in control_wires:
                qml.Hadamard(wire)
            ctrl_decomp_zyz(op, control_wires=Wires(control_wires))
            return qml.state()

        res1 = queue_from_list()
        res2 = queue_from_qnode()
        assert np.allclose(res1, res2, atol=tol, rtol=0)

    def test_trivial_ops_in_decomposition(self):
        """Test that an operator decomposition doesn't have trivial rotations."""
        op = qml.RZ(np.pi, wires=0)
        decomp = ctrl_decomp_zyz(op, [1])
        expected = [
            qml.RZ(np.pi, wires=0),
            qml.CNOT(wires=[1, 0]),
            qml.RZ(-np.pi / 2, wires=0),
            qml.CNOT(wires=[1, 0]),
            qml.RZ(-np.pi / 2, wires=0),
        ]

        assert len(decomp) == 5
        assert decomp == expected

    @pytest.mark.parametrize(
        "composite_op, want_decomp",
        [
            (
                qml.ops.Prod(qml.PauliX(0), qml.PauliX(0)),  # type: ignore
                [
                    qml.CNOT(wires=[1, 0]),
                    qml.CNOT(wires=[1, 0]),
                ],
            ),
            (
                qml.s_prod(1j, qml.PauliX(0)),
                [
                    qml.RZ(7 * np.pi / 2, wires=0),
                    qml.RY(np.pi / 2, wires=0),
                    qml.CNOT(wires=[1, 0]),
                    qml.RY(-np.pi / 2, wires=0),
                    qml.RZ(-2 * np.pi, wires=0),
                    qml.CNOT(wires=[1, 0]),
                    qml.RZ(-3 * np.pi / 2, wires=0),
                ],
            ),
            (
                (
                    qml.s_prod(1 / np.sqrt(2), qml.PauliX(0))
                    + qml.s_prod(1 / np.sqrt(2), qml.PauliX(0))
                ),
                [
                    qml.ctrl(qml.GlobalPhase(phi=-np.pi / 2, wires=0), control=1),
                    qml.RZ(np.pi / 2, wires=0),
                    qml.RY(np.pi / 2, wires=0),
                    qml.CNOT(wires=[1, 0]),
                    qml.RY(-np.pi / 2, wires=0),
                    qml.RZ(-2 * np.pi, wires=0),
                    qml.CNOT(wires=[1, 0]),
                    qml.RZ(3 * np.pi / 2, wires=0),
                ],
            ),
        ],
    )
    def test_composite_ops(self, composite_op, want_decomp):
        """Test that ZYZ decomposition is used for composite operators."""
        have_decomp = ctrl_decomp_zyz(composite_op, 1)
        assert have_decomp == want_decomp

    @pytest.mark.parametrize("test_expand", [False, True])
    def test_zyz_decomp_no_control_values(self, test_expand):
        """Test that the ZYZ decomposition is used for single qubit target operations
        when other decompositions aren't available."""

        base = qml.QubitUnitary(
            np.array(
                [
                    [1, 1],
                    [-1, 1],
                ]
            )
            * 2**-0.5,
            wires="a",
        )
        op = Controlled(base, (0,))

        assert op.has_decomposition
        decomp = (
            qml.tape.QuantumScript(op.decomposition()).expand().circuit
            if test_expand
            else op.decomposition()[0].decomposition()
        )
        expected = qml.ops.ctrl_decomp_zyz(base, (0,))  # pylint:disable=no-member
        assert equal_list(decomp, expected)

    @pytest.mark.parametrize("test_expand", [False, True])
    def test_zyz_decomp_control_values(self, test_expand):
        """Test that the ZYZ decomposition is used for single qubit target operations
        when other decompositions aren't available and control values are present."""
        # pylint:disable=no-member
        base = qml.QubitUnitary(
            np.array(
                [
                    [1, 1],
                    [-1, 1],
                ]
            )
            * 2**-0.5,
            wires="a",
        )
        op = Controlled(base, (0,), control_values=[False])

        assert op.has_decomposition
        decomp = (
            qml.tape.QuantumScript(op.decomposition()).circuit
            if test_expand
            else op.decomposition()
        )
        assert len(decomp) == 3
        qml.assert_equal(qml.PauliX(0), decomp[0])
        qml.assert_equal(qml.PauliX(0), decomp[-1])
        decomp = decomp[1]
        decomp = (
            qml.tape.QuantumScript(decomp.decomposition()).circuit
            if test_expand
            else decomp.decomposition()
        )
        expected = qml.ops.ctrl_decomp_zyz(base, (0,))
        assert equal_list(decomp, expected)

    @pytest.mark.torch
    def test_zyz_decomp_with_torch_params(self):
        """Tests that the ZYZ decomposition runs when the target operation parameters
        are of type torch.Tensor"""
        import torch

        target_op1 = qml.RY(torch.Tensor([1.2]), 0)
        target_op2 = qml.RY(1.2, 0)

        torch_decomp = ctrl_decomp_zyz(target_op1, 1)
        decomp = ctrl_decomp_zyz(target_op2, 1)

        for op1, op2 in zip(torch_decomp, decomp):
            qml.assert_equal(op1, op2, check_interface=False)


class TestControlledBisectOD:
    """tests for qml.ops._ctrl_decomp_bisect_od"""

    def test_invalid_op_error(self):
        """Tests that an error is raised when an invalid operation is passed"""
        with pytest.raises(ValueError, match="Target operation's matrix must have real"):
            _ = _ctrl_decomp_bisect_od(_convert_to_su2(qml.Hadamard.compute_matrix()), 0, [1, 2])

    su2_od_ops = [
        qml.QubitUnitary(
            np.array(
                [
                    [0, 1],
                    [-1, 0],
                ]
            ),
            wires=0,
        ),
        qml.QubitUnitary(
            np.array(
                [
                    [1, 1],
                    [-1, 1],
                ]
            )
            * 2**-0.5,
            wires=0,
        ),
        qml.QubitUnitary(
            np.array(
                [
                    [1j, 0],
                    [0, -1j],
                ]
            ),
            wires=0,
        ),
        qml.QubitUnitary(
            np.array(
                [
                    [1, 0],
                    [0, 1],
                ]
            ),
            wires=0,
        ),
        qml.QubitUnitary(
            np.array(
                [
                    [-1, 0],
                    [0, -1],
                ]
            ),
            wires=0,
        ),
    ]

    od_ops = [
        qml.PauliZ(0),
    ]

    @pytest.mark.parametrize("op", su2_od_ops + od_ops)
    @pytest.mark.parametrize("control_wires", cw5)
    def test_decomposition_circuit(self, op, control_wires, tol):
        """Tests that the controlled decomposition of a single-qubit operation
        behaves as expected in a quantum circuit"""
        dev = qml.device("default.qubit", wires=max(control_wires) + 1)

        @qml.qnode(dev)
        def decomp_circuit():
            for wire in control_wires:
                qml.Hadamard(wire)
            record_from_list(_ctrl_decomp_bisect_od)(
                _convert_to_su2(op.matrix()), op.wires, Wires(control_wires)
            )
            return qml.probs()

        @qml.qnode(dev)
        def expected_circuit():
            for wire in control_wires:
                qml.Hadamard(wire)
            qml.ctrl(op, control_wires)
            return qml.probs()

        res = decomp_circuit()
        expected = expected_circuit()
        assert np.allclose(res, expected, atol=tol, rtol=tol)

    @pytest.mark.parametrize("op", su2_od_ops)
    @pytest.mark.parametrize("control_wires", cw5)
    def test_decomposition_matrix(self, op, control_wires, tol):
        """Tests that the matrix representation of the controlled decomposition
        of a single-qubit operation is correct"""
        assert np.allclose(op.matrix(), _convert_to_su2(op.matrix()), atol=tol, rtol=tol)

        expected_op = qml.ctrl(op, control_wires)
        res = qml.matrix(record_from_list(_ctrl_decomp_bisect_od), wire_order=control_wires + [0])(
            op.matrix(), op.wires, Wires(control_wires)
        )
        expected = expected_op.matrix()

        assert np.allclose(res, expected, atol=tol, rtol=tol)

    @pytest.mark.parametrize("op", su2_od_ops)
    def test_decomposed_operators(self, op, tol):
        """Tests that the operators in the decomposition match expectations."""
        control_wires = [1, 2, 3, 4, 5]

        su = op.matrix()
        sx = qml.PauliX.compute_matrix()
        op_seq = _ctrl_decomp_bisect_od(op.matrix(), op.wires, Wires(control_wires))

        assert len(op_seq) == 8

        mcx1 = qml.MultiControlledX(wires=Wires([1, 2, 3, 0]), work_wires=Wires([4, 5]))
        qml.assert_equal(mcx1, op_seq[0])
        qml.assert_equal(mcx1, op_seq[4])

        mcx2 = qml.Toffoli(wires=[4, 5, 0])
        qml.assert_equal(mcx2, op_seq[2])
        qml.assert_equal(mcx2, op_seq[6])

        a = op_seq[1].matrix()
        at = op_seq[3].matrix()
        a2 = op_seq[5].matrix()
        at2 = op_seq[7].matrix()
        assert np.array_equal(a, a2)
        assert np.array_equal(at, at2)

        i2 = np.identity(2)
        assert np.allclose(a @ at, i2, atol=tol, rtol=tol)
        assert np.allclose(at @ a, i2, atol=tol, rtol=tol)

        assert np.allclose(at @ sx @ a @ sx @ at @ sx @ a @ sx, su, atol=tol, rtol=tol)

    @pytest.mark.parametrize("op", su2_od_ops)
    def test_a_matrix(self, op, tol):
        """Tests that the A matrix subroutine returns a correct A matrix."""
        su = op.matrix()
        sx = qml.PauliX.compute_matrix()
        a = _bisect_compute_a(su)
        at = _matrix_adjoint(a)
        assert np.allclose(at @ sx @ a @ sx @ at @ sx @ a @ sx, su, atol=tol, rtol=tol)


class TestControlledBisectMD:
    """tests for qml.ops._ctrl_decomp_bisect_md"""

    def test_invalid_op_error(self):
        """Tests that an error is raised when an invalid operation is passed"""
        with pytest.raises(ValueError, match="Target operation's matrix must have real"):
            _ = _ctrl_decomp_bisect_md(_convert_to_su2(qml.Hadamard.compute_matrix()), 0, [1, 2])

    su2_md_ops = [
        qml.QubitUnitary(
            np.array(
                [
                    [0, 1j],
                    [1j, 0],
                ]
            ),
            wires=0,
        ),
        qml.QubitUnitary(
            np.array(
                [
                    [0, 1],
                    [-1, 0],
                ]
            ),
            wires=0,
        ),
        qml.QubitUnitary(
            np.array(
                [
                    [1, 1],
                    [-1, 1],
                ]
            )
            * 2**-0.5,
            wires=0,
        ),
        qml.QubitUnitary(
            np.array(
                [
                    [-1, 0],
                    [0, -1],
                ]
            ),
            wires=0,
        ),
    ]

    md_ops = [
        qml.QubitUnitary(
            np.array(
                [
                    [0, 1],
                    [1, 0],
                ]
            ),
            wires=0,
        ),
        qml.QubitUnitary(
            np.array(
                [
                    [1j, 1j],
                    [-1j, 1j],
                ]
            )
            * 2**-0.5,
            wires=0,
        ),
    ]

    @pytest.mark.parametrize("op", su2_md_ops + md_ops)
    @pytest.mark.parametrize("control_wires", cw5)
    def test_decomposition_circuit(self, op, control_wires, tol):
        """Tests that the controlled decomposition of a single-qubit operation
        behaves as expected in a quantum circuit"""
        dev = qml.device("default.qubit", wires=max(control_wires) + 1)

        @qml.qnode(dev)
        def decomp_circuit():
            for wire in control_wires:
                qml.Hadamard(wire)
            record_from_list(_ctrl_decomp_bisect_md)(
                _convert_to_su2(op.matrix()), op.wires, Wires(control_wires)
            )
            return qml.probs()

        @qml.qnode(dev)
        def expected_circuit():
            for wire in control_wires:
                qml.Hadamard(wire)
            qml.ctrl(op, control_wires)
            return qml.probs()

        res = decomp_circuit()
        expected = expected_circuit()
        assert np.allclose(res, expected, atol=tol, rtol=tol)

    @pytest.mark.parametrize("op", su2_md_ops)
    @pytest.mark.parametrize("control_wires", cw5)
    def test_decomposition_matrix(self, op, control_wires, tol):
        """Tests that the matrix representation of the controlled decomposition
        of a single-qubit operation is correct"""
        assert np.allclose(op.matrix(), _convert_to_su2(op.matrix()), atol=tol, rtol=tol)

        expected_op = qml.ctrl(op, control_wires)
        res = qml.matrix(record_from_list(_ctrl_decomp_bisect_md), wire_order=control_wires + [0])(
            op.matrix(), op.wires, control_wires
        )
        expected = expected_op.matrix()

        assert np.allclose(res, expected, atol=tol, rtol=tol)

    @pytest.mark.parametrize("op", su2_md_ops)
    def test_b_matrix(self, op, tol):
        """Tests that the B matrix subroutine returns a correct A matrix."""
        su = op.matrix()
        sx = qml.PauliX.compute_matrix()
        sh = qml.Hadamard.compute_matrix()
        b = _bisect_compute_b(su)
        bt = _matrix_adjoint(b)
        assert np.allclose(sh @ bt @ sx @ b @ sx @ sh, su, atol=tol, rtol=tol)


class TestControlledBisectGeneral:
    """tests for qml.ops._ctrl_decomp_bisect_general"""

    def test_invalid_op_error(self):
        """Tests that an error is raised when an invalid operation is passed"""
        with pytest.raises(
            ValueError, match="The target operation must be a single-qubit operation"
        ):
            _ = ctrl_decomp_bisect(qml.CNOT([0, 1]), [2])

    su2_gen_ops = [
        qml.QubitUnitary(
            np.array(
                [
                    [0, 1],
                    [-1, 0],
                ]
            ),
            wires=0,
        ),
        qml.QubitUnitary(
            np.array(
                [
                    [0, 1j],
                    [1j, 0],
                ]
            ),
            wires=0,
        ),
        qml.QubitUnitary(
            np.array(
                [
                    [1j, 1j],
                    [1j, -1j],
                ]
            )
            * 2**-0.5,
            wires=0,
        ),
        qml.QubitUnitary(
            np.array(
                [
                    [1, 1],
                    [-1, 1],
                ]
            )
            * 2**-0.5,
            wires=0,
        ),
        qml.QubitUnitary(
            np.array(
                [
                    [1 + 2j, -3 + 4j],
                    [3 + 4j, 1 - 2j],
                ]
            )
            * 30**-0.5,
            wires=0,
        ),
    ]

    gen_ops = [
        qml.PauliX(0),
        qml.PauliY(0),
        qml.PauliZ(0),
        qml.Hadamard(0),
        qml.Rot(0.123, 0.456, 0.789, wires=0),
    ]

    gen_ops_best = [
        _ctrl_decomp_bisect_md,
        _ctrl_decomp_bisect_od,
        _ctrl_decomp_bisect_od,
        _ctrl_decomp_bisect_general,
        _ctrl_decomp_bisect_general,
    ]

    @pytest.mark.parametrize("op", su2_gen_ops + gen_ops)
    @pytest.mark.parametrize("control_wires", cw5)
    @pytest.mark.parametrize("auto", [False, True])
    def test_decomposition_circuit(self, op, control_wires, auto, tol):
        """Tests that the controlled decomposition of a single-qubit operation
        behaves as expected in a quantum circuit"""
        dev = qml.device("default.qubit", wires=max(control_wires) + 1)

        @qml.qnode(dev)
        def decomp_circuit():
            for wire in control_wires:
                qml.Hadamard(wire)
            if auto:
                ctrl_decomp_bisect(op, Wires(control_wires))
            else:
                record_from_list(_ctrl_decomp_bisect_general)(
                    _convert_to_su2(op.matrix()), op.wires, Wires(control_wires)
                )
            return qml.probs()

        @qml.qnode(dev)
        def expected_circuit():
            for wire in control_wires:
                qml.Hadamard(wire)
            qml.ctrl(op, control_wires)
            return qml.probs()

        res = decomp_circuit()
        expected = expected_circuit()
        assert np.allclose(res, expected, atol=tol, rtol=tol)

    @pytest.mark.parametrize("op", zip(gen_ops, gen_ops_best))
    @pytest.mark.parametrize("control_wires", cw5)
    @pytest.mark.parametrize("all_the_way_from_ctrl", [False, True])
    def test_auto_select(self, op, control_wires, all_the_way_from_ctrl):
        """
        Test that the auto selection is correct and optimal.
        """
        op, best = op
        if all_the_way_from_ctrl:
            if not isinstance(op, qml.Rot):
                pytest.xfail(reason="decompositions do not match")
            if isinstance(op, qml.PauliX):
                # X has its own special case
                pytest.skip()
            res = qml.ctrl(op, control_wires).decomposition()
        else:
            res = ctrl_decomp_bisect(op, Wires(control_wires))
        expected = best(_convert_to_su2(op.matrix()), op.wires, Wires(control_wires))
        assert equal_list(res, expected)

    @pytest.mark.parametrize("op", su2_gen_ops)
    @pytest.mark.parametrize("control_wires", cw5)
    def test_decomposition_matrix(self, op, control_wires, tol):
        """Tests that the matrix representation of the controlled decomposition
        of a single-qubit operation is correct"""
        assert np.allclose(op.matrix(), _convert_to_su2(op.matrix()), atol=tol, rtol=tol)

        expected_op = qml.ctrl(op, control_wires)
        res = qml.matrix(
            record_from_list(_ctrl_decomp_bisect_general),
            wire_order=control_wires + [0],
        )(op.matrix(), op.wires, control_wires)
        expected = expected_op.matrix()

        assert np.allclose(res, expected, atol=tol, rtol=tol)


class TestMultiControlledUnitary:
    """tests for qml.ops._ops_math.controlled_decompositions._decompose_multicontrolled_unitary"""

    def test_invalid_op_size_error(self):
        """Tests that an error is raised when op acts on more than one wire"""
        with pytest.raises(
            ValueError, match="The target operation must be a single-qubit operation"
        ):
            _ = _decompose_multicontrolled_unitary(qml.CNOT([0, 1]), [2])

    def test_invalid_op_matrix(self):
        """Tests that an error is raised when op does not define a matrix"""

        # pylint: disable=too-few-public-methods
        class MyOp(qml.operation.Operator):
            num_wires = 1

        with pytest.raises(
            ValueError, match="The target operation must be a single-qubit operation"
        ):
            _ = _decompose_multicontrolled_unitary(MyOp, [1])

    su2_gen_ops = [
        qml.QubitUnitary(
            np.array(
                [
                    [0, 1],
                    [-1, 0],
                ]
            ),
            wires=0,
        ),
        qml.QubitUnitary(
            np.array(
                [
                    [0, 1j],
                    [1j, 0],
                ]
            ),
            wires=0,
        ),
        qml.QubitUnitary(
            np.array(
                [
                    [1j, 1j],
                    [1j, -1j],
                ]
            )
            * 2**-0.5,
            wires=0,
        ),
        qml.QubitUnitary(
            np.array(
                [
                    [1, 1],
                    [-1, 1],
                ]
            )
            * 2**-0.5,
            wires=0,
        ),
        qml.QubitUnitary(
            np.array(
                [
                    [1 + 2j, -3 + 4j],
                    [3 + 4j, 1 - 2j],
                ]
            )
            * 30**-0.5,
            wires=0,
        ),
    ]

    gen_ops = [
        qml.PauliX(0),
        qml.PauliZ(0),
        qml.Hadamard(0),
        qml.Rot(0.123, 0.456, 0.789, wires=0),
    ]

    @pytest.mark.parametrize("op", gen_ops + su2_gen_ops)
    @pytest.mark.parametrize("control_wires", cw5)
    def test_decomposition_circuit(self, op, control_wires, tol):
        """Tests that the controlled decomposition of a single-qubit operation
        behaves as expected in a quantum circuit"""
        dev = qml.device("default.qubit", wires=max(control_wires) + 1)

        @qml.qnode(dev)
        def decomp_circuit():
            for wire in control_wires:
                qml.Hadamard(wire)
            _decompose_multicontrolled_unitary(op, Wires(control_wires))
            return qml.probs()

        @qml.qnode(dev)
        def expected_circuit():
            for wire in control_wires:
                qml.Hadamard(wire)
            qml.ctrl(op, control_wires)
            return qml.probs()

        res = decomp_circuit()
        expected = expected_circuit()
        assert np.allclose(res, expected, atol=tol, rtol=tol)

    controlled_wires = tuple(list(range(1, 1 + n)) for n in range(0, 2))

    @pytest.mark.parametrize("op", gen_ops + su2_gen_ops)
    @pytest.mark.parametrize("control_wires", controlled_wires)
    def test_auto_select_wires(self, op, control_wires):
        """
        Test that the auto selection is correct and optimal.
        """
        if len(control_wires) == 0:
            expected = [op]
        else:  # we only have zero or one control wires
            expected = ctrl_decomp_zyz(op, Wires(control_wires))

        res = _decompose_multicontrolled_unitary(op, Wires(control_wires))
        assert equal_list(res, expected)

    @pytest.mark.parametrize(
        "op, controlled_wires, work_wires",
        [
            (qml.RX(0.123, wires=1), [0, 2], [3, 4, 5]),
            (qml.Rot(0.123, 0.456, 0.789, wires=0), [1, 2, 3], [4, 5]),
        ],
    )
    def test_with_many_workers(self, op, controlled_wires, work_wires):
        """Tests ctrl_decomp_zyz with multiple workers"""

        dev = qml.device("default.qubit", wires=6)

        @qml.qnode(dev)
        def decomp_circuit(op):
            ctrl_decomp_zyz(op, controlled_wires, work_wires=work_wires)
            return qml.probs()

        @qml.qnode(dev)
        def expected_circuit(op):
            qml.ctrl(op, controlled_wires, work_wires=work_wires)
            return qml.probs()

        assert np.allclose(decomp_circuit(op), expected_circuit(op))

    controlled_wires = tuple(list(range(2, 1 + n)) for n in range(3, 7))

    @pytest.mark.parametrize("op", gen_ops + su2_gen_ops)
    @pytest.mark.parametrize("control_wires", controlled_wires)
    def test_auto_select_su2(self, op, control_wires):
        """
        Test that the auto selection is correct and optimal.
        """
        if _is_single_qubit_special_unitary(op):
            expected = ctrl_decomp_bisect(op, Wires(control_wires))
        else:
            target_wire = op.wires
            expected = _decompose_recursive(op, 1.0, Wires(control_wires), target_wire, Wires([]))

        res = _decompose_multicontrolled_unitary(op, Wires(control_wires))
        assert equal_list(res, expected)

    @pytest.mark.parametrize("op", gen_ops + su2_gen_ops)
    @pytest.mark.parametrize("control_wires", cw5)
    def test_decomposition_matrix(self, op, control_wires, tol):
        """Tests that the matrix representation of the controlled decomposition
        of a single-qubit operation is correct"""

        expected_op = qml.ctrl(op, control_wires)
        res = qml.matrix(
            record_from_list(_decompose_multicontrolled_unitary),
            wire_order=control_wires + [0],
        )(op, Wires(control_wires))
        expected = expected_op.matrix()

        assert np.allclose(res, expected, atol=tol, rtol=tol)


class TestControlledUnitaryRecursive:
    """tests for qml.ops._decompose_recursive"""

    gen_ops = [
        qml.PauliX(0),
        qml.PauliZ(0),
        qml.Hadamard(0),
    ]
    controlled_wires = tuple(list(range(1, 1 + n)) for n in range(1, 6))

    @pytest.mark.parametrize("op", gen_ops)
    @pytest.mark.parametrize("control_wires", controlled_wires)
    def test_decomposition_circuit(self, op, control_wires, tol):
        """Tests that the controlled decomposition of a single-qubit operation
        behaves as expected in a quantum circuit"""
        dev = qml.device("default.qubit", wires=max(control_wires) + 1)

        @qml.qnode(dev)
        def decomp_circuit():
            for wire in control_wires:
                qml.Hadamard(wire)
            record_from_list(_decompose_recursive)(
                op, 1.0, Wires(control_wires), op.wires, Wires([])
            )
            return qml.probs()

        @qml.qnode(dev)
        def expected_circuit():
            for wire in control_wires:
                qml.Hadamard(wire)
            qml.ctrl(op, control_wires)
            return qml.probs()

        res = decomp_circuit()
        expected = expected_circuit()
        assert np.allclose(res, expected, atol=tol, rtol=tol)

    @pytest.mark.parametrize("op", gen_ops)
    @pytest.mark.parametrize("control_wires", controlled_wires)
    def test_decomposition_matrix(self, op, control_wires, tol):
        """Tests that the matrix representation of the controlled decomposition
        of a single-qubit operation is correct"""

        expected_op = qml.ctrl(op, control_wires)
        res = qml.matrix(record_from_list(_decompose_recursive), wire_order=control_wires + [0])(
            op, 1.0, Wires(control_wires), op.wires, Wires([])
        )
        expected = expected_op.matrix()

        assert np.allclose(res, expected, atol=tol, rtol=tol)


class TestMCXDecomposition:

    def test_wrong_work_wire_type(self):
        """Test that an error is raised if the work wire type is not 'clean' or 'dirty'."""

        # pylint: disable=protected-access
        control_wires = [0, 1]
        target_wire = 2

        # one worker:
        work_wires = 3
        with pytest.raises(ValueError, match="work_wire_type must be either 'clean' or 'dirty'"):
            qml.MultiControlledX(
                wires=control_wires + [target_wire],
                work_wires=work_wires,
                work_wire_type="blah",
            )

        with pytest.raises(ValueError, match="work_wire_type must be either 'clean' or 'dirty'"):
            qml.MultiControlledX.compute_decomposition(
                wires=control_wires + [target_wire],
                work_wires=work_wires,
                work_wire_type="blah",
            )

    @pytest.mark.parametrize("n_ctrl_wires", range(3, 6))
    def test_decomposition_with_many_workers(self, n_ctrl_wires):
        """Test that the decomposed MultiControlledX gate performs the same unitary as the
        matrix-based version by checking if U^dagger U applies the identity to each basis
        state. This test focuses on the case where there are many work wires."""
        # pylint: disable=protected-access
        control_wires = range(n_ctrl_wires)
        target_wire = n_ctrl_wires
        work_wires = range(n_ctrl_wires + 1, 2 * n_ctrl_wires + 1)

        dev = qml.device("default.qubit", wires=2 * n_ctrl_wires + 1)

        with qml.queuing.AnnotatedQueue() as q:
            _decompose_mcx_with_many_workers(control_wires, target_wire, work_wires)
        tape = qml.tape.QuantumScript.from_queue(q)
        assert all(isinstance(op, qml.Toffoli) for op in tape.operations)

        @qml.qnode(dev)
        def f(bitstring):
            qml.BasisState(bitstring, wires=range(n_ctrl_wires + 1))
            qml.MultiControlledX(wires=list(control_wires) + [target_wire])
            for op in tape.operations:
                op.queue()
            return qml.probs(wires=range(n_ctrl_wires + 1))

        u = np.array(
            [f(np.array(b)) for b in itertools.product(range(2), repeat=n_ctrl_wires + 1)]
        ).T
        assert np.allclose(u, np.eye(2 ** (n_ctrl_wires + 1)))

    @pytest.mark.parametrize("n_ctrl_wires", range(3, 6))
    def test_decomposition_with_one_worker_b95(self, n_ctrl_wires):
        """Test that the decomposed MultiControlledX gate performs the same unitary as the
        matrix-based version by checking if U^dagger U applies the identity to each basis
        state. This test focuses on the case where there is one work wire."""

        # pylint: disable=protected-access
        control_wires = Wires(range(n_ctrl_wires))
        target_wire = n_ctrl_wires
        work_wires = n_ctrl_wires + 1

        dev = qml.device("default.qubit", wires=n_ctrl_wires + 2)

        with qml.queuing.AnnotatedQueue() as q:
            _decompose_mcx_with_one_worker_b95(control_wires, target_wire, work_wires)
        tape = qml.tape.QuantumScript.from_queue(q)
        tape = tape.expand(depth=1)

        @qml.qnode(dev)
        def f(bitstring):
            qml.BasisState(bitstring, wires=range(n_ctrl_wires + 1))
            qml.MultiControlledX(wires=list(control_wires) + [target_wire])
            for op in tape.operations:
                op.queue()
            return qml.probs(wires=range(n_ctrl_wires + 1))

        u = np.array(
            [f(np.array(b)) for b in itertools.product(range(2), repeat=n_ctrl_wires + 1)]
        ).T
        assert np.allclose(u, np.eye(2 ** (n_ctrl_wires + 1)))

    @pytest.mark.parametrize("n_ctrl_wires", range(3, 10))
    def test_decomposition_with_one_clean_worker_kg24(self, n_ctrl_wires):
        """Test that the decomposed MCX gate using 1 clean ancilla performs the same unitary as the
        matrix-based version by checking if U^dagger U applies the identity to each basis
        state. This test focuses on the case where there is one work wire."""

        # pylint: disable=protected-access
        control_wires = Wires(range(n_ctrl_wires))
        target_wire = n_ctrl_wires
        work_wires = n_ctrl_wires + 1

        dev = qml.device("default.qubit", wires=n_ctrl_wires + 2)

        @qml.qnode(dev)
        def f(bitstring):
            qml.BasisState(bitstring, wires=range(n_ctrl_wires + 1))
            qml.MultiControlledX(wires=list(control_wires) + [target_wire])
            record_from_list(_decompose_mcx_with_one_worker_kg24)(
                control_wires, target_wire, work_wires, work_wire_type="clean"
            )
            return qml.probs(wires=range(n_ctrl_wires + 1))

        u = np.array(
            [f(np.array(b)) for b in itertools.product(range(2), repeat=n_ctrl_wires + 1)]
        ).T
        assert np.allclose(u, np.eye(2 ** (n_ctrl_wires + 1)))

    @pytest.mark.parametrize("n_ctrl_wires", range(3, 10))
    def test_decomposition_with_one_dirty_worker_kg24(self, n_ctrl_wires):
        """Test that the decomposed MCX gate using 1 dirty ancilla performs the same unitary as the
        matrix-based version by checking if U^dagger U applies the identity to each basis
        state. This test focuses on the case where there is one work wire."""

        # pylint: disable=protected-access
        control_wires = Wires(range(n_ctrl_wires))
        target_wire = n_ctrl_wires
        work_wires = n_ctrl_wires + 1

        dev = qml.device("default.qubit", wires=n_ctrl_wires + 2)

        @qml.qnode(dev)
        def f(bitstring):
            qml.BasisState(bitstring, wires=range(n_ctrl_wires + 2))
            qml.MultiControlledX(wires=list(control_wires) + [target_wire])
            record_from_list(_decompose_mcx_with_one_worker_kg24)(
                control_wires, target_wire, work_wires, work_wire_type="dirty"
            )
            return qml.probs(wires=range(n_ctrl_wires + 2))

        u = np.array(
            [f(np.array(b)) for b in itertools.product(range(2), repeat=n_ctrl_wires + 2)]
        ).T
        assert np.allclose(u, np.eye(2 ** (n_ctrl_wires + 2)))

    @pytest.mark.parametrize("n_ctrl_wires", range(3, 10))
    def test_decomposition_with_two_clean_workers(self, n_ctrl_wires):
        """Test that the decomposed MCX gate using 2 clean ancillae performs the same unitary as the
        matrix-based version by checking if U^dagger U applies the identity to each basis
        state. This test focuses on the case where there is one work wire."""

        # pylint: disable=protected-access
        control_wires = Wires(range(n_ctrl_wires))
        target_wire = n_ctrl_wires
        work_wires = Wires([n_ctrl_wires + 1, n_ctrl_wires + 2])

        dev = qml.device("default.qubit", wires=n_ctrl_wires + 3)

        @qml.qnode(dev)
        def f(bitstring):
            qml.BasisState(bitstring, wires=range(n_ctrl_wires + 1))
            qml.MultiControlledX(wires=list(control_wires) + [target_wire])
            record_from_list(_decompose_mcx_with_two_workers)(
                control_wires, target_wire, work_wires, work_wire_type="clean"
            )
            return qml.probs(wires=range(n_ctrl_wires + 1))

        u = np.array(
            [f(np.array(b)) for b in itertools.product(range(2), repeat=n_ctrl_wires + 1)]
        ).T
        assert np.allclose(u, np.eye(2 ** (n_ctrl_wires + 1)))

    @pytest.mark.parametrize("n_ctrl_wires", range(3, 10))
    def test_decomposition_with_two_dirty_workers(self, n_ctrl_wires):
        """Test that the decomposed MCX gate using 2 dirty ancillae performs the same unitary as the
        matrix-based version by checking if U^dagger U applies the identity to each basis
        state. This test focuses on the case where there is one work wire."""

        # pylint: disable=protected-access
        control_wires = Wires(range(n_ctrl_wires))
        target_wire = n_ctrl_wires
        work_wires = Wires([n_ctrl_wires + 1, n_ctrl_wires + 2])

        dev = qml.device("default.qubit", wires=n_ctrl_wires + 3)

        @qml.qnode(dev)
        def f(bitstring):
            qml.BasisState(bitstring, wires=range(n_ctrl_wires + 3))
            qml.MultiControlledX(wires=list(control_wires) + [target_wire])
            record_from_list(_decompose_mcx_with_two_workers)(
                control_wires, target_wire, work_wires, work_wire_type="dirty"
            )
            return qml.probs(wires=range(n_ctrl_wires + 3))

        u = np.array(
            [f(np.array(b)) for b in itertools.product(range(2), repeat=n_ctrl_wires + 3)]
        ).T
        assert np.allclose(u, np.eye(2 ** (n_ctrl_wires + 3)))

    @pytest.mark.parametrize("n_ctrl_wires", range(3, 8))
    def test_decomposition_with_no_workers(self, n_ctrl_wires):
        """Test that the decomposed MultiControlledX gate performs the same unitary as the
        matrix-based version by checking if U^dagger U applies the identity to each basis
        state. This test focuses on the case where there is no work wires."""

        # pylint: disable=protected-access
        control_wires = Wires(range(n_ctrl_wires))
        target_wire = n_ctrl_wires

        dev = qml.device("default.qubit", wires=n_ctrl_wires + 1)

        @qml.qnode(dev)
        def f(bitstring):
            qml.BasisState(bitstring, wires=range(n_ctrl_wires + 1))
            qml.MultiControlledX(wires=list(control_wires) + [target_wire])
            record_from_list(decompose_mcx)(control_wires, Wires(target_wire), work_wires=Wires([]))
            return qml.probs(wires=range(n_ctrl_wires + 1))

        u = np.array(
            [f(np.array(b)) for b in itertools.product(range(2), repeat=n_ctrl_wires + 1)]
        ).T
        assert np.allclose(u, np.eye(2 ** (n_ctrl_wires + 1)))

    @pytest.mark.parametrize("work_wire_type", ["clean", "dirty"])
    @pytest.mark.parametrize("n_ctrl_wires", range(3, 10))
    def test_integration_multi_controlled_x(self, n_ctrl_wires, work_wire_type):
        """Test that the new decompositions are integrated with the operation."""

        # pylint: disable=protected-access
        control_wires = list(range(n_ctrl_wires))
        target_wire = n_ctrl_wires

        # one worker:
        work_wires = n_ctrl_wires + 1
        op = qml.MultiControlledX(
            wires=control_wires + [target_wire],
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )
        computed_decomp = op.decomposition()

        if n_ctrl_wires > 3:
            expected_decomp = _decompose_mcx_with_one_worker_kg24(
                Wires(control_wires),
                target_wire,
                work_wires,
                work_wire_type,
            )
        else:
            expected_decomp = _decompose_mcx_with_many_workers(
                Wires(control_wires),
                target_wire,
                Wires(work_wires),
            )

        assert computed_decomp == expected_decomp

        # two worker:
        work_wires = [n_ctrl_wires + 1, n_ctrl_wires + 2]
        op = qml.MultiControlledX(
            wires=control_wires + [target_wire],
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )
        computed_decomp = op.decomposition()

        if n_ctrl_wires > 4:
            expected_decomp = _decompose_mcx_with_two_workers(
                Wires(control_wires),
                target_wire,
                Wires(work_wires),
                work_wire_type,
            )
        else:
            expected_decomp = _decompose_mcx_with_many_workers(
                Wires(control_wires),
                target_wire,
                Wires(work_wires),
            )

        assert computed_decomp == expected_decomp

    def test_private_mcx_decomposition_raises_error(self):
        """Test that an error is raised if not enough work wires are provided"""

        # pylint: disable=protected-access
        control_wires = Wires(range(5))
        target_wire = 5
        work_wires = Wires([6])

        with pytest.raises(ValueError, match="At least 2 work wires are needed"):
            _ = _decompose_mcx_with_two_workers(
                control_wires, target_wire, work_wires, work_wire_type="clean"
            )


def test_ControlledQubitUnitary_has_decomposition_correct():
    """Test that ControlledQubitUnitary reports has_decomposition=False if it is False"""
    U = qml.Toffoli(wires=[0, 1, 2]).matrix()
    op = qml.ControlledQubitUnitary(U, wires=[0, 1, 2, 3])

    assert not op.has_decomposition
    with pytest.raises(qml.operation.DecompositionUndefinedError):
        op.decomposition()


def test_ControlledQubitUnitary_has_decomposition_super_False(mocker):
    """Test that has_decomposition returns False if super() returns False"""
    spy = mocker.spy(qml.QueuingManager, "stop_recording")
    op = qml.ControlledQubitUnitary(np.diag((1.0,) * 8), wires=[0, 1, 2, 3, 4])
    assert not op.has_decomposition
    spy.assert_not_called()
