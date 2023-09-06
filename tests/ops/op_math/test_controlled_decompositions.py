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
import pytest

import numpy as np
import pennylane as qml
from pennylane.ops import ctrl_decomp_zyz
from pennylane.wires import Wires
from pennylane.ops.op_math.controlled_decompositions import (
    _ctrl_decomp_bisect_od,
    _ctrl_decomp_bisect_md,
    _ctrl_decomp_bisect_general,
    ctrl_decomp_bisect,
    _convert_to_su2,
    _bisect_compute_a,
    _bisect_compute_b,
)
from pennylane.ops.op_math.controlled import (
    Controlled,
)
from pennylane import math

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

    unitary_ops = [
        qml.Hadamard(0),
        qml.PauliZ(0),
        qml.S(0),
        qml.PhaseShift(1.5, wires=0),
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
    ]

    @pytest.mark.parametrize("op", su2_ops + unitary_ops)
    @pytest.mark.parametrize("control_wires", ([1], [1, 2], [1, 2, 3]))
    def test_decomposition_circuit(self, op, control_wires, tol):
        """Tests that the controlled decomposition of a single-qubit operation
        behaves as expected in a quantum circuit"""
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def decomp_circuit():
            qml.broadcast(unitary=qml.Hadamard, pattern="single", wires=control_wires)
            ctrl_decomp_zyz(op, Wires(control_wires))
            return qml.probs()

        @qml.qnode(dev)
        def expected_circuit():
            qml.broadcast(unitary=qml.Hadamard, pattern="single", wires=control_wires)
            qml.ctrl(op, control_wires)
            return qml.probs()

        res = decomp_circuit()
        expected = expected_circuit()

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("op", su2_ops)
    @pytest.mark.parametrize("control_wires", ([1], [1, 2], [1, 2, 3]))
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
            qml.RZ(0.123, wires=0),
            qml.RY(0.456 / 2, wires=0),
            qml.MultiControlledX(wires=control_wires + [0]),
            qml.RY(-0.456 / 2, wires=0),
            qml.RZ(-(0.123 + 0.789) / 2, wires=0),
            qml.MultiControlledX(wires=control_wires + [0]),
            qml.RZ((0.789 - 0.123) / 2, wires=0),
        ]
        assert all(
            qml.equal(decomp_op, expected_op)
            for decomp_op, expected_op in zip(decomps, expected_ops)
        )
        assert len(decomps) == 7

    @pytest.mark.parametrize("op", su2_ops + unitary_ops)
    @pytest.mark.parametrize("control_wires", ([1], [1, 2], [1, 2, 3]))
    def test_decomp_queues_correctly(self, op, control_wires, tol):
        """Test that any incorrect operations aren't queued when using
        ``ctrl_decomp_zyz``."""
        decomp = ctrl_decomp_zyz(op, control_wires=Wires(control_wires))
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def queue_from_list():
            qml.broadcast(unitary=qml.Hadamard, pattern="single", wires=control_wires)
            for o in decomp:
                qml.apply(o)
            return qml.state()

        @qml.qnode(dev)
        def queue_from_qnode():
            qml.broadcast(unitary=qml.Hadamard, pattern="single", wires=control_wires)
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
            qml.MultiControlledX(wires=[1, 0]),
            qml.RZ(-np.pi / 2, wires=0),
            qml.MultiControlledX(wires=[1, 0]),
            qml.RZ(-np.pi / 2, wires=0),
        ]

        assert len(decomp) == 5
        assert all(qml.equal(o, e) for o, e in zip(decomp, expected))

    @pytest.mark.xfail
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
            * 2 ** -0.5,
            wires="a",
        )
        op = Controlled(base, (0,))

        assert op.has_decomposition
        decomp = (
            op.expand().expand().circuit if test_expand else op.decomposition()[0].decomposition()
        )
        expected = qml.ops.ctrl_decomp_zyz(base, (0,))  # pylint:disable=no-member
        assert equal_list(decomp, expected)

    @pytest.mark.xfail
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
            * 2 ** -0.5,
            wires="a",
        )
        op = Controlled(base, (0,), control_values=[False])

        assert op.has_decomposition
        decomp = op.expand().circuit if test_expand else op.decomposition()
        assert len(decomp) == 3
        assert qml.equal(qml.PauliX(0), decomp[0])
        assert qml.equal(qml.PauliX(0), decomp[-1])
        decomp = decomp[1]
        decomp = decomp.expand().circuit if test_expand else decomp.decomposition()
        expected = qml.ops.ctrl_decomp_zyz(base, (0,))
        assert equal_list(decomp, expected)


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
            * 2 ** -0.5,
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
            qml.broadcast(unitary=qml.Hadamard, pattern="single", wires=control_wires)
            record_from_list(_ctrl_decomp_bisect_od)(
                _convert_to_su2(op.matrix()), op.wires, Wires(control_wires)
            )
            return qml.probs()

        @qml.qnode(dev)
        def expected_circuit():
            qml.broadcast(unitary=qml.Hadamard, pattern="single", wires=control_wires)
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

        mcx1 = qml.MultiControlledX(
            control_wires=Wires([1, 2, 3]), wires=Wires(0), work_wires=Wires([4, 5])
        )
        assert qml.equal(mcx1, op_seq[0])
        assert qml.equal(mcx1, op_seq[4])

        mcx2 = qml.MultiControlledX(
            control_wires=Wires([4, 5]), wires=Wires(0), work_wires=Wires([1, 2, 3])
        )
        assert qml.equal(mcx2, op_seq[2])
        assert qml.equal(mcx2, op_seq[6])

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
            * 2 ** -0.5,
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
            * 2 ** -0.5,
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
            qml.broadcast(unitary=qml.Hadamard, pattern="single", wires=control_wires)
            record_from_list(_ctrl_decomp_bisect_md)(
                _convert_to_su2(op.matrix()), op.wires, Wires(control_wires)
            )
            return qml.probs()

        @qml.qnode(dev)
        def expected_circuit():
            qml.broadcast(unitary=qml.Hadamard, pattern="single", wires=control_wires)
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
            * 2 ** -0.5,
            wires=0,
        ),
        qml.QubitUnitary(
            np.array(
                [
                    [1, 1],
                    [-1, 1],
                ]
            )
            * 2 ** -0.5,
            wires=0,
        ),
        qml.QubitUnitary(
            np.array(
                [
                    [1 + 2j, -3 + 4j],
                    [3 + 4j, 1 - 2j],
                ]
            )
            * 30 ** -0.5,
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
            qml.broadcast(unitary=qml.Hadamard, pattern="single", wires=control_wires)
            if auto:
                ctrl_decomp_bisect(op, Wires(control_wires))
            else:
                record_from_list(_ctrl_decomp_bisect_general)(
                    _convert_to_su2(op.matrix()), op.wires, Wires(control_wires)
                )
            return qml.probs()

        @qml.qnode(dev)
        def expected_circuit():
            qml.broadcast(unitary=qml.Hadamard, pattern="single", wires=control_wires)
            qml.ctrl(op, control_wires)
            return qml.probs()

        res = decomp_circuit()
        expected = expected_circuit()
        assert np.allclose(res, expected, atol=tol, rtol=tol)

    @pytest.mark.xfail
    @pytest.mark.parametrize("op", zip(gen_ops, gen_ops_best))
    @pytest.mark.parametrize("control_wires", cw5)
    @pytest.mark.parametrize("all_the_way_from_ctrl", [False, True])
    def test_auto_select(self, op, control_wires, all_the_way_from_ctrl):
        """
        Test that the auto selection is correct and optimal.
        """
        op, best = op
        if all_the_way_from_ctrl:
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
            record_from_list(_ctrl_decomp_bisect_general), wire_order=control_wires + [0]
        )(op.matrix(), op.wires, control_wires)
        expected = expected_op.matrix()

        assert np.allclose(res, expected, atol=tol, rtol=tol)


def test_ControlledQubitUnitary_has_decomposition_correct():
    """Test that ControlledQubitUnitary reports has_decomposition=False if it is False"""
    U = qml.Toffoli(wires=[0, 1, 2]).matrix()
    op = qml.ControlledQubitUnitary(U, wires=[1, 2, 3], control_wires=[0])

    assert not op.has_decomposition
    with pytest.raises(qml.operation.DecompositionUndefinedError):
        op.decomposition()


def test_ControlledQubitUnitary_has_decomposition_super_False(mocker):
    """Test that has_decomposition returns False if super() returns False"""
    spy = mocker.spy(qml.QueuingManager, "stop_recording")
    op = qml.ControlledQubitUnitary(np.diag((1.0,) * 8), wires=[2, 3, 4], control_wires=[0, 1])
    assert not op.has_decomposition
    spy.assert_not_called()
