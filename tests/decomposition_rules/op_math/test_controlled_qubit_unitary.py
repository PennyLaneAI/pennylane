# Copyright 2026 Xanadu Quantum Technologies Inc.

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
Tests for the controlled QubitUnitary decomposition rules.
"""

import numpy as np
import pytest

import pennylane as qp
from pennylane import math
from pennylane.decomposition_rules import ctrl_decomp_bisect, ctrl_decomp_zyz
from pennylane.decomposition_rules.op_math import controlled_qubit_unitary
from pennylane.decomposition_rules.op_math.controlled_qubit_unitary import (
    _bisect_compute_a,
    _bisect_compute_b,
    _ctrl_decomp_bisect_md,
    _ctrl_decomp_bisect_od,
)

cw5 = tuple(list(range(1, 1 + n)) for n in range(2, 6))


def _matrix_adjoint(matrix: np.ndarray):
    return math.transpose(math.conj(matrix))


def assert_equal_list(lhs, rhs):
    if not isinstance(lhs, list):
        lhs = [lhs]
    if not isinstance(rhs, list):
        rhs = [rhs]
    assert len(lhs) == len(rhs)
    for l, r in zip(lhs, rhs):
        qp.assert_equal(l, r)


class TestControlledDecompositionZYZ:
    """tests for qp.decomposition_rules.ctrl_decomp_zyz"""

    @pytest.mark.unit
    def test_invalid_op_error(self):
        """Tests that an error is raised when an invalid operation is passed"""
        with pytest.raises(
            ValueError, match="The target operation must be a single-qubit operation"
        ):
            _ = ctrl_decomp_zyz(qp.CNOT([0, 1]), [2])

    su2_ops = [
        qp.RX(0.123, wires=0),
        qp.RY(0.123, wires=0),
        qp.RZ(0.123, wires=0),
        qp.Rot(0.123, 0.456, 0.789, wires=0),
    ]

    non_su2_ops = [
        qp.QubitUnitary(
            np.array(
                [
                    [-0.28829348 - 0.78829734j, 0.30364367 + 0.45085995j],
                    [0.53396245 - 0.10177564j, 0.76279558 - 0.35024096j],
                ]
            ),
            wires=0,
        ),
        qp.DiagonalQubitUnitary(np.array([1, -1]), wires=0),
        qp.Hadamard(0),
        qp.PauliZ(0),
        qp.S(0),
        qp.PhaseShift(1.5, wires=0),
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("op", su2_ops + non_su2_ops)
    @pytest.mark.parametrize("control_wires", ([1], [1, 2], [1, 2, 3]))
    def test_decomposition_zyz(self, op, control_wires):
        """Tests that the controlled decomposition of a single-qubit operation is correct."""

        with qp.queuing.AnnotatedQueue() as q:
            decomp = ctrl_decomp_zyz(op, control_wires)

        queued_ops = q.queue
        assert_equal_list(queued_ops, decomp)

        all_wires = control_wires + op.wires
        decomp_matrix = qp.matrix(qp.tape.QuantumScript(decomp), wire_order=all_wires)
        expected_matrix = qp.matrix(qp.ctrl(op, control=control_wires), wire_order=all_wires)

        assert qp.math.allclose(decomp_matrix, expected_matrix)

    @pytest.mark.system
    @pytest.mark.parametrize("control_wires", ([1], [1, 2], [1, 2, 3]))
    def test_decomposition_circuit_gradient(self, control_wires):
        """Tests that the controlled decomposition of a single-qubit operation
        behaves as expected in a quantum circuit"""

        dev = qp.device("default.qubit", wires=4)

        def circuit(p):
            U = qp.Rot.compute_matrix(*p)
            ctrl_decomp_zyz(qp.QubitUnitary(U, wires=[0]), control_wires=control_wires)
            return qp.expval(qp.Z(0))

        circ_ad = qp.QNode(circuit, dev, diff_method="adjoint")
        circ_bp = qp.QNode(circuit, dev, diff_method="backprop")
        par = qp.numpy.array([0.123, 0.234, 0.345])
        jac_ad = qp.jacobian(circ_ad)(par)
        jac_bp = qp.jacobian(circ_bp)(par)

        # different methods must agree
        assert math.allclose(jac_ad, jac_bp)

    @pytest.mark.unit
    def test_trivial_ops_in_decomposition(self):
        """Test that an operator decomposition doesn't have trivial rotations."""

        op = qp.RZ(np.pi, wires=0)
        decomp = ctrl_decomp_zyz(op, [1])
        expected = [
            qp.RZ(np.pi, wires=0),
            # an RY should be omitted here.
            qp.CNOT(wires=[1, 0]),
            # an RY should be omitted here.
            qp.RZ(-np.pi / 2, wires=0),
            qp.CNOT(wires=[1, 0]),
            qp.RZ(-np.pi / 2, wires=0),
        ]
        assert len(decomp) == 5
        assert decomp == expected

    @pytest.mark.parametrize(
        "composite_op, want_decomp",
        [
            (
                qp.ops.Prod(qp.PauliX(0), qp.PauliX(0)),  # type: ignore
                [
                    qp.CNOT(wires=[1, 0]),
                    qp.CNOT(wires=[1, 0]),
                ],
            ),
            (
                qp.s_prod(1j, qp.PauliX(0)),
                [
                    qp.RZ(7 * np.pi / 2, wires=0),
                    qp.RY(np.pi / 2, wires=0),
                    qp.CNOT(wires=[1, 0]),
                    qp.RY(-np.pi / 2, wires=0),
                    qp.RZ(-2 * np.pi, wires=0),
                    qp.CNOT(wires=[1, 0]),
                    qp.RZ(-3 * np.pi / 2, wires=0),
                ],
            ),
            (
                (qp.s_prod(1 / np.sqrt(2), qp.PauliX(0)) + qp.s_prod(1 / np.sqrt(2), qp.PauliX(0))),
                [
                    qp.RZ(np.pi / 2, wires=0),
                    qp.RY(np.pi / 2, wires=0),
                    qp.CNOT(wires=[1, 0]),
                    qp.RY(-np.pi / 2, wires=0),
                    qp.RZ(-2 * np.pi, wires=0),
                    qp.CNOT(wires=[1, 0]),
                    qp.RZ(3 * np.pi / 2, wires=0),
                    qp.ctrl(qp.GlobalPhase(phi=-np.pi / 2), control=1),
                ],
            ),
        ],
    )
    def test_composite_ops(self, composite_op, want_decomp):
        """Test that ZYZ decomposition is used for composite operators."""
        have_decomp = ctrl_decomp_zyz(composite_op, 1)
        for actual, expected in zip(have_decomp, want_decomp, strict=True):
            qp.assert_equal(actual, expected)

    @pytest.mark.torch
    def test_zyz_decomp_with_torch_params(self):
        """Tests that the ZYZ decomposition runs when the target operation parameters
        are of type torch.Tensor"""

        import torch

        target_op1 = qp.RY(torch.Tensor([1.2]), 0)
        target_op2 = qp.RY(np.array([1.2]), 0)

        torch_decomp = ctrl_decomp_zyz(target_op1, 1)
        decomp = ctrl_decomp_zyz(target_op2, 1)

        for op1, op2 in zip(torch_decomp, decomp):
            qp.assert_equal(op1, op2, check_interface=False)


class TestControlledDecompBisect:
    """tests for ctrl_decomp_bisect"""

    su2_od_ops = [
        qp.QubitUnitary(
            np.array(
                [
                    [0, 1],
                    [-1, 0],
                ]
            ),
            wires=0,
        ),
        qp.QubitUnitary(
            np.array(
                [
                    [1, 1],
                    [-1, 1],
                ]
            )
            * 2**-0.5,
            wires=0,
        ),
        qp.QubitUnitary(
            np.array(
                [
                    [1j, 0],
                    [0, -1j],
                ]
            ),
            wires=0,
        ),
        qp.QubitUnitary(
            np.array(
                [
                    [1, 0],
                    [0, 1],
                ]
            ),
            wires=0,
        ),
        qp.QubitUnitary(
            np.array(
                [
                    [-1, 0],
                    [0, -1],
                ]
            ),
            wires=0,
        ),
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("op", su2_od_ops)
    @pytest.mark.parametrize("control_wires", cw5)
    def test_decomposition_matrix_od(self, op, control_wires):
        """Tests that the controlled decomposition produces an equivalent matrix."""

        with qp.queuing.AnnotatedQueue() as q:
            _ctrl_decomp_bisect_od(op.matrix(), control_wires + op.wires)

        all_wires = control_wires + op.wires
        decomp_matrix = qp.matrix(qp.tape.QuantumScript.from_queue(q), wire_order=all_wires)
        expected_matrix = qp.matrix(qp.ctrl(op, control=control_wires), wire_order=all_wires)

        assert qp.math.allclose(decomp_matrix, expected_matrix)

    @pytest.mark.unit
    @pytest.mark.parametrize("op", su2_od_ops)
    def test_decomposed_operators(self, op, tol):
        """Tests that the operators in the decomposition match expectations."""

        control_wires = [1, 2, 3, 4, 5]

        su = op.matrix()
        sx = qp.PauliX.compute_matrix()

        with qp.queuing.AnnotatedQueue() as q:
            _ctrl_decomp_bisect_od(op.matrix(), control_wires + op.wires)
        op_seq = q.queue

        assert len(op_seq) == 8

        mcx1 = qp.MultiControlledX([1, 2, 3, 0], work_wires=[4, 5])
        qp.assert_equal(mcx1, op_seq[0])
        qp.assert_equal(mcx1, op_seq[4])

        mcx2 = qp.MultiControlledX([4, 5, 0], work_wires=[1, 2, 3])
        qp.assert_equal(mcx2, op_seq[2])
        qp.assert_equal(mcx2, op_seq[6])

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

    @pytest.mark.unit
    @pytest.mark.parametrize("op", su2_od_ops)
    def test_a_matrix(self, op, tol):
        """Tests that the A matrix subroutine returns a correct A matrix."""
        su = op.matrix()
        sx = qp.PauliX.compute_matrix()
        a = _bisect_compute_a(su)
        at = _matrix_adjoint(a)
        assert np.allclose(at @ sx @ a @ sx @ at @ sx @ a @ sx, su, atol=tol, rtol=tol)

    su2_md_ops = [
        qp.QubitUnitary(
            np.array(
                [
                    [0, 1j],
                    [1j, 0],
                ]
            ),
            wires=0,
        ),
        qp.QubitUnitary(
            np.array(
                [
                    [0, 1],
                    [-1, 0],
                ]
            ),
            wires=0,
        ),
        qp.QubitUnitary(
            np.array(
                [
                    [1, 1],
                    [-1, 1],
                ]
            )
            * 2**-0.5,
            wires=0,
        ),
        qp.QubitUnitary(
            np.array(
                [
                    [-1, 0],
                    [0, -1],
                ]
            ),
            wires=0,
        ),
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("op", su2_md_ops)
    @pytest.mark.parametrize("control_wires", cw5)
    def test_decomposition_matrix_md(self, op, control_wires):
        """Tests that the controlled decomposition produces an equivalent matrix."""

        with qp.queuing.AnnotatedQueue() as q:
            _ctrl_decomp_bisect_md(op.matrix(), control_wires + op.wires)

        all_wires = control_wires + op.wires
        decomp_matrix = qp.matrix(qp.tape.QuantumScript.from_queue(q), wire_order=all_wires)
        expected_matrix = qp.matrix(qp.ctrl(op, control=control_wires), wire_order=all_wires)

        assert qp.math.allclose(decomp_matrix, expected_matrix)

    @pytest.mark.unit
    @pytest.mark.parametrize("op", su2_md_ops)
    def test_b_matrix(self, op, tol):
        """Tests that the B matrix subroutine returns a correct A matrix."""
        su = op.matrix()
        sx = qp.PauliX.compute_matrix()
        sh = qp.Hadamard.compute_matrix()
        b = _bisect_compute_b(su)
        bt = _matrix_adjoint(b)
        assert np.allclose(sh @ bt @ sx @ b @ sx @ sh, su, atol=tol, rtol=tol)

    @pytest.mark.unit
    def test_bisect_compute_a_jit_compatible(self, tol):
        """Test that _bisect_compute_a works under JAX tracing (jax.jit)."""
        jax = pytest.importorskip("jax")

        @jax.jit
        def f(u):
            return _bisect_compute_a(u)

        # SU(2) matrix with non-zero off-diagonal
        U = jax.numpy.array([[1 + 0j, 1 + 0j], [-1 + 0j, 1 + 0j]]) * 2**-0.5
        result = f(U)
        assert result.shape == (2, 2)
        assert jax.numpy.iscomplexobj(result)

        sx = qp.PauliX.compute_matrix()
        at = _matrix_adjoint(np.array(result))
        a = np.array(result)
        assert np.allclose(at @ sx @ a @ sx @ at @ sx @ a @ sx, np.array(U), atol=tol, rtol=tol)

    @pytest.mark.unit
    def test_bisect_compute_b_jit_compatible(self, tol):
        """Test that _bisect_compute_b works under JAX tracing (jax.jit)."""

        jax = pytest.importorskip("jax")

        @jax.jit
        def f(u):
            return _bisect_compute_b(u)

        # SU(2) matrix with zero main-diagonal
        U = jax.numpy.array([[0 + 0j, 0 + 1j], [0 + 1j, 0 + 0j]])
        result = f(U)
        assert result.shape == (2, 2)
        assert jax.numpy.iscomplexobj(result)

        sx = qp.PauliX.compute_matrix()
        sh = qp.Hadamard.compute_matrix()
        bt = _matrix_adjoint(np.array(result))
        b = np.array(result)
        assert np.allclose(sh @ bt @ sx @ b @ sx @ sh, np.array(U), atol=tol, rtol=tol)

    @pytest.mark.unit
    def test_invalid_op_error(self):
        """Tests that an error is raised when an invalid operation is passed"""
        with pytest.raises(
            ValueError, match="The target operation must be a single-qubit operation"
        ):
            _ = ctrl_decomp_bisect(qp.CNOT([0, 1]), [2])

    su2_gen_ops = [
        qp.QubitUnitary(
            np.array(
                [
                    [0, 1],
                    [-1, 0],
                ]
            ),
            wires=0,
        ),
        qp.QubitUnitary(
            np.array(
                [
                    [0, 1j],
                    [1j, 0],
                ]
            ),
            wires=0,
        ),
        qp.QubitUnitary(
            np.array(
                [
                    [1j, 1j],
                    [1j, -1j],
                ]
            )
            * 2**-0.5,
            wires=0,
        ),
        qp.QubitUnitary(
            np.array(
                [
                    [1, 1],
                    [-1, 1],
                ]
            )
            * 2**-0.5,
            wires=0,
        ),
        qp.QubitUnitary(
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
        qp.PauliX(0),
        qp.PauliY(0),
        qp.PauliZ(0),
        qp.Hadamard(0),
        qp.Rot(0.123, 0.456, 0.789, wires=0),
    ]

    gen_ops_best = [
        "_ctrl_decomp_bisect_md",
        "_ctrl_decomp_bisect_od",
        "_ctrl_decomp_bisect_od",
        "_ctrl_decomp_bisect_general",
        "_ctrl_decomp_bisect_general",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("op", su2_gen_ops + gen_ops)
    @pytest.mark.parametrize("control_wires", cw5)
    def test_decomposition_matrix_general(self, op, control_wires):
        """Tests that the controlled decomposition produces an equivalent matrix."""

        with qp.queuing.AnnotatedQueue() as q:
            decomp = ctrl_decomp_bisect(op, control_wires)

        queued_ops = q.queue
        assert_equal_list(queued_ops, decomp)

        all_wires = control_wires + op.wires
        decomp_matrix = qp.matrix(qp.tape.QuantumScript(decomp), wire_order=all_wires)
        expected_matrix = qp.matrix(qp.ctrl(op, control=control_wires), wire_order=all_wires)

        assert qp.math.allclose(decomp_matrix, expected_matrix)

    @pytest.mark.unit
    @pytest.mark.parametrize("op", list(zip(gen_ops, gen_ops_best)))
    @pytest.mark.parametrize("control_wires", cw5)
    def test_auto_select(self, op, control_wires, mocker):
        """Tests that the correct shortcut is chosen if possible."""

        op, best_rule = op
        spy = mocker.spy(controlled_qubit_unitary, best_rule)
        ctrl_decomp_bisect(op, control_wires)
        spy.assert_called_once()
