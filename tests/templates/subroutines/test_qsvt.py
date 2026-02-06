# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Tests for the QSVT template and qsvt wrapper function.
"""
# pylint: disable=too-many-arguments, import-outside-toplevel, no-self-use
from copy import copy

import pytest
from numpy.linalg import matrix_power
from numpy.polynomial.chebyshev import Chebyshev

import pennylane as qp
from pennylane import numpy as np
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.templates.subroutines.qsvt import (
    _cheby_pol,
    _complementary_poly,
    _poly_func,
    _qsp_iterate,
    _qsp_iterate_broadcast,
    _qsp_optimization,
    _W_of_x,
    _z_rotation,
)


def qfunc(A):
    """Used to test queuing in the next test."""
    return qp.RX(A[0][0], wires=0)


def qfunc2(A):
    """Used to test queuing in the next test."""
    return qp.prod(qp.PauliX(wires=0), qp.RZ(A[0][0], wires=0))


def lst_phis(phis):
    """Used to test queuing in the next test."""
    return [qp.PCPhase(i, 2, wires=[0, 1]) for i in phis]


def generate_polynomial_coeffs(degree, parity=None):
    rng = np.random.default_rng(seed=123)
    if parity is None:
        polynomial_coeffs_in_canonical_basis = rng.normal(size=degree + 1)
        return polynomial_coeffs_in_canonical_basis / np.sum(
            np.abs(polynomial_coeffs_in_canonical_basis)
        )
    if parity == 0:
        assert degree % 2 == 0.0
        polynomial_coeffs_in_canonical_basis = np.zeros(degree + 1)
        polynomial_coeffs_in_canonical_basis[0::2] = rng.normal(size=degree // 2 + 1)
        return polynomial_coeffs_in_canonical_basis / np.sum(
            np.abs(polynomial_coeffs_in_canonical_basis)
        )

    if parity == 1:
        assert degree % 2 == 1.0
        polynomial_coeffs_in_canonical_basis = np.zeros(degree + 1)
        polynomial_coeffs_in_canonical_basis[1::2] = rng.uniform(size=degree // 2 + 1)
        return polynomial_coeffs_in_canonical_basis / np.sum(
            np.abs(polynomial_coeffs_in_canonical_basis)
        )

    raise ValueError(f"parity must be None, 0 or 1 but got {parity}")


class TestQSVT:
    """Test the qp.QSVT template."""

    @pytest.mark.jax
    def test_standard_validity(self):
        """Test standard validity criteria with assert_valid."""
        projectors = [qp.PCPhase(0.2, dim=1, wires=0), qp.PCPhase(0.3, dim=1, wires=0)]
        op = qp.QSVT(qp.PauliX(wires=0), projectors)
        qp.ops.functions.assert_valid(op)

    def test_init_error(self):
        """Test that an error is raised if a non-operation object is passed
        for the block-encoding."""
        with pytest.raises(ValueError, match="Input block encoding must be an Operator"):
            qp.QSVT(1.23, [qp.Identity(wires=0)])

    @pytest.mark.parametrize(
        ("U_A", "lst_projectors", "wires", "operations"),
        [
            (
                qp.BlockEncode([[0.1, 0.2], [0.3, 0.4]], wires=[0, 1]),
                [qp.PCPhase(0.5, dim=2, wires=[0, 1]), qp.PCPhase(0.5, dim=2, wires=[0, 1])],
                [0, 1],
                [
                    qp.PCPhase(0.5, dim=2, wires=[0, 1]),
                    qp.BlockEncode([[0.1, 0.2], [0.3, 0.4]], wires=[0, 1]),
                    qp.PCPhase(0.5, dim=2, wires=[0, 1]),
                ],
            ),
            (
                qp.BlockEncode([[0.3, 0.1], [0.2, 0.9]], wires=[0, 1]),
                [qp.PCPhase(0.5, dim=2, wires=[0, 1]), qp.PCPhase(0.3, dim=2, wires=[0, 1])],
                [0, 1],
                [
                    qp.PCPhase(0.5, dim=2, wires=[0, 1]),
                    qp.BlockEncode([[0.3, 0.1], [0.2, 0.9]], wires=[0, 1]),
                    qp.PCPhase(0.3, dim=2, wires=[0, 1]),
                ],
            ),
            (
                qp.Hadamard(wires=0),
                [qp.RZ(-2 * theta, wires=0) for theta in [1.23, -0.5, 4]],
                [0],
                [
                    qp.RZ(-2.46, wires=0),
                    qp.Hadamard(0),
                    qp.RZ(1, wires=0),
                    qp.Hadamard(0),
                    qp.RZ(-8, wires=0),
                ],
            ),
        ],
    )
    def test_output(self, U_A, lst_projectors, wires, operations):
        """Test that qp.QSVT produces the intended measurements."""
        dev = qp.device("default.qubit", wires=len(wires))

        @qp.qnode(dev)
        def circuit():
            qp.QSVT(U_A, lst_projectors)
            return qp.expval(qp.PauliZ(wires=0))

        @qp.qnode(dev)
        def circuit_correct():
            for op in operations:
                qp.apply(op)
            return qp.expval(qp.PauliZ(wires=0))

        assert np.isclose(circuit(), circuit_correct())

    @pytest.mark.parametrize(
        ("U_A", "lst_projectors", "results"),
        [
            (
                qp.BlockEncode(0.1, wires=0),
                [qp.PCPhase(0.2, dim=1, wires=0), qp.PCPhase(0.3, dim=1, wires=0)],
                [
                    qp.PCPhase(0.2, dim=2, wires=[0]),
                    qp.BlockEncode(np.array([[0.1]]), wires=[0]),
                    qp.PCPhase(0.3, dim=2, wires=[0]),
                ],
            ),
            (
                qp.PauliZ(wires=0),
                [qp.RZ(0.1, wires=0), qp.RY(0.2, wires=0), qp.RZ(0.3, wires=1)],
                [
                    qp.RZ(0.1, wires=[0]),
                    qp.change_op_basis(qp.PauliZ(wires=[0]), qp.RY(0.2, wires=[0])),
                    qp.RZ(0.3, wires=[1]),
                ],
            ),
        ],
    )
    def test_queuing_ops(self, U_A, lst_projectors, results):
        """Test that qp.QSVT queues operations in the correct order."""
        with qp.tape.QuantumTape() as tape:
            qp.QSVT(U_A, lst_projectors)

        for idx, val in enumerate(tape.expand().operations):
            assert val.name == results[idx].name
            assert val.parameters == results[idx].parameters

    @pytest.mark.parametrize(
        ("U_A", "lst_projectors", "results"),
        [
            (
                qp.PauliX(wires=0),
                [qp.PCPhase(0.2, dim=1, wires=0), qp.PCPhase(0.3, dim=1, wires=0)],
                [
                    qp.PCPhase(0.2, dim=1, wires=[0]),
                    qp.PauliX(wires=0),
                    qp.PCPhase(0.3, dim=1, wires=[0]),
                ],
            ),
            (
                qp.PauliZ(wires=0),
                [qp.RZ(0.1, wires=0), qp.RY(0.2, wires=0), qp.RZ(0.3, wires=1)],
                [
                    qp.RZ(0.1, wires=[0]),
                    qp.change_op_basis(qp.PauliZ(wires=[0]), qp.RY(0.2, wires=[0])),
                    qp.RZ(0.3, wires=[1]),
                ],
            ),
        ],
    )
    def test_queuing_ops_defined_in_circuit(self, U_A, lst_projectors, results):
        """Test that qp.QSVT queues operations correctly."""

        with qp.queuing.AnnotatedQueue() as q:
            qp.QSVT(U_A, lst_projectors)

        tape = qp.tape.QuantumScript.from_queue(q)

        with qp.queuing.AnnotatedQueue() as q:
            qp.QSVT.compute_decomposition(UA=U_A, projectors=lst_projectors)

        tape2 = qp.tape.QuantumScript.from_queue(q)

        for expected, val1, val2 in zip(results, tape.expand().operations, tape2.operations):
            qp.assert_equal(expected, val1)
            qp.assert_equal(expected, val2)

    def test_decomposition_queues_its_contents(self):
        """Test that the decomposition method queues the decomposition in the correct order."""
        lst_projectors = [qp.PCPhase(0.2, dim=1, wires=0), qp.PCPhase(0.3, dim=1, wires=0)]
        op = qp.QSVT(qp.PauliX(wires=0), lst_projectors)
        with qp.queuing.AnnotatedQueue() as q:
            decomp = op.decomposition()

        ops, _ = qp.queuing.process_queue(q)
        for op1, op2 in zip(ops, decomp):
            qp.assert_equal(op1, op2)

    @pytest.mark.capture
    @pytest.mark.parametrize(
        ("UA", "projectors"),
        [
            (
                qp.BlockEncode([[0.1, 0.2], [0.3, 0.4]], wires=[0, 1]),
                [qp.PCPhase(0.5, dim=2, wires=[0, 1]), qp.PCPhase(0.5, dim=2, wires=[0, 1])],
            ),
            (
                qp.BlockEncode([[0.3, 0.1], [0.2, 0.9]], wires=[0, 1]),
                [qp.PCPhase(0.5, dim=2, wires=[0, 1]), qp.PCPhase(0.3, dim=2, wires=[0, 1])],
            ),
            (
                qp.Hadamard(wires=0),
                [qp.RZ(-2 * theta, wires=0) for theta in [1.23, -0.5, 4]],
            ),
        ],
    )
    def test_decomposition_new(self, UA, projectors):
        """Test the decomposition of the QSVT template."""
        op = qp.QSVT(UA, projectors)
        for rule in qp.list_decomps(qp.QSVT):
            _test_decomposition_rule(op, rule)

    @pytest.mark.parametrize(
        ("UA", "projectors"),
        [
            (
                qp.BlockEncode([[0.1, 0.2], [0.3, 0.4]], wires=[0, 1]),
                [qp.PCPhase(0.5, dim=2, wires=[0, 1]), qp.PCPhase(0.5, dim=2, wires=[0, 1])],
            ),
            (
                qp.BlockEncode([[0.3, 0.1], [0.2, 0.9]], wires=[0, 1]),
                [qp.PCPhase(0.5, dim=2, wires=[0, 1]), qp.PCPhase(0.3, dim=2, wires=[0, 1])],
            ),
            (
                qp.Hadamard(wires=0),
                [qp.RZ(-2 * theta, wires=0) for theta in [1.23, -0.5, 4]],
            ),
        ],
    )
    def test_decomposition(self, UA, projectors):
        with qp.queuing.AnnotatedQueue() as q:
            qp.QSVT.compute_decomposition(UA=UA, projectors=projectors)
        tape = qp.tape.QuantumScript.from_queue(q)

        # Tests that the decomposition produces the right matrix
        op_matrix = qp.QSVT.compute_matrix(UA=UA, projectors=projectors)
        decomp_matrix = qp.matrix(tape, wire_order=tape.wires)
        assert qp.math.allclose(
            op_matrix, decomp_matrix
        ), "decomposition must produce the same matrix as the operator."

    def test_wire_order(self):
        """Test that the wire order is preserved."""

        op1 = qp.GroverOperator(wires=[0, 3])
        op2 = qp.QFT(wires=[2, 1])
        qsvt_wires = qp.QSVT(op2, [op1]).wires
        assert qsvt_wires == op1.wires + op2.wires

    @pytest.mark.parametrize(
        ("quantum_function", "phi_func", "A", "phis", "results"),
        [
            (
                qfunc,
                lst_phis,
                np.array([[0.1, 0.2], [0.3, 0.4]]),
                np.array([0.2, 0.3]),
                [
                    qp.PCPhase(0.2, dim=2, wires=[0]),
                    qp.RX(0.1, wires=[0]),
                    qp.PCPhase(0.3, dim=2, wires=[0]),
                ],
            ),
            (
                qfunc2,
                lst_phis,
                np.array([[0.1, 0.2], [0.3, 0.4]]),
                np.array([0.1, 0.2]),
                [
                    qp.PCPhase(0.1, dim=2, wires=[0]),
                    qp.prod(qp.PauliX(wires=0), qp.RZ(0.1, wires=0)),
                    qp.PCPhase(0.2, dim=2, wires=[0]),
                ],
            ),
        ],
    )
    def test_queuing_callables(self, quantum_function, phi_func, A, phis, results):
        """Test that qp.QSVT queues operations correctly when a function is called"""

        with qp.tape.QuantumTape() as tape:
            qp.QSVT(quantum_function(A), phi_func(phis))

        for idx, val in enumerate(tape.expand().operations):
            assert val.name == results[idx].name
            assert val.parameters == results[idx].parameters

    @pytest.mark.torch
    @pytest.mark.parametrize(
        ("input_matrix", "poly", "wires"),
        [([[0.1, 0.2], [0.3, 0.4]], [0.1, 0, 0.2], [0, 1])],
    )
    def test_ltorch(self, input_matrix, poly, wires):
        """Test that the qsvt function matrix is correct for torch."""
        import torch

        angles = qp.poly_to_angles(poly, "QSVT")
        default_matrix = qp.matrix(
            qp.qsvt(input_matrix, poly, encoding_wires=wires, block_encoding="embedding")
        )

        input_matrix = torch.tensor(input_matrix, dtype=float)
        angles = torch.tensor(angles, dtype=float)

        op = qp.QSVT(
            qp.BlockEncode(input_matrix, wires),
            [qp.PCPhase(phi, 2, wires) for phi in angles],
        )

        assert np.allclose(qp.matrix(op), default_matrix)
        assert qp.math.get_interface(qp.matrix(op)) == "torch"

    @pytest.mark.jax
    @pytest.mark.parametrize(
        ("input_matrix", "poly", "wires"),
        [([[0.1, 0.2], [0.3, 0.4]], [0.1, 0, 0.2], [0, 1])],
    )
    def test_QSVT_jax(self, input_matrix, poly, wires):
        """Test that the qsvt function matrix is correct for jax."""
        import jax.numpy as jnp

        angles = qp.poly_to_angles(poly, "QSVT")
        default_matrix = qp.matrix(
            qp.qsvt(input_matrix, poly, encoding_wires=wires, block_encoding="embedding")
        )

        input_matrix = jnp.array(input_matrix)
        angles = jnp.array(angles)

        op = qp.QSVT(
            qp.BlockEncode(input_matrix, wires),
            [qp.PCPhase(phi, 2, wires) for phi in angles],
        )

        assert np.allclose(qp.matrix(op), default_matrix)
        assert qp.math.get_interface(qp.matrix(op)) == "jax"

    @pytest.mark.jax
    @pytest.mark.parametrize(
        ("input_matrix", "poly", "wires"),
        [([[0.1, 0.2], [0.3, 0.4]], [0.1, 0, 0.2], [0, 1])],
    )
    def test_QSVT_jax_with_identity(self, input_matrix, poly, wires):
        """Test that applying the identity operation before the qsvt function in
        a JIT context does not affect the matrix for jax.

        The main purpose of this test is to ensure that the types of the block
        encoding and projector-controlled phase shift data in a QSVT instance
        are taken into account when inferring the backend of a QuantumScript.
        """
        import jax

        def identity_and_qsvt(angles):
            qp.Identity(wires=wires[0])
            qp.QSVT(
                qp.BlockEncode(input_matrix, wires=wires),
                [
                    qp.PCPhase(angles[i], dim=len(input_matrix), wires=wires)
                    for i in range(len(angles))
                ],
            )

        @jax.jit
        def get_matrix_with_identity(angles):
            return qp.matrix(identity_and_qsvt, wire_order=wires)(angles)

        angles = qp.poly_to_angles(poly, "QSVT")
        matrix = qp.matrix(qp.qsvt(input_matrix, poly, wires, "embedding"))
        matrix_with_identity = get_matrix_with_identity(angles)

        assert np.allclose(matrix, matrix_with_identity)

    @pytest.mark.tf
    @pytest.mark.parametrize(
        ("input_matrix", "poly", "wires"),
        [([[0.1, 0.2], [0.3, 0.4]], [0.1, 0, 0.2], [0, 1])],
    )
    def test_QSVT_tensorflow(self, input_matrix, poly, wires):
        """Test that the qsvt function matrix is correct for tensorflow."""
        import tensorflow as tf

        angles = qp.poly_to_angles(poly, "QSVT")
        default_matrix = qp.matrix(qp.qsvt(input_matrix, poly, wires, "embedding"))

        input_matrix = tf.Variable(input_matrix)
        angles = tf.Variable(angles)

        op = qp.QSVT(
            qp.BlockEncode(input_matrix, wires),
            [qp.PCPhase(phi, 2, wires) for phi in angles],
        )

        assert np.allclose(qp.matrix(op), default_matrix)
        assert qp.math.get_interface(qp.matrix(op)) == "tensorflow"

    @pytest.mark.parametrize(
        ("A", "phis"),
        [
            (
                [[0.1, 0.2], [0.3, 0.4]],
                [0.1, 0.2, 0.3],
            )
        ],
    )
    def test_QSVT_grad(self, A, phis):
        """Test that qp.grad results are the same as finite difference results"""

        @qp.qnode(qp.device("default.qubit", wires=2))
        def circuit(A, phis):
            qp.QSVT(
                qp.BlockEncode(A, wires=[0, 1]),
                [qp.PCPhase(phi, 2, wires=[0, 1]) for phi in phis],
            )
            return qp.expval(qp.PauliZ(wires=0))

        A = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=complex, requires_grad=True)
        phis = np.array([0.1, 0.2, 0.3], dtype=complex, requires_grad=True)
        y = circuit(A, phis)

        mat_grad_results, phi_grad_results = qp.grad(circuit)(A, phis)

        diff = 1e-8

        manual_mat_results = [
            (circuit(A + np.array([[diff, 0], [0, 0]]), phis) - y) / diff,
            (circuit(A + np.array([[0, diff], [0, 0]]), phis) - y) / diff,
            (circuit(A + np.array([[0, 0], [diff, 0]]), phis) - y) / diff,
            (circuit(A + np.array([[0, 0], [0, diff]]), phis) - y) / diff,
        ]

        for idx, result in enumerate(manual_mat_results):
            assert np.isclose(result, np.real(mat_grad_results.flatten()[idx]), atol=1e-6)

        manual_phi_results = [
            (circuit(A, phis + np.array([diff, 0, 0])) - y) / diff,
            (circuit(A, phis + np.array([0, diff, 0])) - y) / diff,
            (circuit(A, phis + np.array([0, 0, diff])) - y) / diff,
        ]

        for idx, result in enumerate(manual_phi_results):
            assert np.isclose(result, np.real(phi_grad_results[idx]), atol=1e-6)

    def test_label(self):
        """Test that the label method returns the correct string label"""
        op = qp.QSVT(qp.Hadamard(0), [qp.Identity(0)])
        assert op.label() == "QSVT"
        assert op.label(base_label="custom_label") == "custom_label"

    def test_data(self):
        """Test that the data property gets and sets the correct values"""
        op = qp.QSVT(qp.RX(1, wires=0), [qp.RY(2, wires=0), qp.RZ(3, wires=0)])
        assert op.data == (1, 2, 3)
        op.data = [4, 5, 6]
        assert op.data == (4, 5, 6)

    def test_copy(self):
        """Test that a QSVT operator can be copied."""
        orig_op = qp.QSVT(qp.RX(1, wires=0), [qp.RY(2, wires=0), qp.RZ(3, wires=0)])
        copy_op = copy(orig_op)
        qp.assert_equal(orig_op, copy_op)

        # Ensure the (nested) operations are copied instead of aliased.
        assert orig_op is not copy_op
        assert orig_op.hyperparameters["UA"] is not copy_op.hyperparameters["UA"]

        orig_projectors = orig_op.hyperparameters["projectors"]
        copy_projectors = copy_op.hyperparameters["projectors"]
        assert all(p1 is not p2 for p1, p2 in zip(orig_projectors, copy_projectors))


phase_angle_data = (
    (
        [0, 0, 0],
        [3 * np.pi / 4, np.pi / 2, -np.pi / 4],
    ),
    (
        [1.0, 2.0, 3.0, 4.0],
        [1.0 + 3 * np.pi / 4, 2.0 + np.pi / 2, 3.0 + np.pi / 2, 4.0 - np.pi / 4],
    ),
)


class Testqsvt:
    """Test the qp.qsvt function."""

    @pytest.mark.parametrize(
        ("A", "poly", "block_encoding", "encoding_wires"),
        [
            (
                [[0.1, 0.2], [0.2, -0.4]],
                [0.2, 0, 0.3],
                "fable",
                [0, 1, 2],
            ),
            (
                [[0.1, 0.2], [0.2, -0.4]],
                [0.2, 0, 0.3],
                "embedding",
                [0, 1],
            ),
            (
                [[-0.1, 0, 0, 0.1], [0, 0.2, 0, 0], [0, 0, -0.2, -0.2], [0.1, 0, -0.2, -0.1]],
                [0.2, 0, 0.3],
                "embedding",
                [0, 1, 2],
            ),
            (
                [[-0.1, 0, 0, 0.1], [0, 0.2, 0, 0], [0, 0, -0.2, -0.2], [0.1, 0, -0.2, -0.1]],
                [0.2, 0, 0.3],
                "fable",
                [0, 1, 2, 3, 4],
            ),
            (
                0.3,
                [0.2, 0, 0.3],
                "embedding",
                [0],
            ),
        ],
    )
    def test_matrix_input(self, A, poly, encoding_wires, block_encoding):
        """Test that qp.qsvt produces the correct output when A is a matrix."""
        dev = qp.device("default.qubit", wires=encoding_wires)

        @qp.qnode(dev)
        def circuit():
            qp.qsvt(A, poly, encoding_wires, block_encoding)
            return qp.state()

        A_matrix = qp.math.atleast_2d(A)
        # Calculation of the polynomial transformation on the input matrix
        expected = sum(coef * matrix_power(A_matrix, i) for i, coef in enumerate(poly))

        assert np.allclose(qp.matrix(circuit)()[: len(A_matrix), : len(A_matrix)].real, expected)

    @pytest.mark.parametrize(
        ("A", "poly", "block_encoding", "encoding_wires"),
        [
            (
                qp.Z(1) + qp.X(1),
                [0.2, 0, 0.3],
                "prepselprep",
                [0],
            ),
            (
                qp.Z(2) + qp.X(2) - 0.2 * qp.X(3) @ qp.Z(2),
                [0, -0.2, 0, 0.5],
                "prepselprep",
                [0, 1],
            ),
            (
                qp.Z(1) + qp.X(1),
                [0.2, 0, 0.3],
                "qubitization",
                [0],
            ),
            (
                qp.Z(2) + qp.X(2) - 0.2 * qp.X(3) @ qp.Z(2),
                [0, -0.2, 0, 0.5],
                "qubitization",
                [0, 1],
            ),
        ],
    )
    def test_ham_input(self, A, poly, encoding_wires, block_encoding):
        """Test that qp.qsvt produces the correct output when A is a hamiltonian."""

        coeffs = A.terms()[0]
        coeffs /= np.linalg.norm(coeffs, 1)

        A = qp.dot(coeffs, A.terms()[1])
        A_matrix = qp.matrix(A)
        dev = qp.device("default.qubit", wires=encoding_wires + A.wires)

        @qp.qnode(dev)
        def circuit():
            qp.qsvt(A, poly, encoding_wires, block_encoding)
            return qp.state()

        # Calculation of the polynomial transformation on the input matrix
        expected = sum(coef * matrix_power(A_matrix, i) for i, coef in enumerate(poly))

        assert np.allclose(qp.matrix(circuit)()[: len(A_matrix), : len(A_matrix)].real, expected)

    @pytest.mark.parametrize(
        ("A", "poly", "block_encoding", "encoding_wires", "msg_match"),
        [
            (
                [[0.1, 0], [0, -0.1]],
                [0.3, 0, 0.4],
                "prepselprep",
                [0, 1],
                "block_encoding should take",
            ),
            (
                [[1, 0], [0, 1]],
                [0.3, 0, 0.4],
                "fable",
                [0, 1],
                "The subnormalization factor should be lower than 1",
            ),
            (qp.Z(0) - qp.X(0), [0.3, 0, 0.4], "fable", [1], "block_encoding should take"),
            (qp.Z(0) - qp.X(0), [0.3, 0, 0.4], "prepselprep", [0], "Control wires in"),
        ],
    )
    def test_raise_error(
        self, A, poly, block_encoding, encoding_wires, msg_match
    ):  # pylint: disable=too-many-arguments
        """Test that proper errors are raised"""

        with pytest.raises(ValueError, match=msg_match):

            qp.qsvt(A, poly, encoding_wires=encoding_wires, block_encoding=block_encoding)

    @pytest.mark.torch
    def test_qsvt_torch(self):
        """Test that the qsvt function generates the correct matrix with torch."""
        import torch

        poly = [-0.1, 0, 0.2, 0, 0.5]
        A = [[-0.1, 0, 0, 0.1], [0, 0.2, 0, 0], [0, 0, -0.2, -0.2], [0.1, 0, -0.2, -0.1]]

        default_op = qp.qsvt(A, poly, [0, 1, 2], "embedding")
        default_matrix = torch.tensor(qp.matrix(default_op))

        torch_op = qp.qsvt(torch.tensor(A), torch.tensor(poly), [0, 1, 2], "embedding")
        torch_matrix = qp.matrix(torch_op)

        assert qp.math.allclose(default_matrix, torch_matrix, atol=1e-6)
        assert qp.math.get_interface(torch_matrix) == "torch"

    @pytest.mark.jax
    def test_qsvt_jax(self):
        """Test that the qsvt function generates the correct matrix with jax."""
        import jax.numpy as jnp

        poly = [-0.1, 0, 0.2, 0, 0.5]
        A = [[-0.1, 0, 0, 0.1], [0, 0.2, 0, 0], [0, 0, -0.2, -0.2], [0.1, 0, -0.2, -0.1]]

        default_op = qp.qsvt(A, poly, [0, 1, 2], "embedding")
        default_matrix = jnp.array(qp.matrix(default_op))

        jax_op = qp.qsvt(jnp.array(A), jnp.array(poly), [0, 1, 2], "embedding")
        jax_matrix = qp.matrix(jax_op)

        assert qp.math.allclose(default_matrix, jax_matrix, atol=1e-6)
        assert qp.math.get_interface(jax_matrix) == "jax"

    @pytest.mark.tf
    def test_qsvt_tensorflow(self):
        """Test that the qsvt function generates the correct matrix with tensorflow."""
        import tensorflow as tf

        poly = [-0.1, 0, 0.2, 0, 0.5]
        A = [[-0.1, 0, 0, 0.1], [0, 0.2, 0, 0], [0, 0, -0.2, -0.2], [0.1, 0, -0.2, -0.1]]

        default_op = qp.qsvt(A, poly, [0, 1, 2], "embedding")
        default_matrix = qp.matrix(default_op)

        tf_op = qp.qsvt(tf.Variable(A), poly, [0, 1, 2], "embedding")
        tf_matrix = qp.matrix(tf_op)

        assert qp.math.allclose(default_matrix, tf_matrix, atol=1e-6)
        assert qp.math.get_interface(tf_matrix) == "tensorflow"

    @pytest.mark.jax
    def test_qsvt_grad(self):
        """Test that the qsvt function generates the correct output with qp.grad and jax.grad."""
        import jax
        import jax.numpy as jnp

        poly = [-0.1, 0, 0.2, 0, 0.5]
        A = [[-0.1, 0, 0, 0.1], [0, 0.2, 0, 0], [0, 0, -0.2, -0.2], [0.1, 0, -0.2, -0.1]]

        dev = qp.device("default.qubit")

        @qp.qnode(dev)
        def circuit(A):
            qp.qsvt(A, poly, [0, 1, 2], "embedding")
            return qp.expval(qp.Z(0) @ qp.Z(1))

        assert np.allclose(qp.grad(circuit)(np.array(A)), jax.grad(circuit)(jnp.array(A)))
        assert not np.allclose(qp.grad(circuit)(np.array(A)), 0.0)

    @pytest.mark.jax
    def test_qsvt_jit(self):
        """
        Test that the qsvt function works with jax.jit.
        Note that the traceable argument is A.
        """

        import jax
        import jax.numpy as jnp

        poly = [-0.1, 0, 0.2, 0, 0.5]
        A = [[-0.1, 0, 0, 0.1], [0, 0.2, 0, 0], [0, 0, -0.2, -0.2], [0.1, 0, -0.2, -0.1]]

        dev = qp.device("default.qubit")

        @qp.qnode(dev)
        def circuit(A):
            qp.qsvt(A, poly, [0, 1, 2], "embedding")
            return qp.expval(qp.Z(0) @ qp.Z(1))

        not_jitted_output = circuit(jnp.array(A))

        jitted_circuit = jax.jit(circuit)
        jitted_output = jitted_circuit(jnp.array(A))
        assert jnp.allclose(not_jitted_output, jitted_output)


class TestRootFindingSolver:

    @pytest.mark.parametrize(
        "P",
        [
            ([0.1, 0, 0.3, 0, -0.1]),
            ([0, 0.2, 0, 0.3]),
            ([-0.4, 0, 0.4, 0, -0.1, 0, 0.1]),
        ],
    )
    def test_complementary_polynomial(self, P):
        """Checks that |P(z)|^2 + |Q(z)|^2 = 1 in the unit circle"""

        Q = _complementary_poly(P)  # Calculate complementary polynomial Q

        # Define points on the unit circle
        theta_vals = np.linspace(0, 2 * np.pi, 100)
        unit_circle_points = np.exp(1j * theta_vals)

        for z in unit_circle_points:
            P_val = np.polyval(P, z)
            P_magnitude_squared = np.abs(P_val) ** 2

            Q_val = np.polyval(Q, z)
            Q_magnitude_squared = np.abs(Q_val) ** 2

            assert np.isclose(P_magnitude_squared + Q_magnitude_squared, 1, atol=1e-7)

    @pytest.mark.parametrize(
        "angles",
        [
            (generate_polynomial_coeffs(4, None)),
            (generate_polynomial_coeffs(5, None)),
            (generate_polynomial_coeffs(6, None)),
        ],
    )
    def test_transform_angles(self, angles):
        """Test the transform_angles function"""

        new_angles = qp.transform_angles(angles, "QSP", "QSVT")
        assert np.allclose(angles, qp.transform_angles(new_angles, "QSVT", "QSP"))

        new_angles = qp.transform_angles(angles, "QSVT", "QSP")
        assert np.allclose(angles, qp.transform_angles(new_angles, "QSP", "QSVT"))

        with pytest.raises(AssertionError, match="Invalid conversion"):
            _ = qp.transform_angles(angles, "QFT", "QSVT")

    @pytest.mark.parametrize(
        "poly",
        [
            (generate_polynomial_coeffs(4, 0)),
            (generate_polynomial_coeffs(3, 1)),
            (generate_polynomial_coeffs(11, 1)),
            (generate_polynomial_coeffs(100, 0)),
        ],
    )
    @pytest.mark.parametrize(
        "angle_solver",
        [
            ("root-finding"),
            ("iterative"),
        ],
    )
    def test_correctness_QSP_angles_finding(self, poly, angle_solver):
        """Tests that angles generate desired poly"""

        angles = qp.poly_to_angles(list(poly), "QSP", angle_solver=angle_solver)
        rng = np.random.default_rng(123)
        x = rng.uniform(low=-1.0, high=1.0)

        @qp.qnode(qp.device("default.qubit"))
        def circuit_qsp():
            qp.RX(2 * angles[0], wires=0)
            for angle in angles[1:]:
                qp.RZ(-2 * np.arccos(x), wires=0)
                qp.RX(2 * angle, wires=0)

            return qp.state()

        output = qp.matrix(circuit_qsp, wire_order=[0])()[0, 0]
        expected = sum(coef * (x**i) for i, coef in enumerate(poly))
        assert np.isclose(output.real, expected.real)

    @pytest.mark.jax
    @pytest.mark.parametrize(
        "poly",
        [
            (generate_polynomial_coeffs(4, 0)),
            (generate_polynomial_coeffs(3, 1)),
            (generate_polynomial_coeffs(6, 0)),
            (generate_polynomial_coeffs(100, 0)),
        ],
    )
    @pytest.mark.parametrize(
        "angle_solver",
        [
            ("root-finding"),
            ("iterative"),
        ],
    )
    def test_correctness_QSP_angles_finding_with_jax(self, poly, angle_solver):
        """Tests that angles generate desired poly"""

        angles = qp.poly_to_angles(list(poly), "QSP", angle_solver=angle_solver)
        rng = np.random.default_rng(123)
        x = rng.uniform(low=-1.0, high=1.0)

        @qp.qnode(qp.device("default.qubit"))
        def circuit_qsp():
            qp.RX(2 * angles[0], wires=0)
            for angle in angles[1:]:
                qp.RZ(-2 * np.arccos(x), wires=0)
                qp.RX(2 * angle, wires=0)

            return qp.state()

        output = qp.matrix(circuit_qsp, wire_order=[0])()[0, 0]
        expected = sum(coef * (x**i) for i, coef in enumerate(poly))
        assert np.isclose(output.real, expected.real)

    @pytest.mark.parametrize(
        "poly",
        [
            (generate_polynomial_coeffs(4, 0)),
            (generate_polynomial_coeffs(3, 1)),
            (generate_polynomial_coeffs(6, 0)),
            (generate_polynomial_coeffs(100, 0)),
        ],
    )
    @pytest.mark.parametrize(
        "angle_solver",
        [
            ("root-finding"),
            ("iterative"),
        ],
    )
    def test_correctness_QSVT_angles(self, poly, angle_solver):
        """Tests that angles generate desired poly"""

        angles = qp.poly_to_angles(list(poly), "QSVT", angle_solver=angle_solver)
        rng = np.random.default_rng(123)
        x = rng.uniform(low=-1.0, high=1.0)

        block_encoding = qp.RX(-2 * np.arccos(x), wires=0)
        projectors = [qp.PCPhase(angle, dim=1, wires=0) for angle in angles]

        @qp.qnode(qp.device("default.qubit"))
        def circuit_qsvt():
            qp.QSVT(block_encoding, projectors)
            return qp.state()

        output = qp.matrix(circuit_qsvt, wire_order=[0])()[0, 0]
        expected = sum(coef * (x**i) for i, coef in enumerate(poly))
        assert qp.math.isclose(output.real, expected.real)

    @pytest.mark.parametrize(
        ("poly", "routine", "angle_solver", "msg_match"),
        [
            (
                [0.0, 0.1, 0.2],
                "QSVT",
                "root-finding",
                "The polynomial has no definite parity",
            ),
            (
                [0, 0.1j, 0, 0.3, 0, 0.2, 0.0],
                "QSVT",
                "root-finding",
                "Array must not have an imaginary part",
            ),
            (
                [0, 0.1, 0, 0.3, 0, 0.2],
                "QFT",
                "root-finding",
                "Invalid routine",
            ),
            (
                [0, 0.1, 0, 0.3, 0, 0.2],
                "QSVT",
                "Pitagoras",
                "Invalid angle solver",
            ),
            (
                [0, 0.1, 0, 0.3, 0, 0.2],
                "QSP",
                "Pitagoras",
                "Invalid angle solver",
            ),
            (
                [0, 2, 0, 0.3, 0, 0.2],
                "QSP",
                "root-finding",
                "The polynomial must satisfy that |P(x)| ≤ 1",
            ),
            (
                [1],
                "QSP",
                "root-finding",
                "The polynomial must have at least degree 1",
            ),
        ],
    )
    def test_raise_error(self, poly, routine, angle_solver, msg_match):
        """Test that proper errors are raised"""

        with pytest.raises(AssertionError, match=msg_match):
            _ = qp.poly_to_angles(poly, routine, angle_solver)


class TestIterativeSolver:
    @pytest.mark.parametrize(
        "polynomial_coeffs_in_cheby_basis",
        [
            (generate_polynomial_coeffs(10, 0)),
            (generate_polynomial_coeffs(7, 1)),
            (generate_polynomial_coeffs(12, 0)),
        ],
    )
    def test_qsp_on_poly_with_parity(self, polynomial_coeffs_in_cheby_basis):
        """Test that _qsp_optimization returns correct angles"""
        degree = len(polynomial_coeffs_in_cheby_basis) - 1
        parity = degree % 2
        if parity:
            target_polynomial_coeffs = polynomial_coeffs_in_cheby_basis[1::2]
        else:
            target_polynomial_coeffs = polynomial_coeffs_in_cheby_basis[0::2]
        phis, cost_func = _qsp_optimization(degree, target_polynomial_coeffs)

        rng = np.random.default_rng(123)
        x_point = rng.uniform(size=1, low=-1.0, high=1.0)

        x_point = x_point.item()
        # Theorem 4: |\alpha_i-\beta_i|\leq 2\sqrt(cost_func) https://arxiv.org/pdf/2002.11649
        # which \implies |target_poly(x)-approx_poly(x)|\leq 2\sqrt(cost_func) \sum_i |T_i(x)|
        tolerance = (
            np.sum(
                np.array(
                    [
                        2 * np.sqrt(cost_func) * abs(_cheby_pol(degree=2 * i, x=x_point))
                        for i in range(len(target_polynomial_coeffs))
                    ]
                )
            )
            if not parity
            else np.sum(
                np.array(
                    [
                        2 * np.sqrt(cost_func) * abs(_cheby_pol(degree=2 * i + 1, x=x_point))
                        for i in range(len(target_polynomial_coeffs))
                    ]
                )
            )
        )

        assert qp.math.isclose(
            _qsp_iterate_broadcast(phis, x_point, None),
            _poly_func(coeffs=target_polynomial_coeffs, parity=parity, x=x_point),
            atol=tolerance,
        )

    @pytest.mark.parametrize(
        "x, degree",
        [
            (0.27885, 4),
            (0.4831, 32),
            (-0.5535, 13),
            (-0.79500, 11),
        ],
    )
    def test_cheby_pol(self, x, degree):
        """Test internal function _cheby_pol"""
        coeffs = [0.0] * (degree) + [1.0]
        assert np.isclose(_cheby_pol(x, degree), Chebyshev(coeffs)(x))

    @pytest.mark.parametrize(
        "coeffs, parity, x",
        [
            (generate_polynomial_coeffs(100, 0), 0, 0.1),
            (generate_polynomial_coeffs(7, 1), 1, 0.2),
            (generate_polynomial_coeffs(12, 0), 0, 0.3),
        ],
    )
    def test_poly_func(self, coeffs, parity, x):
        """Test internal function _poly_func"""
        val = _poly_func(coeffs=coeffs[parity::2], parity=parity, x=x)
        ref = Chebyshev(coeffs)(x)
        assert np.isclose(val, ref)

    @pytest.mark.parametrize("angle", list([0.1, 0.2, 0.3, 0.4]))
    def test_z_rotation(self, angle):
        """Test internal function _z_rotation"""
        assert np.allclose(_z_rotation(angle, None), qp.RZ.compute_matrix(-2 * angle))

    @pytest.mark.parametrize("phi", [0.1, 0.2, 0.3, 0.4])
    def test_qsp_iterate(self, phi):
        """Test internal function _qsp_iterate"""
        mtx = _qsp_iterate(0.0, phi, None)
        ref = qp.RX.compute_matrix(-2 * np.arccos(phi))
        assert np.allclose(mtx, ref)

    @pytest.mark.jax
    @pytest.mark.parametrize(
        "x",
        list([0.1, 0.2, 0.3, 0.4]),
    )
    @pytest.mark.parametrize("degree", range(2, 6))
    def test_qsp_iterate_broadcast(self, x, degree):
        """Test internal function _qsp_iterate_broadcast"""
        from jax import numpy as jnp

        phis = jnp.array([np.pi / 4] + [0.0] * (degree - 1) + [-np.pi / 4])
        qsp_be = _qsp_iterate_broadcast(phis, x, "jax")
        ref = qp.RX.compute_matrix(-2 * (degree) * np.arccos(x))[0, 0]
        assert jnp.isclose(qsp_be, ref)

    @pytest.mark.parametrize("x", [0.1, 0.2, 0.3, 0.4])
    def test_W_of_x(self, x):
        """Test internal function _W_of_x"""
        mtx = _W_of_x(x, None)
        ref = qp.RX.compute_matrix(-2 * np.arccos(x))
        assert np.allclose(mtx, ref)

    def test_immutable_input(self):
        """Test `poly_to_angles` does not modify the input"""

        poly = [0, 1.0, 0, -1 / 2, 0, 1 / 3, 0]
        poly_copy = poly.copy()
        qp.poly_to_angles(poly, "QSVT")

        assert len(poly) == len(poly_copy)
        assert np.allclose(poly, poly_copy)

    def test_interface_numpy(self):
        """Test `poly_to_angles` works with numpy"""

        poly = [0, 1.0, 0, -1 / 2, 0, 1 / 3, 0]
        angles = qp.poly_to_angles(poly, "QSVT")

        poly_numpy = np.array(poly)
        angles_numpy = qp.poly_to_angles(poly_numpy, "QSVT")

        assert qp.math.allclose(angles, angles_numpy)

    @pytest.mark.jax
    def test_interface_jax(self):
        """Test `poly_to_angles` works with jax"""

        import jax

        poly = [0, 1.0, 0, -1 / 2, 0, 1 / 3, 0]
        angles = qp.poly_to_angles(poly, "QSVT")

        poly_jax = jax.numpy.array(poly)
        angles_jax = qp.poly_to_angles(poly_jax, "QSVT")

        assert qp.math.allclose(angles, angles_jax)

    @pytest.mark.torch
    def test_interface_torch(self):
        """Test `poly_to_angles` works with torch"""

        import torch

        poly = [0, 1.0, 0, -1 / 2, 0, 1 / 3, 0]
        angles = qp.poly_to_angles(poly, "QSVT")

        poly_torch = torch.tensor(poly)
        angles_torch = qp.poly_to_angles(poly_torch, "QSVT")

        assert qp.math.allclose(angles, angles_torch)

    @pytest.mark.tf
    def test_interface_tf(self):
        """Test `poly_to_angles` works with tensorflow"""

        import tensorflow as tf

        poly = [0, 1.0, 0, -1 / 2, 0, 1 / 3, 0]
        angles = qp.poly_to_angles(poly, "QSVT")

        poly_tf = tf.Variable(poly)
        angles_tf = qp.poly_to_angles(poly_tf, "QSVT")

        assert qp.math.allclose(angles, angles_tf)
