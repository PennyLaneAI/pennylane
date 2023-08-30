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
import pytest
import pennylane as qml
from pennylane import numpy as np


def qfunc(A):
    """Used to test queuing in the next test."""
    return qml.RX(A[0][0], wires=0)


def qfunc2(A):
    """Used to test queuing in the next test."""
    return qml.prod(qml.PauliX(wires=0), qml.RZ(A[0][0], wires=0))


def lst_phis(phis):
    """Used to test queuing in the next test."""
    return [qml.PCPhase(i, 2, wires=[0, 1]) for i in phis]


class TestQSVT:
    """Test the qml.QSVT template."""

    # pylint: disable=protected-access
    def test_flatten_unflatten(self):
        projectors = [qml.PCPhase(0.2, dim=1, wires=0), qml.PCPhase(0.3, dim=1, wires=0)]
        op = qml.QSVT(qml.PauliX(wires=0), projectors)
        data, metadata = op._flatten()
        assert qml.equal(data[0], qml.PauliX(0))
        assert len(data[1]) == len(projectors)
        assert all(qml.equal(op1, op2) for op1, op2 in zip(data[1], projectors))

        assert metadata == tuple()

        new_op = type(op)._unflatten(*op._flatten())
        assert qml.equal(op, new_op)
        assert op is not new_op

    def test_init_error(self):
        """Test that an error is raised if a non-operation object is passed
        for the block-encoding."""
        with pytest.raises(ValueError, match="Input block encoding must be an Operator"):
            qml.QSVT(1.23, [qml.Identity(wires=0)])

    @pytest.mark.parametrize(
        ("U_A", "lst_projectors", "wires", "operations"),
        [
            (
                qml.BlockEncode([[0.1, 0.2], [0.3, 0.4]], wires=[0, 1]),
                [qml.PCPhase(0.5, dim=2, wires=[0, 1]), qml.PCPhase(0.5, dim=2, wires=[0, 1])],
                [0, 1],
                [
                    qml.PCPhase(0.5, dim=2, wires=[0, 1]),
                    qml.BlockEncode([[0.1, 0.2], [0.3, 0.4]], wires=[0, 1]),
                    qml.PCPhase(0.5, dim=2, wires=[0, 1]),
                ],
            ),
            (
                qml.BlockEncode([[0.3, 0.1], [0.2, 0.9]], wires=[0, 1]),
                [qml.PCPhase(0.5, dim=2, wires=[0, 1]), qml.PCPhase(0.3, dim=2, wires=[0, 1])],
                [0, 1],
                [
                    qml.PCPhase(0.5, dim=2, wires=[0, 1]),
                    qml.BlockEncode([[0.3, 0.1], [0.2, 0.9]], wires=[0, 1]),
                    qml.PCPhase(0.3, dim=2, wires=[0, 1]),
                ],
            ),
            (
                qml.Hadamard(wires=0),
                [qml.RZ(-2 * theta, wires=0) for theta in [1.23, -0.5, 4]],
                [0],
                [
                    qml.RZ(-2.46, wires=0),
                    qml.Hadamard(0),
                    qml.RZ(1, wires=0),
                    qml.Hadamard(0),
                    qml.RZ(-8, wires=0),
                ],
            ),
        ],
    )
    def test_output(self, U_A, lst_projectors, wires, operations):
        """Test that qml.QSVT produces the intended measurements."""
        dev = qml.device("default.qubit", wires=len(wires))

        @qml.qnode(dev)
        def circuit():
            qml.QSVT(U_A, lst_projectors)
            return qml.expval(qml.PauliZ(wires=0))

        @qml.qnode(dev)
        def circuit_correct():
            for op in operations:
                qml.apply(op)
            return qml.expval(qml.PauliZ(wires=0))

        assert np.isclose(circuit(), circuit_correct())

    @pytest.mark.parametrize(
        ("U_A", "lst_projectors", "results"),
        [
            (
                qml.BlockEncode(0.1, wires=0),
                [qml.PCPhase(0.2, dim=1, wires=0), qml.PCPhase(0.3, dim=1, wires=0)],
                [
                    qml.PCPhase(0.2, dim=2, wires=[0]),
                    qml.BlockEncode(np.array([[0.1]]), wires=[0]),
                    qml.PCPhase(0.3, dim=2, wires=[0]),
                ],
            ),
            (
                qml.PauliZ(wires=0),
                [qml.RZ(0.1, wires=0), qml.RY(0.2, wires=0), qml.RZ(0.3, wires=1)],
                [
                    qml.RZ(0.1, wires=[0]),
                    qml.PauliZ(wires=[0]),
                    qml.RY(0.2, wires=[0]),
                    qml.adjoint(qml.PauliZ(wires=[0])),
                    qml.RZ(0.3, wires=[1]),
                ],
            ),
        ],
    )
    def test_queuing_ops(self, U_A, lst_projectors, results):
        """Test that qml.QSVT queues operations in the correct order."""
        with qml.tape.QuantumTape() as tape:
            qml.QSVT(U_A, lst_projectors)

        for idx, val in enumerate(tape.expand().operations):
            assert val.name == results[idx].name
            assert val.parameters == results[idx].parameters

    def test_queuing_ops_defined_in_circuit(self):
        """Test that qml.QSVT queues operations correctly when they are called in the qnode."""
        lst_projectors = [qml.PCPhase(0.2, dim=1, wires=0), qml.PCPhase(0.3, dim=1, wires=0)]
        results = [
            qml.PCPhase(0.2, dim=1, wires=[0]),
            qml.PauliX(wires=[0]),
            qml.PCPhase(0.3, dim=1, wires=[0]),
        ]

        with qml.queuing.AnnotatedQueue() as q:
            qml.QSVT(qml.PauliX(wires=0), lst_projectors)

        tape = qml.tape.QuantumScript.from_queue(q)

        for expected, val in zip(results, tape.expand().operations):
            assert qml.equal(expected, val)

    def test_decomposition_queues_its_contents(self):
        """Test that the decomposition method queues the decomposition in the correct order."""
        lst_projectors = [qml.PCPhase(0.2, dim=1, wires=0), qml.PCPhase(0.3, dim=1, wires=0)]
        op = qml.QSVT(qml.PauliX(wires=0), lst_projectors)
        with qml.queuing.AnnotatedQueue() as q:
            decomp = op.decomposition()

        ops, _ = qml.queuing.process_queue(q)
        assert all(qml.equal(op1, op2) for op1, op2 in zip(ops, decomp))

    @pytest.mark.parametrize(
        ("quantum_function", "phi_func", "A", "phis", "results"),
        [
            (
                qfunc,
                lst_phis,
                np.array([[0.1, 0.2], [0.3, 0.4]]),
                np.array([0.2, 0.3]),
                [
                    qml.PCPhase(0.2, dim=2, wires=[0]),
                    qml.RX(0.1, wires=[0]),
                    qml.PCPhase(0.3, dim=2, wires=[0]),
                ],
            ),
            (
                qfunc2,
                lst_phis,
                np.array([[0.1, 0.2], [0.3, 0.4]]),
                np.array([0.1, 0.2]),
                [
                    qml.PCPhase(0.1, dim=2, wires=[0]),
                    qml.prod(qml.PauliX(wires=0), qml.RZ(0.1, wires=0)),
                    qml.PCPhase(0.2, dim=2, wires=[0]),
                ],
            ),
        ],
    )
    def test_queuing_callables(self, quantum_function, phi_func, A, phis, results):
        """Test that qml.QSVT queues operations correctly when a function is called"""

        with qml.tape.QuantumTape() as tape:
            qml.QSVT(quantum_function(A), phi_func(phis))

        for idx, val in enumerate(tape.expand().operations):
            assert val.name == results[idx].name
            assert val.parameters == results[idx].parameters

    @pytest.mark.torch
    @pytest.mark.parametrize(
        ("input_matrix", "angles", "wires"),
        [([[0.1, 0.2], [0.3, 0.4]], [0.1, 0.2], [0, 1])],
    )
    def test_QSVT_torch(self, input_matrix, angles, wires):
        """Test that the qsvt function matrix is correct for torch."""
        import torch

        default_matrix = qml.matrix(qml.qsvt(input_matrix, angles, wires))

        input_matrix = torch.tensor(input_matrix, dtype=float)
        angles = torch.tensor(angles[::-1], dtype=float)

        op = qml.QSVT(
            qml.BlockEncode(input_matrix, wires),
            [qml.PCPhase(phi, 2, wires) for phi in angles],
        )

        assert np.allclose(qml.matrix(op), default_matrix)
        assert qml.math.get_interface(qml.matrix(op)) == "torch"

    @pytest.mark.jax
    @pytest.mark.parametrize(
        ("input_matrix", "angles", "wires"),
        [([[0.1, 0.2], [0.3, 0.4]], [0.1, 0.2], [0, 1])],
    )
    def test_QSVT_jax(self, input_matrix, angles, wires):
        """Test that the qsvt function matrix is correct for jax."""
        import jax.numpy as jnp

        default_matrix = qml.matrix(qml.qsvt(input_matrix, angles, wires))

        input_matrix = jnp.array(input_matrix)
        angles = jnp.array(angles[::-1])

        op = qml.QSVT(
            qml.BlockEncode(input_matrix, wires),
            [qml.PCPhase(phi, 2, wires) for phi in angles],
        )

        assert np.allclose(qml.matrix(op), default_matrix)
        assert qml.math.get_interface(qml.matrix(op)) == "jax"

    @pytest.mark.tf
    @pytest.mark.parametrize(
        ("input_matrix", "angles", "wires"),
        [([[0.1, 0.2], [0.3, 0.4]], [0.1, 0.2], [0, 1])],
    )
    def test_QSVT_tensorflow(self, input_matrix, angles, wires):
        """Test that the qsvt function matrix is correct for tensorflow."""
        import tensorflow as tf

        default_matrix = qml.matrix(qml.qsvt(input_matrix, angles, wires))

        input_matrix = tf.Variable(input_matrix)
        angles = tf.Variable(angles[::-1])

        op = qml.QSVT(
            qml.BlockEncode(input_matrix, wires),
            [qml.PCPhase(phi, 2, wires) for phi in angles],
        )

        assert np.allclose(qml.matrix(op), default_matrix)
        assert qml.math.get_interface(qml.matrix(op)) == "tensorflow"

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
        """Test that qml.grad results are the same as finite difference results"""

        @qml.qnode(qml.device("default.qubit", wires=2))
        def circuit(A, phis):
            qml.QSVT(
                qml.BlockEncode(A, wires=[0, 1]),
                [qml.PCPhase(phi, 2, wires=[0, 1]) for phi in phis],
            )
            return qml.expval(qml.PauliZ(wires=0))

        A = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=complex, requires_grad=True)
        phis = np.array([0.1, 0.2, 0.3], dtype=complex, requires_grad=True)
        y = circuit(A, phis)

        mat_grad_results, phi_grad_results = qml.grad(circuit)(A, phis)

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
        op = qml.QSVT(qml.Hadamard(0), [qml.Identity(0)])
        assert op.label() == "QSVT"
        assert op.label(base_label="custom_label") == "custom_label"


class Testqsvt:
    """Test the qml.qsvt function."""

    @pytest.mark.parametrize(
        ("A", "phis", "wires", "true_mat"),
        [
            (
                [[0.1, 0.2], [0.3, 0.4]],
                [0.2, 0.3],
                [0, 1],
                # mathematical order of gates:
                qml.matrix(qml.PCPhase(0.2, dim=2, wires=[0, 1]))
                @ qml.matrix(qml.BlockEncode([[0.1, 0.2], [0.3, 0.4]], wires=[0, 1]))
                @ qml.matrix(qml.PCPhase(0.3, dim=2, wires=[0, 1])),
            ),
            (
                [[0.3, 0.1], [0.2, 0.9]],
                [0.1, 0.2, 0.3],
                [0, 1],
                # mathematical order of gates:
                qml.matrix(qml.PCPhase(0.1, dim=2, wires=[0, 1]))
                @ qml.matrix(qml.adjoint(qml.BlockEncode([[0.3, 0.1], [0.2, 0.9]], wires=[0, 1])))
                @ qml.matrix(qml.PCPhase(0.2, dim=2, wires=[0, 1]))
                @ qml.matrix(qml.BlockEncode([[0.3, 0.1], [0.2, 0.9]], wires=[0, 1]))
                @ qml.matrix(qml.PCPhase(0.3, dim=2, wires=[0, 1])),
            ),
        ],
    )
    def test_output(self, A, phis, wires, true_mat):
        """Test that qml.qsvt produces the correct output."""
        dev = qml.device("default.qubit", wires=len(wires))

        @qml.qnode(dev)
        def circuit():
            qml.qsvt(A, phis, wires)
            return qml.expval(qml.PauliZ(wires=0))

        observable_mat = np.kron(qml.matrix(qml.PauliZ(0)), np.eye(2))
        true_expval = (np.conj(true_mat).T @ observable_mat @ true_mat)[0, 0]

        assert np.isclose(circuit(), true_expval)
        assert np.allclose(qml.matrix(circuit)(), true_mat)

    @pytest.mark.parametrize(
        ("A", "phis", "wires", "result"),
        [
            (
                [[0.1, 0.2], [0.3, 0.4]],
                [-1.520692517929803, 0.05010380886509347],
                [0, 1],
                0.01,
            ),  # angles from pyqsp give 0.1*x
            (
                0.3,
                [-0.8104500678299933, 1.520692517929803, 0.7603462589648997],
                [0],
                0.009,
            ),  # angles from pyqsp give 0.1*x**2
            (
                -1,
                [-1.164, 0.3836, 0.383, 0.406],
                [0],
                -1,
            ),  # angles from pyqsp give 0.5 * (5 * x**3 - 3 * x)
        ],
    )
    def test_output_wx(self, A, phis, wires, result):
        """Test that qml.qsvt produces the correct output."""
        dev = qml.device("default.qubit", wires=len(wires))

        @qml.qnode(dev)
        def circuit():
            qml.qsvt(A, phis, wires, convention="Wx")
            return qml.expval(qml.PauliZ(wires=0))

        assert np.isclose(np.real(qml.matrix(circuit)())[0][0], result, rtol=1e-3)

    @pytest.mark.parametrize(
        ("A", "phis", "wires", "result"),
        [
            (
                [[0.1, 0.2], [0.3, 0.4]],
                [-1.520692517929803, 0.05010380886509347],
                [0, 1],
                0.01,
            ),  # angles from pyqsp give 0.1*x
            (
                0.3,
                [-0.8104500678299933, 1.520692517929803, 0.7603462589648997],
                [0],
                0.009,
            ),  # angles from pyqsp give 0.1*x**2
            (
                -1,
                [-1.164, 0.3836, 0.383, 0.406],
                [0],
                -1,
            ),  # angles from pyqsp give 0.5 * (5 * x**3 - 3 * x)
        ],
    )
    def test_matrix_wx(self, A, phis, wires, result):
        """Assert that the matrix method produces the expected result using both call signatures."""
        m1 = qml.matrix(qml.qsvt(A, phis, wires, convention="Wx"))
        m2 = qml.matrix(qml.qsvt)(A, phis, wires, convention="Wx")

        assert np.isclose(np.real(m1[0, 0]), result, rtol=1e-3)
        assert np.allclose(m1, m2)

    @pytest.mark.torch
    @pytest.mark.parametrize(
        ("input_matrix", "angles", "wires"),
        [([[0.1, 0.2], [0.3, 0.4]], [0.1, 0.2], [0, 1])],
    )
    def test_qsvt_torch(self, input_matrix, angles, wires):
        """Test that the qsvt function matrix is correct for torch."""
        import torch

        default_matrix = qml.matrix(qml.qsvt(input_matrix, angles, wires))

        input_matrix = torch.tensor(input_matrix, dtype=float)
        angles = torch.tensor(angles, dtype=float)

        op = qml.qsvt(input_matrix, angles, wires)

        assert np.allclose(qml.matrix(op), default_matrix)
        assert qml.math.get_interface(qml.matrix(op)) == "torch"

    @pytest.mark.jax
    @pytest.mark.parametrize(
        ("input_matrix", "angles", "wires"),
        [([[0.1, 0.2], [0.3, 0.4]], [0.1, 0.2], [0, 1])],
    )
    def test_qsvt_jax(self, input_matrix, angles, wires):
        """Test that the qsvt function matrix is correct for jax."""
        import jax.numpy as jnp

        default_matrix = qml.matrix(qml.qsvt(input_matrix, angles, wires))

        input_matrix = jnp.array(input_matrix)
        angles = jnp.array(angles)

        op = qml.qsvt(input_matrix, angles, wires)

        assert np.allclose(qml.matrix(op), default_matrix)
        assert qml.math.get_interface(qml.matrix(op)) == "jax"

    @pytest.mark.tf
    @pytest.mark.parametrize(
        ("input_matrix", "angles", "wires"),
        [([[0.1, 0.2], [0.3, 0.4]], [0.1, 0.2], [0, 1])],
    )
    def test_qsvt_tensorflow(self, input_matrix, angles, wires):
        """Test that the qsvt function matrix is correct for tensorflow."""
        import tensorflow as tf

        default_matrix = qml.matrix(qml.qsvt(input_matrix, angles, wires))

        input_matrix = tf.Variable(input_matrix)
        angles = tf.Variable(angles)

        op = qml.qsvt(input_matrix, angles, wires)

        assert np.allclose(qml.matrix(op), default_matrix)
        assert qml.math.get_interface(qml.matrix(op)) == "tensorflow"

    def test_qsvt_grad(self):
        """Test that qml.grad results are the same as finite difference results"""

        @qml.qnode(qml.device("default.qubit", wires=2))
        def circuit(A, phis):
            qml.qsvt(
                A,
                phis,
                wires=[0, 1],
            )
            return qml.expval(qml.PauliZ(wires=0))

        A = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=complex, requires_grad=True)
        phis = np.array([0.1, 0.2, 0.3], dtype=complex, requires_grad=True)
        y = circuit(A, phis)

        mat_grad_results, phi_grad_results = qml.grad(circuit)(A, phis)

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
