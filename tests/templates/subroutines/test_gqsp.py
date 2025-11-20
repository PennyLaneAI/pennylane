# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Tests for the GQSP template.
"""
# pylint: disable=too-many-arguments, import-outside-toplevel, no-self-use

import pytest
from numpy.linalg import matrix_power

import pennylane as qml
from pennylane import numpy as np
from pennylane.ops.functions.assert_valid import _test_decomposition_rule


class TestGQSP:
    """Test the qml.GQSP template."""

    @pytest.mark.jax
    def test_standard_validity(self):
        """Test standard validity criteria with assert_valid."""

        angles = np.ones([3, 5])

        @qml.prod
        def unitary(wires):
            qml.RX(0.3, wires)
            qml.RZ(0.6, wires)

        op = qml.GQSP(unitary(1), angles, control=(0,))
        qml.ops.functions.assert_valid(op, skip_differentiation=True)

    @pytest.mark.parametrize(
        ("unitary", "poly"),
        [
            (qml.RX(0.3, wires=1), [0.3, -0.2j, 0.1]),
            (qml.RZ(1.3, wires=1), [-0.3j, -0.2j, 0, 0, 0.2]),
            (qml.RY(0.2, wires=1), [0.4, -0, 0.2, 0, 0.2]),
        ],
    )
    def test_correct_algorithm(self, unitary, poly):
        """Test that poly_to_angles and GQSP produce the correct solution"""

        angles = qml.poly_to_angles(poly, "GQSP")

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(angles):
            qml.GQSP(unitary, angles, control=0)
            return qml.expval(qml.Z(0))

        unitary_matrix = qml.matrix(unitary)
        expected_output = sum(
            [coeff * matrix_power(unitary_matrix, i) for i, coeff in enumerate(poly)]
        )
        generated_output = qml.matrix(circuit, wire_order=[0, 1])(angles)[:2, :2]

        assert np.allclose(expected_output, generated_output)

    @pytest.mark.parametrize(
        ("unitary"),
        [
            (qml.RX(0.3, wires=1)),
            (qml.RZ(1.3, wires=1)),
            (qml.RY(0.2, wires=1)),
        ],
    )
    def test_correct_template(self, unitary):
        """Test that GQSP produce the correct solution"""

        # Precalucated angles for polynomial p(x) = 0.1 + 0.2x + 0.3x^2
        angles = [
            np.array([0.10798862, 0.22107159, 1.25635543]),
            np.array([-3.14159265, 0.0, 0.0]),
            np.array([3.14159265, 0.0, 0.0]),
        ]

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(angles):
            qml.GQSP(unitary, angles, control=0)
            return qml.expval(qml.Z(0))

        unitary_matrix = qml.matrix(unitary)
        expected_output = sum(
            [coeff * matrix_power(unitary_matrix, i) for i, coeff in enumerate([0.1, 0.2, 0.3])]
        )
        generated_output = qml.matrix(circuit, wire_order=[0, 1])(angles)[:2, :2]

        assert np.allclose(expected_output, generated_output)

    def test_queueing(self):
        """Test that no additional gates are being queued"""

        with qml.queuing.AnnotatedQueue() as q:
            qml.GQSP(qml.Z(1), np.ones([3, 3]), control=0)

        assert len(q.queue) == 1
        assert q.queue[0].name == "GQSP"

    def test_decomposition(self):

        angles = np.array([[1, 2], [3, 4], [5, 6]])

        qml.GQSP(qml.Z(1), angles, control=0)

        decomposition = qml.GQSP(qml.Z(1), angles, control=0).decomposition()

        expected = [
            qml.PauliX(wires=0),
            qml.U3(2 * angles[0, 0], angles[1, 0], angles[2, 0], wires=0),
            qml.PauliX(wires=0),
            qml.PauliZ(wires=0),
            qml.ctrl(qml.PauliZ(wires=1), control=0, control_values=[0]),
            qml.PauliX(wires=0),
            qml.U3(2 * angles[0, 1], angles[1, 1], angles[2, 1], wires=0),
            qml.PauliX(wires=0),
            qml.PauliZ(wires=0),
        ]

        for op1, op2 in zip(decomposition, expected):
            qml.assert_equal(op1, op2)

    @pytest.mark.capture
    def test_decomposition_new_capture(self):
        """Tests the decomposition rule implemented with the new system."""
        angles = np.array([[1, 2], [3, 4], [5, 6]])
        op = qml.GQSP(qml.Z(1), angles, control=0)

        for rule in qml.list_decomps(qml.GQSP):
            _test_decomposition_rule(op, rule)

    def test_decomposition_new(self):
        """Tests the decomposition rule implemented with the new system."""
        angles = np.array([[1, 2], [3, 4], [5, 6]])
        op = qml.GQSP(qml.Z(1), angles, control=0)

        for rule in qml.list_decomps(qml.GQSP):
            _test_decomposition_rule(op, rule)

    @pytest.mark.jax
    def test_gqsp_jax(self):
        """Test that GQSP works with jax"""

        import jax.numpy as jnp

        angles = np.array([[1, 2], [3, 4], [5, 6]])

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(angles):
            qml.GQSP(qml.RX(0.3, wires=1), angles, control=0)
            return qml.expval(qml.Z(0))

        expected_output = jnp.array(qml.matrix(circuit, wire_order=[0, 1])(angles))
        generated_output = qml.matrix(circuit, wire_order=[0, 1])(jnp.array(angles))

        assert np.allclose(expected_output, generated_output)
        assert qml.math.get_interface(generated_output) == "jax"

    @pytest.mark.torch
    def test_gqsp_torch(self):
        """Test that GQSP works with torch"""

        import torch

        angles = np.array([[1, 2], [3, 4], [5, 6]])

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(angles):
            qml.GQSP(qml.RX(0.3, wires=1), angles, control=0)
            return qml.expval(qml.Z(0))

        expected_output = torch.tensor(qml.matrix(circuit, wire_order=[0, 1])(angles))
        generated_output = qml.matrix(circuit, wire_order=[0, 1])(torch.tensor(angles))

        assert np.allclose(expected_output, generated_output)
        assert qml.math.get_interface(generated_output) == "torch"

    @pytest.mark.tf
    def test_gqsp_tensorflow(self):
        """Test that GQSP works with tensorflow"""

        import tensorflow as tf

        angles = np.array([[1, 2], [3, 4], [5, 6]])

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(angles):
            qml.GQSP(qml.RX(0.3, wires=1), angles, control=0)
            return qml.expval(qml.Z(0))

        expected_output = tf.Variable(qml.matrix(circuit, wire_order=[0, 1])(angles))
        generated_output = qml.matrix(circuit, wire_order=[0, 1])(tf.Variable(angles))

        assert np.allclose(expected_output, generated_output)
        assert qml.math.get_interface(generated_output) == "tensorflow"

    @pytest.mark.jax
    def test_gqsp_jax_jit(self):
        """Test that GQSP works with jax"""

        import jax
        import jax.numpy as jnp

        angles = jnp.array([[1, 2], [3, 4], [5, 6]])

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(angles):
            qml.GQSP(qml.RX(0.3, wires=1), angles, control=0)
            return qml.expval(qml.Z(0) @ qml.Z(1))

        expected_output = circuit(angles)

        jit_circuit = jax.jit(circuit)
        generated_output = jit_circuit(angles)

        assert np.allclose(expected_output, generated_output)
        assert qml.math.get_interface(generated_output) == "jax"
