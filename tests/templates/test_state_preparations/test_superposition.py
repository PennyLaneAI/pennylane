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
Unit tests for the ArbitraryStatePreparation template.
"""
from unicodedata import decomposition

import numpy as np

# pylint: disable=too-many-arguments,too-few-public-methods
import pytest

import pennylane as qml
from pennylane import numpy as pnp


def test_standard_validity():
    """Check the operation using the assert_valid function."""

    coeffs = np.array([0.5, 0.5, -0.5, -0.5])
    basis = np.array(
        [[0, 0, 0, 0, 0], [0, 1, 0, 1, 1], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [1, 1, 0, 1, 1]]
    )

    op = qml.Superposition(coeffs, basis=basis, wires=range(5), work_wire=5)
    qml.ops.functions.assert_valid(op)


class TestSuperposition:
    """Test the Superposition template."""

    @pytest.mark.parametrize(
        "probs, basis",
        [
            (
                [0.1, 0.2, 0.3, 0.2, 0.2],
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0],
                    [1, 1, 0, 1, 1],
                ],
            ),
            (
                [0.1, 0.05, 0.25, 0.6],
                [
                    [0, 0, 0, 0],
                    [0, 1, 1, 1],
                    [0, 0, 0, 1],
                    [1, 1, 1, 1],
                ],
            ),
            (
                [0.1, 0.2, 0.3, 0.2, 0.2],
                [
                    [0, 1, 1, 1, 0],
                    [1, 1, 0, 1, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0],
                    [1, 0, 0, 1, 1],
                ],
            ),
            ([0.1, 0.9], [[0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 1, 1, 1]]),
        ],
    )
    def test_correct_output(self, probs, basis):
        """Test the correct output of the Superposition template."""

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            qml.Superposition(
                np.sqrt(probs), basis=basis, wires=range(len(basis[0])), work_wire=len(basis[0])
            )
            return qml.probs(range(len(basis[0])))

        output = circuit()
        for i, base in enumerate(basis):
            dec = int("".join(map(str, base)), 2)
            assert np.isclose(output[dec], probs[i])

    def test_decomposition(self):
        """Test the decomposition of the Superposition template."""
        decomposition = qml.Superposition(
            np.sqrt([0.5, 0.5]), [[0, 0], [1, 1]], wires=range(2), work_wire=2
        ).decomposition()

        expected = [
            qml.StatePrep(np.array([0.70710678, 0.70710678]), pad_with=0, wires=[1]),
            qml.MultiControlledX(wires=[0, 1, 2], control_values=[0, 1]),
            qml.CNOT(wires=[2, 0]),
            qml.Toffoli(wires=[0, 1, 2]),
        ]

        for op1, op2 in zip(decomposition, expected):
            assert qml.equal(op1, op2)

    @pytest.mark.parametrize(
        "state_vector",
        [
            pnp.array([0.70710678, 0.70710678], requires_grad=True),
            pnp.array([0.70710678, 0.70710678j], requires_grad=True),
        ],
    )
    def test_gradient_evaluated(self, state_vector):
        """Test that the gradient is successfully calculated for a simple example. This test only
        checks that the gradient is calculated without an error."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(state_vector):
            qml.Superposition(state_vector, basis=[[1], [0]], wires=range(1), work_wire=1)
            return qml.expval(qml.PauliZ(0))

        qml.grad(circuit)(state_vector)


@pytest.mark.parametrize(
    "probs, basis",
    [
        (
            [0.1, 0.2, 0.3, 0.2, 0.2],
            [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 1, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0],
                [1, 1, 0, 1, 1],
            ],
        ),
        (
            [0.1, 0.2, 0.3, 0.2, 0.2],
            [
                [0, 1, 1, 1, 0],
                [1, 1, 0, 1, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0],
                [1, 0, 0, 1, 1],
            ],
        ),
        ([0.1, 0.9], [[0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 1, 1, 1]]),
    ],
)
class TestInterfaces:
    """Test that the Superposition template ensures the compatibility with
    interfaces"""

    @pytest.mark.jax
    def test_jax(self, probs, basis):
        """Test that MottonenStatePreparation can be correctly used with the JAX interface."""
        from jax import numpy as jnp

        probs = jnp.array(probs)

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            qml.Superposition(
                jnp.sqrt(probs), basis=basis, wires=range(len(basis[0])), work_wire=len(basis[0])
            )
            return qml.probs(range(len(basis[0])))

        output = circuit()
        for i, base in enumerate(basis):
            dec = int("".join(map(str, base)), 2)
            assert jnp.isclose(output[dec], probs[i])

    @pytest.mark.jax
    def test_jax_jit(self, probs, basis):
        """Test that MottonenStatePreparation can be correctly used with the JAX-JIT interface."""
        import jax
        from jax import numpy as jnp

        probs = jnp.array(probs)

        dev = qml.device("default.qubit")

        @jax.jit
        @qml.qnode(dev)
        def circuit():
            qml.Superposition(
                jnp.sqrt(probs), basis=basis, wires=range(len(basis[0])), work_wire=len(basis[0])
            )
            return qml.probs(range(len(basis[0])))

        output = circuit()
        for i, base in enumerate(basis):
            dec = int("".join(map(str, base)), 2)
            assert jnp.isclose(output[dec], probs[i])

    @pytest.mark.tf
    def test_tensorflow(self, probs, basis):
        """Test that MottonenStatePreparation can be correctly used with the TensorFlow interface."""
        import tensorflow as tf

        probs = tf.Variable(probs)
        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            qml.Superposition(
                tf.sqrt(probs), basis=basis, wires=range(len(basis[0])), work_wire=len(basis[0])
            )
            return qml.probs(range(len(basis[0])))

        output = circuit()
        for i, base in enumerate(basis):
            dec = int("".join(map(str, base)), 2)
            assert np.isclose(output[dec], probs[i])

    @pytest.mark.torch
    def test_torch(self, probs, basis):
        """Test that MottonenStatePreparation can be correctly used with the Torch interface."""
        import torch

        probs = torch.tensor(probs, dtype=torch.float64)

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            qml.Superposition(
                torch.sqrt(probs), basis=basis, wires=range(len(basis[0])), work_wire=len(basis[0])
            )
            return qml.probs(range(len(basis[0])))

        output = circuit()
        for i, base in enumerate(basis):
            dec = int("".join(map(str, base)), 2)
            assert qml.math.isclose(output[dec], probs[i])
