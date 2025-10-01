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
Unit tests for the Superposition template.
"""

import numpy as np

# pylint: disable=too-many-arguments,too-few-public-methods
import pytest

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.templates.state_preparations.superposition import order_states

PROBS_BASES = [
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
]


def int_to_state(i, length):
    return tuple(map(int, f"{i:0{length}b}"))


@pytest.mark.parametrize(
    "basis_states, exp_map",
    (
        [  # Examples where all basis states are fixed points
            (
                [int_to_state(i, L) for i in range(m)],
                {int_to_state(i, L): int_to_state(i, L) for i in range(m)},
            )
            for L, m in [(1, 2), (2, 4), (2, 3), (3, 7), (4, 3), (4, 16)]
        ]
        + [  # Examples from docstring
            (
                [[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1]],
                {
                    (1, 1, 0, 0): (0, 0, 0, 0),
                    (1, 0, 1, 0): (0, 0, 0, 1),
                    (0, 1, 0, 1): (0, 0, 1, 0),
                    (1, 0, 0, 1): (0, 0, 1, 1),
                },
            ),
            (
                [[1, 1, 0, 0], [0, 1, 0, 1], [0, 0, 0, 1], [1, 0, 0, 1]],
                {
                    (0, 0, 0, 1): (0, 0, 0, 1),
                    (1, 1, 0, 0): (0, 0, 0, 0),
                    (0, 1, 0, 1): (0, 0, 1, 0),
                    (1, 0, 0, 1): (0, 0, 1, 1),
                },
            ),
        ]
        + [  # Other examples
            (
                [[1, 1, 0, 1], [0, 1, 0, 0], [1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 0]],
                {
                    (0, 0, 0, 0): (0, 0, 0, 0),
                    (0, 0, 1, 0): (0, 0, 1, 0),
                    (0, 1, 0, 0): (0, 1, 0, 0),
                    (1, 1, 0, 1): (0, 0, 0, 1),
                    (1, 1, 1, 1): (0, 0, 1, 1),
                },
            ),
            ([[1, 1, 0], [0, 0, 1]], {(0, 0, 1): (0, 0, 1), (1, 1, 0): (0, 0, 0)}),
        ]
    ),
)
def test_order_states(basis_states, exp_map):
    assert order_states(basis_states) == exp_map


@pytest.mark.jax
def test_standard_validity():
    """Check the operation using the assert_valid function."""

    coeffs = np.array([0.5, 0.5, -0.5, -0.5])
    bases = np.array(
        [[0, 0, 0, 0, 0], [0, 1, 0, 1, 1], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [1, 1, 0, 1, 1]]
    )

    op = qml.Superposition(coeffs, bases=bases, wires=range(5), work_wire=5)
    qml.ops.functions.assert_valid(op)


class TestSuperposition:
    """Test the Superposition template."""

    @pytest.mark.parametrize("probs, bases", PROBS_BASES)
    def test_correct_output(self, probs, bases):
        """Test the correct output of the Superposition template."""

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            qml.Superposition(
                np.sqrt(probs), bases=bases, wires=range(len(bases[0])), work_wire=len(bases[0])
            )
            return qml.probs(range(len(bases[0])))

        output = circuit()
        for i, base in enumerate(bases):
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

    @pytest.mark.parametrize(("probs", "bases"), PROBS_BASES)
    def test_decomposition_new(self, probs, bases):
        """Test the decomposition of the Superposition template."""
        op = qml.Superposition(
            np.sqrt(probs), bases, wires=range(len(bases[0])), work_wire=len(bases[0])
        )
        for rule in qml.list_decomps(qml.Superposition):
            _test_decomposition_rule(op, rule)

    @pytest.mark.parametrize(
        ("coeffs", "bases", "msg_match"),
        [
            (
                np.sqrt([0.1, 0.2, 0.3, 0.4]),
                [[2, 0, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 1, 1]],
                "The elements of the basis states must be either 0 or 1",
            ),
            (
                np.sqrt([0.3, 0.2, 0.3, 0.4]),
                [[0, 0, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 1, 1]],
                "The input superposition must be normalized",
            ),
            (
                np.sqrt([0.1, 0.2, 0.3, 0.4]),
                [[0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 1, 0], [1, 1, 1, 1]],
                "The basis states must be unique",
            ),
            (
                np.sqrt([0.1, 0.2, 0.3, 0.4]),
                [[0, 0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 0], [1, 1, 1, 1]],
                "All basis states must have the same length",
            ),
        ],
    )
    def test_raise_error(self, coeffs, bases, msg_match):
        """Test that the Superposition template raises an error if a bases state is repeated."""

        with pytest.raises(ValueError, match=msg_match):
            n_wires = len(bases[0])
            qml.Superposition(coeffs, bases, wires=range(n_wires), work_wire=n_wires)

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
            qml.Superposition(state_vector, bases=[[1], [0]], wires=range(1), work_wire=1)
            return qml.expval(qml.PauliZ(0))

        qml.grad(circuit)(state_vector)

    def test_access_work_wire(self):
        """Test that the work_wire can be accessed."""
        op = qml.Superposition(np.sqrt([0.5, 0.5]), [[0, 0], [1, 1]], wires=range(2), work_wire=2)
        assert op.work_wire == qml.wires.Wires(2)


@pytest.mark.parametrize(
    "probs, bases",
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
    def test_jax(self, probs, bases):
        """Test that Superposition can be correctly used with the JAX interface."""
        from jax import numpy as jnp

        probs = jnp.array(probs)

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            qml.Superposition(
                jnp.sqrt(probs), bases=bases, wires=range(len(bases[0])), work_wire=len(bases[0])
            )
            return qml.probs(range(len(bases[0])))

        output = circuit()
        for i, base in enumerate(bases):
            dec = int("".join(map(str, base)), 2)
            assert jnp.isclose(output[dec], probs[i])

    @pytest.mark.jax
    def test_jax_jit(self, probs, bases):
        """Test that Superposition can be correctly used with the JAX-JIT interface."""
        import jax
        from jax import numpy as jnp

        probs = jnp.array(probs)

        dev = qml.device("default.qubit")

        @jax.jit
        @qml.qnode(dev)
        def circuit():
            qml.Superposition(
                jnp.sqrt(probs), bases=bases, wires=range(len(bases[0])), work_wire=len(bases[0])
            )
            return qml.probs(range(len(bases[0])))

        output = circuit()
        for i, base in enumerate(bases):
            dec = int("".join(map(str, base)), 2)
            assert jnp.isclose(output[dec], probs[i])

    @pytest.mark.tf
    def test_tensorflow(self, probs, bases):
        """Test that Superposition can be correctly used with the TensorFlow interface."""
        import tensorflow as tf

        probs = tf.Variable(probs)
        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            qml.Superposition(
                tf.sqrt(probs), bases=bases, wires=range(len(bases[0])), work_wire=len(bases[0])
            )
            return qml.probs(range(len(bases[0])))

        output = circuit()
        for i, base in enumerate(bases):
            dec = int("".join(map(str, base)), 2)
            assert np.isclose(output[dec], probs[i])

    @pytest.mark.torch
    def test_torch(self, probs, bases):
        """Test that Superposition can be correctly used with the Torch interface."""
        import torch

        probs = torch.tensor(probs, dtype=torch.float64)

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            qml.Superposition(
                torch.sqrt(probs), bases=bases, wires=range(len(bases[0])), work_wire=len(bases[0])
            )
            return qml.probs(range(len(bases[0])))

        output = circuit()
        for i, base in enumerate(bases):
            dec = int("".join(map(str, base)), 2)
            assert qml.math.isclose(output[dec], probs[i])
