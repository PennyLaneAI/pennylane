# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Tests for the MPSPrep template.
"""

import numpy as np
import pytest

import pennylane as qml
from pennylane.templates.state_preparations.state_prep_mps import _complete_unitary


def test_complete_unitary_func():
    """Check that the auxiliar function _complete_unitary works properly."""

    columns = np.array([[1.0, 0, 0, 0], [0, 1, 0, 0]])
    unitary = _complete_unitary(columns)
    identity = np.eye(unitary.shape[0])

    assert np.allclose(np.conj(unitary.T) @ unitary, identity, atol=0.1)


class TestMPSPrep:

    def test_standard_validity(self):
        """Check the operation using the assert_valid function."""
        mps = [
            np.array([[0.0, 0.107], [0.994, 0.0]]),
            np.array(
                [
                    [[0.0, 0.0, 0.0, -0.0], [1.0, 0.0, 0.0, -0.0]],
                    [[0.0, 1.0, 0.0, -0.0], [0.0, 0.0, 0.0, -0.0]],
                ]
            ),
            np.array(
                [
                    [[-1.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 1.0]],
                    [[0.0, -1.0], [0.0, 0.0]],
                    [[0.0, 0.0], [1.0, 0.0]],
                ]
            ),
            np.array([[-1.0, -0.0], [-0.0, -1.0]]),
        ]

        op = qml.MPSPrep(mps, wires=[0, 1, 2, 3], work_wires=[4, 5])
        qml.ops.functions.assert_valid(op, skip_differentiation=True)

    def test_access_to_param(self):
        """tests that the parameter is accessible."""
        mps = [
            np.array([[0.0, 0.107], [0.994, 0.0]]),
            np.array(
                [
                    [[0.0, 0.0, 0.0, -0.0], [1.0, 0.0, 0.0, -0.0]],
                    [[0.0, 1.0, 0.0, -0.0], [0.0, 0.0, 0.0, -0.0]],
                ]
            ),
            np.array(
                [
                    [[-1.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 1.0]],
                    [[0.0, -1.0], [0.0, 0.0]],
                    [[0.0, 0.0], [1.0, 0.0]],
                ]
            ),
            np.array([[-1.0, -0.0], [-0.0, -1.0]]),
        ]
        op = qml.MPSPrep(mps, wires=[0, 1, 2])

        for arr1, arr2 in zip(mps, op.mps):
            assert np.allclose(arr1, arr2)

    @pytest.mark.parametrize(
        ("mps", "msg_match"),
        [
            (
                [
                    np.array([[0.0, 0.107, 0.0], [0.994, 0.0, 0.0]]),
                    np.array(
                        [
                            [[0.0, 0.0, 0.0, -0.0], [1.0, 0.0, 0.0, -0.0]],
                            [[0.0, 1.0, 0.0, -0.0], [0.0, 0.0, 0.0, -0.0]],
                        ]
                    ),
                    np.array(
                        [
                            [[-1.0, 0.0], [0.0, 0.0]],
                            [[0.0, 0.0], [0.0, 1.0]],
                            [[0.0, -1.0], [0.0, 0.0]],
                            [[0.0, 0.0], [1.0, 0.0]],
                        ]
                    ),
                    np.array([[-1.0, -0.0], [-0.0, -1.0]]),
                ],
                "The second dimension of the first tensor must be a power of 2.",
            ),
            (
                [
                    np.array([0.0, 0.107, 0.994, 0.0]),
                    np.array(
                        [
                            [[0.0, 0.0, 0.0, -0.0], [1.0, 0.0, 0.0, -0.0]],
                            [[0.0, 1.0, 0.0, -0.0], [0.0, 0.0, 0.0, -0.0]],
                        ]
                    ),
                    np.array(
                        [
                            [[-1.0, 0.0], [0.0, 0.0]],
                            [[0.0, 0.0], [0.0, 1.0]],
                            [[0.0, -1.0], [0.0, 0.0]],
                            [[0.0, 0.0], [1.0, 0.0]],
                        ]
                    ),
                    np.array([[-1.0, -0.0], [-0.0, -1.0]]),
                ],
                "The first tensor must have exactly 2 dimensions",
            ),
            (
                [
                    np.array([[0.0, 0.107], [0.994, 0.0]]),
                    np.array(
                        [
                            [[0.0, 0.0, 0.0, -0.0], [1.0, 0.0, 0.0, -0.0]],
                            [[0.0, 1.0, 0.0, -0.0], [0.0, 0.0, 0.0, -0.0]],
                        ]
                    ),
                    np.array(
                        [[-1.0, 0.0], [0.0, 0.0]],
                    ),
                    np.array([[-1.0, -0.0], [-0.0, -1.0]]),
                ],
                "Tensor 2 must have exactly 3 dimensions.",
            ),
            (
                [
                    np.array([[0.0, 0.107], [0.994, 0.0]]),
                    np.array(
                        [
                            [[0.0, 0.0, 0.0, -0.0], [1.0, 0.0, 0.0, -0.0]],
                            [[0.0, 1.0, 0.0, -0.0], [0.0, 0.0, 0.0, -0.0]],
                        ]
                    ),
                    np.array(
                        [
                            [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                            [[0.0, -1.0, 0.0], [0.0, 0.0, 0.0]],
                        ]
                    ),
                    np.array([[-1.0, -0.0], [-0.0, -1.0]]),
                ],
                "The first dimension of tensor 2 must be a power of 2.",
            ),
            (
                [
                    np.array([[0.0, 0.107], [0.994, 0.0]]),
                    np.array(
                        [
                            [[-1.0, 0.0], [0.0, 0.0]],
                            [[0.0, 0.0], [0.0, 1.0]],
                            [[0.0, -1.0], [0.0, 0.0]],
                            [[0.0, 0.0], [1.0, 0.0]],
                        ]
                    ),
                    np.array(
                        [
                            [[-1.0, 0.0], [0.0, 0.0]],
                            [[0.0, 0.0], [0.0, 1.0]],
                            [[0.0, -1.0], [0.0, 0.0]],
                            [[0.0, 0.0], [1.0, 0.0]],
                        ]
                    ),
                    np.array([[-1.0, -0.0], [-0.0, -1.0]]),
                ],
                "Dimension mismatch:",
            ),
            (
                [
                    np.array([[0.0, 0.107], [0.994, 0.0]]),
                    np.array(
                        [
                            [[0.0, 0.0, 0.0, -0.0], [1.0, 0.0, 0.0, -0.0]],
                            [[0.0, 1.0, 0.0, -0.0], [0.0, 0.0, 0.0, -0.0]],
                        ]
                    ),
                    np.array(
                        [
                            [[-1.0, 0.0], [0.0, 0.0]],
                            [[0.0, 0.0], [0.0, 1.0]],
                            [[0.0, -1.0], [0.0, 0.0]],
                            [[0.0, 0.0], [1.0, 0.0]],
                        ]
                    ),
                    np.array([-1.0, -0.0, -0.0, -1.0]),
                ],
                "The last tensor must have exactly 2 dimensions.",
            ),
            (
                [
                    np.array([[0.0, 0.107], [0.994, 0.0]]),
                    np.array(
                        [
                            [[0.0, 0.0, 0.0, -0.0], [1.0, 0.0, 0.0, -0.0]],
                            [[0.0, 1.0, 0.0, -0.0], [0.0, 0.0, 0.0, -0.0]],
                        ]
                    ),
                    np.array(
                        [
                            [[-1.0, 0.0], [0.0, 0.0]],
                            [[0.0, 0.0], [0.0, 1.0]],
                            [[0.0, -1.0], [0.0, 0.0]],
                            [[0.0, 0.0], [1.0, 0.0]],
                        ]
                    ),
                    np.array([[-1.0, -0.0, 0.0], [-0.0, -1.0, 0.0]]),
                ],
                "The second dimension of the last tensor must be exactly 2.",
            ),
        ],
    )
    def test_MPSPrep_error(self, mps, msg_match):
        """Test that proper errors are raised for MPSPrep"""
        with pytest.raises(AssertionError, match=msg_match):
            qml.MPSPrep(mps, wires=[0, 1, 2])

    @pytest.mark.jax
    def test_jax_mps(self):
        """Check the operation works with jax."""

        import jax
        from jax import numpy as jnp

        mps = [
            jnp.array([[0.0, 0.107j], [0.994, 0.0]]),
            jnp.array(
                [
                    [[0.0, 0.0], [1.0, 0.0]],
                    [[0.0, 1.0], [0.0, 0.0]],
                ]
            ),
            jnp.array([[-1.0, -0.0], [-0.0, -1.0]]),
        ]

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            qml.MPSPrep(mps, wires=range(2, 5), work_wires=[0, 1])
            return qml.state()

        output = circuit()[:8]

        state = [
            0.0 + 0.0j,
            -0.10705513j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            -0.99451217 + 0.0j,
            0.0 + 0.0j,
        ]

        assert jax.numpy.allclose(output, jax.numpy.array(state), rtol=0.01)

    @pytest.mark.parametrize(
        ("mps", "state", "num_wires"),
        [
            (
                [
                    np.array([[0.70710678, 0.0], [0.0, 0.70710678]]),
                    np.array(
                        [
                            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                            [[0.0, 0.0, -0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]],
                        ]
                    ),
                    np.array(
                        [
                            [[0.00000000e00, 1.74315280e-32], [-7.07106781e-01, -7.07106781e-01]],
                            [[7.07106781e-01, 7.07106781e-01], [0.00000000e00, 0.00000000e00]],
                            [[0.00000000e00, 0.00000000e00], [-7.07106781e-01, 7.07106781e-01]],
                            [[-7.07106781e-01, 7.07106781e-01], [0.00000000e00, 0.00000000e00]],
                        ]
                    ),
                    np.array([[1.0, 0.0], [0.0, 1.0]]),
                ],
                np.array([1 / 2, 1 / 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 / 2, 1 / 2]),
                4,
            ),
            (
                [
                    np.array([[0.0, 0.107], [0.994, 0.0]]),
                    np.array(
                        [
                            [[0.0, 0.0], [1.0, 0.0]],
                            [[0.0, 1.0], [0.0, 0.0]],
                        ]
                    ),
                    np.array([[-1.0, -0.0], [-0.0, -1.0]]),
                ],
                np.array(
                    [
                        0.0 + 0.0j,
                        -0.10705513 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        -0.99451217 + 0.0j,
                        0.0 + 0.0j,
                    ]
                ),
                3,
            ),
        ],
    )
    def test_correctness(self, mps, state, num_wires):
        """Test correctness of the solution

        Data was generated using the functionality `decompose_dense` in
        pennylane-lightning/pennylane_lightning/lightning_tensor/_tensornet.py
        """

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            qml.MPSPrep(mps, wires=range(2, 2 + num_wires), work_wires=[0, 1])
            return qml.state()

        output = circuit()[: 2**num_wires]

        assert np.allclose(state, output, rtol=0.01)

    def test_decomposition(self):
        """Tests that the template defines the correct decomposition."""

        mps = [
            np.array([[0.70710678, 0.0], [0.0, 0.70710678]]),
            np.array(
                [
                    [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                    [[0.0, 0.0, -0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]],
                ]
            ),
            np.array(
                [
                    [[0.00000000e00, 1.74315280e-32], [-7.07106781e-01, -7.07106781e-01]],
                    [[7.07106781e-01, 7.07106781e-01], [0.00000000e00, 0.00000000e00]],
                    [[0.00000000e00, 0.00000000e00], [-7.07106781e-01, 7.07106781e-01]],
                    [[-7.07106781e-01, 7.07106781e-01], [0.00000000e00, 0.00000000e00]],
                ]
            ),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
        ]

        ops = qml.MPSPrep.compute_decomposition(mps, wires=range(2, 6), work_wires=[0, 1])

        for ind, op in enumerate(ops):
            assert op.wires == qml.wires.Wires([2 + ind] + [0, 1])
            assert op.name == "QubitUnitary"

    @pytest.mark.jax
    def test_jax_jit_mps(self):
        """Check the operation works with jax and jit."""

        import jax
        from jax import numpy as jnp

        mps = [
            jnp.array([[0.0, 0.107j], [0.994, 0.0]], dtype=complex),
            jnp.array(
                [
                    [[0.0, 0.0], [1.0, 0.0]],
                    [[0.0, 1.0], [0.0, 0.0]],
                ],
                dtype=complex,
            ),
            jnp.array([[-1.0, -0.0], [-0.0, -1.0]], dtype=complex),
        ]

        dev = qml.device("default.qubit")

        @jax.jit
        @qml.qnode(dev)
        def circuit():
            qml.MPSPrep(mps, wires=range(2, 5), work_wires=[0, 1])
            return qml.state()

        output = circuit()[:8]

        state = [
            0.0 + 0.0j,
            -0.10705513j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            -0.99451217 + 0.0j,
            0.0 + 0.0j,
        ]

        assert jax.numpy.allclose(output, jax.numpy.array(state), rtol=0.01)

    def test_wires_decomposition(self):
        """Checks that error is shown if no `work_wires` are given in decomposition"""

        mps = [
            np.array([[0.0, 0.107j], [0.994, 0.0]], dtype=complex),
            np.array(
                [
                    [[0.0, 0.0], [1.0, 0.0]],
                    [[0.0, 1.0], [0.0, 0.0]],
                ],
                dtype=complex,
            ),
            np.array([[-1.0, -0.0], [-0.0, -1.0]], dtype=complex),
        ]

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            qml.MPSPrep(mps, wires=range(2, 5))
            return qml.state()

        with pytest.raises(AssertionError, match="To decompose MPSPrep you must specify"):
            circuit()
