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
                    np.array([[-0.6734052, 0.73927359], [0.73927359, 0.6734052]]),
                    np.array(
                        [
                            [
                                [-7.20105790e-01, -1.89741090e-01, 5.82838055e-01, -3.25185749e-01],
                                [4.29632311e-01, -8.68380029e-01, 2.47651976e-01, -8.37507469e-04],
                            ],
                            [
                                [
                                    -1.00441887e-01,
                                    -2.24394589e-01,
                                    -6.15113611e-01,
                                    -7.49128654e-01,
                                ],
                                [-5.35513918e-01, -3.99451514e-01, -4.69684517e-01, 5.77113322e-01],
                            ],
                        ]
                    ),
                    np.array(
                        [
                            [
                                [-0.80632643, -0.47294461, 0.23182434, -0.03532423],
                                [0.19104774, 0.18355921, 0.33180043, 0.15166789],
                            ],
                            [
                                [-0.12789955, 0.15909324, -0.28329785, -0.23830545],
                                [-0.5332595, 0.74593332, -0.19189508, 0.10234046],
                            ],
                            [
                                [0.01881935, 0.30423266, 0.28341988, 0.32859376],
                                [0.02322847, 0.21707857, 0.68904115, -0.58895129],
                            ],
                            [
                                [-0.07230691, 0.14463917, 0.05603212, -0.3091061],
                                [0.08057569, 0.01735283, -0.40104915, -0.59841618],
                            ],
                        ]
                    ),
                    np.array(
                        [
                            [[-0.22294664, -0.00953879], [-0.80210505, 0.03741124]],
                            [[-0.53296709, -0.07688778], [0.1478749, -0.03842062]],
                            [[-0.00305516, 0.34730904], [-0.00997211, -0.68175905]],
                            [[-0.02820048, 0.56757052], [0.0098664, 0.28907832]],
                        ]
                    ),
                    np.array([[-0.85785372, 0.18059319], [0.09911105, 0.47079729]]),
                ],
                np.array(
                    [
                        -0.30838839,
                        0.00913474,
                        -0.14842667,
                        0.06058075,
                        -0.00464704,
                        0.16533888,
                        0.08806463,
                        0.19535211,
                        0.17301488,
                        -0.20745747,
                        0.30696199,
                        0.01131135,
                        -0.08844845,
                        -0.12525685,
                        -0.13280471,
                        0.29048878,
                        0.19283544,
                        0.02670539,
                        0.39244675,
                        0.00316343,
                        -0.12347408,
                        0.15527045,
                        0.02744977,
                        -0.01096892,
                        -0.01303293,
                        0.06124989,
                        0.08509908,
                        -0.08363092,
                        -0.24638746,
                        -0.00924703,
                        0.44259161,
                        -0.07738196,
                    ]
                ),
                5,
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

        with pytest.raises(AssertionError, match="The qml.MPSPREP decomposition requires"):
            circuit()
