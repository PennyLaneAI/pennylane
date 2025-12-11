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
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.templates.state_preparations.state_prep_mps import (
    _mps_prep_decomposition,
    _validate_mps_shape,
    right_canonicalize_mps,
)


class TestMPSPrep:

    @pytest.mark.jax
    def test_standard_validity(self):
        """Check the template using the `assert_valid` function."""
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
        """Tests that the template parameters are accessible."""
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
    def test_mps_validate_function(self, mps, msg_match):
        """Test that proper errors are raised in mps_validate_shape"""
        with pytest.raises(AssertionError, match=msg_match):
            _validate_mps_shape(mps)

    @pytest.mark.jax
    def test_jax_jit_mps(self):
        """Check the operation works with jax and jit."""

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

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            qml.MPSPrep(mps, wires=range(2, 5), work_wires=[0, 1])
            return qml.state()

        output = circuit()[:8]
        output_jit = jax.jit(circuit)()[:8]

        assert jax.numpy.allclose(output, jax.numpy.array(state), rtol=0.01)
        assert jax.numpy.allclose(output_jit, jax.numpy.array(state), rtol=0.01)

    @pytest.mark.parametrize(
        ("mps", "state", "num_wires", "num_work_wires"),
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
                2,
            ),
            (
                [
                    np.array([[0.53849604, -0.44389787], [-0.59116842, -0.40434711]]),
                    np.array(
                        [
                            [
                                [-6.05052107e-01, 1.34284016e-01, -2.84018989e-01, -1.12416345e-01],
                                [3.60988555e-01, 6.14571922e-01, -1.20681653e-01, -2.89527967e-04],
                            ],
                            [
                                [-1.12393068e-01, 2.11496619e-01, 3.99193070e-01, -3.44891522e-01],
                                [-5.99232567e-01, 3.76491467e-01, 3.04813277e-01, 2.65697349e-01],
                            ],
                        ]
                    ),
                    np.array(
                        [
                            [
                                [0.87613189, -0.34254341, -0.12704983, -0.0161698],
                                [-0.20758717, 0.1329479, -0.18184107, 0.06942658],
                            ],
                            [
                                [-0.16499137, -0.13680142, -0.18432824, 0.12950892],
                                [-0.68790868, -0.64141472, -0.12485688, -0.0556177],
                            ],
                            [
                                [0.0352582, -0.37993402, 0.26781956, -0.25935129],
                                [0.04351872, -0.27109361, 0.65111429, 0.4648453],
                            ],
                            [
                                [0.1909576, 0.25461839, -0.07463641, -0.34390477],
                                [-0.21279487, 0.0305474, 0.53420894, -0.66578494],
                            ],
                        ]
                    ),
                    np.array(
                        [
                            [[-0.26771292, -0.00628612], [-0.96316273, 0.02465422]],
                            [[0.96011241, 0.07601506], [-0.2663889, 0.03798452]],
                            [[-0.00727353, 0.4537835], [-0.02374101, -0.89076596]],
                            [[0.08038064, -0.88784161], [-0.02812246, -0.45220057]],
                        ]
                    ),
                    np.array([[-0.97855153, 0.2060022], [0.2060022, 0.97855153]]),
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
                2,
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
                2,
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
                2,
            ),
        ],
    )
    def test_correctness(self, mps, state, num_wires, num_work_wires):
        """Test correctness of the state approximation with MPS

        Developer Note:
        This data was generated by adapting the functionality `decompose_dense` in
        the `pennylane-lightning/pennylane_lightning/lightning_tensor/_tensornet.py`

        Args:
            mps (List[Array]):  list of arrays of rank-3 and rank-2 tensors representing an MPS state as a
                product of site matrices.
            state (Array): target state that the mps is approaching
            num_wires(int): number of wires to encode the state
            num_work_wires(int): number of auxiliary wires used in the mps decomposition
        """

        wires = qml.registers({"work": num_work_wires, "state": num_wires})
        dev = qml.device("default.qubit")

        qs = qml.tape.QuantumScript(
            qml.MPSPrep.compute_decomposition(
                mps, wires=wires["state"], work_wires=wires["work"], right_canonicalize=True
            ),
            [qml.state()],
        )
        output = dev.execute(qs)[: 2**num_wires]

        assert np.allclose(state, output, rtol=0.01)

    @pytest.mark.parametrize(
        ("mps", "num_wires"),
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
                4,
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
                    np.array([[-1.0, -0.0], [-0.0, -1.0]]),
                ],
                4,
            ),
            (
                [
                    np.array([[0.53849604, -0.44389787], [-0.59116842, -0.40434711]]),
                    np.array(
                        [
                            [
                                [-6.05052107e-01, 1.34284016e-01, -2.84018989e-01, -1.12416345e-01],
                                [3.60988555e-01, 6.14571922e-01, -1.20681653e-01, -2.89527967e-04],
                            ],
                            [
                                [-1.12393068e-01, 2.11496619e-01, 3.99193070e-01, -3.44891522e-01],
                                [-5.99232567e-01, 3.76491467e-01, 3.04813277e-01, 2.65697349e-01],
                            ],
                        ]
                    ),
                    np.array(
                        [
                            [
                                [0.87613189, -0.34254341, -0.12704983, -0.0161698],
                                [-0.20758717, 0.1329479, -0.18184107, 0.06942658],
                            ],
                            [
                                [-0.16499137, -0.13680142, -0.18432824, 0.12950892],
                                [-0.68790868, -0.64141472, -0.12485688, -0.0556177],
                            ],
                            [
                                [0.0352582, -0.37993402, 0.26781956, -0.25935129],
                                [0.04351872, -0.27109361, 0.65111429, 0.4648453],
                            ],
                            [
                                [0.1909576, 0.25461839, -0.07463641, -0.34390477],
                                [-0.21279487, 0.0305474, 0.53420894, -0.66578494],
                            ],
                        ]
                    ),
                    np.array(
                        [
                            [[-0.26771292, -0.00628612], [-0.96316273, 0.02465422]],
                            [[0.96011241, 0.07601506], [-0.2663889, 0.03798452]],
                            [[-0.00727353, 0.4537835], [-0.02374101, -0.89076596]],
                            [[0.08038064, -0.88784161], [-0.02812246, -0.45220057]],
                        ]
                    ),
                    np.array([[-0.97855153, 0.2060022], [0.2060022, 0.97855153]]),
                ],
                5,
            ),
        ],
    )
    def test_decomposition_new(self, mps, num_wires):
        """Tests the decomposition rule implemented with the new system."""
        op = qml.MPSPrep(
            mps, wires=range(2, num_wires + 2), work_wires=[0, 1], right_canonicalize=True
        )
        for rule in qml.list_decomps(qml.MPSPrep):
            _test_decomposition_rule(op, rule)

    @pytest.mark.capture
    @pytest.mark.usefixtures("enable_graph_decomposition")
    def test_decomposition_capture(self):
        """Tests that the new decomposition works with capture."""

        from pennylane.tape.plxpr_conversion import CollectOpsandMeas

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
        num_wires = 4

        def circuit(*_mps):
            _mps_prep_decomposition(
                *_mps, wires=range(2, num_wires + 2), work_wires=[0, 1], right_canonicalize=True
            )

        plxpr = qml.capture.make_plxpr(circuit)(*mps)
        collector = CollectOpsandMeas()
        collector.eval(plxpr.jaxpr, plxpr.consts, *mps)
        assert len(collector.state["ops"]) == 4
        for op in collector.state["ops"]:
            assert isinstance(op, qml.QubitUnitary)

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

    @pytest.mark.parametrize(
        ("work_wires", "msg"),
        [(None, "The qml.MPSPrep decomposition requires"), (1, "Incorrect number of `work_wires`")],
    )
    def test_wires_decomposition(self, work_wires, msg):
        """Checks that error is shown if no `work_wires` are given in decomposition"""

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

        op = qml.MPSPrep(mps, wires=range(2, 5), work_wires=work_wires)
        with pytest.raises(ValueError, match=msg):
            op.decomposition()

    def test_right_canonical(self):
        """Checks that the function `right_canonicalize_mps` generates MPS in the correct format"""

        n_sites = 4
        mps = (
            [np.ones((2, 4))]
            + [np.ones((4, 2, 4)) for _ in range(1, n_sites - 1)]
            + [np.ones((4, 2))]
        )
        mps_rc = right_canonicalize_mps(mps)

        for i in range(1, n_sites - 1):
            tensor = mps_rc[i]

            # Right-canonical definition
            contraction_matrix = np.tensordot(tensor, tensor.conj(), axes=([1, 2], [1, 2]))
            assert np.allclose(contraction_matrix, np.eye(tensor.shape[0]))

    @pytest.mark.jax
    def test_right_canonical_jax_jit(self):
        """Checks that the function `right_canonicalize_mps` works with JAX and JIT"""

        import jax
        from jax import numpy as jnp

        n_sites = 4
        mps = (
            [jnp.ones((2, 4))]
            + [jnp.ones((4, 2, 4)) for _ in range(1, n_sites - 1)]
            + [jnp.ones((4, 2))]
        )
        mps_rc = jax.jit(right_canonicalize_mps)(mps)

        for i in range(1, n_sites - 1):
            tensor = mps_rc[i]

            # Right-canonical definition
            contraction_matrix = np.tensordot(tensor, tensor.conj(), axes=([1, 2], [1, 2]))
            assert qml.math.allclose(contraction_matrix, qml.math.eye(tensor.shape[0]))

    def test_immutable_input(self):
        """Verifies that the input MPS remains unchanged after processing."""

        n_sites = 4
        mps = (
            [np.ones((2, 4))]
            + [np.ones((4, 2, 4)) for _ in range(1, n_sites - 1)]
            + [np.ones((4, 2))]
        )
        mps_copy = mps.copy()
        _ = right_canonicalize_mps(mps)

        for tensor1, tensor2 in zip(mps, mps_copy):
            assert qml.math.allclose(tensor1, tensor2)

        wires = qml.registers({"work": 2, "state": n_sites})
        dev = qml.device("default.qubit")

        qs = qml.tape.QuantumScript(
            qml.MPSPrep.compute_decomposition(
                mps, wires=wires["state"], work_wires=wires["work"], right_canonicalize=True
            ),
            [qml.state()],
        )
        _ = dev.execute(qs)

        for tensor1, tensor2 in zip(mps, mps_copy):
            assert qml.math.allclose(tensor1, tensor2)
