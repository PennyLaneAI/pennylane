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


def test_standard_validity():
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
    op = qml.MPSPrep(mps, wires=[0, 1, 2])
    qml.ops.functions.assert_valid(op, skip_differentiation=True)


def test_access_to_param():
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
def test_MPSPrep_error(mps, msg_match):
    """Test that proper errors are raised for MPSPrep"""
    with pytest.raises(AssertionError, match=msg_match):
        qml.MPSPrep(mps, wires=[0, 1, 2])


@pytest.mark.jax
def test_jax_mps():
    """Check the operation works with jax."""

    from jax import numpy as jnp

    mps = [
        jnp.array([[0.0, 0.107], [0.994, 0.0]]),
        jnp.array(
            [
                [[0.0, 0.0, 0.0, -0.0], [1.0, 0.0, 0.0, -0.0]],
                [[0.0, 1.0, 0.0, -0.0], [0.0, 0.0, 0.0, -0.0]],
            ]
        ),
        jnp.array(
            [
                [[-1.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
                [[0.0, -1.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0]],
            ]
        ),
        jnp.array([[-1.0, -0.0], [-0.0, -1.0]]),
    ]
    _ = qml.MPSPrep(mps, wires=[0, 1, 2])
