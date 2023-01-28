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
Tests for the MERA template.
"""
import pytest
import numpy as np
import pennylane as qml
from pennylane.templates.tensornetworks.mera import *


def circuit0_block(wires):
    qml.PauliX(wires=wires[1])
    qml.PauliZ(wires=wires[0])


def circuit1_block(weights1, weights2, weights3, wires):
    qml.RX(weights1, wires=wires[0])
    qml.RX(weights2, wires=wires[1])
    qml.RY(weights3, wires=wires[1])


def circuit2_block(weights, wires):
    qml.RY(weights[0], wires=wires[0])
    qml.RY(weights[1], wires=wires[1])


def circuit2_MERA(weights, wires):
    qml.RY(weights[0][0], wires=wires[1])
    qml.RY(weights[0][1], wires=wires[0])
    qml.RY(weights[1][0], wires=wires[2])
    qml.RY(weights[1][1], wires=wires[3])
    qml.RY(weights[2][0], wires=wires[3])
    qml.RY(weights[2][1], wires=wires[1])
    qml.RY(weights[3][0], wires=wires[0])
    qml.RY(weights[3][1], wires=wires[2])
    qml.RY(weights[4][0], wires=wires[0])
    qml.RY(weights[4][1], wires=wires[1])


def circuit3_block(weights, wires):
    SELWeights = np.array(
        [[[weights[0], weights[1], weights[2]], [weights[0], weights[1], weights[2]]]]
    )
    qml.StronglyEntanglingLayers(SELWeights, wires)


def circuit3_MERA(weights, wires):
    SELWeights1 = np.array(
        [
            [
                [weights[0][0], weights[0][1], weights[0][2]],
                [weights[0][0], weights[0][1], weights[0][2]],
            ]
        ]
    )
    SELWeights2 = np.array(
        [
            [
                [weights[1][0], weights[1][1], weights[1][2]],
                [weights[1][0], weights[1][1], weights[1][2]],
            ]
        ]
    )
    SELWeights3 = np.array(
        [
            [
                [weights[2][0], weights[2][1], weights[2][2]],
                [weights[2][0], weights[2][1], weights[2][2]],
            ]
        ]
    )
    SELWeights4 = np.array(
        [
            [
                [weights[3][0], weights[3][1], weights[3][2]],
                [weights[3][0], weights[3][1], weights[3][2]],
            ]
        ]
    )
    SELWeights5 = np.array(
        [
            [
                [weights[4][0], weights[4][1], weights[4][2]],
                [weights[4][0], weights[4][1], weights[4][2]],
            ]
        ]
    )
    qml.StronglyEntanglingLayers(SELWeights1, wires=wires[1::-1])
    qml.StronglyEntanglingLayers(SELWeights2, wires=wires[2::])
    qml.StronglyEntanglingLayers(SELWeights3, wires=[wires[3], wires[1]])
    qml.StronglyEntanglingLayers(SELWeights4, wires=[wires[0], wires[2]])
    qml.StronglyEntanglingLayers(SELWeights5, wires=[wires[0], wires[1]])


class TestIndicesMERA:
    """Test function that computes MERA indices"""

    @pytest.mark.parametrize(
        ("n_wires", "n_block_wires"),
        [
            (5, 3),
            (9, 5),
            (11, 7),
        ],
    )
    def test_exception_n_block_wires_uneven(self, n_wires, n_block_wires):
        """Verifies that an exception is raised if n_block_wires is not even."""

        with pytest.raises(
            ValueError, match=f"n_block_wires must be an even integer; got {n_block_wires}"
        ):
            compute_indices(range(n_wires), n_block_wires)

    @pytest.mark.parametrize(
        ("n_wires", "n_block_wires"),
        [
            (3, 4),
            (6, 8),
            (10, 14),
        ],
    )
    def test_exception_n_block_wires_large(self, n_wires, n_block_wires):
        """Verifies that an exception is raised when n_block_wires is too large."""

        with pytest.raises(
            ValueError,
            match="n_block_wires must be smaller than or equal to the number of wires; "
            f"got n_block_wires = {n_block_wires} and number of wires = {n_wires}",
        ):
            compute_indices(range(n_wires), n_block_wires)

    def test_exception_n_block_wires_small(self):
        """Verifies that an exception is raised when n_block_wires is less than 2."""

        n_wires = 2
        n_block_wires = 0
        with pytest.raises(
            ValueError,
            match=f"number of wires in each block must be larger than or equal to 2; "
            f"got n_block_wires = {n_block_wires}",
        ):
            compute_indices(range(n_wires), n_block_wires)

    @pytest.mark.parametrize(
        ("wires", "n_block_wires"),
        [(range(5), 2), (range(12), 4), (range(16), 6)],
    )
    def test_warning_many_wires(self, wires, n_block_wires):
        """Verifies that a warning is raised if n_wires doesn't correspond to n_block_wires."""
        n_wires = len(wires)
        with pytest.warns(
            Warning,
            match=f"The number of wires should be n_block_wires times 2\\^n; got n_wires/n_block_wires = {n_wires/n_block_wires}",
        ):
            compute_indices(range(n_wires), n_block_wires)

    @pytest.mark.parametrize(
        ("wires", "n_block_wires", "expected_indices"),
        [
            ([1, 2, 3, 4], 2, [[2, 1], [3, 4], [4, 2], [1, 3], [1, 2]]),
            (
                range(12),
                6,
                [
                    [3, 4, 5, 0, 1, 2],
                    [6, 7, 8, 9, 10, 11],
                    [9, 10, 11, 3, 4, 5],
                    [0, 1, 2, 6, 7, 8],
                    [0, 1, 2, 3, 4, 5],
                ],
            ),
            (["a", "b", "c", "d"], 2, [["b", "a"], ["c", "d"], ["d", "b"], ["a", "c"], ["a", "b"]]),
        ],
    )
    def test_indices_output(self, wires, n_block_wires, expected_indices):
        """Verifies the indices are correct for both integer and string wire labels."""
        indices = compute_indices(wires, n_block_wires)

        assert indices == expected_indices


class TestTemplateInputs:
    """Test template inputs and pre-processing (ensure the correct exceptions are thrown for the inputs)"""

    @pytest.mark.parametrize(
        ("block", "n_params_block", "wires", "n_block_wires", "msg_match"),
        [
            (None, None, [1, 2, 3, 4], 7, "n_block_wires must be an even integer; got 7"),
            (
                None,
                None,
                [1, 2, 3, 4],
                6,
                "n_block_wires must be smaller than or equal to the number of wires; "
                "got n_block_wires = 6 and number of wires = 4",
            ),
            (
                None,
                None,
                [1, 2, 3, 4],
                0,
                "number of wires in each block must be larger than or equal to 2; "
                "got n_block_wires = 0",
            ),
        ],
    )
    def test_exception_wrong_input(self, block, n_params_block, wires, n_block_wires, msg_match):
        """Verifies that an exception is raised if the number of wires or n_block_wires is incorrect."""
        with pytest.raises(ValueError, match=msg_match):
            MERA(wires, n_block_wires, block, n_params_block)

    def test_warning_many_wires(self):
        """Verifies that a warning is raised if n_wires doesn't correspond to n_block_wires."""

        n_block_wires = 4
        wires = [1, 2, 3, 4, 5]
        n_wires = len(wires)
        n_params_block = 1
        with pytest.warns(
            Warning,
            match=f"The number of wires should be n_block_wires times 2\\^n; "
            f"got n_wires/n_block_wires = {n_wires/n_block_wires}",
        ):
            MERA(wires, n_block_wires, block=None, n_params_block=n_params_block)

    @pytest.mark.parametrize(
        ("block", "n_params_block", "wires", "n_block_wires", "block_weights", "msg_match"),
        [
            (
                None,
                2,
                [1, 2, 3, 4],
                2,
                [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
                "Weights tensor must have last dimension of length 2; got 3",
            ),
            (
                None,
                2,
                [1, 2, 3, 4],
                2,
                [[1, 2], [2, 3], [4, 5], [6, 7]],
                "Weights tensor must have first dimension of length 5; got 4",
            ),
        ],
    )
    def test_exception_wrong_weight_shape(
        self, block, n_params_block, wires, n_block_wires, block_weights, msg_match
    ):
        """Verifies that an exception is raised if the weights shape is incorrect."""
        with pytest.raises(ValueError, match=msg_match):
            MERA(wires, n_block_wires, block, n_params_block, block_weights)

    @pytest.mark.parametrize(
        ("block", "n_params_block", "wires", "n_block_wires", "template_weights"),
        [
            (circuit0_block, 0, [1, 2, 3, 4], 2, None),
            (
                circuit1_block,
                3,
                [1, 2, 3, 4],
                2,
                [
                    [0.1, 0.1, 0.2],
                    [0.2, 0.2, 0.3],
                    [0.2, 0.3, 0.1],
                    [0.1, 0.1, 0.2],
                    [0.2, 0.2, 0.3],
                ],
            ),
            (
                circuit2_block,
                2,
                [0, 1, 2, 3],
                2,
                [[0.1, 0.2], [-0.2, 0.3], [0.3, 0.4], [-0.2, 0.3], [0.1, 0.2]],
            ),
            (
                circuit3_block,
                3,
                [1, 2, 3, 4],
                2,
                [
                    [0.1, 0.2, 0.3],
                    [0.2, 0.3, -0.4],
                    [0.5, 0.2, 0.3],
                    [0.2, 0.3, -0.4],
                    [0.1, 0.2, 0.3],
                ],
            ),
        ],
    )
    def test_block_params(self, block, n_params_block, wires, n_block_wires, template_weights):
        """Verify that the template works with arbitrary block parameters"""
        dev = qml.device("default.qubit", wires=wires)

        @qml.qnode(dev)
        def circuit():
            qml.MERA(wires, n_block_wires, block, n_params_block, template_weights)
            return qml.expval(qml.PauliZ(wires=wires[-1]))

        circuit()


class TestAttributes:
    """Tests additional methods and attributes"""

    @pytest.mark.parametrize(
        ("wires", "n_block_wires"),
        [(range(7), 4), (range(13), 6)],
    )
    def test_get_n_blocks_warning(self, wires, n_block_wires):
        """Test that get_n_blocks() warns the user when there are too many wires."""
        with pytest.warns(
            Warning,
            match=f"The number of wires should be n_block_wires times 2\\^n; "
            f"got n_wires/n_block_wires = {len(wires)/n_block_wires}",
        ):
            qml.TTN.get_n_blocks(wires, n_block_wires)

    @pytest.mark.filterwarnings("ignore")
    @pytest.mark.parametrize(
        ("wires", "n_block_wires", "expected_n_blocks"),
        [
            (range(4), 2, 5),
            (range(5), 2, 5),
            (range(6), 2, 5),
            (range(10), 4, 5),
            (range(25), 6, 13),
        ],
    )
    def test_get_n_blocks(self, wires, n_block_wires, expected_n_blocks):
        """Test that the number of blocks attribute returns the correct number of blocks."""
        assert qml.MERA.get_n_blocks(wires, n_block_wires) == expected_n_blocks

    @pytest.mark.filterwarnings("ignore")
    @pytest.mark.parametrize(
        ("wires", "n_block_wires"),
        [(range(4), 5), (range(9), 20)],
    )
    def test_get_n_blocks_error(self, wires, n_block_wires):
        """Test that the number of blocks attribute raises an error when n_block_wires is too large."""

        with pytest.raises(
            ValueError,
            match=f"n_block_wires must be smaller than or equal to the number of wires; "
            f"got n_block_wires = {n_block_wires} and number of wires = {len(wires)}",
        ):
            qml.MERA.get_n_blocks(wires, n_block_wires)


class TestDifferentiability:
    """Test that the template is differentiable."""

    @pytest.mark.parametrize(
        ("block", "n_params_block", "wires", "n_block_wires", "template_weights"),
        [
            (
                circuit2_block,
                2,
                [0, 1, 2, 3],
                2,
                [[0.1, 0.2], [-0.2, 0.3], [0.3, 0.4], [0.1, 0.2], [-0.2, 0.3]],
            )
        ],
    )
    def test_template_differentiable(
        self, block, n_params_block, wires, n_block_wires, template_weights
    ):
        """Test that the template is differentiable for different inputs."""
        dev = qml.device("default.qubit", wires=wires)

        @qml.qnode(dev)
        def circuit(template_weights):
            qml.MERA(wires, n_block_wires, block, n_params_block, template_weights)
            return qml.expval(qml.PauliZ(wires=wires[-1]))

        qml.grad(circuit)(qml.numpy.array(template_weights, requires_grad=True))


class TestTemplateOutputs:
    @pytest.mark.parametrize(
        (
            "block",
            "n_params_block",
            "wires",
            "n_block_wires",
            "template_weights",
            "expected_circuit",
        ),
        [
            (
                circuit2_block,
                2,
                [0, 1, 2, 3],
                2,
                [[0.1, 0.2], [-0.2, 0.3], [0.3, 0.4], [0.1, 0.2], [-0.2, 0.3]],
                circuit2_MERA,
            ),
            (
                circuit3_block,
                3,
                [1, 2, 3, 4],
                2,
                [
                    [0.1, 0.2, 0.3],
                    [0.2, 0.3, -0.4],
                    [0.5, 0.2, 0.3],
                    [0.1, 0.2, 0.3],
                    [0.2, 0.3, -0.4],
                ],
                circuit3_MERA,
            ),
        ],
    )
    def test_output(
        self, block, n_params_block, wires, n_block_wires, template_weights, expected_circuit
    ):
        """Verifies that the output of the circuits is correct."""
        dev = qml.device("default.qubit", wires=wires)

        @qml.qnode(dev)
        def circuit():
            qml.MERA(wires, n_block_wires, block, n_params_block, template_weights)
            return qml.expval(qml.PauliZ(wires=wires[1]))

        template_result = circuit()

        @qml.qnode(dev)
        def circuit():
            expected_circuit(template_weights, wires)
            return qml.expval(qml.PauliZ(wires=wires[1]))

        manual_result = circuit()
        assert np.isclose(template_result, manual_result)
