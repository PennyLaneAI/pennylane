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
Tests for the MPS template.
"""
import math
import pytest
import numpy as np
import pennylane as qml
from pennylane.templates.tensornetworks.mps import *


class TestIndicesMPS:
    """Test function that computes MPS indices"""

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
            compute_indices_MPS(range(n_wires), n_block_wires)

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
            compute_indices_MPS(range(n_wires), n_block_wires)

    def test_exception_n_block_wires_small(self):
        """Verifies that an exception is raised when n_block_wires is less than 2."""

        n_wires = 2
        n_block_wires = 0
        with pytest.raises(
            ValueError,
            match=f"number of wires in each block must be larger than or equal to 2; "
            f"got n_block_wires = {n_block_wires}",
        ):
            compute_indices_MPS(range(n_wires), n_block_wires)

    @pytest.mark.parametrize(
        ("n_wires", "n_block_wires"),
        [
            (5, 4),
            (9, 4),
            (7, 6),
        ],
    )
    def test_warning_many_wires(self, n_wires, n_block_wires):
        """Verifies that a warning is raised if n_wires doesn't correspond to n_block_wires."""

        with pytest.warns(
            Warning,
            match=f"The number of wires should be a multiple of {int(n_block_wires/2)}; "
            f"got {n_wires}",
        ):
            compute_indices_MPS(range(n_wires), n_block_wires)

    @pytest.mark.parametrize(
        ("wires", "n_block_wires", "expected_indices"),
        [
            ([1, 2, 3, 4], 2, [[1, 2], [2, 3], [3, 4]]),
            (["a", "b", "c", "d"], 2, [["a", "b"], ["b", "c"], ["c", "d"]]),
        ],
    )
    def test_indices_output(self, wires, n_block_wires, expected_indices):
        """Verifies the indices are correct for both integer and string wire labels."""
        indices = compute_indices_MPS(wires, n_block_wires)
        for i in range(len(expected_indices)):
            assert all(indices[i] == expected_indices[i])


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
            MPS(wires, n_block_wires, block, n_params_block)

    def test_warning_many_wires(self):
        """Verifies that a warning is raised if n_wires doesn't correspond to n_block_wires."""

        n_block_wires = 4
        wires = [1, 2, 3, 4, 5]
        n_wires = len(wires)
        n_params_block = 1
        with pytest.warns(
            Warning,
            match=f"The number of wires should be a multiple of {int(n_block_wires/2)}; got {n_wires}",
        ):
            MPS(wires, n_block_wires, block=None, n_params_block=n_params_block)

    @pytest.mark.parametrize(
        ("block", "n_params_block", "wires", "n_block_wires", "block_weights", "msg_match"),
        [
            (
                None,
                2,
                [1, 2, 3, 4],
                2,
                [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
                "Weights tensor must have last dimension of length 2; got 3",
            ),
            (
                None,
                2,
                [1, 2, 3, 4],
                2,
                [[1, 2], [2, 3], [4, 5], [6, 7]],
                "Weights tensor must have first dimension of length 3; got 4",
            ),
        ],
    )
    def test_exception_wrong_weight_shape(
        self, block, n_params_block, wires, n_block_wires, block_weights, msg_match
    ):
        """Verifies that an exception is raised if the weights shape is incorrect."""
        with pytest.raises(ValueError, match=msg_match):
            MPS(wires, n_block_wires, block, n_params_block, block_weights)


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
            match=f"The number of wires should be a multiple of {int(n_block_wires/2)}; "
            f"got {len(wires)}",
        ):
            qml.MPS.get_n_blocks(wires, n_block_wires)

    @pytest.mark.filterwarnings("ignore")
    @pytest.mark.parametrize(
        ("wires", "n_block_wires", "expected_n_blocks"),
        [
            (range(4), 2, 3),
            (range(5), 2, 4),
            (range(6), 2, 5),
            (range(10), 4, 4),
            (range(11), 4, 4),
        ],
    )
    def test_get_n_blocks(self, wires, n_block_wires, expected_n_blocks):
        """Test that the number of blocks attribute returns the correct number of blocks."""

        assert qml.MPS.get_n_blocks(wires, n_block_wires) == expected_n_blocks

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
            qml.MPS.get_n_blocks(wires, n_block_wires)


class TestTemplateOutputs:
    def circuit1_block(weights, wires):
        qml.RZ(weights[0], wires=wires[0])
        qml.RZ(weights[1], wires=wires[1])

    def circuit1_MPS(weights, wires):
        qml.RZ(weights[0][0], wires=wires[0])
        qml.RZ(weights[0][1], wires=wires[1])
        qml.RZ(weights[1][0], wires=wires[1])
        qml.RZ(weights[1][1], wires=wires[2])
        qml.RZ(weights[2][0], wires=wires[2])
        qml.RZ(weights[2][1], wires=wires[3])

    def circuit2_block(weights, wires):
        SELWeights = np.array(
            [[[weights[0], weights[1], weights[2]], [weights[0], weights[1], weights[2]]]]
        )
        qml.StronglyEntanglingLayers(SELWeights, wires)

    def circuit2_MPS(weights, wires):
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
        qml.StronglyEntanglingLayers(SELWeights1, wires=wires[0:2])
        qml.StronglyEntanglingLayers(SELWeights2, wires=wires[1:3])

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
                circuit1_block,
                2,
                [1, 2, 3, 4],
                2,
                [[0.1, 0.2], [-0.2, 0.3], [0.3, 0.4]],
                circuit1_MPS,
            ),
            (circuit2_block, 3, [1, 2, 3], 2, [[0.1, 0.2, 0.3], [0.2, 0.3, -0.4]], circuit2_MPS),
        ],
    )
    def test_output(
        self, block, n_params_block, wires, n_block_wires, template_weights, expected_circuit
    ):
        """Verifies that the output of the circuits is correct."""
        dev = qml.device("default.qubit", wires=wires)

        @qml.qnode(dev)
        def circuit():
            qml.MPS(wires, n_block_wires, block, n_params_block, template_weights)
            return qml.expval(qml.PauliZ(wires=wires[-1]))

        template_result = circuit()

        @qml.qnode(dev)
        def circuit():
            expected_circuit(template_weights, wires)
            return qml.expval(qml.PauliZ(wires=wires[-1]))

        manual_result = circuit()
        assert np.isclose(template_result, manual_result)
