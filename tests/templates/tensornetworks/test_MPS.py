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
import numpy as np

# pylint: disable=too-many-arguments
import pytest

import pennylane as qml
from pennylane.templates.tensornetworks.mps import MPS, compute_indices_MPS


# pylint: disable=protected-access
def test_flatten_unflatten():
    """Test the flatten and unflatten methods."""

    def block(weights, wires, use_CNOT=True):
        if use_CNOT:
            qml.CNOT(wires=[wires[0], wires[1]])
        qml.RY(weights[0], wires=wires[0])
        qml.RY(weights[1], wires=wires[1])

    n_wires = 4
    n_block_wires = 2
    n_params_block = 2
    n_blocks = qml.MPS.get_n_blocks(range(n_wires), n_block_wires)
    template_weights = [[0.1, -0.3]] * n_blocks

    wires = qml.wires.Wires((0, 1, 2, 3))

    op = qml.MPS(wires, n_block_wires, block, n_params_block, template_weights, use_CNOT=True)

    data, metadata = op._flatten()
    assert len(data) == 1
    assert qml.math.allclose(data[0], template_weights)

    assert metadata[0] == wires
    assert dict(metadata[1]) == op.hyperparameters

    # make sure metadata hashable
    assert hash(metadata)

    new_op = qml.MPS._unflatten(*op._flatten())
    qml.assert_equal(new_op, op)
    assert new_op._name == "MPS"  # make sure acutally initialized
    assert new_op is not op


class TestIndicesMPS:
    """Test function that computes MPS indices"""

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
        ("n_wires", "n_block_wires", "offset"),
        [
            (18, 6, 6),
            (12, 4, 0),
            (10, 4, 4),
        ],
    )
    def test_exception_offset(self, n_wires, n_block_wires, offset):
        """Verifies that an exception is raised when abs(offset) is more than n_block_wires / 2."""

        with pytest.raises(
            ValueError,
            match="Provided offset is outside the expected range; "
            f"the expected range for n_block_wires = {n_block_wires}",
        ):
            compute_indices_MPS(range(n_wires), n_block_wires, offset)

    @pytest.mark.parametrize(
        ("wires", "n_block_wires", "offset", "expected_indices"),
        [
            ([1, 2, 3, 4], 2, 1, ((1, 2), (2, 3), (3, 4))),
            (["a", "b", "c", "d"], 2, 1, (("a", "b"), ("b", "c"), ("c", "d"))),
            ([1, 2, 3, 4, 5, 6, 7, 8], 4, 3, ((1, 2, 3, 4), (4, 5, 6, 7))),
            (
                ["a", "b", "c", "d", "e", "f", "g", "h"],
                4,
                1,
                (
                    ("a", "b", "c", "d"),
                    ("b", "c", "d", "e"),
                    ("c", "d", "e", "f"),
                    ("d", "e", "f", "g"),
                    ("e", "f", "g", "h"),
                ),
            ),
        ],
    )
    def test_indices_output(self, wires, n_block_wires, offset, expected_indices):
        """Verifies the indices are correct for both integer and string wire labels."""
        indices = compute_indices_MPS(wires, n_block_wires, offset)
        assert indices == expected_indices


class TestTemplateInputs:
    """Test template inputs and pre-processing (ensure the correct exceptions are thrown for the inputs)"""

    @pytest.mark.parametrize(
        ("block", "n_params_block", "wires", "n_block_wires", "offset", "msg_match"),
        [
            (
                None,
                None,
                [1, 2, 3, 4],
                6,
                None,
                "n_block_wires must be smaller than or equal to the number of wires; "
                "got n_block_wires = 6 and number of wires = 4",
            ),
            (
                None,
                None,
                [1, 2, 3, 4],
                0,
                None,
                "The number of wires in each block must be larger than or equal to 2; "
                "got n_block_wires = 0",
            ),
            (
                None,
                None,
                [1, 2, 3, 4, 5, 6, 7, 8],
                4,
                4,
                "Provided offset is outside the expected range; ",
            ),
        ],
    )
    def test_exception_wrong_input(
        self, block, n_params_block, wires, n_block_wires, offset, msg_match
    ):
        """Verifies that an exception is raised if the number of wires or n_block_wires is incorrect."""
        with pytest.raises(ValueError, match=msg_match):
            MPS(wires, n_block_wires, block, n_params_block, offset=offset)

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
        """Test that the number of blocks attribute raises an error when
        n_block_wires is too large."""

        with pytest.raises(
            ValueError,
            match=f"n_block_wires must be smaller than or equal to the number of wires; "
            f"got n_block_wires = {n_block_wires} and number of wires = {len(wires)}",
        ):
            qml.MPS.get_n_blocks(wires, n_block_wires)

    @pytest.mark.filterwarnings("ignore")
    @pytest.mark.parametrize(
        ("wires", "n_block_wires", "offset", "expected_n_blocks"),
        [
            (range(14), 4, 1, 11),
            (range(15), 4, 3, 4),
            (range(18), 6, 1, 13),
            (range(20), 6, 5, 3),
        ],
    )
    def test_get_n_blocks_with_offset(self, wires, n_block_wires, offset, expected_n_blocks):
        """Test that the number of blocks attribute returns the correct number of blocks with offset."""

        assert qml.MPS.get_n_blocks(wires, n_block_wires, offset) == expected_n_blocks

    @pytest.mark.filterwarnings("ignore")
    @pytest.mark.parametrize(
        ("wires", "n_block_wires", "offset"),
        [(range(12), 6, 6), (range(9), 4, 0)],
    )
    def test_get_n_blocks_error_with_offset(self, wires, offset, n_block_wires):
        """Test that the number of blocks attribute raises an error when offset is out of bounds."""

        with pytest.raises(
            ValueError,
            match=r"Provided offset is outside the expected range; "
            f"the expected range for n_block_wires = {n_block_wires}",
        ):
            qml.MPS.get_n_blocks(wires, n_block_wires, offset)


class TestTemplateOutputs:
    """Test the output of the MPS template."""

    @staticmethod
    def circuit1_block(weights, wires):
        qml.RZ(weights[0], wires=wires[0])
        qml.RZ(weights[1], wires=wires[1])

    @staticmethod
    def circuit1_MPS(weights, wires):
        qml.RZ(weights[0][0], wires=wires[0])
        qml.RZ(weights[0][1], wires=wires[1])
        qml.RZ(weights[1][0], wires=wires[1])
        qml.RZ(weights[1][1], wires=wires[2])
        qml.RZ(weights[2][0], wires=wires[2])
        qml.RZ(weights[2][1], wires=wires[3])

    @staticmethod
    def circuit2_block(weights, wires):
        SELWeights = np.array(
            [[[weights[0], weights[1], weights[2]], [weights[0], weights[1], weights[2]]]]
        )
        qml.StronglyEntanglingLayers(SELWeights, wires)

    @staticmethod
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

    @staticmethod
    def circuit3_block(wires, k=None):
        qml.MultiControlledX(wires=[wires[i] for i in range(len(wires))])
        assert k == 2

    @staticmethod
    def circuit3_MPS(wires, **kwargs):  # pylint: disable=unused-argument
        qml.MultiControlledX(wires=[wires[0], wires[1], wires[2], wires[3]])
        qml.MultiControlledX(wires=[wires[1], wires[2], wires[3], wires[4]])
        qml.MultiControlledX(wires=[wires[2], wires[3], wires[4], wires[5]])
        qml.MultiControlledX(wires=[wires[3], wires[4], wires[5], wires[6]])
        qml.MultiControlledX(wires=[wires[4], wires[5], wires[6], wires[7]])

    @staticmethod
    def circuit4_MPS(wires, **kwargs):  # pylint: disable=unused-argument
        qml.MultiControlledX(wires=[wires[0], wires[1], wires[2], wires[3]])
        qml.MultiControlledX(wires=[wires[2], wires[3], wires[4], wires[5]])
        qml.MultiControlledX(wires=[wires[4], wires[5], wires[6], wires[7]])

    @pytest.mark.parametrize(
        (
            "block",
            "n_params_block",
            "wires",
            "n_block_wires",
            "template_weights",
            "offset",
            "kwargs",
            "expected_circuit",
        ),
        [
            (
                "circuit1_block",
                2,
                [1, 2, 3, 4],
                2,
                [[0.1, 0.2], [-0.2, 0.3], [0.3, 0.4]],
                None,
                {},
                "circuit1_MPS",
            ),
            (
                "circuit2_block",
                3,
                [1, 2, 3],
                2,
                [[0.1, 0.2, 0.3], [0.2, 0.3, -0.4]],
                None,
                {},
                "circuit2_MPS",
            ),
            (
                "circuit3_block",
                0,
                [1, 2, 3, 4, 5, 6, 7, 8],
                4,
                None,
                1,
                {"k": 2},
                "circuit3_MPS",
            ),
            (
                "circuit3_block",
                0,
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
                5,
                None,
                2,
                {"k": 2},
                "circuit4_MPS",
            ),
        ],
    )
    def test_output(
        self,
        block,
        n_params_block,
        wires,
        n_block_wires,
        template_weights,
        offset,
        kwargs,
        expected_circuit,
    ):
        """Verifies that the output of the circuits is correct."""
        dev = qml.device("default.qubit", wires=wires)
        block = getattr(self, block)
        expected_circuit = getattr(self, expected_circuit)

        @qml.qnode(dev)
        def circuit_template():
            qml.MPS(
                wires,
                n_block_wires,
                block,
                n_params_block,
                template_weights,
                offset=offset,
                **kwargs,
            )
            return qml.expval(qml.PauliZ(wires=wires[-1]))

        template_result = circuit_template()

        @qml.qnode(dev)
        def circuit_manual():
            expected_circuit(weights=template_weights, wires=wires)
            return qml.expval(qml.PauliZ(wires=wires[-1]))

        manual_result = circuit_manual()
        assert np.isclose(template_result, manual_result)

    @pytest.mark.parametrize(
        (
            "block",
            "n_params_block",
            "wires",
            "n_block_wires",
            "template_weights",
            "offset",
            "kwargs",
            "expected_circuit",
        ),
        [
            (
                "circuit1_block",
                2,
                [1, 2, 3, 4],
                2,
                [[0.1, 0.2], [-0.2, 0.3], [0.3, 0.4]],
                None,
                {},
                "circuit1_MPS",
            ),
            (
                "circuit2_block",
                3,
                [1, 2, 3],
                2,
                [[0.1, 0.2, 0.3], [0.2, 0.3, -0.4]],
                None,
                {},
                "circuit2_MPS",
            ),
            (
                "circuit3_block",
                0,
                [1, 2, 3, 4, 5, 6, 7, 8],
                4,
                None,
                1,
                {"k": 2},
                "circuit3_MPS",
            ),
            (
                "circuit3_block",
                0,
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
                5,
                None,
                2,
                {"k": 2},
                "circuit4_MPS",
            ),
        ],
    )
    @pytest.mark.jax
    def test_jax_jit(
        self,
        block,
        n_params_block,
        wires,
        n_block_wires,
        template_weights,
        offset,
        kwargs,
        expected_circuit,
    ):
        import jax

        dev = qml.device("default.qubit", wires=wires)
        block = getattr(self, block)
        expected_circuit = getattr(self, expected_circuit)

        @qml.qnode(dev)
        def circuit():
            qml.MPS(
                wires,
                n_block_wires,
                block,
                n_params_block,
                template_weights,
                offset=offset,
                **kwargs,
            )
            return qml.expval(qml.PauliZ(wires=wires[-1]))

        jit_circuit = jax.jit(circuit)

        assert qml.math.isclose(circuit(), jit_circuit())
