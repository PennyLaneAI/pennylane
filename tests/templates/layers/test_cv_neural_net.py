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
Unit tests for the CVNeuralNetLayers template.
"""

import numpy as np

# pylint: disable=too-few-public-methods,protected-access
import pytest

import pennylane as qp


def expected_shapes(n_layers, n_wires):
    # compute the expected shapes for a given number of wires
    n_if = n_wires * (n_wires - 1) // 2
    expected = (
        [(n_layers, n_if)] * 2
        + [(n_layers, n_wires)] * 3
        + [(n_layers, n_if)] * 2
        + [(n_layers, n_wires)] * 4
    )
    return expected


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    QUEUES = [
        (1, ["Rotation", "Squeezing", "Rotation", "Displacement", "Kerr"], [[0]] * 5),
        (
            2,
            [
                "Beamsplitter",  # Interferometer 1
                "Rotation",  # Interferometer 1
                "Rotation",  # Interferometer 1
                "Squeezing",
                "Squeezing",
                "Beamsplitter",  # Interferometer 2
                "Rotation",  # Interferometer 2
                "Rotation",  # Interferometer 2
                "Displacement",
                "Displacement",
                "Kerr",
                "Kerr",
            ],
            [[0, 1], [0], [1], [0], [1], [0, 1], [0], [1], [0], [1], [0], [1]],
        ),
    ]

    @pytest.mark.parametrize("n_wires, expected_names, expected_wires", QUEUES)
    def test_expansion(self, n_wires, expected_names, expected_wires):
        """Checks the queue for the default settings."""

        shapes = expected_shapes(1, n_wires)
        weights = [np.random.random(shape) for shape in shapes]

        op = qp.CVNeuralNetLayers(*weights, wires=range(n_wires))
        tape = qp.tape.QuantumScript(op.decomposition())

        i = 0
        for gate in tape.operations:
            if gate.name != "Interferometer":
                assert gate.name == expected_names[i]
                assert gate.wires.labels == tuple(expected_wires[i])
                i = i + 1
            else:
                for gate_inter in gate.decomposition():
                    assert gate_inter.name == expected_names[i]
                    assert gate_inter.wires.labels == tuple(expected_wires[i])
                    i = i + 1


class TestAttributes:
    """Test methods and attributes."""

    @pytest.mark.parametrize(
        "n_layers, n_wires",
        [
            (2, 3),
            (2, 1),
            (2, 2),
        ],
    )
    def test_shapes(self, n_layers, n_wires, tol):
        """Test that the shape method returns the correct shapes for
        the weight tensors"""

        shapes = qp.CVNeuralNetLayers.shape(n_layers, n_wires)
        expected = expected_shapes(n_layers, n_wires)

        assert np.allclose(shapes, expected, atol=tol, rtol=0)
