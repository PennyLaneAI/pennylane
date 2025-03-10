# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the decomposition graph."""

# pylint: disable=protected-access

from unittest.mock import patch

import numpy as np
import pytest

import pennylane as qml
from pennylane.decomposition import DecompositionGraph
from conftest import Hadamard, decompositions


@pytest.mark.unit
def test_get_decomp_rule():
    """Tests the internal method that gets the decomposition rules for an operator."""

    @qml.register_resources({qml.PhaseShift: 2, qml.RX: 1})
    def custom_hadamard(wires):
        qml.PhaseShift(np.pi / 2, wires=wires)
        qml.RX(np.pi / 2, wires=wires)
        qml.PhaseShift(np.pi / 2, wires=wires)

    @qml.register_resources({qml.PhaseShift: 1, qml.RY: 1})
    def custom_hadamard_2(wires):
        qml.PhaseShift(np.pi / 2, wires=wires)
        qml.RY(np.pi / 2, wires=wires)

    graph = DecompositionGraph(operations=[qml.H(0)], target_gate_set={"RX", "RY", "RZ"})
    assert graph._get_decompositions(qml.H) == qml.list_decomps(qml.H)

    graph = DecompositionGraph(
        operations=[qml.H(0)],
        target_gate_set={"RX", "RY", "RZ"},
        fixed_decomps={qml.H: custom_hadamard},
    )
    assert graph._get_decompositions(qml.H) == [custom_hadamard]

    graph = DecompositionGraph(
        operations=[qml.H(0)],
        target_gate_set={"RX", "RY", "RZ"},
        alt_decomps={qml.H: [custom_hadamard, custom_hadamard_2]},
    )
    assert graph._get_decompositions(qml.H) == [
        custom_hadamard,
        custom_hadamard_2,
    ] + qml.list_decomps(qml.H)


@patch(
    "pennylane.decomposition.decomposition_graph.list_decomps",
    side_effect=lambda x: decompositions[x],
)
class TestGraphConstruction:  # pylint: disable=too-few-public-methods
    """Unit tests for constructing the graph."""

    def test_single_op_construction(self, _):
        """Tests constructing a graph from a single Hadamard."""

        op = Hadamard(wires=[0])
        graph = DecompositionGraph(operations=[op], target_gate_set={"RX", "RZ", "GlobalPhase"})
        assert len(graph._graph.nodes()) == 8
        assert len(graph._graph.edges()) == 11

        # Check that graph construction stops at gates in the target gate set.
        graph2 = DecompositionGraph(operations=[op], target_gate_set={"RY", "RZ", "GlobalPhase"})
        assert len(graph2._graph.nodes()) == 7
        assert len(graph2._graph.edges()) == 8
