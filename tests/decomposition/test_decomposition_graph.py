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
from conftest import (
    CustomGlobalPhase,
    CustomHadamard,
    CustomPhaseShift,
    CustomRX,
    CustomRY,
    CustomRZ,
    decompositions,
)

import pennylane as qml
from pennylane.decomposition import DecompositionGraph, Resources
from pennylane.decomposition.decomposition_graph import DecompositionError


@patch(
    "pennylane.decomposition.decomposition_graph.list_decomps",
    side_effect=lambda x: decompositions[x],
)
class TestDecompositionGraph:

    def test_get_decomp_rule(self, _):
        """Tests the internal method that gets the decomposition rules for an operator."""

        @qml.register_resources({CustomPhaseShift: 2, CustomRX: 1})
        def custom_hadamard(wires):
            CustomPhaseShift(np.pi / 2, wires=wires)
            CustomRX(np.pi / 2, wires=wires)
            CustomPhaseShift(np.pi / 2, wires=wires)

        @qml.register_resources({CustomPhaseShift: 1, CustomRY: 1})
        def custom_hadamard_2(wires):
            CustomPhaseShift(np.pi / 2, wires=wires)
            CustomRY(np.pi / 2, wires=wires)

        graph = DecompositionGraph(
            operations=[CustomHadamard(0)], target_gate_set={"RX", "RY", "RZ"}
        )
        assert graph._get_decompositions(CustomHadamard) == decompositions[CustomHadamard]

        graph = DecompositionGraph(
            operations=[CustomHadamard(0)],
            target_gate_set={"RX", "RY", "RZ"},
            fixed_decomps={CustomHadamard: custom_hadamard},
        )
        assert graph._get_decompositions(CustomHadamard) == [custom_hadamard]

        graph = DecompositionGraph(
            operations=[CustomHadamard(0)],
            target_gate_set={"RX", "RY", "RZ"},
            alt_decomps={CustomHadamard: [custom_hadamard, custom_hadamard_2]},
        )
        assert (
            graph._get_decompositions(CustomHadamard)
            == [
                custom_hadamard,
                custom_hadamard_2,
            ]
            + decompositions[CustomHadamard]
        )

    def test_graph_construction(self, _):
        """Tests constructing a graph from a single Hadamard."""

        op = CustomHadamard(wires=[0])
        graph = DecompositionGraph(operations=[op], target_gate_set={"RX", "RZ", "GlobalPhase"})
        assert len(graph._graph.nodes()) == 8
        assert len(graph._graph.edges()) == 11

        # Check that graph construction stops at gates in the target gate set.
        graph2 = DecompositionGraph(operations=[op], target_gate_set={"RY", "RZ", "GlobalPhase"})
        assert len(graph2._graph.nodes()) == 7
        assert len(graph2._graph.edges()) == 8

    def test_graph_solve(self, _):
        """Tests solving a simple graph for the optimal decompositions."""

        op = CustomHadamard(wires=[0])
        graph = DecompositionGraph(operations=[op], target_gate_set={"RX", "RZ", "GlobalPhase"})
        graph.solve()
        expected_resource = Resources(
            num_gates=4,
            gate_counts={
                qml.resource_rep(CustomRX): 1,
                qml.resource_rep(CustomRZ): 2,
                qml.resource_rep(CustomGlobalPhase): 1,
            },
        )
        assert graph.resource_estimates(op) == expected_resource
        assert graph.decomposition(op).compute_resources() == expected_resource

    def test_decomposition_not_found(self, _):
        """Tests that the correct error is raised if a decomposition isn't found."""

        op = CustomHadamard(wires=[0])
        graph = DecompositionGraph(operations=[op], target_gate_set={"RX", "RY", "GlobalPhase"})
        with pytest.raises(DecompositionError, match="Decomposition not found for {'Hadamard'}"):
            graph.solve()
