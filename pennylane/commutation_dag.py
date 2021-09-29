
# Copyright 2021 Xanadu Quantum Technologies Inc.

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
This module contains the CommutationDAG class which is used to generate a DAG (directed acyclic graph)
representation of a quantum circuit from an Operator queue.
"""

from collections import OrderedDict
import heapq
import networkx as nx
import pennylane as qml

position = OrderedDict(
    {
        "Hadamard": 0,
        "PauliX": 1,
        "PauliY": 2,
        "PauliZ": 3,
        "ctrl": 4,
    }
)

commutation_map = OrderedDict(
    {
        "Hadamard": [
            1,
            0,
            0,
            0,
            0,
        ],
        "PauliX": [
            0,
            1,
            0,
            0,
            0,
        ],
        "PauliY": [
            0,
            0,
            1,
            0,
            0,
        ],
        "PauliZ": [
            0,
            0,
            0,
            1,
            1,
        ],
        "ctrl": [
            0,
            0,
            0,
            1,
            1,
        ],
    }
)


def is_commuting(operation1, operation2):
    # Case 1 operations are disjoints
    if not bool(operation1.wires.toset().intersection(operation2.wires.toset())):
        return True

    # Case 2 both operations are controlled
    if operation1.is_controlled and operation2.is_controlled:
        control_control = bool(
            operation1.control_wires.toset().intersection(operation2.control_wires.toset())
        )
        target_target = bool(
            operation1.target_wires.toset().intersection(operation2.target_wires.toset())
        )
        control_target = bool(
            operation1.control_wires.toset().intersection(operation2.target_wires.toset())
        )
        target_control = bool(
            operation1.target_wires.toset().intersection(operation2.control_wires.toset())
        )

        # Case 2.1: disjoint targets
        if control_control and not target_target and not control_target and not target_control:
            return True

        # Case 2.2: disjoint controls
        if not control_control and target_target and not control_target and not target_control:
            return bool(
                commutation_map[operation1.is_controlled][position[operation2.is_controlled]]
            )

        # Case 2.3: targets overlap and controls overalp
        if target_target and control_control and not control_target and not target_control:
            return bool(
                commutation_map[operation1.is_controlled][position[operation2.is_controlled]]
            )

        # Case 2.4: targets and controls overlap
        if control_target and target_control and not target_target:
            return bool(commutation_map["ctrl"][position[operation2.is_controlled]]) and bool(
                commutation_map[operation1.is_controlled][position["ctrl"]]
            )

        # Case 2.5: targets overlap with and controls and targets
        if control_target and not target_control and target_target:
            return bool(commutation_map["ctrl"][position[operation2.is_controlled]]) and bool(
                commutation_map[operation1.is_controlled][position[operation2.is_controlled]]
            )

        # Case 2.6: targets overlap with and controls and targets
        if target_control and not control_target and target_target:
            return bool(commutation_map[operation1.is_controlled][position["ctrl"]]) and bool(
                commutation_map[operation1.is_controlled][position[operation2.is_controlled]]
            )

        # Case 2.7: targets overlap with control
        if target_control and not control_target and not target_target:
            return bool(commutation_map[operation1.is_controlled][position["ctrl"]])

        # Case 2.8: targets overlap with control
        if not target_control and control_target and not target_target:
            return bool(commutation_map["ctrl"][position[operation2.is_controlled]])

        # Case 2.9: targets and controls overlap with targets and controls
        if target_control and control_target and target_target:
            return (
                bool(commutation_map[operation1.is_controlled][position["ctrl"]])
                and bool(commutation_map["ctrl"][position[operation2.is_controlled]])
                and bool(
                    commutation_map[operation1.is_controlled][
                        position[operation2.is_controlled]
                    ]
                )
            )

    # Case 3: only operation 1 is controlled
    elif operation1.is_controlled:

        control_target = bool(
            operation1.control_wires.toset().intersection(operation2.wires.toset())
        )
        target_target = bool(
            operation1.target_wires.toset().intersection(operation2.wires.toset())
        )

        # Case 3.1: control and target 1 overlap with target 2
        if control_target and target_target:
            return bool(
                commutation_map[operation1.is_controlled][position[operation2.name]]
            ) and bool(commutation_map["ctrl"][position[operation2.name]])

        # Case 3.2: control operation 1 overlap with target 2
        if control_target and not target_target:
            return bool(commutation_map["ctrl"][position[operation2.name]])

        # Case 3.3: target 1 overlaps with target 2
        if not control_target and target_target:
            return bool(commutation_map[operation1.is_controlled][position[operation2.name]])

    # Case 4: only operation 2 is controlled
    elif operation2.is_controlled:
        target_control = bool(
            operation1.wires.toset().intersection(operation2.control_wires.toset())
        )
        target_target = bool(
            operation1.wires.toset().intersection(operation2.target_wires.toset())
        )

        # Case 4.1: control and target 2 overlap with target 1
        if target_control and target_target:
            return bool(
                commutation_map[operation1.name][position[operation2.is_controlled]]
            ) and bool(commutation_map[operation1.name][position[operation2.is_controlled]])

        # Case 4.2: control operation 2 overlap with target 1
        if target_control and not target_target:
            return bool(commutation_map[operation1.name][position["ctrl"]])

        # Case 4.3: target 1 overlaps with target 2
        if not target_control and target_target:
            return bool(commutation_map[operation1.name][position[operation2.is_controlled]])

    # Case 5: no controlled operations
    else:
        # Case 5.1: no controlled operations we simply check the commutation table
        return bool(commutation_map[operation1.name][position[operation2.name]])

def merge_no_duplicates(*iterables):
    """Merge K list without duplicate using python heapq ordered merging

    Args:
        *iterables: A list of k sorted lists

    Yields:
        Iterator: List from the merging of the k ones (without duplicates
    """
    last = object()
    for val in heapq.merge(*iterables):
        if val != last:
            last = val
            yield val

class CommutationDAGNode:

    __slots__ = [
        "op",
        "name",
        "wires",
        "node_id",
        "successors",
        "predecessors",
        "reachable",
        "matchedwith",
        "isblocked",
        "successorstovisit",
    ]

    def __init__(
        self,
        op=None,
        name=None,
        wires=None,
        successors=None,
        predecessors=None,
        reachable=None,
        matchedwith=None,
        successorstovisit=None,
        isblocked=None,
        node_id=-1,
    ):

        self.op = op
        self.name = name
        self.wires = wires
        self.node_id = node_id
        self.successors = successors if successors is not None else []
        self.predecessors = predecessors if predecessors is not None else []
        self.reachable = reachable
        self.matchedwith = matchedwith if matchedwith is not None else []
        self.isblocked = isblocked
        self.successorstovisit = successorstovisit if successorstovisit is not None else []

class CommutationDAG:
    """Represents a quantum circuit as a directed acyclic graph, a node represent a quantum operation."""

    def __init__(self, wires):
        self.wires = wires
        self.num_wires = len(wires)
        self.node_id = -1
        self._multi_graph = nx.MultiDiGraph()

    def _add_node(self, node):
        self.node_id += 1
        node.node_id = self.node_id
        print(node.node_id)
        print(node.name)
        print("_______________________")
        self._multi_graph.add_node(node)

    def add_node(self, operation):
        new_node = qml.commutation_dag.CommutationDAGNode(
            op=operation,
            wires=operation.wires,
            name=operation.name,
            successors=[],
            predecessors=[],
        )
        self._add_node(new_node)
        #self.update_edges()

    @property
    def graph(self):
        return self._multi_graph