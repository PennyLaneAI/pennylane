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
        "SWAP": 4,
        "ctrl": 5,
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
            0,
        ],
        "PauliX": [
            0,
            1,
            0,
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
            0,
        ],
        "PauliZ": [
            0,
            0,
            0,
            1,
            0,
            1,
        ],
        "SWAP": [
            0,
            0,
            0,
            0,
            1,
            0,
        ],
        "ctrl": [
            0,
            0,
            0,
            1,
            0,
            1,
        ],
    }
)


def is_commuting(operation1, operation2):
    r"""Check if two operations are commuting

    Args:
        operation1 (pennylane.Operation): A first quantum operation.
        operation2 (pennylane.Operation): A second quantum operation.

    Returns:
         bool: True if the operations commute, False otherwise.

    **Example**

    >>> qml.is_commuting(qml.PauliX(wires=0), qml.PauliZ(wires=0))
    True
    """
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-return-statements

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
                    commutation_map[operation1.is_controlled][position[operation2.is_controlled]]
                )
            )

    # Case 3: only operation 1 is controlled
    elif operation1.is_controlled:

        control_target = bool(
            operation1.control_wires.toset().intersection(operation2.wires.toset())
        )
        target_target = bool(operation1.target_wires.toset().intersection(operation2.wires.toset()))

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
        target_target = bool(operation1.wires.toset().intersection(operation2.target_wires.toset()))

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
    r"""Class to store information about a quantum operation in a node of the
    commutation DAG.

    Args:
        op (qml.Operation): PennyLane operation.
        name (str): Name of the operation.
        wires (qml.Wires): Wires on which the operation acts on.
        node_id (int): Id of the node in the DAG.
        successors (array[int]): List of the node's successors in the DAG.
        predecessors (array[int]): List of the node's predecessors in the DAG.
        reachable (bool): Attribute used to check reachability by pairewise commutation.
        matchedwith (array[int]): Id of the matched node in the pattern.
        isblocked (bool): Id of the matched node in the pattern.
        successortovisit (array[int]): List of nodes (ids) to visit in the forward part of the algorithm.
    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    # pylint: disable=too-few-public-methods

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
    r"""Class to represent a quantum circuit as a directed acyclic graph (DAG).

    **Example:**

    **Reference:**

    [1] Iten, R., Moyard, R., Metger, T., Sutter, D. and Woerner, S., 2020.
    Exact and practical pattern matching for quantum circuit optimization.
    `arXiv:1909.05270 <https://arxiv.org/abs/1909.05270>`_

    """

    def __init__(self, wires, observables=None):
        self.wires = wires
        self.num_wires = len(wires)
        self.node_id = -1
        self._multi_graph = nx.MultiDiGraph()
        self.observables = observables if observables is not None else []

    def _add_node(self, node):
        self.node_id += 1
        node.node_id = self.node_id
        self._multi_graph.add_node(node.node_id, node=node)

    def add_node(self, operation):
        """Add the operation as a node in the DAG and updates the edges.

        Args:
            operation (qml.operation): PennyLane quantum operation to add to the DAG.
        """

        new_node = qml.commutation_dag.CommutationDAGNode(
            op=operation,
            wires=operation.wires,
            name=operation.name,
            successors=[],
            predecessors=[],
        )
        self._add_node(new_node)
        self._update_edges()

    def get_node(self, node_id):
        """Add the operation as a node in the DAG and updates the edges.

        Args:
            node_id (int): PennyLane quantum operation to add to the DAG.

        Returns:
            qml.commutation_dag.CommutationDAGNOde: The node with the given id.
        """
        return self._multi_graph.nodes(data="node")[node_id]

    def get_nodes(self):
        """Return iterable to loop through all the nodes in the DAG

        Returns:
            networkx.classes.reportviews.NodeDataView: Iterable nodes.
        """
        return self._multi_graph.nodes(data="node")

    def add_edge(self, node_in, node_out):
        """Add an edge (non commutation) between node_in and node_out.

        Args:
            node_in (int): Id of the ingoing node.
            node_out (int): Id of the outgoing node.

        Returns:
            int: Id of the created edge.
        """
        return self._multi_graph.add_edge(node_in, node_out, commute=False)

    def get_edge(self, node_in, node_out):
        """Get the edge between two nodes if it exists.

        Args:
            node_in (int): Id of the ingoing node.
            node_out (int): Id of the outgoing node.

        Returns:
            dict or None: Default weight is 0, it returns None when there is no edge.
        """
        return self._multi_graph.get_edge_data(node_in, node_out)

    def get_edges(self):
        """Get all edges as an iterable.

        Returns:
            networkx.classes.reportviews.OutMultiEdgeDataView: Iterable over all edges.
        """
        return self._multi_graph.edges.data()

    def _update_edges(self):

        max_node_id = len(self._multi_graph) - 1
        max_node = self.get_node(max_node_id).op

        for current_node_id in range(0, max_node_id):
            self.get_node(current_node_id).reachable = True

        for prev_node_id in range(max_node_id - 1, -1, -1):
            if self.get_node(prev_node_id).reachable and not is_commuting(
                self.get_node(prev_node_id).op, max_node
            ):
                self.add_edge(prev_node_id, max_node_id)
                self._pred_update(max_node_id)
                list_predecessors = self.get_node(max_node_id).predecessors
                for pred_id in list_predecessors:
                    self.get_node(pred_id).reachable = False

    def direct_predecessors(self, node_id):
        """Return the direct predecessors of the given node.

        Args:
            node_id (int): Id of the node in the DAG.

        Returns:
            list[int]: List of the direct predecessors of the given node.
        """
        dir_pred = list(self._multi_graph.pred[node_id].keys())
        dir_pred.sort()
        return dir_pred

    def predecessors(self, node_id):
        """Return the predecessors of the given node.

        Args:
            node_id (int): Id of the node in the DAG.

        Returns:
            list[int]: List of the predecessors of the given node.
        """
        pred = list(nx.ancestors(self._multi_graph, node_id))
        pred.sort()
        return pred

    def direct_successors(self, node_id):
        """Return the direct successors of the given node.

        Args:
            node_id (int): Id of the node in the DAG.

        Returns:
            list[int]: List of the direct successors of the given node.
        """
        dir_succ = list(self._multi_graph.succ[node_id].keys())
        dir_succ.sort()
        return dir_succ

    def successors(self, node_id):
        """Return the successors of the given node.

        Args:
            node_id (int): Id of the node in the DAG.

        Returns:
            list[int]: List of the successors of the given node.
        """
        succ = list(nx.descendants(self._multi_graph, node_id))
        succ.sort()
        return succ

    @property
    def graph(self):
        """Return the DAG object.

        Returns:
            networkx.MultiDiGraph(): Networkx representation of the DAG.
        """
        return self._multi_graph

    def _pred_update(self, node_id):
        self.get_node(node_id).predecessors = []

        for d_pred in self.direct_predecessors(node_id):
            self.get_node(node_id).predecessors.append([d_pred])
            self.get_node(node_id).predecessors.append(self.get_node(d_pred).predecessors)

        self.get_node(node_id).predecessors = list(
            merge_no_duplicates(*(self.get_node(node_id).predecessors))
        )

    def _add_successors(self):

        for node_id in range(len(self._multi_graph) - 1, -1, -1):
            direct_successors = self.direct_successors(node_id)

            for d_succ in direct_successors:
                self.get_node(node_id).successors.append([d_succ])
                self.get_node(node_id).successors.append(self.get_node(d_succ).successors)

            self.get_node(node_id).successors = list(
                merge_no_duplicates(*(self.get_node(node_id).successors))
            )
