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
A transform to obtain the commutation DAG of a quantum circuit.
"""
import heapq
from collections import OrderedDict
from functools import partial
from typing import Sequence, Callable

import networkx as nx
from networkx.drawing.nx_pydot import to_pydot

import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.wires import Wires
from pennylane.transforms import transform


@partial(transform, is_informative=True)
def commutation_dag(tape: QuantumTape) -> (Sequence[QuantumTape], Callable):
    r"""Construct the pairwise-commutation DAG (directed acyclic graph) representation of a quantum circuit.

    In the DAG, each node represents a quantum operation, and edges represent
    non-commutation between two operations.

    This transform takes into account that not all
    operations can be moved next to each other by pairwise commutation.

    Args:
        tape (QNode or QuantumTape or Callable): The quantum circuit.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the commutation DAG.

    **Example**

    >>> dev = qml.device("default.qubit")

    .. code-block:: python

        @qml.qnode(device=dev)
        def circuit(x, y, z):
            qml.RX(x, wires=0)
            qml.RX(y, wires=0)
            qml.CNOT(wires=[1, 2])
            qml.RY(y, wires=1)
            qml.Hadamard(wires=2)
            qml.CRZ(z, wires=[2, 0])
            qml.RY(-y, wires=1)
            return qml.expval(qml.Z(0))

    The commutation dag can be returned by using the following code:

    >>> dag_fn = commutation_dag(circuit)
    >>> dag = dag_fn(np.pi / 4, np.pi / 3, np.pi / 2)

    Nodes in the commutation DAG can be accessed via the :meth:`~.get_nodes` method, returning a list of
    the  form ``(ID, CommutationDAGNode)``:

    >>> nodes = dag.get_nodes()
    >>> nodes
    NodeDataView({0: <pennylane.transforms.commutation_dag.CommutationDAGNode object at 0x7f461c4bb580>, ...}, data='node')

    You can also access specific nodes (of type :class:`~.CommutationDAGNode`) by using the :meth:`~.get_node`
    method. See :class:`~.CommutationDAGNode` for a list of available
    node attributes.

    >>> second_node = dag.get_node(2)
    >>> second_node
    <pennylane.transforms.commutation_dag.CommutationDAGNode object at 0x136f8c4c0>
    >>> second_node.op
    CNOT(wires=[1, 2])
    >>> second_node.successors
    [3, 4, 5, 6]
    >>> second_node.predecessors
    []

    For more details, see:

    * Iten, R., Moyard, R., Metger, T., Sutter, D., Woerner, S.
      "Exact and practical pattern matching for quantum circuit optimization" `doi.org/10.1145/3498325
      <https://dl.acm.org/doi/abs/10.1145/3498325>`_

    """

    def processing_fn(res):
        """Processing function that returns the circuit as a commutation DAG."""
        # Initialize DAG
        dag = CommutationDAG(res[0])
        return dag

    return [tape], processing_fn


def _merge_no_duplicates(*iterables):
    """Merge K list without duplicate using python heapq ordered merging.

    Args:
        *iterables: A list of k sorted lists.

    Yields:
        Iterator: List from the merging of the k ones (without duplicates).
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
        op (.Operation): PennyLane operation.
        wires (.Wires): Wires on which the operation acts on.
        node_id (int): ID of the node in the DAG.
        successors (array[int]): List of the node's successors in the DAG.
        predecessors (array[int]): List of the node's predecessors in the DAG.
        reachable (bool): Attribute used to check reachability by pairwise commutation.
    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    # pylint: disable=too-few-public-methods

    __slots__ = [
        "op",
        "wires",
        "target_wires",
        "control_wires",
        "node_id",
        "successors",
        "predecessors",
        "reachable",
    ]

    def __init__(
        self,
        op=None,
        wires=None,
        target_wires=None,
        control_wires=None,
        successors=None,
        predecessors=None,
        reachable=None,
        node_id=-1,
    ):
        self.op = op
        """Operation: The operation represented by the nodes."""
        self.wires = wires
        """Wires: The wires that the operation acts on."""
        self.target_wires = target_wires
        """Wires: The target wires of the operation."""
        self.control_wires = control_wires if control_wires is not None else []
        """Wires: The control wires of the operation."""
        self.node_id = node_id
        """int: The ID of the operation in the DAG."""
        self.successors = successors if successors is not None else []
        """list(int): List of the node's successors."""
        self.predecessors = predecessors if predecessors is not None else []
        """list(int): List of the node's predecessors."""
        self.reachable = reachable
        """bool: Useful attribute to create the commutation DAG."""


class CommutationDAG:
    r"""Class to represent a quantum circuit as a directed acyclic graph (DAG). This class is useful to build the
    commutation DAG and set up all nodes attributes. The construction of the DAG should be used through the
    transform :class:`qml.transforms.commutation_dag`.

    Args:
        tape (.QuantumTape): PennyLane quantum tape representing a quantum circuit.

    **Reference:**

    [1] Iten, R., Moyard, R., Metger, T., Sutter, D. and Woerner, S., 2020.
    Exact and practical pattern matching for quantum circuit optimization.
    `doi.org/10.1145/3498325 <https://dl.acm.org/doi/abs/10.1145/3498325>`_

    """

    def __init__(self, tape: QuantumTape):
        self.num_wires = len(tape.wires)
        self.node_id = -1
        self._multi_graph = nx.MultiDiGraph()

        consecutive_wires = Wires(range(len(tape.wires)))
        wires_map = OrderedDict(zip(tape.wires, consecutive_wires))

        for operation in tape.operations:
            operation = qml.map_wires(operation, wire_map=wires_map)
            self.add_node(operation)

        self._add_successors()

        self.observables = [qml.map_wires(obs, wire_map=wires_map) for obs in tape.observables]

    def _add_node(self, node):
        self.node_id += 1
        node.node_id = self.node_id
        self._multi_graph.add_node(node.node_id, node=node)

    def add_node(self, operation):
        """Add the operation as a node in the DAG and updates the edges.

        Args:
            operation (qml.operation): PennyLane quantum operation to add to the DAG.
        """
        target_wires = [w for w in operation.wires if w not in operation.control_wires]

        new_node = CommutationDAGNode(
            op=operation,
            wires=operation.wires.tolist(),
            target_wires=target_wires,
            control_wires=operation.control_wires.tolist(),
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
            CommutationDAGNOde: The node with the given id.
        """
        return self._multi_graph.nodes(data="node")[node_id]

    def get_nodes(self):
        """Return iterable to loop through all the nodes in the DAG.

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

    @property
    def size(self):
        """Return the size of the DAG object.

        Returns:
            int: Number of nodes in the DAG.
        """
        return len(self._multi_graph)

    # pylint: disable=no-member
    def draw(self, filename="dag.png"):  # pragma: no cover
        """Draw the DAG object.

        Args:
            filename (str): The file name which is in PNG format. Default = 'dag.png'
        """
        draw_graph = nx.MultiDiGraph()

        for node in self.get_nodes():
            wires = ",".join([" " + str(elem) for elem in node[1].op.wires.tolist()])
            label = (
                "ID: "
                + str(node[0])
                + "\n"
                + "Op: "
                + node[1].op.name
                + "\n"
                + "Wires: ["
                + wires[1::]
                + "]"
            )
            draw_graph.add_node(
                node[0], label=label, color="blue", style="filled", fillcolor="lightblue"
            )

        for edge in self.get_edges():
            draw_graph.add_edge(edge[0], edge[1])

        dot = to_pydot(draw_graph)
        dot.write_png(filename)

    def _pred_update(self, node_id):
        self.get_node(node_id).predecessors = []

        for d_pred in self.direct_predecessors(node_id):
            self.get_node(node_id).predecessors.append([d_pred])
            self.get_node(node_id).predecessors.append(self.get_node(d_pred).predecessors)

        self.get_node(node_id).predecessors = list(
            _merge_no_duplicates(*self.get_node(node_id).predecessors)
        )

    def _add_successors(self):
        for node_id in range(len(self._multi_graph) - 1, -1, -1):
            direct_successors = self.direct_successors(node_id)

            for d_succ in direct_successors:
                self.get_node(node_id).successors.append([d_succ])
                self.get_node(node_id).successors.append(self.get_node(d_succ).successors)

            self.get_node(node_id).successors = list(
                _merge_no_duplicates(*self.get_node(node_id).successors)
            )

    def _update_edges(self):
        max_node_id = len(self._multi_graph) - 1
        max_node = self.get_node(max_node_id).op

        for current_node_id in range(0, max_node_id):
            self.get_node(current_node_id).reachable = True

        for prev_node_id in range(max_node_id - 1, -1, -1):
            if self.get_node(prev_node_id).reachable and not qml.is_commuting(
                self.get_node(prev_node_id).op, max_node
            ):
                self.add_edge(prev_node_id, max_node_id)
                self._pred_update(max_node_id)
                list_predecessors = self.get_node(max_node_id).predecessors
                for pred_id in list_predecessors:
                    self.get_node(pred_id).reachable = False
