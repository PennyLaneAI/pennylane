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
This module contains the CircuitGraph class which is used to generate a DAG (directed acyclic graph)
representation of a quantum circuit from an Operator queue.
"""
import warnings
from collections import defaultdict, namedtuple
from collections.abc import Sequence
from functools import cached_property

import numpy as np
import rustworkx as rx

from pennylane.exceptions import PennyLaneDeprecationWarning
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operator
from pennylane.ops.identity import I
from pennylane.queuing import QueuingManager, WrappedObj
from pennylane.resource import ResourcesOperation
from pennylane.wires import Wires


def _get_wires(obj, all_wires):
    return all_wires if len(obj.wires) == 0 else obj.wires


Layer = namedtuple("Layer", ["ops", "param_inds", "ops_inds"])
"""Parametrized layer of the circuit.

Args:

    ops (list[Operator]): parametrized operators in the layer
    param_inds (list[int]): corresponding free parameter indices
    ops_inds (list[int]): the indices into the circuit for ops
"""
# TODO define what a layer is

LayerData = namedtuple("LayerData", ["pre_ops", "ops", "param_inds", "post_ops"])
"""Parametrized layer of the circuit.

Args:
    pre_ops (list[Operator]): operators that precede the layer
    ops (list[Operator]): parametrized operators in the layer
    param_inds (tuple[int]): corresponding free parameter indices
    post_ops (list[Operator]): operators that succeed the layer
"""


def _construct_graph_from_queue(queue, all_wires):
    inds_for_objs = defaultdict(list)  # dict from wrappedobjs to all indices for the objs
    nodes_on_wires = defaultdict(list)  # wire to list of nodes

    graph = rx.PyDiGraph(multigraph=False)

    for i, obj in enumerate(queue):
        inds_for_objs[WrappedObj(obj)].append(i)

        obj_node = graph.add_node(i)
        for w in _get_wires(obj, all_wires):
            if w in nodes_on_wires:
                graph.add_edge(nodes_on_wires[w][-1], obj_node, "")
            nodes_on_wires[w].append(obj_node)

    return graph, inds_for_objs, nodes_on_wires


# pylint: disable=too-many-instance-attributes, too-many-public-methods
class CircuitGraph:
    """Represents a quantum circuit as a directed acyclic graph.

    In this representation the :class:`~.Operator` instances are the nodes of the graph,
    and each directed edge represent a subsystem (or a group of subsystems) on which the two
    Operators act subsequently. This representation can describe the causal relationships
    between arbitrary quantum channels and measurements, not just unitary gates.

    Args:
        ops (Iterable[.Operator]): quantum operators constituting the circuit, in temporal order
        obs (list[Union[MeasurementProcess, Operator]]): terminal measurements, in temporal order
        wires (.Wires): The addressable wire registers of the device that will be executing this graph
        par_info (Optional[list[dict]]): Parameter information. For each index, the entry is a dictionary containing an operation
        and an index into that operation's parameters.
        trainable_params (Optional[set[int]]): A set containing the indices of parameters that support
            differentiability. The indices provided match the order of appearance in the
            quantum circuit.
    """

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        ops: list[Operator | MeasurementProcess],
        obs: list[MeasurementProcess | Operator],
        wires: Wires,
        par_info: list[dict] | None = None,
        trainable_params: set[int] | None = None,
    ):
        self._operations = ops
        self._observables = obs
        self.par_info = par_info
        self.trainable_params = trainable_params

        self._queue = ops + obs

        self.wires = Wires(wires)
        """Wires: wires that are addressed in the operations.
        Required to translate between wires and indices of the wires on the device."""
        self.num_wires = len(wires)
        """int: number of wires the circuit contains"""

        self._graph, self._inds_for_objs, self._nodes_on_wires = _construct_graph_from_queue(
            self._queue, wires
        )

        # Required to keep track if we need to handle multiple returned
        # observables per wire
        self._max_simultaneous_measurements = None

    def __str__(self):
        """The string representation of the class."""
        string = "Operations\n==========\n"
        string += "\n".join(repr(op) for op in self.operations)

        string += "\n\nObservables\n===========\n"
        string += "\n".join(repr(obs) for obs in self.observables)
        string += "\n"

        return string

    def print_contents(self):
        """Prints the contents of the quantum circuit."""
        warnings.warn(
            "``CircuitGraph.print_contents`` is deprecated and will be removed in v0.44. "
            "Instead, please use ``print(circuit_graph_obj)``.",
            PennyLaneDeprecationWarning,
        )
        print(self)

    def serialize(self) -> str:
        """Serialize the quantum circuit graph based on the operations and
        observables in the circuit graph and the index of the variables
        used by them.

        The string that is produced can be later hashed to assign a unique value to the circuit graph.

        Returns:
            string: serialized quantum circuit graph
        """
        serialization_string = ""
        delimiter = "!"

        for op in self.operations:
            serialization_string += op.name

            for param in op.data:
                serialization_string += delimiter
                serialization_string += str(param)
                serialization_string += delimiter

            serialization_string += str(op.wires.tolist())

        # Adding a distinct separating string that could not occur by any combination of the
        # name of the operation and wires
        serialization_string += "|||"

        for mp in self.observables:
            obs = mp.obs or mp
            data, name = ([], "Identity") if obs is mp else (obs.data, str(obs.name))
            serialization_string += mp.__class__.__name__
            serialization_string += delimiter
            serialization_string += name
            for param in data:
                serialization_string += delimiter
                serialization_string += str(param)
                serialization_string += delimiter

            serialization_string += str(obs.wires.tolist())
        return serialization_string

    @property
    def hash(self) -> int:
        """Creating a hash for the circuit graph based on the string generated by serialize.

        Returns:
            int: the hash of the serialized quantum circuit graph
        """
        return hash(self.serialize())

    @property
    def observables_in_order(self):
        """Observables in the circuit, in a fixed topological order.

        The topological order used by this method is guaranteed to be the same
        as the order in which the measured observables are returned by the quantum function.
        Currently the topological order is determined by the queue index.

        Returns:
            list[Union[MeasurementProcess, Operator]]: observables
        """
        warnings.warn(
            "``CircuitGraph.observables_in_order`` is deprecated and will be removed in v0.44. "
            "Instead, please use ``CircuitGraph.observables``",
            PennyLaneDeprecationWarning,
        )
        return self._observables

    @property
    def observables(self):
        """Observables in the circuit."""
        return self._observables

    @property
    def operations_in_order(self):
        """Operations in the circuit, in a fixed topological order.

        Currently the topological order is determined by the queue index.

        The complement of :meth:`QNode.observables`. Together they return every :class:`Operator`
        instance in the circuit.

        Returns:
            list[Operation]: operations
        """
        warnings.warn(
            "``CircuitGraph.operations_in_order`` is deprecated and will be removed in v0.44. "
            "Instead, please use ``CircuitGraph.operations``",
            PennyLaneDeprecationWarning,
        )
        return self._operations

    @property
    def operations(self):
        """Operations in the circuit."""
        return self._operations

    @property
    def graph(self):
        """The graph representation of the quantum circuit.

        The graph has nodes representing indices into the queue,
        and directed edges pointing from nodes to their immediate dependents/successors.

        Returns:
            rustworkx.PyDiGraph: the directed acyclic graph representing the quantum circuit
        """
        return self._graph

    def wire_indices(self, wire):
        """Operator indices on the given wire.

        Args:
            wire (int): wire to examine

        Returns:
            list[int]: indices of operators on the wire, in temporal order
        """
        return self._nodes_on_wires[wire]

    def ancestors(self, ops, sort=False):
        """Ancestors of a given set of operators.

        Args:
            ops (Iterable[Operator]): set of operators in the circuit
            sort=False (bool): if ``True``, sort the operators according
            to the topological order determined by the queue index

        Returns:
            list[Operator]: ancestors of the given operators
        """
        if isinstance(ops, (Operator, MeasurementProcess)):
            raise ValueError(
                "CircuitGraph.ancestors accepts an iterable of"
                " operators and measurements, not operators and measurements themselves."
            )
        if any(len(self._inds_for_objs[WrappedObj(op)]) > 1 for op in ops):
            raise ValueError(
                "Cannot calculate ancestors for an operator that occurs multiple times."
                "Please use ancestors_of_indexes instead."
            )
        ancestors = set()
        for op in ops:
            ind = self._inds_for_objs[WrappedObj(op)][0]
            op_ancestors = rx.ancestors(self._graph, ind)
            ancestors.update(set(op_ancestors))
        if sort:
            ancestors = sorted(ancestors)
        return [self._queue[ind] for ind in ancestors]

    def ancestors_of_indexes(self, indexes: Sequence[int], sort=False):
        """Ancestors of a given set of operators.

        Args:
            indexes (Sequence[int]) : the index into the queue for the operator
            sort=False (bool): if ``True``, sort the operators according
            to the topological order determined by the queue index

        Returns:
            list[Operator]: ancestors of the given operators
        """

        ancestors = {i for ind in indexes for i in rx.ancestors(self._graph, ind)}
        if sort:
            ancestors = sorted(ancestors)
        return [self._queue[ind] for ind in ancestors]

    def descendants_of_indexes(self, indexes: Sequence[int], sort=False):
        """Descendants of a given set of operators.

        Args:
            indexes (Sequence[int]) : the index into the queue for the operator
            sort=False (bool): if ``True``, sort the operators according
            to the topological order determined by the queue index

        Returns:
            list[Operator]: descendants of the given operators
        """

        ancestors = {i for ind in indexes for i in rx.descendants(self._graph, ind)}
        if sort:
            ancestors = sorted(ancestors)
        return [self._queue[ind] for ind in ancestors]

    def descendants(self, ops, sort=False):
        """Descendants of a given set of operators.

        Args:
            ops (Iterable[Operator]): set of operators in the circuit
            sort=False (bool): if ``True``, sort the operators according
            to the topological order determined by the queue index

        Returns:
            list[Operator]: descendants of the given operators
        """
        if isinstance(ops, (Operator, MeasurementProcess)):
            raise ValueError(
                "CircuitGraph.descendants accepts an iterable of"
                " operators and measurements, not operators and measurements themselves."
            )
        if any(len(self._inds_for_objs[WrappedObj(op)]) > 1 for op in ops):
            raise ValueError(
                "cannot calculate decendents for an operator that occurs multiple times. "
                "Please use descendants_of_indexes instead."
            )
        descendants = set()
        for op in ops:
            ind = self._inds_for_objs[WrappedObj(op)][0]
            op_descendants = rx.descendants(self._graph, ind)
            descendants.update(set(op_descendants))
        if sort:
            descendants = sorted(descendants)
        return [self._queue[ind] for ind in descendants]

    def ancestors_in_order(self, ops):
        """Operator ancestors in a topological order.

        Currently the topological order is determined by the queue index.

        Args:
            ops (Iterable[Operator]): set of operators in the circuit

        Returns:
            list[Operator]: ancestors of the given operators, topologically ordered
        """
        warnings.warn(
            "``CircuitGraph.ancestors_in_order`` is deprecated and will be removed in v0.44. "
            "Instead, please use ``CircuitGraph.ancestors(ops, sort=True)``",
            PennyLaneDeprecationWarning,
        )
        return self.ancestors(ops, sort=True)

    def descendants_in_order(self, ops):
        """Operator descendants in a topological order.

        Currently the topological order is determined by the queue index.

        Args:
            ops (Iterable[Operator]): set of operators in the circuit

        Returns:
            list[Operator]: descendants of the given operators, topologically ordered
        """
        warnings.warn(
            "``CircuitGraph.descendants_in_order`` is deprecated and will be removed in v0.44. "
            "Instead, please use ``CircuitGraph.descendants(ops, sort=True)``",
            PennyLaneDeprecationWarning,
        )
        return self.descendants(ops, sort=True)

    def nodes_between(self, a, b):
        r"""Nodes on all the directed paths between the two given nodes.

        Returns the set of all nodes ``s`` that fulfill :math:`a \le s \le b`.
        There is a directed path from ``a`` via ``s`` to ``b`` iff the set is nonempty.
        The endpoints belong to the path.

        Args:
            a (Operator): initial node
            b (Operator): final node

        Returns:
            list[Operator]: nodes on all the directed paths between a and b
        """
        A = self.descendants([a])
        A.append(a)
        B = self.ancestors([b])
        B.append(b)

        return [B.pop(i) for op1 in A for i, op2 in enumerate(B) if op1 is op2]

    @property
    def parametrized_layers(self):
        """Identify the parametrized layer structure of the circuit.

        Returns:
            list[Layer]: layers of the circuit
        """
        # FIXME maybe layering should be greedier, for example [a0 b0 c1 d1] should layer as [a0
        # c1], [b0, d1] and not [a0], [b0 c1], [d1] keep track of the current layer
        current = Layer([], [], [])
        layers = [current]

        for idx, info in enumerate(self.par_info):
            if self.trainable_params and idx in self.trainable_params:
                op = info["op"]

                # get all predecessor ops of the op
                sub = self.ancestors_of_indexes((info["op_idx"],))

                # check if any of the dependents are in the
                # currently assembled layer
                if any(o1 is o2 for o1 in current.ops for o2 in sub):
                    # operator depends on current layer, start a new layer
                    current = Layer([], [], [])
                    layers.append(current)

                # store the parameters and ops indices for the layer
                current.ops.append(op)
                current.ops_inds.append(info["op_idx"])
                current.param_inds.append(idx)

        return layers

    def iterate_parametrized_layers(self):
        """Parametrized layers of the circuit.

        Returns:
            Iterable[LayerData]: layers with extra metadata
        """
        # iterate through each layer
        for ops, param_inds, indexes in self.parametrized_layers:
            pre_queue = self.ancestors_of_indexes(indexes, sort=True)
            post_queue = self.descendants_of_indexes(indexes, sort=True)
            yield LayerData(pre_queue, ops, tuple(param_inds), post_queue)

    def update_node(self, old, new):
        """Replaces the given circuit graph node with a new one.

        Args:
            old (Operator): node to replace
            new (Operator): replacement

        Raises:
            ValueError: if the new :class:`~.Operator` does not act on the same wires as the old one
        """
        # NOTE Does not alter the graph edges in any way. variable_deps is not changed, Dangerous!
        if new.wires != old.wires:
            raise ValueError("The new Operator must act on the same wires as the old one.")

        self._inds_for_objs[WrappedObj(new)] = self._inds_for_objs.pop(WrappedObj(old))

        for i, op in enumerate(self._operations):
            if op is old:
                self._operations[i] = new
        for i, mp in enumerate(self._observables):
            if mp is old:
                self._observables[i] = new
        for i, obj in enumerate(self._queue):
            if obj is old:
                self._queue[i] = new

    def get_depth(self):
        """Depth of the quantum circuit (longest path in the DAG)."""
        return self._depth

    @cached_property
    def _depth(self):
        # If there are no operations in the circuit, the depth is 0
        if not self.operations:
            return 0
        with QueuingManager.stop_recording():
            ops_with_initial_I = [
                I(self.wires)
            ] + self.operations  # add identity wire to end the graph
        operation_graph, _, _ = _construct_graph_from_queue(ops_with_initial_I, self.wires)

        # pylint: disable=unused-argument
        def weight_fn(in_idx, out_idx, w):
            out_op = ops_with_initial_I[out_idx]
            if isinstance(out_op, ResourcesOperation):
                return out_op.resources().depth
            return 1

        return rx.dag_longest_path_length(operation_graph, weight_fn=weight_fn)

    def has_path_idx(self, a_idx: int, b_idx: int) -> bool:
        """Checks if a path exists between the two given nodes.

        Args:
            a_idx (int): initial node index
            b_idx (int): final node index

        Returns:
            bool: returns ``True`` if a path exists
        """
        if a_idx == b_idx:
            return True

        return (
            len(
                rx.digraph_dijkstra_shortest_paths(
                    self._graph,
                    a_idx,
                    b_idx,
                    weight_fn=None,
                    default_weight=1.0,
                    as_undirected=False,
                )
            )
            != 0
        )

    def has_path(self, a, b) -> bool:
        """Checks if a path exists between the two given nodes.

        Args:
            a (Operator): initial node
            b (Operator): final node

        Returns:
            bool: returns ``True`` if a path exists
        """

        if a is b:
            return True

        if any(len(self._inds_for_objs[WrappedObj(o)]) > 1 for o in (a, b)):
            raise ValueError(
                "CircuitGraph.has_path does not work with operations that have been repeated. "
                "Consider using has_path_idx instead."
            )

        return (
            len(
                rx.digraph_dijkstra_shortest_paths(
                    self._graph,
                    self._inds_for_objs[WrappedObj(a)][0],
                    self._inds_for_objs[WrappedObj(b)][0],
                    weight_fn=None,
                    default_weight=1.0,
                    as_undirected=False,
                )
            )
            != 0
        )

    @cached_property
    def max_simultaneous_measurements(self):
        """Returns the maximum number of measurements on any wire in the circuit graph.

        This method counts the number of measurements for each wire and returns
        the maximum.

        **Examples**


        >>> dev = qml.device('default.qubit', wires=3)
        >>> def circuit_measure_max_once():
        ...     return qml.expval(qml.X(0))
        >>> qnode = qml.QNode(circuit_measure_max_once, dev)
        >>> tape = qml.workflow.construct_tape(qnode)()
        >>> print(tape.graph.max_simultaneous_measurements)
        1
        >>> def circuit_measure_max_twice():
        ...     return qml.expval(qml.X(0)), qml.probs(wires=0)
        >>> qnode = qml.QNode(circuit_measure_max_twice, dev)
        >>> tape = qml.workflow.construct_tape(qnode)()
        >>> print(tape.graph.max_simultaneous_measurements)
        2

        Returns:
            int: the maximum number of measurements
        """
        all_wires = []

        for obs in self.observables:
            all_wires.extend(obs.wires.tolist())

        a = np.array(all_wires)
        _, counts = np.unique(a, return_counts=True)
        return counts.max() if counts.size != 0 else 1  # qml.state() will result in an empty array
