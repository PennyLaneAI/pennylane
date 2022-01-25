# Copyright 2022 Xanadu Quantum Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module provides the circuit cutting functionality that allows large
circuits to be distributed across multiple devices
"""

from networkx import MultiDiGraph

from pennylane.measure import MeasurementProcess
from pennylane.operation import Observable, Operation, Tensor
from pennylane.ops.qubit.non_parametric_ops import WireCut
from pennylane.tape import QuantumTape


class MeasureNode(Operation):
    num_wires = 1
    grad_method = None


class PrepareNode(Operation):
    num_wires = 1
    grad_method = None


def replace_wire_cut_node(node: WireCut, graph: MultiDiGraph):
    """
    Replace a `WireCut` nodes in the graph with `MeasureNode` and `PrepareNode`
    """
    predecessors = graph.pred[node]
    successors = graph.succ[node]

    predecessor_on_wire = {}
    for op, data in predecessors.items():
        for d in data.values():
            wire = d["wire"]
            predecessor_on_wire[wire] = op

    successor_on_wire = {}
    for op, data in successors.items():
        for d in data.values():
            wire = d["wire"]
            successor_on_wire[wire] = op

    order = graph.nodes[node]["order"]
    graph.remove_node(node)

    for wire in node.wires:
        predecessor = predecessor_on_wire.get(wire, None)
        successor = successor_on_wire.get(wire, None)

        meas = MeasureNode(wires=wire)
        prep = PrepareNode(wires=wire)
        graph.add_node(meas, order=order)
        graph.add_node(prep, order=order + 0.5)

        graph.add_edge(meas, prep, wire=wire)

        if predecessor is not None:
            graph.add_edge(predecessor, meas, wire=wire)
        if successor is not None:
            graph.add_edge(prep, successor, wire=wire)


def replace_wire_cut_nodes(graph: MultiDiGraph):
    """
    Remove all `WireCut`s in the graph with `MeasureNode`s and `PrepareNode`s
    """
    for op in list(graph.nodes):
        if isinstance(op, WireCut):
            replace_wire_cut_node(op, graph)


def add_operator_node(
    graph: MultiDiGraph, op: Observable, order: int, wire_latest_node: dict
) -> None:
    """
    Helper function to add operators as nodes during tape to graph connversion
    """
    graph.add_node(op, order=order)
    for wire in op.wires:
        if wire_latest_node[wire] is not None:
            parent_op = wire_latest_node[wire]
            graph.add_edge(parent_op, op, wire=wire)
        wire_latest_node[wire] = op


def tape_to_graph(tape: QuantumTape) -> MultiDiGraph:
    """Converts a quantum tape to a directed multigraph."""
    graph = MultiDiGraph()

    wire_latest_node = {w: None for w in tape.wires}
    state_preps = ["BasisState", "QubitStateVector"]

    for order, op in enumerate(tape.operations):
        if op.name in state_preps:
            sub_ops = op.expand().expand(depth=1).operations
            for sub_op in sub_ops:
                add_operator_node(graph, sub_op, order, wire_latest_node)
            order += 1
        else:
            add_operator_node(graph, op, order, wire_latest_node)
            order += 1

    for m in tape.measurements:
        obs = getattr(m, "obs", None)
        if obs is not None and isinstance(obs, Tensor):
            for o in obs.obs:
                m_ = MeasurementProcess(m.return_type, obs=o)

                graph.add_node(m_, order=order)
                order += 1
                for wire in o.wires:
                    parent_op = wire_latest_node[wire]
                    graph.add_edge(parent_op, m_, wire=wire)
        else:
            graph.add_node(m, order=order)
            order += 1

            for wire in m.wires:
                parent_op = wire_latest_node[wire]
                graph.add_edge(parent_op, m, wire=wire)

    return graph
