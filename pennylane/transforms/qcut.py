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
from pennylane.operation import Tensor
from pennylane.tape import QuantumTape


def tape_to_graph(tape: QuantumTape) -> MultiDiGraph:
    """Converts a quantum tape to a directed multigraph."""
    graph = MultiDiGraph()

    wire_latest_node = {w: None for w in tape.wires}
    state_preps = ["BasisState", "QubitStateVector"]

    for order, op in enumerate(tape.operations):
        if op.name in state_preps:
            sub_ops = op.expand().expand(depth=1).operations
            for op in sub_ops:
                graph.add_node(op, order=order)
                for wire in op.wires:
                    if wire_latest_node[wire] is not None:
                        parent_op = wire_latest_node[wire]
                        graph.add_edge(parent_op, op, wire=wire)
                    wire_latest_node[wire] = op
            order += 1
        else:
            graph.add_node(op, order=order)
            for wire in op.wires:
                if wire_latest_node[wire] is not None:
                    parent_op = wire_latest_node[wire]
                    graph.add_edge(parent_op, op, wire=wire)
                wire_latest_node[wire] = op

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
