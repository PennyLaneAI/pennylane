# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transforms for interacting with PyZX, framework for ZX calculus."""
# pylint: disable=too-many-statements, too-many-branches

from collections import OrderedDict
import numpy as np
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.wires import Wires


def to_zx(tape, backend=None):
    """It converts a PennyLane quantum tape to a ZX-Graph in PyZX.
    Args:
        tape(QuantumTape): The PennyLane quantum tape.
        backend(str): Backend for the PyZX graph. "Simple" is default and use Python backend from PyZX. The backend
            'igraph' is not complete but can be used for the package python-igraph.
    """
    # Avoid to make PyZX a requirement for PennyLane.
    try:
        # pylint: disable=import-outside-toplevel
        import pyzx
        from pyzx.utils import VertexType
        from pyzx.circuit.gates import TargetMapper
        from pyzx.graph import Graph

    except ImportError as Error:
        raise ImportError(
            "This feature requires PyZX. It can be installed with: pip install pyzx"
        ) from Error

    # Dictionary of gates (PennyLane to PyZX circuit)
    gate_types = {
        "PauliX": pyzx.circuit.gates.NOT,
        "PauliZ": pyzx.circuit.gates.Z,
        "S": pyzx.circuit.gates.S,
        "T": pyzx.circuit.gates.T,
        "Hadamard": pyzx.circuit.gates.HAD,
        "RX": pyzx.circuit.gates.XPhase,
        "RZ": pyzx.circuit.gates.ZPhase,
        "PhaseShift": pyzx.circuit.gates.ZPhase,
        "SWAP": pyzx.circuit.gates.SWAP,
        "CNOT": pyzx.circuit.gates.CNOT,
        "CZ": pyzx.circuit.gates.CZ,
        "CRZ": pyzx.circuit.gates.CRZ,
        "CH": pyzx.circuit.gates.CHAD,
        "CCZ": pyzx.circuit.gates.CCZ,
        "Toffoli": pyzx.circuit.gates.Tofolli,
    }

    # Create the graph, a qubit mapper, the classical mapper stays empty as PennyLane do not support classical bits.
    graph = Graph(backend)
    q_mapper = TargetMapper()
    c_mapper = TargetMapper()

    # Map the wires to consecutive wires
    consecutive_wires = Wires(range(len(tape.wires)))
    consecutive_wires_map = OrderedDict(zip(tape.wires, consecutive_wires))
    tape = qml.map_wires(input=tape, wire_map=consecutive_wires_map)

    inputs = []

    # Create the qubits in the graph and the qubit mapper
    for i in range(len(tape.wires)):
        vertex = graph.add_vertex(VertexType.BOUNDARY, i, 0)
        inputs.append(vertex)
        q_mapper.set_prev_vertex(i, vertex)
        q_mapper.set_next_row(i, 1)
        q_mapper.set_qubit(i, i)

    # Expand the tape to be compatible with PyZX and add rotations first for measurements
    stop_crit = qml.BooleanFn(lambda obj: obj.name in gate_types)
    tape = qml.tape.tape.expand_tape(tape, depth=10, stop_at=stop_crit, expand_measurements=True)

    expanded_tape = []

    # Define specific decompositions
    for op in tape.operations:
        if op.name == "PauliY":
            decomp = [
                qml.S(wires=0),
                qml.RX(np.pi / 2, wires=0),
                qml.RZ(np.pi + np.pi, wires=0),
                qml.RX(np.pi / 2, wires=0),
                qml.RZ(3 * np.pi, wires=0),
                qml.S(wires=0),
            ]
            expanded_tape.extend(decomp)
        elif op.name == "RY":
            theta = op.data[0]
            decomp = [
                qml.RX(np.pi / 2, wires=0),
                qml.RZ(theta + np.pi, wires=0),
                qml.RX(np.pi / 2, wires=0),
                qml.RZ(3 * np.pi, wires=0),
            ]
            expanded_tape.extend(decomp)
        else:
            expanded_tape.append(op)

    expanded_tape = qml.tape.QuantumTape(expanded_tape, tape.measurements, [])

    # Create graph from circuit in the tape (operations, measurements)
    for op in expanded_tape.operations:

        # Check that the gate is compatible with PyZX
        name = op.name
        if name not in gate_types:
            raise qml.QuantumFunctionError(
                "The expansion of the tape failed, PyZX does not support", name
            )

        # Apply wires and parameters
        map_gate = gate_types[name]

        par = [param / np.pi for param in op.parameters]
        wires = list(op.wires)

        # Max number of parameter is one
        if par:
            args = []
            args.extend(wires)
            args.extend(par)
        else:
            args = wires

        gate = map_gate(*args)
        gate.to_graph(graph, q_mapper, c_mapper)

    row = max(q_mapper.max_row(), c_mapper.max_row())

    outputs = []
    for mapper in (q_mapper, c_mapper):
        for label in mapper.labels():
            qubit = mapper.to_qubit(label)
            vertex = graph.add_vertex(VertexType.BOUNDARY, qubit, row)
            outputs.append(vertex)
            pre_vertex = mapper.prev_vertex(label)
            graph.add_edge(graph.edge(pre_vertex, vertex))

    graph.set_inputs(tuple(inputs))
    graph.set_outputs(tuple(outputs))

    return graph


def from_zx(graph, split_phases=True):
    """It converts a graph from PyZX to a PennyLane tape, if graph is diagram-like.
    Args:
        graph(Graph): ZX graph in PyZX.
        split_phases(bool): If True the phases are split.
    """

    # Avoid to make PyZX a requirement for PennyLane.
    try:
        # pylint: disable=import-outside-toplevel, unused-import
        import pyzx
        from pyzx.utils import EdgeType, VertexType, FloatInt
        from pyzx.graph.base import VT

    except ImportError as Error:
        raise ImportError(
            "This feature requires PyZX. It can be installed with: pip install pyzx"
        ) from Error

    # List of PennyLane operations
    operations = []

    qubits = graph.qubits()
    graph_rows = graph.rows()
    types = graph.types()

    # Parameters are phases in the ZX framework
    params = graph.phases()
    rows = {}

    inputs = graph.inputs()

    # Set up the rows dictionary
    for vertex in graph.vertices():
        if vertex in inputs:
            continue
        row_index = graph.row(vertex)
        if row_index in rows:
            rows[row_index].append(vertex)
        else:
            rows[row_index] = [vertex]

    for row_key in sorted(rows.keys()):
        for vertex in rows[row_key]:

            qubit_1 = qubits[vertex]
            param = params[vertex]
            type_1 = types[vertex]
            neighbors = [w for w in graph.neighbors(vertex) if graph_rows[w] < row_key]

            # The graph is not diagram like.
            if len(neighbors) != 1:
                raise qml.QuantumFunctionError(
                    "Graph doesn't seem circuit like: multiple parents. Use extract circuit function."
                )

            neighbor_0 = neighbors[0]

            if qubits[neighbor_0] != qubit_1:
                raise qml.QuantumFunctionError(
                    "Graph doesn't seem circuit like: cross qubit connections"
                )

            if graph.edge_type(graph.edge(neighbor_0, vertex)) == EdgeType.HADAMARD:
                operations.append(qml.Hadamard(wires=qubit_1))

            if type_1 == VertexType.BOUNDARY:
                continue

            # Given the phase add a 1 qubit gate
            if param != 0 and not split_phases:
                if type_1 == VertexType.Z:
                    param = np.pi * float(param)
                    operations.append(qml.RZ(param, wires=qubit_1))
                else:
                    param = np.pi * float(param)
                    operations.append(qml.RX(param, wires=qubit_1))
            elif type_1 == VertexType.Z and param.denominator == 2:
                if param.numerator == 3:
                    operations.append(qml.adjoint(qml.S(wires=qubit_1)))
                else:
                    operations.append(qml.S(wires=qubit_1))
            elif type_1 == VertexType.Z and param.denominator == 4:
                if param.numerator in (1, 7):
                    if param.numerator == 7:
                        operations.append(qml.adjoint(qml.T(wires=qubit_1)))
                    else:
                        operations.append(qml.T(wires=qubit_1))
                if param.numerator in (3, 5):
                    operations.append(qml.PauliZ(wires=qubit_1))
                    if param.numerator == 3:
                        operations.append(qml.adjoint(qml.T(wires=qubit_1)))
                    else:
                        operations.append(qml.T(wires=qubit_1))
            elif param == 1:
                if type_1 == VertexType.Z:
                    operations.append(qml.PauliZ(wires=qubit_1))
                else:
                    operations.append(qml.PauliX(wires=qubit_1))
            elif param != 0:
                if type_1 == VertexType.Z:
                    param = np.pi * float(param)
                    operations.append(qml.RZ(param, wires=qubit_1))
                else:
                    param = np.pi * float(param)
                    operations.append(qml.RX(param, wires=qubit_1))

            # Given the neighbors add two qubits gates
            neighbors = [
                w for w in graph.neighbors(vertex) if graph_rows[w] == row_key and w < vertex
            ]

            for neighbor in neighbors:

                type_2 = types[neighbor]
                qubit_2 = qubits[neighbor]

                if type_1 == type_2:

                    if graph.edge_type(graph.edge(vertex, neighbor)) != EdgeType.HADAMARD:
                        raise qml.QuantumFunctionError(
                            "Invalid vertical connection between vertices of the same type."
                        )

                    if type_1 == VertexType.Z:
                        operations.append(qml.CZ(wires=[qubit_2, qubit_1]))
                    else:
                        operations.append(qml.Hadamard(wires=qubit_2))
                        operations.append(qml.CNOT(wires=[qubit_2, qubit_1]))
                        operations.append(qml.Hadamard(wires=qubit_2))

                else:
                    if graph.edge_type(graph.edge(vertex, neighbor)) != EdgeType.SIMPLE:
                        raise qml.QuantumFunctionError(
                            "Invalid vertical connection between vertices of different type."
                        )

                    if type_1 == VertexType.Z:
                        operations.append(qml.CNOT(wires=[qubit_1, qubit_2]))
                    else:
                        operations.append(qml.CNOT(wires=[qubit_2, qubit_1]))

    tape = QuantumTape(operations, [], prep=[])
    return tape
