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

import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.wires import Wires


def tape_to_graph_zx(tape, backend=None):
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
        "RY": pyzx.circuit.gates.YPhase,
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

    # Create graph from circuit in the tape (operations, measurements)
    for op in tape.operations:

        # Check that the gate is compatible with PyZX
        name = op.name
        if name not in gate_types:
            raise qml.QuantumFunctionError(
                "The expansion of the tape failed, PyZX does not support", name
            )

        # Apply wires and parameters
        map_gate = gate_types[name]
        par = op.parameters
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


def graph_zx_to_tape(graph, split_phases=True):
    """From PyZX graph to a PennyLane tape."""

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
            neigh = [w for w in graph.neighbors(vertex) if graph_rows[w] < row_key]
            if len(neigh) != 1:
                raise TypeError("Graph doesn't seem circuit like: multiple parents")
            n = neigh[0]
            if qubits[n] != qubit_1:
                raise TypeError("Graph doesn't seem circuit like: cross qubit connections")
            if graph.edge_type(graph.edge(n, vertex)) == EdgeType.HADAMARD:
                operations.append(qml.Hadamard(wires=qubit_1))
            if type_1 == VertexType.BOUNDARY:  # vertex is an output
                continue
            if param != 0 and not split_phases:
                if type_1 == VertexType.Z:
                    param = float(param)
                    operations.append(qml.RZ(param, wires=qubit_1))
                else:
                    param = float(param)
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
                    param = float(param)
                    operations.append(qml.RZ(param, wires=qubit_1))
                else:
                    param = float(param)
                    operations.append(qml.RX(param, wires=qubit_1))

            neigh = [w for w in graph.neighbors(vertex) if graph_rows[w] == row_key and w < vertex]
            for n in neigh:
                t2 = types[n]
                qubit_2 = qubits[n]
                if type_1 == t2:
                    if graph.edge_type(graph.edge(vertex, n)) != EdgeType.HADAMARD:
                        raise TypeError(
                            "Invalid vertical connection between vertices of the same type"
                        )
                    if type_1 == VertexType.Z:
                        operations.append(qml.CZ(wires=[qubit_2, qubit_1]))
                    else:
                        operations.append(qml.Hadamard(wires=qubit_2))
                        operations.append(qml.CNOT(wires=[qubit_2, qubit_1]))
                        operations.append(qml.Hadamard(wires=qubit_2))
                else:
                    if graph.edge_type(graph.edge(vertex, n)) != EdgeType.SIMPLE:
                        raise TypeError(
                            "Invalid vertical connection between vertices of different type"
                        )
                    if type_1 == VertexType.Z:
                        operations.append(qml.CNOT(wires=[qubit_1, qubit_2]))
                    else:
                        operations.append(qml.CNOT(wires=[qubit_2, qubit_1]))
    tape = QuantumTape(operations, [], prep=[])
    return tape
