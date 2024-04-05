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
# pylint: disable=too-many-statements, too-many-branches, too-many-return-statements, too-many-arguments

from functools import partial
from typing import Sequence, Callable
from collections import OrderedDict
import numpy as np
import pennylane as qml
from pennylane.operation import Operator
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.transforms import TransformError
from pennylane.transforms import transform
from pennylane.wires import Wires


class VertexType:  # pylint: disable=too-few-public-methods
    """Type of a vertex in the graph.

    This class is copied from PyZX as we do not make PyZX a Pennylane requirement.

    Copyright (C) 2018 - Aleks Kissinger and John van de Wetering"""

    BOUNDARY = 0
    Z = 1
    X = 2
    H_BOX = 3


class EdgeType:  # pylint: disable=too-few-public-methods
    """Type of an edge in the graph.

    This class is copied from PyZX as we do not make PyZX a Pennylane requirement.

    Copyright (C) 2018 - Aleks Kissinger and John van de Wetering"""

    SIMPLE = 1
    HADAMARD = 2


def to_zx(tape, expand_measurements=False):  # pylint: disable=unused-argument
    """This transform converts a PennyLane quantum tape to a ZX-Graph in the `PyZX framework <https://pyzx.readthedocs.io/en/latest/>`_.
    The graph can be optimized and transformed by well-known ZX-calculus reductions.

    Args:
        tape(QNode or QuantumTape or Callable or Operation): The PennyLane quantum circuit.
        expand_measurements(bool): The expansion will be applied on measurements that are not in the Z-basis and
            rotations will be added to the operations.

    Returns:
        graph (pyzx.Graph) or qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the ZX graph in the form of a PyZX graph.

    **Example**

    You can use the transform decorator directly on your :class:`~.QNode`, quantum function and executing it will produce a
    PyZX graph. You can also use the transform directly on the :class:`~.QuantumTape`.

    .. code-block:: python

        import pyzx
        dev = qml.device('default.qubit', wires=2)

        @qml.transforms.to_zx
        @qml.qnode(device=dev)
        def circuit(p):
            qml.RZ(p[0], wires=1),
            qml.RZ(p[1], wires=1),
            qml.RX(p[2], wires=0),
            qml.Z(0),
            qml.RZ(p[3], wires=1),
            qml.X(1),
            qml.CNOT(wires=[0, 1]),
            qml.CNOT(wires=[1, 0]),
            qml.SWAP(wires=[0, 1]),
            return qml.expval(qml.Z(0) @ qml.Z(1))

        params = [5 / 4 * np.pi, 3 / 4 * np.pi, 0.1, 0.3]
        g = circuit(params)

    >>> g
    Graph(20 vertices, 23 edges)

    It is now a PyZX graph and can apply function from the framework on your Graph, for example you can draw it:

    >>> pyzx.draw_matplotlib(g)
    <Figure size 800x200 with 1 Axes>

    Alternatively you can use the transform directly on a quantum tape and get PyZX graph.

    .. code-block:: python

        operations = [
                qml.RZ(5 / 4 * np.pi, wires=1),
                qml.RZ(3 / 4 * np.pi, wires=1),
                qml.RX(0.1, wires=0),
                qml.Z(0),
                qml.RZ(0.3, wires=1),
                qml.X(1),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 0]),
                qml.SWAP(wires=[0, 1]),
            ]

        tape = qml.tape.QuantumTape(operations)
        g = qml.transforms.to_zx(tape)

    >>> g
    Graph(20 vertices, 23 edges)

    .. details::
        :title: Usage Details

        Here we give an example of how to use optimization techniques from ZX calculus to reduce the T count of a
        quantum circuit and get back a PennyLane circuit.

        Let's start by starting with the mod 5 4 circuit from a known benchmark `library <https://github.com/njross/optimizer>`_
        the expanded circuit before optimization is the following QNode:

        .. code-block:: python

            dev = qml.device("default.qubit", wires=5)

            @qml.transforms.to_zx
            @qml.qnode(device=dev)
            def mod_5_4():
                qml.X(4),
                qml.Hadamard(wires=4),
                qml.CNOT(wires=[3, 4]),
                qml.adjoint(qml.T(wires=[4])),
                qml.CNOT(wires=[0, 4]),
                qml.T(wires=[4]),
                qml.CNOT(wires=[3, 4]),
                qml.adjoint(qml.T(wires=[4])),
                qml.CNOT(wires=[0, 4]),
                qml.T(wires=[3]),
                qml.T(wires=[4]),
                qml.CNOT(wires=[0, 3]),
                qml.T(wires=[0]),
                qml.adjoint(qml.T(wires=[3]))
                qml.CNOT(wires=[0, 3]),
                qml.CNOT(wires=[3, 4]),
                qml.adjoint(qml.T(wires=[4])),
                qml.CNOT(wires=[2, 4]),
                qml.T(wires=[4]),
                qml.CNOT(wires=[3, 4]),
                qml.adjoint(qml.T(wires=[4])),
                qml.CNOT(wires=[2, 4]),
                qml.T(wires=[3]),
                qml.T(wires=[4]),
                qml.CNOT(wires=[2, 3]),
                qml.T(wires=[2]),
                qml.adjoint(qml.T(wires=[3]))
                qml.CNOT(wires=[2, 3]),
                qml.Hadamard(wires=[4]),
                qml.CNOT(wires=[3, 4]),
                qml.Hadamard(wires=4),
                qml.CNOT(wires=[2, 4]),
                qml.adjoint(qml.T(wires=[4]),)
                qml.CNOT(wires=[1, 4]),
                qml.T(wires=[4]),
                qml.CNOT(wires=[2, 4]),
                qml.adjoint(qml.T(wires=[4])),
                qml.CNOT(wires=[1, 4]),
                qml.T(wires=[4]),
                qml.T(wires=[2]),
                qml.CNOT(wires=[1, 2]),
                qml.T(wires=[1]),
                qml.adjoint(qml.T(wires=[2]))
                qml.CNOT(wires=[1, 2]),
                qml.Hadamard(wires=[4]),
                qml.CNOT(wires=[2, 4]),
                qml.Hadamard(wires=4),
                qml.CNOT(wires=[1, 4]),
                qml.adjoint(qml.T(wires=[4])),
                qml.CNOT(wires=[0, 4]),
                qml.T(wires=[4]),
                qml.CNOT(wires=[1, 4]),
                qml.adjoint(qml.T(wires=[4])),
                qml.CNOT(wires=[0, 4]),
                qml.T(wires=[4]),
                qml.T(wires=[1]),
                qml.CNOT(wires=[0, 1]),
                qml.T(wires=[0]),
                qml.adjoint(qml.T(wires=[1])),
                qml.CNOT(wires=[0, 1]),
                qml.Hadamard(wires=[4]),
                qml.CNOT(wires=[1, 4]),
                qml.CNOT(wires=[0, 4]),
                return qml.expval(qml.Z(0))

        The circuit contains 63 gates; 28 :func:`qml.T` gates, 28 :func:`qml.CNOT`, 6 :func:`qml.Hadmard` and
        1 :func:`qml.X`. We applied the ``qml.transforms.to_zx`` decorator in order to transform our circuit to
        a ZX graph.

        You can get the PyZX graph by simply calling the QNode:

        >>> g = mod_5_4()
        >>> pyzx.tcount(g)
        28

        PyZX gives multiple options for optimizing ZX graphs (:func:`pyzx.full_reduce`, :func:`pyzx.teleport_reduce`, ...).
        The :func:`pyzx.full_reduce` applies all optimization passes, but the final result may not be circuit-like.
        Converting back to a quantum circuit from a fully reduced graph may be difficult to impossible.
        Therefore we instead recommend using :func:`pyzx.teleport_reduce`, as it preserves the circuit structure.

        >>> g = pyzx.simplify.teleport_reduce(g)
        >>> pyzx.tcount(g)
        8

        If you give a closer look, the circuit contains now 53 gates; 8 :func:`qml.T` gates, 28 :func:`qml.CNOT`, 6 :func:`qml.Hadmard` and
        1 :func:`qml.X` and 10 :func:`qml.S`. We successfully reduced the T-count by 20 and have ten additional
        S gates. The number of CNOT gates remained the same.

        The :func:`from_zx` transform can now convert the optimized circuit back into PennyLane operations:

        .. code-block:: python

            tape_opt = qml.transforms.from_zx(g)

            wires = qml.wires.Wires([4, 3, 0, 2, 1])
            wires_map = dict(zip(tape_opt.wires, wires))
            tapes_opt_reorder, fn = qml.map_wires(input=tape_opt, wire_map=wires_map)[0][0]
            tape_opt_reorder = fn(tapes_opt_reorder)

            @qml.qnode(device=dev)
            def mod_5_4():
                for g in tape_opt_reorder:
                    qml.apply(g)
                return qml.expval(qml.Z(0))

        >>> mod_5_4()
        tensor(1., requires_grad=True)

    .. note::

        It is a PennyLane adapted and reworked `circuit_to_graph <https://github.com/Quantomatic/pyzx/blob/master/pyzx/circuit/graphparser.py>`_
        function.

        Copyright (C) 2018 - Aleks Kissinger and John van de Wetering
    """
    # If it is a simple operation just transform it to a tape
    if not isinstance(tape, Operator):
        if not isinstance(tape, (qml.tape.QuantumScript, qml.QNode)) and not callable(tape):
            raise TransformError("Input is not an Operator, tape, QNode, or quantum function")
        return _to_zx_transform(tape, expand_measurements=expand_measurements)

    return to_zx(QuantumScript([tape]))


@partial(transform, is_informative=True)
def _to_zx_transform(
    tape: QuantumTape, expand_measurements=False
) -> (Sequence[QuantumTape], Callable):
    """Private function to convert a PennyLane tape to a `PyZX graph <https://pyzx.readthedocs.io/en/latest/>`_ ."""
    # Avoid to make PyZX a requirement for PennyLane.
    try:
        # pylint: disable=import-outside-toplevel
        import pyzx
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

    def processing_fn(res):
        # Create the graph, a qubit mapper, the classical mapper stays empty as PennyLane does not support classical bits.
        graph = Graph(None)
        q_mapper = TargetMapper()
        c_mapper = TargetMapper()

        # Map the wires to consecutive wires

        consecutive_wires = Wires(range(len(res[0].wires)))
        consecutive_wires_map = OrderedDict(zip(res[0].wires, consecutive_wires))
        mapped_tapes, fn = qml.map_wires(input=res[0], wire_map=consecutive_wires_map)
        mapped_tape = fn(mapped_tapes)

        inputs = []

        # Create the qubits in the graph and the qubit mapper
        for i in range(len(mapped_tape.wires)):
            vertex = graph.add_vertex(VertexType.BOUNDARY, i, 0)
            inputs.append(vertex)
            q_mapper.set_prev_vertex(i, vertex)
            q_mapper.set_next_row(i, 1)
            q_mapper.set_qubit(i, i)

        # Expand the tape to be compatible with PyZX and add rotations first for measurements
        stop_crit = qml.BooleanFn(lambda obj: isinstance(obj, Operator) and obj.name in gate_types)
        mapped_tape = qml.tape.tape.expand_tape(
            mapped_tape, depth=10, stop_at=stop_crit, expand_measurements=expand_measurements
        )

        expanded_operations = []

        # Define specific decompositions
        for op in mapped_tape.operations:
            if op.name == "RY":
                theta = op.data[0]
                decomp = [
                    qml.RX(np.pi / 2, wires=op.wires),
                    qml.RZ(theta + np.pi, wires=op.wires),
                    qml.RX(np.pi / 2, wires=op.wires),
                    qml.RZ(3 * np.pi, wires=op.wires),
                ]
                expanded_operations.extend(decomp)
            else:
                expanded_operations.append(op)

        expanded_tape = QuantumScript(expanded_operations, mapped_tape.measurements)

        _add_operations_to_graph(expanded_tape, graph, gate_types, q_mapper, c_mapper)

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

    return [tape], processing_fn


def _add_operations_to_graph(tape, graph, gate_types, q_mapper, c_mapper):
    """Add the tape operation to the PyZX graph."""
    # Create graph from circuit in the quantum tape (operations, measurements)
    for op in tape.operations:
        # Check that the gate is compatible with PyZX
        name = op.name
        if name not in gate_types:
            raise qml.QuantumFunctionError(
                "The expansion of the quantum tape failed, PyZX does not support", name
            )

        # Apply wires and parameters
        map_gate = gate_types[name]

        args = [*op.wires, *(p / np.pi for p in op.parameters)]

        gate = map_gate(*args)
        gate.to_graph(graph, q_mapper, c_mapper)


def from_zx(graph, decompose_phases=True):
    """Converts a graph from `PyZX <https://pyzx.readthedocs.io/en/latest/>`_ to a PennyLane tape, if the graph is
    diagram-like.

    Args:
        graph (Graph): ZX graph in PyZX.
        decompose_phases (bool): If True the phases are decomposed, meaning that :func:`qml.RZ` and :func:`qml.RX` are
            simplified into other gates (e.g. :func:`qml.T`, :func:`qml.S`, ...).

    **Example**

    From the example for the :func:`~.to_zx` function, one can convert back the PyZX graph to a PennyLane by using the
    function :func:`~.from_zx`.

    .. code-block:: python

        import pyzx
        dev = qml.device('default.qubit', wires=2)

        @qml.transforms.to_zx
        def circuit(p):
            qml.RZ(p[0], wires=0),
            qml.RZ(p[1], wires=0),
            qml.RX(p[2], wires=1),
            qml.Z(1),
            qml.RZ(p[3], wires=0),
            qml.X(0),
            qml.CNOT(wires=[1, 0]),
            qml.CNOT(wires=[0, 1]),
            qml.SWAP(wires=[1, 0]),
            return qml.expval(qml.Z(0) @ qml.Z(1))

        params = [5 / 4 * np.pi, 3 / 4 * np.pi, 0.1, 0.3]
        g = circuit(params)

        pennylane_tape = qml.transforms.from_zx(g)

    You can check that the operations are similar but some were decomposed in the process.

    >>> pennylane_tape.operations
    [Z(0),
     T(wires=[0]),
     RX(0.1, wires=[1]),
     Z(0),
     Adjoint(T(wires=[0])),
     Z(1),
     RZ(0.3, wires=[0]),
     X(0),
     CNOT(wires=[1, 0]),
     CNOT(wires=[0, 1]),
     CNOT(wires=[1, 0]),
     CNOT(wires=[0, 1]),
     CNOT(wires=[1, 0])]

    .. warning::

        Be careful because not all graphs are circuit-like, so the process might not be successful
        after you apply some optimization on your PyZX graph. You can extract a circuit by using the dedicated
        PyZX function.

    .. note::

        It is a PennyLane adapted and reworked `graph_to_circuit <https://github.com/Quantomatic/pyzx/blob/master/pyzx/circuit/graphparser.py>`_
        function.

        Copyright (C) 2018 - Aleks Kissinger and John van de Wetering

    """

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
                    "Graph doesn't seem circuit like: multiple parents. Try to use the PyZX function `extract_circuit`."
                )

            neighbor_0 = neighbors[0]

            if qubits[neighbor_0] != qubit_1:
                raise qml.QuantumFunctionError(
                    "Cross qubit connections, the graph is not circuit-like."
                )

            # Add Hadamard gate (written in the edge)
            if graph.edge_type(graph.edge(neighbor_0, vertex)) == EdgeType.HADAMARD:
                operations.append(qml.Hadamard(wires=qubit_1))

            # Vertex is a boundary
            if type_1 == VertexType.BOUNDARY:
                continue

            # Add the one qubits gate
            operations.extend(_add_one_qubit_gate(param, type_1, qubit_1, decompose_phases))

            # Given the neighbors on the same rowadd two qubits gates
            neighbors = [
                w for w in graph.neighbors(vertex) if graph_rows[w] == row_key and w < vertex
            ]

            for neighbor in neighbors:
                type_2 = types[neighbor]
                qubit_2 = qubits[neighbor]

                operations.extend(
                    _add_two_qubit_gates(graph, vertex, neighbor, type_1, type_2, qubit_1, qubit_2)
                )

    return QuantumScript(operations)


def _add_one_qubit_gate(param, type_1, qubit_1, decompose_phases):
    """Return the list of one qubit gates, that will be added to the tape."""
    if decompose_phases:
        type_z = type_1 == VertexType.Z
        if type_z and param.denominator == 2:
            op = qml.adjoint(qml.S(wires=qubit_1)) if param.numerator == 3 else qml.S(wires=qubit_1)
            return [op]
        if type_z and param.denominator == 4:
            if param.numerator in (1, 7):
                op = (
                    qml.adjoint(qml.T(wires=qubit_1))
                    if param.numerator == 7
                    else qml.T(wires=qubit_1)
                )
                return [op]
            if param.numerator in (3, 5):
                op1 = qml.Z(qubit_1)
                op2 = (
                    qml.adjoint(qml.T(wires=qubit_1))
                    if param.numerator == 3
                    else qml.T(wires=qubit_1)
                )
                return [op1, op2]
        if param == 1:
            op = qml.Z(qubit_1) if type_1 == VertexType.Z else qml.X(qubit_1)
            return [op]
        if param != 0:
            scaled_param = np.pi * float(param)
            op_class = qml.RZ if type_1 == VertexType.Z else qml.RX
            return [op_class(scaled_param, wires=qubit_1)]
    # Phases are not decomposed
    if param != 0:
        scaled_param = np.pi * float(param)
        op_class = qml.RZ if type_1 == VertexType.Z else qml.RX
        return [op_class(scaled_param, wires=qubit_1)]

    # No gate is added
    return []


def _add_two_qubit_gates(graph, vertex, neighbor, type_1, type_2, qubit_1, qubit_2):
    """Return the list of two qubit gates giveeen the vertex and its neighbor."""
    if type_1 == type_2:
        if graph.edge_type(graph.edge(vertex, neighbor)) != EdgeType.HADAMARD:
            raise qml.QuantumFunctionError(
                "Two green or respectively two red nodes connected by a simple edge does not have a "
                "circuit representation."
            )

        if type_1 == VertexType.Z:
            op = qml.CZ(wires=[qubit_2, qubit_1])
            return [op]

        op_1 = qml.Hadamard(wires=qubit_2)
        op_2 = qml.CNOT(wires=[qubit_2, qubit_1])
        op_3 = qml.Hadamard(wires=qubit_2)
        return [op_1, op_2, op_3]

    if graph.edge_type(graph.edge(vertex, neighbor)) != EdgeType.SIMPLE:
        raise qml.QuantumFunctionError(
            "A green and red node connected by a Hadamard edge does not have a circuit representation."
        )
    # Type1 is always of type Z therefore the qubits are already ordered.
    op = qml.CNOT(wires=[qubit_1, qubit_2])
    return [op]
