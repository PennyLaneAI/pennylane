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
"""Transform finding all maximal matches of a pattern in a quantum circuit and optimizing the circuit by
substitution."""

import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.transforms import qfunc_transform


def _import_pyzx():
    """Import PyZx."""
    try:
        # pylint: disable=import-outside-toplevel, unused-import, multiple-imports
        import pyzx
    except ImportError as Error:
        raise ImportError(
            "This feature requires pyzx. It can be installed with: pip install pyzx"
        ) from Error

    return pyzx


@qfunc_transform
def zx_optimization(tape: QuantumTape):
    r"""Quantum function transform to optimize a circuit given a list of patterns (templates).

    Args:
        qfunc (function): A quantum function to be optimized.

    Returns:
        function: the transformed quantum function

    **Example**

    .. code-block:: python

        def circuit(x):
            qml.Hadamard(wires=0)
            qml.PauliX(wires=0)
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            qml.PauliZ(wires=0)
            return qml.expval(qml.PauliZ(0))

    from pennylane.transforms.optimization.zx_optimization import zx_optimization

    optimized_qfunc = zx_optimization(circuit)

    qnode_opt = qml.QNode(optimized_qfunc, dev)
    >>> print(qml.draw(qnode_opt)(0.4))
    0: ──RZ(3.14)──H──RZ(3.14)─┤  <Z>
    """
    # Try import pyzx
    zx = _import_pyzx()

    # Save the measurements
    measurements = tape.measurements

    # PennyLane to QASM
    open_qasm_circuit = tape.to_openqasm()

    # QASM -> ZX circuit -> ZX Graph
    zx_g = zx.qasm(open_qasm_circuit).to_graph()

    # Optimization pass
    zx.full_reduce(zx_g)

    # ZX graph -> ZX circuit
    zx_simplified_c = zx.extract_circuit(zx_g)

    # ZX circuit -> QASM
    zx_simplified_c_qasm = zx_simplified_c.to_qasm()

    # QASM to PennyLane
    circuit_simplified = qml.from_qasm(zx_simplified_c_qasm)

    # PennyLane function -> PennyLane tape
    with qml.tape.QuantumTape(do_queue=False) as tape_new:
        circuit_simplified()

    # Queueing of the optimised tape
    for op in tape_new.operations:
        qml.apply(op)

    for m in measurements:
        qml.apply(m)


from collections import OrderedDict
from typing import Dict, List, Optional, Type

from pennylane.wires import Wires

from pyzx.circuit import Gate
from pyzx.circuit.gates import (
    TargetMapper,
    NOT,
    Z,
    S,
    T,
    HAD,
    XPhase,
    YPhase,
    ZPhase,
    SWAP,
    CNOT,
    CZ,
    CRZ,
    CHAD,
    CCZ,
    Tofolli,
    Measurement,
)
from pyzx.utils import EdgeType, VertexType, FloatInt, FractionLike
from pyzx.graph import Graph
from pyzx.graph.base import BaseGraph, VT, ET

gate_types: Dict[str, Type[Gate]] = {
    "PauliX": NOT,
    "PauliZ": Z,
    "S": S,
    "T": T,
    "Hadamard": HAD,
    "RX": XPhase,
    "RZ": ZPhase,
    "PhaseShift": ZPhase,
    "RY": YPhase,
    "SWAP": SWAP,
    "CNOT": CNOT,
    "CZ": CZ,
    "CRZ": CRZ,
    "CH": CHAD,
    "CCZ": CCZ,
    "Toffoli": Tofolli,
    "Measurement": Measurement,
}


#    "ParityPhase": ParityPhase,


def circuit_to_graph(
    tape: QuantumTape, compress_rows: bool = True, backend: Optional[str] = None
) -> BaseGraph[VT, ET]:
    """Turns a PennyLane quantum tape into a ZX-Graph.
    If ``compress_rows`` is set, it tries to put single qubit gates on different qubits,
    on the same row."""
    g = Graph(backend)
    q_mapper: TargetMapper[VT] = TargetMapper()
    c_mapper: TargetMapper[VT] = TargetMapper()
    inputs = []
    outputs = []

    # Map the wires to consecutive wires
    consecutive_wires = Wires(range(len(tape.wires)))
    consecutive_wires_map = OrderedDict(zip(tape.wires, consecutive_wires))
    tape = qml.map_wires(input=tape, wire_map=consecutive_wires_map)

    # Create the qubit
    for i in range(len(tape.wires)):
        v = g.add_vertex(VertexType.BOUNDARY, i, 0)
        inputs.append(v)
        q_mapper.set_prev_vertex(i, v)
        q_mapper.set_next_row(i, 1)
        q_mapper.set_qubit(i, i)

    # Expand the tape and add rotations first for measurements
    stop_crit = qml.BooleanFn(lambda obj: obj.name in gate_types.keys())
    tape = qml.tape.tape.expand_tape(tape, depth=10, stop_at=stop_crit, expand_measurements=True)

    # Create graph from circuit in the tape (operations, measurements)
    for op in tape.operations:

        # Map the gate
        name = op.name
        if name not in gate_types.keys():
            raise qml.QuantumFunctionError(
                "The expansion of the tape failed, PyZx does not support", name
            )

        # Apply wires and parameters
        map_gate = gate_types[name]
        par = op.parameters
        wires = list(op.wires)
        # Max number of parameter is one
        if par:
            # Single wires is int
            if len(wires) == 1:
                wires = wires[0]
            args = [wires, par[0]]
        else:
            args = wires

        gate = map_gate(*args)

        # Create node in the graph
        if not compress_rows:  # or not isinstance(gate, (ZPhase, XPhase, HAD)):
            r = max(q_mapper.max_row(), c_mapper.max_row())
            q_mapper.set_all_rows(r)
            c_mapper.set_all_rows(r)
        gate.to_graph(g, q_mapper, c_mapper)
        if not compress_rows:  # or not isinstance(gate, (ZPhase, XPhase, HAD)):
            r = max(q_mapper.max_row(), c_mapper.max_row())
            q_mapper.set_all_rows(r)
            c_mapper.set_all_rows(r)

    r = max(q_mapper.max_row(), c_mapper.max_row())
    for mapper in (q_mapper, c_mapper):
        for l in mapper.labels():
            o = mapper.to_qubit(l)
            v = g.add_vertex(VertexType.BOUNDARY, o, r)
            outputs.append(v)
            u = mapper.prev_vertex(l)
            g.add_edge(g.edge(u, v))

    g.set_inputs(tuple(inputs))
    g.set_outputs(tuple(outputs))

    return g


def graph_to_circuit(g: BaseGraph[VT, ET], split_phases: bool = True) -> qml.tape.QuantumTape:
    """From PyZX graph to a PennyLane tape."""

    operations = []
    qs = g.qubits()
    rs = g.rows()
    ty = g.types()
    phases = g.phases()
    rows: Dict[FloatInt, List[VT]] = {}

    inputs = g.inputs()
    for v in g.vertices():
        if v in inputs:
            continue
        r = g.row(v)
        if r in rows:
            rows[r].append(v)
        else:
            rows[r] = [v]
    for r in sorted(rows.keys()):
        for v in rows[r]:
            q = qs[v]
            phase = phases[v]
            t = ty[v]
            neigh = [w for w in g.neighbors(v) if rs[w] < r]
            if len(neigh) != 1:
                raise TypeError("Graph doesn't seem circuit like: multiple parents")
            n = neigh[0]
            if qs[n] != q:
                raise TypeError("Graph doesn't seem circuit like: cross qubit connections")
            if g.edge_type(g.edge(n, v)) == EdgeType.HADAMARD:
                operations.append(qml.Hadamard(wires=q))
            if t == VertexType.BOUNDARY:  # vertex is an output
                continue
            if phase != 0 and not split_phases:
                if t == VertexType.Z:
                    operations.append(qml.RZ(phase, wires=q))
                else:
                    operations.append(qml.RX(phase, wires=q))
            elif t == VertexType.Z and phase.denominator == 2:
                if phase.numerator == 3:
                    operations.append(qml.adjoint(qml.S(wires=q)))
                else:
                    operations.append(qml.S(wires=q))
            elif t == VertexType.Z and phase.denominator == 4:
                if phase.numerator in (1, 7):
                    if phase.numerator == 7:
                        operations.append(qml.adjoint(qml.T(wires=q)))
                    else:
                        operations.append(qml.T(wires=q))
                if phase.numerator in (3, 5):
                    operations.append(qml.PauliZ(wires=q))
                    if phase.numerator == 3:
                        operations.append(qml.adjoint(qml.T(wires=q)))
                    else:
                        operations.append(qml.T(wires=q))
            elif phase == 1:
                if t == VertexType.Z:
                    operations.append(qml.PauliZ(wires=q))
                else:
                    operations.append(qml.PauliX(wires=q))
            elif phase != 0:
                if t == VertexType.Z:
                    operations.append(qml.RZ(phase, wires=q))
                else:
                    operations.append(qml.RX(phase, wires=q))

            neigh = [w for w in g.neighbors(v) if rs[w] == r and w < v]
            for n in neigh:
                t2 = ty[n]
                q2 = qs[n]
                if t == t2:
                    if g.edge_type(g.edge(v, n)) != EdgeType.HADAMARD:
                        raise TypeError(
                            "Invalid vertical connection between vertices of the same type"
                        )
                    if t == VertexType.Z:
                        operations.append(qml.CZ(wires=[q2, q]))
                    else:
                        operations.append(qml.Hadamard(wires=q2))
                        operations.append(qml.CNOT(wires=[q2, q]))
                        operations.append(qml.Hadamard(wires=q2))
                else:
                    if g.edge_type(g.edge(v, n)) != EdgeType.SIMPLE:
                        raise TypeError(
                            "Invalid vertical connection between vertices of different type"
                        )
                    if t == VertexType.Z:
                        operations.append(qml.CNOT(wires=[q, q2]))
                    else:
                        operations.append(qml.CNOT(wires=[q2, q]))
    tape = qml.tape.QuantumTape(operations, [], prep=[])
    return tape
