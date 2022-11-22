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


from typing import Dict, List, Optional

import pyzx
from pyzx import Circuit
from pyzx import TargetMapper
from pyzx.utils import EdgeType, VertexType, FloatInt, FractionLike
from pyzx.graph import Graph
from pyzx.graph.base import BaseGraph, VT, ET


gate_types: Dict[str, Type[Gate]] = {
    "XPhase": XPhase,
    "NOT": NOT,
    "ZPhase": ZPhase,
    "YPhase": YPhase,
    "PauliZ": Z,
    "S": S,
    "T": T,
    "CNOT": CNOT,
    "CZ": CZ,
    "CX": CX,
    "SWAP": SWAP,
    "CRZ": CRZ,
    "HAD": HAD,
    "H": HAD,
    "CHAD": CHAD,
    "Toffoli": Tofolli,
    "CCZ": CCZ,
    "Measurement": Measurement,
}
#    "ParityPhase": ParityPhase,


def circuit_to_graph(
    q: QuantumTape, compress_rows: bool = True, backend: Optional[str] = None
) -> BaseGraph[VT, ET]:
    """Turns a PennyLane quantum tape into a ZX-Graph.
    If ``compress_rows`` is set, it tries to put single qubit gates on different qubits,
    on the same row."""
    g = Graph(backend)
    q_mapper: TargetMapper[VT] = TargetMapper()
    c_mapper: TargetMapper[VT] = TargetMapper()
    inputs = []
    outputs = []

    # Tape wires
    for i in range(q.qubits):
        v = g.add_vertex(VertexType.BOUNDARY, i, 0)
        inputs.append(v)
        q_mapper.set_prev_vertex(i, v)
        q_mapper.set_next_row(i, 1)
        q_mapper.set_qubit(i, i)

    # Expand the tape: add rotation first: only Z measurement I think

    # Create graph from circuit in the tape (operations, measurements)
    for op in q.circuit:
        # Map the gate

        # Control, target, phase
        gate = gate_types[op.name]
        gate = gate(*op.wires)

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
