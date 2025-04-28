import pennylane as qml
from pennylane.transforms import transform
from pennylane.tape import QuantumScript
import networkx as nx
from typing import Dict, List, Optional, Tuple

@transform
def qubit_mapping(
    tape,
    graph: Dict[int, List[int]],
    init_mapping: Optional[Dict[int, int]] = None
) -> Tuple[List[QuantumScript], callable]:
    """
    Fast, shortest-path-only qubit mapping:
    - Builds a NetworkX graph once.
    - Maintains a mapping dict, defaulting to identity.
    - For each 2-qubit gate, finds the BFS shortest path, inserts SWAPs along it,
      updates the mapping, then emits the remapped gate.
    - Single-qubit and measurement wires are just renamed.
    """
    # 1) Build NX graph once
    G_nx = nx.Graph()
    for u, nbrs in graph.items():
        for v in nbrs:
            G_nx.add_edge(u, v)

    # 2) Initialize logical→physical mapping
    if init_mapping is None:
        wires = {w for op in tape.operations for w in op.wires}
        mapping = {q: q for q in wires}
    else:
        mapping = init_mapping.copy()

    new_ops = []
    # 3) Process operations
    for op in tape.operations:
        w = list(op.wires)
        if len(w) == 2:
            q1, q2 = w
            p1, p2 = mapping[q1], mapping[q2]
            # find shortest physical path
            path = nx.shortest_path(G_nx, p1, p2)
            # insert SWAPs along path (except final adjacency)
            for u, v in zip(path, path[1:]):
                new_ops.append(qml.SWAP(wires=[u, v]))
                # update mapping by swapping the two logical qubits
                inv = {phys: log for log, phys in mapping.items()}
                l_u, l_v = inv[u], inv[v]
                mapping[l_u], mapping[l_v] = v, u
            # now the qubits are adjacent: emit the two-qubit gate
            new_ops.append(op.map_wires({q1: mapping[q1], q2: mapping[q2]}))

        else:
            # single- or multi-qubit: just remap wires
            wire_map = {q: mapping[q] for q in w}
            new_ops.append(op.map_wires(wire_map))

    # 4) Remap measurements
    new_meas = [
        m.map_wires({q: mapping[q] for q in m.wires})
        for m in tape.measurements
    ]

    return [QuantumScript(new_ops, new_meas)], lambda results: results[0]