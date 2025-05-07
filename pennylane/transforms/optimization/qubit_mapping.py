# Copyright 2025 Xanadu Quantum Technologies Inc.
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

import networkx as nx
import pennylane as qml
from pennylane.tape import QuantumScript
from pennylane.transforms import transform


@transform
def qubit_mapping(tape, graph, init_mapping=None, window_size=10):
    """Qubit mapping transform with sliding window for dependency lookahead.

    Args:
        tape (QNode or QuantumTape or Callable): The input quantum circuit to transform.
        graph (dict): Adjacency list describing the connectivity of
            the physical qubits.
        init_mapping (dict or None): Optional initial mapping from logical
            wires to physical qubits. If None, a default mapping is chosen.
        window_size (int): Number of upcoming operations to inspect for dependencies.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    **Example**

    .. code-block:: python

        dev = qml.device('default.qubit')

        graph = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}

        @partial(qml.transforms.qubit_mapping, graph = graph)
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.CNOT([0,2])
            return qml.expval(qml.Z(2))

    >>> print(qml.draw(circuit)())
    a: ──H────╭●─┤
    b: ─╭SWAP─╰X─┤  <Z>
    c: ─╰SWAP────┤
    """
    # Build physical connectivity graph
    phys_graph = nx.Graph()
    for q, nbrs in graph.items():
        phys_graph.add_edges_from((q, nbr) for nbr in nbrs)

    # Initialize mapping
    logical_qubits = tape.wires
    num_logical = len(logical_qubits)
    phys_qubits = list(graph.keys())
    num_phys = len(phys_qubits)
    if init_mapping is None:
        if all(w in phys_qubits for w in logical_qubits):
            mapping = {w: w for w in logical_qubits}
        elif num_logical <= num_phys:
            mapping = {logical_qubits[i]: phys_qubits[i] for i in range(num_logical)}
        else:
            raise ValueError(f"Insufficient physical qubits: {num_phys} < {num_logical}.")
    else:
        mapping = init_mapping.copy()

    # Lazy path & distance caches
    path_cache, dist_cache = {}, {}

    def get_path(u, v):
        if (u, v) not in path_cache:
            p = nx.shortest_path(phys_graph, source=u, target=v)
            path_cache[(u, v)] = p
            path_cache[(v, u)] = list(reversed(p))
            dist_cache[(u, v)] = len(p) - 1
            dist_cache[(v, u)] = len(p) - 1
        return path_cache[(u, v)]

    def get_dist(u, v):
        if (u, v) not in dist_cache:
            get_path(u, v)
        return dist_cache[(u, v)]

    ops_list = list(tape.operations)

    # Sliding-window lookup for next 2-qubit partner
    def find_next_partner(idx, qubit):
        end = min(idx + 1 + window_size, len(ops_list))
        for j in range(idx + 1, end):
            op_j = ops_list[j]
            if len(op_j.wires) == 2 and qubit in op_j.wires:
                w0, w1 = op_j.wires
                return w1 if w0 == qubit else w0
        return None

    new_ops = []

    def longe_range_cnot(phys_path):
        L = len(phys_path) - 1
        if L <= 0:
            return
        if L == 1:
            new_ops.append(qml.CNOT(wires=phys_path))
            return
        mid = L // 2
        for i in range(mid):
            new_ops.append(qml.CNOT(wires=[phys_path[i], phys_path[i + 1]]))
        for i in range(L, mid, -1):
            new_ops.append(qml.CNOT(wires=[phys_path[i], phys_path[i - 1]]))
        new_ops.append(qml.CNOT(wires=[phys_path[mid], phys_path[mid + 1]]))
        for i in range(mid + 1, L + 1):
            new_ops.append(qml.CNOT(wires=[phys_path[i], phys_path[i - 1]]))
        for i in range(mid - 1, -1, -1):
            new_ops.append(qml.CNOT(wires=[phys_path[i], phys_path[i + 1]]))

    # Process each operation
    for idx, op in enumerate(ops_list):
        wires = list(op.wires)
        if len(wires) == 2:
            l0, l1 = wires
            p0, p1 = mapping[l0], mapping[l1]
            phys_path = get_path(p0, p1)
            d = len(phys_path) - 1
            # Handle CNOT specifically
            if op.name == "CNOT":
                if d == 1:
                    new_ops.append(qml.CNOT(wires=[p0, p1]))
                else:
                    mid = d // 2
                    pc = find_next_partner(idx, l0)
                    pt = find_next_partner(idx, l1)
                    # independent optimal positions
                    best_k1 = min(
                        range(mid + 1),
                        key=lambda k: get_dist(phys_path[k], mapping[pc]) if pc else 0,
                    )
                    best_k2 = min(
                        range(mid, d + 1),
                        key=lambda k: get_dist(phys_path[k], mapping[pt]) if pt else 0,
                    )
                    # swaps control
                    for i in range(best_k1):
                        u, v = phys_path[i], phys_path[i + 1]
                        new_ops.append(qml.SWAP(wires=[u, v]))
                        inv = {pos: lg for lg, pos in mapping.items()}
                        if inv.get(u):
                            mapping[inv[u]] = v
                        if inv.get(v):
                            mapping[inv[v]] = u
                    # swaps target
                    for i in range(d, best_k2, -1):
                        u, v = phys_path[i], phys_path[i - 1]
                        new_ops.append(qml.SWAP(wires=[u, v]))
                        inv = {pos: lg for lg, pos in mapping.items()}
                        if inv.get(u):
                            mapping[inv[u]] = v
                        if inv.get(v):
                            mapping[inv[v]] = u
                    longe_range_cnot(phys_path[best_k1 : best_k2 + 1])
            else:
                # generic 2-qubit gate via adjacent swaps
                if d == 1:
                    new_ops.append(op.map_wires({l0: p0, l1: p1}))
                else:
                    npc = find_next_partner(idx, l0)
                    npt = find_next_partner(idx, l1)
                    best_edge = min(
                        range(d),
                        key=lambda b: (get_dist(phys_path[b], mapping[npc]) if npc else 0)
                        + (get_dist(phys_path[b + 1], mapping[npt]) if npt else 0),
                    )
                    left, right = phys_path[best_edge], phys_path[best_edge + 1]
                    while (mapping[l0], mapping[l1]) != (left, right):
                        if mapping[l0] != left:
                            cur = mapping[l0]
                            nxt = phys_path[phys_path.index(cur) + 1]
                            new_ops.append(qml.SWAP(wires=[cur, nxt]))
                            inv = {pos: lg for lg, pos in mapping.items()}
                            if inv.get(cur):
                                mapping[inv[cur]] = nxt
                            if inv.get(nxt):
                                mapping[inv[nxt]] = cur
                        else:
                            cur = mapping[l1]
                            nxt = phys_path[phys_path.index(cur) - 1]
                            new_ops.append(qml.SWAP(wires=[cur, nxt]))
                            inv = {pos: lg for lg, pos in mapping.items()}
                            if inv.get(cur):
                                mapping[inv[cur]] = nxt
                            if inv.get(nxt):
                                mapping[inv[nxt]] = cur
                    new_ops.append(op.map_wires({l0: mapping[l0], l1: mapping[l1]}))
        elif len(wires) > 2:
            raise ValueError("All operations should act in less than 3 wires.")
        else:
            new_ops.append(op.map_wires({q: mapping[q] for q in wires}))

    # Remap measurements
    new_meas = [m.map_wires({q: mapping[q] for q in m.wires}) for m in tape.measurements]
    return [QuantumScript(new_ops, new_meas)], lambda res: res[0]
