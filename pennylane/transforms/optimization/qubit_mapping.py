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

"""Transform for mapping a quantum circuit into a given architecture."""
# pylint: disable=too-many-branches

from functools import lru_cache

import networkx as nx

import pennylane as qml
from pennylane.tape import QuantumScript
from pennylane.transforms import transform


@transform
def qubit_mapping(tape, graph, init_mapping=None):
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

    # On-demand cached shortest path and distance
    @lru_cache(maxsize=None)
    def get_path(u, v):
        return nx.shortest_path(phys_graph, u, v)

    @lru_cache(maxsize=None)
    def get_dist(u, v):
        return len(get_path(u, v)) - 1

    # Initialize logical-to-physical mapping
    logical_qubits = tape.wires
    phys_qubits = list(graph.keys())
    num_logical = len(logical_qubits)
    num_phys = len(phys_qubits)

    if init_mapping is None:
        if all(w in phys_qubits for w in logical_qubits):
            mapping = {w: w for w in logical_qubits}
        elif num_logical <= num_phys:
            mapping = {logical_qubits[i]: phys_qubits[i] for i in range(num_logical)}
        else:
            raise ValueError(
                f"Insufficient physical qubits: {num_phys} < {num_logical} logical wires."
            )
    else:
        mapping = init_mapping.copy()

    # Precompute future two-qubit dependencies
    future = {w: [] for w in logical_qubits}
    ops_list = list(tape.operations)
    for idx, op in enumerate(ops_list):
        if len(op.wires) == 2:
            w0, w1 = op.wires
            future[w0].append((idx, w1))
            future[w1].append((idx, w0))

    next_partner = [{} for _ in ops_list]
    for idx in range(len(ops_list)):
        for q in logical_qubits:
            part = None
            for j, p in future[q]:
                if j > idx:
                    part = p
                    break
            next_partner[idx][q] = part

    new_ops = []

    # Long-range CNOT implementation
    def long_range_cnot(phys_path):
        L = len(phys_path) - 1
        if L <= 0:
            return
        if L == 1:
            new_ops.append(qml.CNOT(wires=phys_path))
            return
        mid = L // 2
        # forward
        for i in range(mid):
            new_ops.append(qml.CNOT(wires=[phys_path[i], phys_path[i + 1]]))
        # backward
        for i in range(L, mid, -1):
            new_ops.append(qml.CNOT(wires=[phys_path[i], phys_path[i - 1]]))
        new_ops.append(qml.CNOT(wires=[phys_path[mid], phys_path[mid + 1]]))
        # re-expand
        for i in range(mid + 1, L + 1):
            new_ops.append(qml.CNOT(wires=[phys_path[i], phys_path[i - 1]]))
        for i in range(mid - 1, -1, -1):
            new_ops.append(qml.CNOT(wires=[phys_path[i], phys_path[i + 1]]))

    # Process each operation
    for idx, op in enumerate(ops_list):
        wires = list(op.wires)
        # Error on operators > 2 wires
        if len(wires) > 2:
            raise ValueError("All operations should act in less than 3 wires.")

        # CNOT routing
        if len(wires) == 2 and op.name == "CNOT":
            lc, lt = wires
            pc, pt = mapping[lc], mapping[lt]
            phys_path = get_path(pc, pt)
            d = len(phys_path) - 1
            if d == 1:
                new_ops.append(qml.CNOT(wires=[pc, pt]))
            else:
                mid = d // 2
                pctrl = next_partner[idx][lc]
                ptrg = next_partner[idx][lt]
                best_score = float("inf")
                best_k1, best_k2 = 0, d
                for k1 in range(mid + 1):
                    pos1 = phys_path[k1]
                    for k2 in range(mid, d + 1):
                        if k2 <= k1:
                            continue
                        pos2 = phys_path[k2]
                        score = 0
                        if pctrl is not None:
                            score += get_dist(pos1, mapping[pctrl])
                        if ptrg is not None:
                            score += get_dist(pos2, mapping[ptrg])
                        if score < best_score or (
                            score == best_score and (k2 - k1) < (best_k2 - best_k1)
                        ):
                            best_score, best_k1, best_k2 = score, k1, k2
                # SWAPs for control
                for i in range(best_k1):
                    u, v = phys_path[i], phys_path[i + 1]
                    new_ops.append(qml.SWAP(wires=[u, v]))
                    inv = {pos: lg for lg, pos in mapping.items()}
                    if inv.get(u) is not None:
                        mapping[inv[u]] = v
                    if inv.get(v) is not None:
                        mapping[inv[v]] = u
                # SWAPs for target
                for i in range(d, best_k2, -1):
                    u, v = phys_path[i], phys_path[i - 1]
                    new_ops.append(qml.SWAP(wires=[u, v]))
                    inv = {pos: lg for lg, pos in mapping.items()}
                    if inv.get(u) is not None:
                        mapping[inv[u]] = v
                    if inv.get(v) is not None:
                        mapping[inv[v]] = u
                # long-range CNOT
                sub = phys_path[best_k1 : best_k2 + 1]
                long_range_cnot(sub)
        # Other 2-qubit gates
        elif len(wires) == 2:
            l0, l1 = wires
            p0, p1 = mapping[l0], mapping[l1]
            phys_path = get_path(p0, p1)
            d = len(phys_path) - 1
            if d == 1:
                new_ops.append(op.map_wires({l0: p0, l1: p1}))
            else:
                npc = next_partner[idx][l0]
                npt = next_partner[idx][l1]
                best_score, best_edge = float("inf"), 0
                for b in range(d):
                    u, v = phys_path[b], phys_path[b + 1]
                    score = 0
                    if npc is not None:
                        score += get_dist(u, mapping[npc])
                    if npt is not None:
                        score += get_dist(v, mapping[npt])
                    if score < best_score:
                        best_score, best_edge = score, b
                left, right = phys_path[best_edge], phys_path[best_edge + 1]
                while (mapping[l0], mapping[l1]) != (left, right):
                    c0, c1 = mapping[l0], mapping[l1]
                    if c0 != left:
                        nxt = phys_path[phys_path.index(c0) + 1]
                        new_ops.append(qml.SWAP(wires=[c0, nxt]))
                        inv = {pos: lg for lg, pos in mapping.items()}
                        if inv.get(c0) is not None:
                            mapping[inv[c0]] = nxt
                        if inv.get(nxt) is not None:
                            mapping[inv[nxt]] = c0
                    else:
                        nxt = phys_path[phys_path.index(c1) - 1]
                        new_ops.append(qml.SWAP(wires=[c1, nxt]))
                        inv = {pos: lg for lg, pos in mapping.items()}
                        if inv.get(c1) is not None:
                            mapping[inv[c1]] = nxt
                        if inv.get(nxt) is not None:
                            mapping[inv[nxt]] = c1
                new_ops.append(op.map_wires({l0: mapping[l0], l1: mapping[l1]}))
        # Single-qubit gates
        else:
            new_ops.append(op.map_wires({q: mapping[q] for q in wires}))

    # Remap measurements
    new_meas = [m.map_wires({q: mapping[q] for q in m.wires}) for m in tape.measurements]
    return [QuantumScript(new_ops, new_meas)], lambda results: results[0]
