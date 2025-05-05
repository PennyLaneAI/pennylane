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

"""Transform for hybrid CNOT routing and logical-to-physical qubit mapping on arbitrary connectivity graphs."""

import networkx as nx

import pennylane as qml
from pennylane.tape import QuantumScript
from pennylane.transforms import transform


@transform
def qubit_mapping(tape, graph, init_mapping=None):
    """Qubit mapping transform.

    Implements a qubit‐mapping scheme that connects nonadjacent logical qubits using SWAP
    operations and long-range CNOTs. Each qubit’s placement is dynamically chosen based on
    the location of its next scheduled gate.

    Supports cases with more physical qubits than logical wires by mapping
    extra physical qubits arbitrarily if no init_mapping is provided.

    Args:
        tape (QNode or QuantumTape or Callable): The input quantum circuit to transform.
        graph (dict): Adjacency list describing the connectivity of
            the physical qubits.
        init_mapping (dict or None): Optional initial mapping from logical
            wires to physical qubits. If None, a default mapping is chosen.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    **Example**

    .. code-block:: python

        dev = qml.device('default.qubit')

        graph = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}
        initial_map = {0: "a", 1: "b", 2: "c"}

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
    # Build physical graph
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
            raise ValueError(
                f"Insufficient physical qubits: {num_phys} < {num_logical} logical wires."
            )
    else:
        mapping = init_mapping.copy()

    # Precompute shortest paths and distances in the graph
    shortest_paths = dict(nx.all_pairs_shortest_path(phys_graph))
    dist = {u: {v: len(p) - 1 for v, p in targets.items()} for u, targets in shortest_paths.items()}
    path = {(u, v): p for u, targets in shortest_paths.items() for v, p in targets.items()}


    # Create a dependency graph of operators (``next_partner``)

    future = {w: [] for w in logical_qubits}
    # future[<qubit1>] --> List[(<ind>, <qubit2>)]
    # This means: the operators where <qubit1> is involved are
    # the <ind>th operators, that whose second action qubit is <qubit2>

    ops_list = list(tape.operations)
    for idx, op in enumerate(ops_list):
        if len(op.wires) == 2:
            wire0, wire1 = op.wires
            future[wire0].append((idx, wire1))
            future[wire1].append((idx, wire0))

    next_partner = [{} for _ in ops_list]
    # next_partner[<ind>][<qubit1>] --> <qubit2>
    # This means: After apply the <ind>th-operator, the next operation
    # that need to be linked with <qubit1> is located in <qubit2>
    for idx in range(len(ops_list)):
        for q in logical_qubits:
            part = None
            for j, p in future[q]:
                if j > idx:
                    part = p
                    break
            next_partner[idx][q] = part
    new_ops = []

    def longe_range_cnot(phys_path):
        L = len(phys_path) - 1
        if L <= 0:
            return
        if L == 1:
            new_ops.append(qml.CNOT(wires=phys_path))
            return

        # We implement the long range CNOT (Fig 4c [https://arxiv.org/pdf/2305.18128])
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

    # Process operations
    for idx, op in enumerate(ops_list):
        w = list(op.wires)
        # A: CNOT routing
        if len(w) == 2 and op.name == "CNOT":
            logical_control, logical_target = w
            phys_control, phys_target = mapping[logical_control], mapping[logical_target]
            phys_path = path[(phys_control, phys_target)]
            d = len(phys_path) - 1
            if d == 1:
                new_ops.append(qml.CNOT(wires=[phys_control, phys_target]))
            else:
                mid = d // 2
                partner_control = next_partner[idx][logical_control]
                partner_target = next_partner[idx][logical_target]
                best_score = float("inf")
                best_k1, best_k2 = 0, d
                for k1 in range(mid + 1):
                    pos1 = phys_path[k1]
                    for k2 in range(mid, d + 1):
                        if k2 <= k1:
                            continue
                        pos2 = phys_path[k2]
                        score = 0

                        # The score is the distance from the current positions to the next connection.
                        if partner_control is not None:
                            score += dist[pos1][mapping[partner_control]]
                        if partner_target is not None:
                            score += dist[pos2][mapping[partner_target]]
                        if score < best_score or (
                            score == best_score and (k2 - k1) < (best_k2 - best_k1)
                        ):
                            best_score = score
                            best_k1, best_k2 = k1, k2

                # swaps to the best positions (control qubit)
                for i in range(best_k1):
                    phys_u, phys_v = phys_path[i], phys_path[i + 1]
                    new_ops.append(qml.SWAP(wires=[phys_u, phys_v]))
                    inv_map = {pos: logical for logical, pos in mapping.items()}
                    logical_u = inv_map.get(phys_u)
                    logical_v = inv_map.get(phys_v)
                    if logical_u is not None:
                        mapping[logical_u] = phys_v
                    if logical_v is not None:
                        mapping[logical_v] = phys_u

                # swaps to the best positions (target qubit)
                for i in range(d, best_k2, -1):
                    phys_u, phys_v = phys_path[i], phys_path[i - 1]
                    new_ops.append(qml.SWAP(wires=[phys_u, phys_v]))
                    inv_map = {pos: logical for logical, pos in mapping.items()}
                    logical_u = inv_map.get(phys_u)
                    logical_v = inv_map.get(phys_v)
                    if logical_u is not None:
                        mapping[logical_u] = phys_v
                    if logical_v is not None:
                        mapping[logical_v] = phys_u

                # long range cnot to connect the operations
                subpath = phys_path[best_k1 : best_k2 + 1]
                longe_range_cnot(subpath)

        # B: other 2-qubit gates (long range handled via optimized adjacent swaps)
        elif len(w) == 2:
            logical_0, logical_1 = w
            phys_0, phys_1 = mapping[logical_0], mapping[logical_1]
            phys_path = path[(phys_0, phys_1)]
            d = len(phys_path) - 1

            if d == 1:
                # Adjacent already: apply gate directly
                new_ops.append(op.map_wires({logical_0: phys_0, logical_1: phys_1}))
            else:
                # Compute next partners and find best meeting edge
                npc = next_partner[idx][logical_0]
                npt = next_partner[idx][logical_1]

                best_score = float("inf")
                best_edge = 0
                for b in range(d):
                    u, v = phys_path[b], phys_path[b + 1]
                    score = 0
                    if npc is not None:
                        score += dist[u][mapping[npc]]
                    if npt is not None:
                        score += dist[v][mapping[npt]]
                    if score < best_score:
                        best_score = score
                        best_edge = b

                # Determine the target adjacent positions
                left_target = phys_path[best_edge]
                right_target = phys_path[best_edge + 1]

                # Move each qubit by adjacent swaps until they occupy the target edge
                while (mapping[logical_0], mapping[logical_1]) != (left_target, right_target):
                    current_0 = mapping[logical_0]
                    current_1 = mapping[logical_1]

                    if current_0 != left_target:
                        # Swap logical_0 one step toward left_target
                        idx0 = phys_path.index(current_0)
                        next_pos = phys_path[idx0 + 1]
                        new_ops.append(qml.SWAP(wires=[current_0, next_pos]))
                        inv_map = {pos: log for log, pos in mapping.items()}
                        lu = inv_map.get(current_0)
                        lv = inv_map.get(next_pos)
                        if lu is not None:
                            mapping[lu] = next_pos
                        if lv is not None:
                            mapping[lv] = current_0

                    else:
                        # Swap logical_1 one step toward right_target
                        idx1 = phys_path.index(current_1)
                        next_pos = phys_path[idx1 - 1]
                        new_ops.append(qml.SWAP(wires=[current_1, next_pos]))
                        inv_map = {pos: log for log, pos in mapping.items()}
                        lu = inv_map.get(current_1)
                        lv = inv_map.get(next_pos)
                        if lu is not None:
                            mapping[lu] = next_pos
                        if lv is not None:
                            mapping[lv] = current_1

                # 4. Now adjacent at optimal edge: apply the two-qubit gate
                new_ops.append(
                    op.map_wires(
                        {
                            logical_0: mapping[logical_0],
                            logical_1: mapping[logical_1],
                        }
                    )
                )

        elif len(w) > 2:
            raise ValueError(f"All operations should act in less than 3 wires.")

        # C: single-qubit
        else:
            new_ops.append(op.map_wires({q: mapping[q] for q in w}))

    new_meas = [m.map_wires({q: mapping[q] for q in m.wires}) for m in tape.measurements]
    return [QuantumScript(new_ops, new_meas)], lambda results: results[0]
