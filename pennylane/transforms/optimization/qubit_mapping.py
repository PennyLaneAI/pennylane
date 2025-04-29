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
    Hybrid routing transform:

    Implements CNOT routing by selectively swapping control and target
    qubits toward optimal split points on the shortest path, then using a
    remote meet-in-the-middle CNOT block on the remaining subpath.

    Steps:
    1) Build the physical connectivity graph.
    2) Initialize logical→physical mapping (identity by default).
    3) Precompute all-pairs shortest paths and distances.
    4) Determine the next logical partner for each qubit at each CNOT index.
    5) For each operation:
       - If non-adjacent CNOT:
         a) Find shortest physical path of length d+1.
         b) Let mid = floor(d/2).  Search k1 in [0..mid], k2 in [mid..d], k2>k1
            minimizing: dist(path[k1], next_partner_of_control) +
                       dist(path[k2], next_partner_of_target).
         c) Apply k1 SWAPs on the control end (path[0..k1]) and k2..d SWAPs
            on the target end (path[d..k2]), updating mapping.
         d) Run the meet-in-the-middle CNOT block on the subpath
            path[k1..k2] (inclusive).
       - Else if other 2-qubit gate: SWAP-based routing + native gate.
       - Else: rename single-qubit ops by mapping.
    6) Remap measurements.
    """
    # 1) Build physical graph
    phys = nx.Graph()
    for q, nbrs in graph.items():
        phys.add_edges_from((q, nbr) for nbr in nbrs)

    # 2) Initialize mapping
    if init_mapping is None:
        logicals = {w for op in tape.operations for w in op.wires}
        mapping = {q: q for q in logicals}
    else:
        mapping = init_mapping.copy()

    # 3) Precompute shortest paths and distances
    spaths = dict(nx.all_pairs_shortest_path(phys))
    dist = {u: {v: len(p)-1 for v,p in targets.items()} for u,targets in spaths.items()}
    path = {(u,v): p for u,targets in spaths.items() for v,p in targets.items()}

    # 4) Precompute next logical partner per CNOT
    ops_list = list(tape.operations)
    future: Dict[int, List[Tuple[int,int]]] = {q: [] for op in ops_list for q in op.wires}
    for idx, op in enumerate(ops_list):
        if op.name == 'CNOT':
            c, t = op.wires
            future[c].append((idx, t))
            future[t].append((idx, c))
    next_partner: List[Dict[int, Optional[int]]] = [{} for _ in ops_list]
    for idx in range(len(ops_list)):
        for q in mapping:
            # find the smallest j>idx where q participates
            part = None
            for j, p in future[q]:
                if j > idx:
                    part = p
                    break
            next_partner[idx][q] = part

    new_ops: List = []

    def meet_remote_block(phys_path: List[int]):
        """Execute meet-in-the-middle CNOTs on phys_path list."""
        L = len(phys_path) - 1
        if L <= 0:
            return
        if L == 1:
            new_ops.append(qml.CNOT(wires=phys_path))
            return
        mid = L // 2
        # forward chain
        for i in range(mid):
            new_ops.append(qml.CNOT(wires=[phys_path[i], phys_path[i+1]]))
        # backward chain
        for i in range(L, mid, -1):
            new_ops.append(qml.CNOT(wires=[phys_path[i], phys_path[i-1]]))
        # central
        new_ops.append(qml.CNOT(wires=[phys_path[mid], phys_path[mid+1]]))
        # uncompute backward
        for i in range(mid+1, L+1):
            new_ops.append(qml.CNOT(wires=[phys_path[i], phys_path[i-1]]))
        # uncompute forward
        for i in range(mid-1, -1, -1):
            new_ops.append(qml.CNOT(wires=[phys_path[i], phys_path[i+1]]))

    # 5) Process each op
    for idx, op in enumerate(ops_list):
        w = list(op.wires)
        # Case A: targeted CNOT routing
        if len(w) == 2 and op.name == 'CNOT':
            lc, lt = w
            pc, pt = mapping[lc], mapping[lt]
            phys_path = path[(pc, pt)]
            d = len(phys_path) - 1
            if d == 1:
                # adjacent
                new_ops.append(qml.CNOT(wires=[pc, pt]))
            else:
                # 5a) select k1, k2
                mid = d // 2
                npc = next_partner[idx][lc]
                npt = next_partner[idx][lt]
                best_score = float('inf')
                best_k1, best_k2 = 0, d
                # search split
                for k1 in range(mid+1):
                    # qc at phys_path[k1]
                    pos1 = phys_path[k1]
                    for k2 in range(mid, d+1):
                        if k2 <= k1:
                            continue
                        pos2 = phys_path[k2]
                        score = 0
                        if npc is not None:
                            score += dist[pos1][mapping[npc]]
                        if npt is not None:
                            score += dist[pos2][mapping[npt]]
                        if score < best_score:
                            best_score = score
                            best_k1, best_k2 = k1, k2
                # 5b) apply k1 swaps on control side
                for i in range(best_k1):
                    u, v = phys_path[i], phys_path[i+1]
                    new_ops.append(qml.SWAP(wires=[u, v]))
                    # update mapping
                    inv_map = {pos: log for log, pos in mapping.items()}
                    lu, lv = inv_map[u], inv_map[v]
                    mapping[lu], mapping[lv] = v, u
                # 5c) apply (d-best_k2) swaps on target side
                for i in range(d, best_k2, -1):
                    u, v = phys_path[i], phys_path[i-1]
                    new_ops.append(qml.SWAP(wires=[u, v]))
                    inv_map = {pos: log for log, pos in mapping.items()}
                    lu, lv = inv_map[u], inv_map[v]
                    mapping[lu], mapping[lv] = v, u
                # 5d) remote CNOT on remainder phys_path[best_k1:best_k2+1]
                subpath = phys_path[best_k1:best_k2+1]
                meet_remote_block(subpath)
        # Case B: other 2-qubit gates
        elif len(w) == 2:
            l0, l1 = w
            p0, p1 = mapping[l0], mapping[l1]
            phys_path = path[(p0, p1)]
            for u, v in zip(phys_path, phys_path[1:]):
                new_ops.append(qml.SWAP(wires=[u, v]))
                inv_map = {pos: log for log, pos in mapping.items()}
                lu, lv = inv_map[u], inv_map[v]
                mapping[lu], mapping[lv] = v, u
            new_ops.append(op.map_wires({l0: mapping[l0], l1: mapping[l1]}))
        # Case C: single-qubit or others
        else:
            new_ops.append(op.map_wires({q: mapping[q] for q in w}))

    # 6) Remap measurements
    new_meas = [m.map_wires({q: mapping[q] for q in m.wires}) for m in tape.measurements]

    return [QuantumScript(new_ops, new_meas)], lambda results: results[0]
