# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Bucket-Brigade QRAM with explicit bus routing for PennyLane, supporting:
- Select/QROM-style MSB prefix (``select_wires``),
- Bucket-brigade QRAM LSBs (``qram_wires``) using 3-qubits-per-node (dir, portL, portR),
- Hybrid mode where MSBs and/or LSBs can be classical constants.

Address loading is performed **layer-by-layer** by routing a single top **bus** qubit
down to the active node using CSWAPs controlled by already-written upper routers,
depositing each low-order address bit into the node's direction qubit.

Data phase routes the target qubits down to the selected leaf for each target bit,
performs the leaf write (classical bit flip), then routes back and restores the target.
"""

from typing import List, Optional, Sequence, Tuple

import pennylane as qml


# -----------------------------
# Utilities
# -----------------------------
def _num_levels(num_leaves: int) -> int:
    n = 0
    x = 1
    while x < num_leaves:
        x <<= 1
        n += 1
    if (1 << n) != num_leaves:
        raise ValueError("Number of bitstrings must be a power of two.")
    return n


def _level_offset(level: int) -> int:
    """Index offset of the first node at a given level (root=0). Offset = 2^level - 1."""
    return (1 << level) - 1


def _node_index(level: int, prefix_value: int) -> int:
    """Return the flat index (level order) of the internal node at `level` with prefix `prefix_value`."""
    return _level_offset(level) + prefix_value


# -----------------------------
# Select-prefix × Bucket-Brigade with explicit bus routing
# -----------------------------
class SelectBucketBrigadeBusQRAM(qml.operation.Operation):
    r"""Bucket-brigade QRAM with **explicit bus routing** using 3 qubits per node,
    and an optional **select (MSB) prefix**, plus **hybrid** support.

    Each internal node (level k, prefix p) has:
      - a direction qubit: ``dir[k,p]`` (stores the routed low-order address bit for level k),
      - two ports: ``portL[k,p]`` and ``portR[k,p]``.

    A single top **bus** qubit is reused for both phases:
      - Address loading (layer-by-layer via CSWAPs; deposit bit into `dir[k,p]`),
      - Data routing (per-target: swap target↔bus, route down, leaf write, route up, swap back).
    """

    grad_method = None

    def __init__(
        self,
        bitstrings: Sequence[str],
        select_wires: Sequence[int],
        qram_wires: Sequence[int],
        target_wires: Sequence[int],
        bus_wire: int,
        dir_wires: Sequence[int],
        portL_wires: Sequence[int],
        portR_wires: Sequence[int],
        *,
        mode: str = "quantum",
        select_value: Optional[int] = None,
        qram_value: Optional[int] = None,
        id: Optional[str] = None,
    ):
        if not bitstrings:
            raise ValueError("'bitstrings' cannot be empty.")
        m_set = {len(s) for s in bitstrings}
        if len(m_set) != 1:
            raise ValueError("All bitstrings must have equal length.")
        self.m = next(iter(m_set))
        self.bitstrings = list(bitstrings)

        self.select_wires = qml.wires.Wires(select_wires)
        self.qram_wires = qml.wires.Wires(qram_wires)
        self.target_wires = qml.wires.Wires(target_wires)
        self.bus_wire = qml.wires.Wires([bus_wire])
        self.dir_wires = qml.wires.Wires(dir_wires)
        self.portL_wires = qml.wires.Wires(portL_wires)
        self.portR_wires = qml.wires.Wires(portR_wires)

        if self.m != len(self.target_wires):
            raise ValueError("len(target_wires) must equal bitstring length.")

        self.k = len(self.select_wires)
        self.n_k = len(self.qram_wires)
        self.n = self.k + self.n_k
        if (1 << self.n) != len(self.bitstrings):
            raise ValueError("len(bitstrings) must be 2^(len(select_wires)+len(qram_wires)).")

        expected_nodes = (1 << self.n_k) - 1 if self.n_k > 0 else 0
        for name, wires in {
            "dir_wires": self.dir_wires,
            "portL_wires": self.portL_wires,
            "portR_wires": self.portR_wires,
        }.items():
            if len(wires) != expected_nodes:
                raise ValueError(f"{name} must have length {expected_nodes}.")

        if mode not in {"quantum", "hybrid"}:
            raise ValueError("mode must be 'quantum' or 'hybrid'.")
        self.mode = mode
        self.select_value = select_value
        self.qram_value = qram_value
        if mode == "hybrid" and (select_value is None and qram_value is None):
            raise ValueError("hybrid mode requires select_value and/or qram_value.")
        if select_value is not None and not (0 <= select_value < (1 << self.k)):
            raise ValueError("select_value out of range.")
        if qram_value is not None and not (0 <= qram_value < (1 << self.n_k)):
            raise ValueError("qram_value out of range.")

        all_wires = (
            self.select_wires
            + self.qram_wires
            + self.target_wires
            + self.bus_wire
            + self.dir_wires
            + self.portL_wires
            + self.portR_wires
        )
        super().__init__(wires=all_wires, id=id)

    # ---------- Tree helpers ----------
    def _node_in_wire(self, level: int, prefix: int):
        """The input wire of node (level, prefix): root input is `bus`, else parent's L/R port."""
        if level == 0:
            return self.bus_wire[0]
        parent = _node_index(level - 1, prefix >> 1)
        return self.portL_wires[parent] if (prefix % 2 == 0) else self.portR_wires[parent]

    def _router(self, level: int, prefix: int):
        return self.dir_wires[_node_index(level, prefix)]

    def _portL(self, level: int, prefix: int):
        return self.portL_wires[_node_index(level, prefix)]

    def _portR(self, level: int, prefix: int):
        return self.portR_wires[_node_index(level, prefix)]

    def _path_ctrls(self, i_low: int):
        """(controls, values) for the router path to leaf `i_low` (MSB-first across n_k)."""
        ctrls, vals = [], []
        for k in range(self.n_k):
            prefix = i_low >> (self.n_k - k)
            ctrls.append(self._router(k, prefix))
            vals.append((i_low >> (self.n_k - 1 - k)) & 1)
        return ctrls, vals

    # ---------- Address Loading via CSWAP routing ----------
    def _mark_routers_via_bus(self) -> list:
        """Write low-order address bits into router directions **layer-by-layer** via the bus.

        For each low bit a_k (k = 0..n_k-1):
          1) SWAP(qram_wires[k], bus)
          2) Route bus down k levels (CSWAPs controlled by routers at levels < k)
          3) At node (k, path-prefix), SWAP(bus, dir[k, path-prefix])
        """
        ops = []
        for k in range(self.n_k):
            # 1) load a_k into the bus
            ops.append(qml.SWAP(wires=[self.qram_wires[k], self.bus_wire[0]]))
            # 2) route down k levels
            ops += self._route_bus_down_first_k_levels(k)
            # 3) deposit at level-k node on the active path
            if k == 0:
                ops.append(qml.SWAP(wires=[self.bus_wire[0], self._router(0, 0)]))
            else:
                for p in range(1 << k):
                    #change to  in_wire later
                    parent = _node_index(k - 1, p >> 1)
                    if p % 2 == 0:
                        ops.append(qml.SWAP(wires =[self.portL_wires[parent], self._router(k, p)]))
                    else:
                        ops.append(qml.SWAP(wires =[self.portR_wires[parent], self._router(k, p)]))
        return ops
    
    def _unmark_routers_via_bus(self) -> list:
        return list(reversed(self._mark_routers_via_bus()))
  

    def _route_bus_down_first_k_levels(self, k_levels: int) -> list:
        """Route the bus down the first `k_levels` of the tree using dir-controlled CSWAPs."""
        ops = []
        for ell in range(k_levels):
            for p in range(1 << ell):
                in_w = self._node_in_wire(ell, p)
                L = self._portL(ell, p)
                R = self._portR(ell, p)
                d = self._router(ell, p)
                # dir==1 ⇒ SWAP(in, R)
                op0 = qml.CSWAP(wires=[d, in_w, R])
                ops.append(op0)
                # dir==0 ⇒ SWAP(in, L)
                op = qml.SWAP(wires=[in_w, L])
                ops.append(qml.ctrl(op, control=[d], control_values=[0]))
        return ops

    def _route_bus_up_first_k_levels(self, k_levels: int) -> list:
        """Inverse of `_route_bus_down_first_k_levels`."""
        return list(reversed(self._route_bus_down_first_k_levels(k_levels)))

    # # ---------- Data routing (full depth) ----------
    # def _route_bus_down(self) -> list:
    #     """Route the bus from root to leaf across all levels using dir-controlled CSWAPs."""
    #     ops = []
    #     for k in range(self.n_k):
    #         for p in range(1 << k):
    #             in_w = self._node_in_wire(k, p)
    #             L = self._portL(k, p)
    #             R = self._portR(k, p)
    #             d = self._router(k, p)
    #             if k == 0:
    #                 upper_ctrls, upper_vals = [], []
    #             else:
    #                 upper_ctrls = [self._router(j, p >> (k - j)) for j in range(k)]
    #                 upper_vals = [(p >> (k - 1 - j)) & 1 for j in range(k)]
    #             op0 = qml.SWAP(wires=[in_w, L])
    #             op1 = qml.SWAP(wires=[in_w, R])
    #             ops.append(
    #                 qml.ctrl(op0, control=[d] + upper_ctrls, control_values=[0] + upper_vals)
    #                 if upper_ctrls
    #                 else qml.ctrl(op0, control=[d], control_values=[0])
    #             )
    #             ops.append(
    #                 qml.ctrl(op1, control=[d] + upper_ctrls, control_values=[1] + upper_vals)
    #                 if upper_ctrls
    #                 else qml.ctrl(op1, control=[d], control_values=[1])
    #             )
    #     return ops

    # def _route_bus_up(self) -> list:
    #     """Inverse of `_route_bus_down`."""
    #     return list(reversed(self._route_bus_down()))

    # # ---------- Select controls----------
    # def _select_ctrls(self, s: int):
    #     if self.k == 0:
    #         return [], []
    #     ctrls = list(self.select_wires)
    #     vals = [(s >> (self.k - 1 - j)) & 1 for j in range(self.k)]
    #     return ctrls, vals

    # ---------- classical data input----------
    def _leaf_ops_for_bit(self, j: int) -> list:
        """Apply the leaf write for target bit index j."""
        ops = []
        for p in range(1 << self.n_k):
            if p % 2 == 0:
                target = self._portL(self.n_k - 1, p >> 1)
            else:
                target = self._portR(self.n_k - 1, p >> 1)
            bit = self.bitstrings[p][j] 
            if bit == "1":
                ops.append(qml.PauliZ(wires = target))
            elif bit == "0":
                pass
        return ops

    # ---------- Decompositions ----------
    def _decomp_quantum(self) -> list:
        ops = []
        # 1) address loading
        ops += self._mark_routers_via_bus()
        # 2) For each target bit: load→route down→leaf op→route up→restore (reuse the route bus function)
        for j, tw in enumerate(self.target_wires):
            ops.append(qml.Hadamard(wires=[tw]))
            ops.append(qml.SWAP(wires=[tw, self.bus_wire[0]]))
            ops += self._route_bus_down_first_k_levels(len(self.qram_wires))
            ops += self._leaf_ops_for_bit(j)
            ops += self._route_bus_up_first_k_levels(len(self.qram_wires))
            ops.append(qml.SWAP(wires=[tw, self.bus_wire[0]]))
            ops.append(qml.Hadamard(wires=[tw]))
        # 3) address unloading
        ops += self._unmark_routers_via_bus()
        return ops
    
    #Not work yet
    def _decomp_hybrid(self) -> list:
        ops = []
        # If LSBs are quantum, loadrouters; else skip loading & routing for LSBs
        if self.qram_value is None:
            ops += self._mark_routers_via_bus()
        for j, tw in enumerate(self.target_wires):
            ops.append(qml.SWAP(wires=[tw, self.bus_wire[0]]))
            if self.qram_value is None:
                ops += self._route_bus_down_first_k_levels(len(self.qram_wires))
            ops += self._leaf_ops_for_bit(j)
            if self.qram_value is None:
                ops += self._route_bus_up()
            ops.append(qml.SWAP(wires=[tw, self.bus_wire[0]]))
        return ops

    #not work yet
    def _decomp_select_only(self) -> list:
        # Degenerate case: n_k == 0, no routers; only select-controlled flips on the bus.
        ops = []
        for j, tw in enumerate(self.target_wires):
            ops.append(qml.SWAP(wires=[tw, self.bus_wire[0]]))
            s_range = [self.select_value] if self.select_value is not None else range(1 << self.k)
            for s in s_range:
                if self.bitstrings[s][j] != "1":
                    continue
                sel_ctrls, sel_vals = (self._select_ctrls(s) if self.select_value is None else ([], []))
                op = qml.PauliX(wires=self.bus_wire[0])
                ops.append(qml.ctrl(op, control=sel_ctrls, control_values=sel_vals) if sel_ctrls else op)
            ops.append(qml.SWAP(wires=[tw, self.bus_wire[0]]))
            return ops

    def decomposition(self) -> List[qml.operation.Operator]:
        if self.n_k == 0:
            return self._decomp_select_only()
        if self.mode == "hybrid":
            return self._decomp_hybrid()
        return self._decomp_quantum()


# Functional wrapper
def select_bucket_brigade_bus_qram(
    bitstrings: Sequence[str],
    select_wires: Sequence[int],
    qram_wires: Sequence[int],
    target_wires: Sequence[int],
    bus_wire: int,
    dir_wires: Sequence[int],
    portL_wires: Sequence[int],
    portR_wires: Sequence[int],
    *,
    mode: str = "quantum",
    select_value: Optional[int] = None,
    qram_value: Optional[int] = None,
):
    """Functional wrapper for SelectBucketBrigadeBusQRAM."""
    return SelectBucketBrigadeBusQRAM(
        bitstrings,
        select_wires=select_wires,
        qram_wires=qram_wires,
        target_wires=target_wires,
        bus_wire=bus_wire,
        dir_wires=dir_wires,
        portL_wires=portL_wires,
        portR_wires=portR_wires,
        mode=mode,
        select_value=select_value,
        qram_value=qram_value,
    )

# -----------------------------
# TODOs / Extensions
# -----------------------------
# - Use two qubits per router (|wait>,|left>,|right| encoding) if strict qutrit emulation is needed.
# - support for select/hybrid qram
# - fat-tree architecture implementation.
# - resource analysis.
# - compatibility with PennyLane qrom implementation.

