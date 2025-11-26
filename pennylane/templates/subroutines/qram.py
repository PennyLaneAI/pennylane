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
- Bucket-brigade QRAM LSBs (``qram_wires``) using 3-qubits-per-node (dir, portL, portR)

Address loading is performed **layer-by-layer** by routing a single top **bus** qubit
down to the active node using CSWAPs controlled by already-written upper routers,
depositing each low-order address bit into the node's direction qubit.

Data phase routes the target qubits down to the selected leaf for each target bit,
performs the leaf write (classical bit flip), then routes back and restores the target.
"""
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Sequence

from pennylane.decomposition import (
    add_decomps,
    controlled_resource_rep,
    register_resources,
    resource_rep,
)
from pennylane.operation import Operation, Operator
from pennylane.ops import CSWAP, SWAP, Hadamard, PauliX, PauliZ, ctrl
from pennylane.wires import Wires, WiresLike


# pylint: disable=consider-using-generator


# -----------------------------
# Wires Data Structure
# -----------------------------
@dataclass
class _QRAMWires:

    qram_wires: Sequence[Wires]
    target_wires: Sequence[Wires]
    bus_wire: Sequence[Wires]
    dir_wires: Sequence[Wires]
    portL_wires: Sequence[Wires]
    portR_wires: Sequence[Wires]

    # ---------- Tree helpers ----------
    def node_in_wire(self, level: int, prefix: int):
        """The input wire of node (level, prefix): root input is `bus`, else parent's L/R port."""
        if level == 0:
            return self.bus_wire[0]
        parent = _node_index(level - 1, prefix >> 1)
        return self.portL_wires[parent] if (prefix % 2 == 0) else self.portR_wires[parent]

    def router(self, level: int, prefix: int):
        """Helps with fetching the routing qubits of a node."""
        return self.dir_wires[_node_index(level, prefix)]

    def portL(self, level: int, prefix: int):
        """Helps with fetching the left port qubit of a node."""
        return self.portL_wires[_node_index(level, prefix)]

    def portR(self, level: int, prefix: int):
        """Helps with fetching the right port qubit of a node."""
        return self.portR_wires[_node_index(level, prefix)]


# -----------------------------
# Utilities
# -----------------------------
def _level_offset(level: int) -> int:
    """Index offset of the first node at a given level (root=0). Offset = 2^level - 1."""
    return (1 << level) - 1


def _node_index(level: int, prefix_value: int) -> int:
    """Return the flat index (level order) of the internal node at `level` with prefix `prefix_value`."""
    return _level_offset(level) + prefix_value


# -----------------------------
# Select-prefix × Bucket-Brigade with explicit bus routing
# -----------------------------
class BBQRAM(Operation):  # pylint: disable=too-many-instance-attributes
    r"""Bucket-brigade QRAM (https://arxiv.org/pdf/0708.1879) with **explicit bus routing** using 3 qubits per node.

    Bucket-brigade QRAM achieves an O(log N) complexity instead of the typical N, where N is the number of
    memory cells addressed. It does this by reducing the number of nodes that need to be visited in a tree
    which converts our binary address into a unary address at the leaves. The approach is simply to keep track
    of the active path as it is constructed by loading the address one bit at a time into a location in the next layer
    of the tree based on the previous address bit.

    In this implementation, each node is composed of three qubits: one direction bit ``dir[k,p]`` which stores the routed
    low-order address bit for level k, and one bit for each child of the node ``portL[k,p]`` and ``portR[k,p]`` that are
    used for loading the next layers' bits.

    The algorithm is composed of five steps:

        1) load
        2) route down
        3) leaf op
        4) route up
        5) restore

    The address is first loaded layer-by-layer via CSWAPs, depositing each address bit into the `dir[k,p]`.
    Data routing is performed per-target. The target is swapped with the bus, routed down, the leaf write operation is
    performed to correlate the data with the qubit at the leaf of the tree, routing is then done in reverse and we swap
    back.

    In the end, the target wires' values correspond to the data at the address specified.

    Args:
        bitstrings (Sequence[str]): the classical data as a sequence of bitstrings
        qram_wires (WiresLike): stores the index for the entry of the classical data we want to access
        target_wires (WiresLike): where the classical data gets loaded
        work_wires (WiresLike): the bus, direction, left port and right port wires in that order. Each node in the
            tree contains one address (direction), one left port and one right port wire. The single bus wire is used
            for address loading and data routing

    Raises:
        ValueError: if the bitstrings are not provided, the bitstrings are of the wrong length, the target wires are
            of the wrong length or if there is not one direction wire, one left port wire and one right port wire per node

    **Example:**

    .. code-block:: python

        from pennylane.measurements import probs
        from pennylane.templates import BasisEmbedding
        from pennylane import device, qnode
        from pennylane.templates.subroutines.qram import BBQRAM

        bitstrings = ["010", "111", "110", "000"]  # 2^2 entries, m=3
        dev = device("default.qubit")

        @qnode(dev)
        def bb_quantum():
            # qram_wires are the 2 LSB address bits.
            qram_wires = [0, 1]  # |i> for 4 leaves
            target_wires = [2, 3, 4]  # m=3
            bus = 5  # single bus at the top

            # For n_k=2 → (2^2 - 1) = 3 internal nodes in level order:
            # (0,0) root; (1,0) left child; (1,1) right child
            dir_wires = [6, 7, 8]
            portL_wires = [9, 10, 11]
            portR_wires = [12, 13, 14]

            # prepare an address, e.g., |10> (index 2)
            BasisEmbedding(2, wires=qram_wires)

            BBQRAM(
                bitstrings,
                qram_wires=qram_wires,  # n_k=2
                target_wires=target_wires,
                work_wires=[bus] + dir_wires + portL_wires + portR_wires,
            )
            return probs(wires=target_wires)

    >>> print(bb_quantum())  # doctest: +SKIP
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    """

    grad_method = None

    resource_keys = {"bitstrings", "num_target_wires", "num_qram_wires", "n_k"}

    @property
    def resource_params(self) -> dict:
        wire_manager = self.hyperparameters["wire_manager"]
        return {
            "bitstrings": self.hyperparameters["bitstrings"],
            "num_target_wires": len(wire_manager.target_wires),
            "num_qram_wires": len(wire_manager.qram_wires),
            "n_k": self.hyperparameters["n_k"],
        }

    def __init__(
        self,
        bitstrings: Sequence[str],
        qram_wires: WiresLike,
        target_wires: WiresLike,
        work_wires: WiresLike,
        id: str | None = None,
    ):  # pylint: disable=too-many-arguments
        if not bitstrings:
            raise ValueError("'bitstrings' cannot be empty.")
        m_set = {len(s) for s in bitstrings}
        if len(m_set) != 1:
            raise ValueError("All bitstrings must have equal length.")
        m = next(iter(m_set))
        bitstrings = list(bitstrings)

        qram_wires = Wires(qram_wires)

        n_k = len(qram_wires)
        if (1 << n_k) != len(bitstrings):
            raise ValueError("len(bitstrings) must be 2^(len(qram_wires)).")

        target_wires = Wires(target_wires)
        if m != len(target_wires):
            raise ValueError("len(target_wires) must equal bitstring length.")

        bus_wire = Wires(work_wires[0])
        divider = len(work_wires[1:]) // 3
        dir_wires = Wires(work_wires[1 : 1 + divider])
        portL_wires = Wires(work_wires[1 + divider : 1 + divider * 2])
        portR_wires = Wires(work_wires[1 + divider * 2 : 1 + divider * 3])

        expected_nodes = (1 << n_k) - 1 if n_k > 0 else 0

        if len(work_wires) != 1 + 3 * expected_nodes:
            raise ValueError(f"work_wires must have length {1 + 3 * expected_nodes}.")

        all_wires = (
            list(qram_wires)
            + list(target_wires)
            + list(bus_wire)
            + list(dir_wires)
            + list(portL_wires)
            + list(portR_wires)
        )

        wire_manager = _QRAMWires(
            qram_wires, target_wires, bus_wire, dir_wires, portL_wires, portR_wires
        )

        self._hyperparameters = {
            "wire_manager": wire_manager,
            "n_k": n_k,
            "bitstrings": bitstrings,
        }

        super().__init__(wires=all_wires, id=id)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    def _path_ctrls(self, i_low: int):
        """(controls, values) for the router path to leaf `i_low` (MSB-first across n_k)."""
        ctrls, vals = [], []
        wire_manager = self.hyperparameters["wire_manager"]
        n_k = self.hyperparameters["n_k"]
        for k in range(n_k):
            prefix = i_low >> (n_k - k)
            ctrls.append(wire_manager.router(k, prefix))
            vals.append((i_low >> (n_k - 1 - k)) & 1)
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
        wire_manager = self.hyperparameters["wire_manager"]
        for k in range(self.hyperparameters["n_k"]):
            # 1) load a_k into the bus
            origin = wire_manager.qram_wires[k]
            target = wire_manager.bus_wire[0]
            ops.append(SWAP(wires=[origin, target]))
            # 2) route down k levels
            ops += self._route_bus_down_first_k_levels(k)
            # 3) deposit at level-k node on the active path
            if k == 0:
                ops.append(SWAP(wires=[wire_manager.bus_wire[0], wire_manager.router(0, 0)]))
            else:
                for p in range(1 << k):
                    # change to  in_wire later
                    parent = _node_index(k - 1, p >> 1)
                    if p % 2 == 0:
                        origin = wire_manager.portL_wires[parent]
                        target = wire_manager.router(k, p)
                        ops.append(SWAP(wires=[origin, target]))
                    else:
                        origin = wire_manager.portR_wires[parent]
                        target = wire_manager.router(k, p)
                        ops.append(SWAP(wires=[origin, target]))
        return ops

    def _unmark_routers_via_bus(self) -> list:
        return list(reversed(self._mark_routers_via_bus()))

    def _route_bus_down_first_k_levels(self, k_levels: int) -> list:
        """Route the bus down the first `k_levels` of the tree using dir-controlled CSWAPs."""
        ops = []
        wire_manager = self.hyperparameters["wire_manager"]
        for ell in range(k_levels):
            for p in range(1 << ell):
                in_w = wire_manager.node_in_wire(ell, p)
                L = wire_manager.portL(ell, p)
                R = wire_manager.portR(ell, p)
                d = wire_manager.router(ell, p)
                # dir==1 ⇒ SWAP(in, R)
                op0 = CSWAP(wires=[d, in_w, R])
                ops.append(op0)
                # dir==0 ⇒ SWAP(in, L)
                op = SWAP(wires=[in_w, L])
                ops.append(ctrl(op, control=[d], control_values=[0]))
        return ops

    def _route_bus_up_first_k_levels(self, k_levels: int) -> list:
        """Inverse of `_route_bus_down_first_k_levels`."""
        return list(reversed(self._route_bus_down_first_k_levels(k_levels)))

    # ---------- classical data input----------
    def _leaf_ops_for_bit(self, j: int) -> list:
        """Apply the leaf write for target bit index j."""
        ops = []
        wire_manager = self.hyperparameters["wire_manager"]
        n_k = self.hyperparameters["n_k"]
        for p in range(1 << n_k):
            if p % 2 == 0:
                target = wire_manager.portL(n_k - 1, p >> 1)
            else:
                target = wire_manager.portR(n_k - 1, p >> 1)
            bit = self.hyperparameters["bitstrings"][p][j]
            if bit == "1":
                ops.append(PauliZ(wires=target))
            elif bit == "0":
                pass
        return ops

    # ---------- Decomposition ----------
    def decomposition(self) -> List[Operator]:
        ops = []
        wire_manager = self.hyperparameters["wire_manager"]
        bus_wire = wire_manager.bus_wire
        qram_wires = wire_manager.qram_wires
        # 1) address loading
        ops += self._mark_routers_via_bus()
        # 2) For each target bit: load→route down→leaf op→route up→restore (reuse the route bus function)
        for j, tw in enumerate(wire_manager.target_wires):
            ops.append(Hadamard(wires=[tw]))
            ops.append(SWAP(wires=[tw, bus_wire[0]]))
            ops += self._route_bus_down_first_k_levels(len(qram_wires))
            ops += self._leaf_ops_for_bit(j)
            ops += self._route_bus_up_first_k_levels(len(qram_wires))
            ops.append(SWAP(wires=[tw, bus_wire[0]]))
            ops.append(Hadamard(wires=[tw]))
        # 3) address unloading
        ops += self._unmark_routers_via_bus()
        return ops


def _bucket_brigade_qram_resources(bitstrings, num_target_wires, num_qram_wires, n_k):
    resources = defaultdict(int)
    resources[resource_rep(SWAP)] = (
        sum([1 if k == 0 else (1 << k) for k in range(n_k)]) + n_k
    ) * 2 + num_target_wires * 2
    resources[resource_rep(CSWAP)] = sum(
        [(1 << ell) for ell in range(num_qram_wires)]
    ) * num_target_wires * 2 + (sum([(1 << ell) for k in range(n_k) for ell in range(k)]) * 2)
    resources[
        controlled_resource_rep(
            base_class=SWAP, base_params={}, num_control_wires=1, num_zero_control_values=1
        )
    ] = sum([(1 << ell) for ell in range(num_qram_wires)]) * num_target_wires * 2 + (
        sum([(1 << ell) for k in range(n_k) for ell in range(k)]) * 2
    )
    resources[resource_rep(Hadamard)] += num_target_wires * 2
    for j in range(num_target_wires):
        for p in range(1 << n_k):
            resources[resource_rep(PauliZ)] += 1 if int(bitstrings[p][j]) else 0
    return resources


def _mark_routers_via_bus_qfunc(wire_manager, n_k):
    """Write low-order address bits into router directions **layer-by-layer** via the bus.

    For each low bit a_k (k = 0..n_k-1):
      1) SWAP(qram_wires[k], bus)
      2) Route bus down k levels (CSWAPs controlled by routers at levels < k)
      3) At node (k, path-prefix), SWAP(bus, dir[k, path-prefix])
    """
    for k in range(n_k):
        # 1) load a_k into the bus
        origin = wire_manager.qram_wires[k]
        target = wire_manager.bus_wire[0]
        SWAP(wires=[origin, target])
        # 2) route down k levels
        _route_bus_down_first_k_levels_qfunc(wire_manager, k)
        # 3) deposit at level-k node on the active path
        if k == 0:
            SWAP(wires=[wire_manager.bus_wire[0], wire_manager.router(0, 0)])
        else:
            for p in range(1 << k):
                # change to  in_wire later
                parent = _node_index(k - 1, p >> 1)
                if p % 2 == 0:
                    origin = wire_manager.portL_wires[parent]
                    target = wire_manager.router(k, p)
                    SWAP(wires=[origin, target])
                else:
                    origin = wire_manager.portR_wires[parent]
                    target = wire_manager.router(k, p)
                    SWAP(wires=[origin, target])


def _unmark_routers_via_bus_qfunc(wire_manager, n_k):
    """
    Operations used to write low-order address bits into router directions **layer-by-layer** via the bus, reversed.
    """
    for k in range(n_k - 1, -1, -1):
        # 1) level-k node on the active path
        if k == 0:
            SWAP(wires=[wire_manager.bus_wire[0], wire_manager.router(0, 0)])
        else:
            for p in range(1 << k - 1, -1, -1):
                # change to  in_wire later
                parent = _node_index(k - 1, p >> 1)
                if p % 2 == 0:
                    origin = wire_manager.portL_wires[parent]
                    target = wire_manager.router(k, p)
                    SWAP(wires=[origin, target])
                else:
                    origin = wire_manager.portR_wires[parent]
                    target = wire_manager.router(k, p)
                    SWAP(wires=[origin, target])
        # 2) route up k levels
        _route_bus_up_first_k_levels_qfunc(wire_manager, k)
        # 3) reverse load
        origin = wire_manager.qram_wires[k]
        target = wire_manager.bus_wire[0]
        SWAP(wires=[origin, target])


def _route_bus_down_first_k_levels_qfunc(wire_manager, k_levels):
    """Route the bus down the first `k_levels` of the tree using dir-controlled CSWAPs."""
    for ell in range(k_levels):
        for p in range(1 << ell):
            in_w = wire_manager.node_in_wire(ell, p)
            L = wire_manager.portL(ell, p)
            R = wire_manager.portR(ell, p)
            d = wire_manager.router(ell, p)
            # dir==1 ⇒ SWAP(in, R)
            CSWAP(wires=[d, in_w, R])
            # dir==0 ⇒ SWAP(in, L)
            ctrl(SWAP(wires=[in_w, L]), control=[d], control_values=[0])


def _route_bus_up_first_k_levels_qfunc(wire_manager, k_levels):
    """Route the bus up the first `k_levels` of the tree using dir-controlled CSWAPs."""
    for ell in range(k_levels - 1, -1, -1):
        for p in range((1 << ell) - 1, -1, -1):
            in_w = wire_manager.node_in_wire(ell, p)
            L = wire_manager.portL(ell, p)
            R = wire_manager.portR(ell, p)
            d = wire_manager.router(ell, p)
            # dir==0 ⇒ SWAP(in, L)
            ctrl(SWAP(wires=[in_w, L]), control=[d], control_values=[0])
            # dir==1 ⇒ SWAP(in, R)
            CSWAP(wires=[d, in_w, R])


def _leaf_ops_for_bit_qfunc(wire_manager, bitstrings, n_k, j):
    """Apply the leaf write for target bit index j."""
    ops = []
    for p in range(1 << n_k):
        if p % 2 == 0:
            target = wire_manager.portL(n_k - 1, p >> 1)
        else:
            target = wire_manager.portR(n_k - 1, p >> 1)
        bit = bitstrings[p][j]
        if bit == "1":
            PauliZ(wires=target)
        elif bit == "0":
            pass
    return ops


@register_resources(_bucket_brigade_qram_resources)
def _bucket_brigade_qram_decomposition(
    wires, wire_manager, bitstrings, n_k
):  # pylint: disable=unused-argument
    bus_wire = wire_manager.bus_wire
    qram_wires = wire_manager.qram_wires
    # 1) address loading
    _mark_routers_via_bus_qfunc(wire_manager, n_k)
    # 2) For each target bit: load→route down→leaf op→route up→restore (reuse the route bus function)
    for j, tw in enumerate(wire_manager.target_wires):
        Hadamard(wires=[tw])
        SWAP(wires=[tw, bus_wire[0]])
        _route_bus_down_first_k_levels_qfunc(wire_manager, len(qram_wires))
        _leaf_ops_for_bit_qfunc(wire_manager, bitstrings, n_k, j)
        _route_bus_up_first_k_levels_qfunc(wire_manager, len(qram_wires))
        SWAP(wires=[tw, bus_wire[0]])
        Hadamard(wires=[tw])
    # 3) address unloading
    _unmark_routers_via_bus_qfunc(wire_manager, n_k)


add_decomps(BBQRAM, _bucket_brigade_qram_decomposition)

class HybridQRAM(Operation):
    r"""Hybrid QRAM combining select-only and bucket-brigade behavior.

    This implements a space–time tradeoff:

    1.Total address bits: n = len(qram_wires)
    2.Choose an integer k with 0 ≤ k < n.
      2.1 The first k address bits (high-order) are "select" bits.
      2.2 The remaining n-k bits (low-order) are routed through a bucket-brigade tree.

    Instead of a full-depth tree of size 2^n leaves, we build a smaller tree of depth n-k
    (2^(n-k) leaves) and reuse it 2^k times:

        For each prefix s \in {0, …, 2^k - 1}:
            1. Multi-controlled-X on a "signal" ancilla, controlled by the k select bits
               being equal to s.
            2. Conditioned on signal==1, perform a BBQRAM query using only the lower
               n-k address bits and the sub-table of bitstrings whose prefix is s.
            3. Uncompute the signal with the same multi-controlled-X.

    In the end, for any full address a = (prefix, suffix), the target wires are loaded with
    bitstrings[a].

    Wire layout:

      qram_wires: [ sel_0, ..., sel_{k-1}, tree_0, ..., tree_{n-k-1} ]
      work_wires: [ signal, bus, dir..., portL..., portR... ]  (tree ancillas)

    Args:
        bitstrings (Sequence[str]): classical data table; must have length 2^n where n = len(qram_wires)
        qram_wires (WiresLike): full address register (length n)
        target_wires (WiresLike): m target qubits; m must equal bitstring length
        work_wires (WiresLike): ancillas: [signal, bus, dir..., portL..., portR...] for a tree of depth (n-k)
        k (int): number of "select" bits taken from the MSB of qram_wires
    """

    grad_method = None

    def __init__(
        self,
        bitstrings: Sequence[str],
        qram_wires: WiresLike,
        target_wires: WiresLike,
        work_wires: WiresLike,
        k: int, #define the select part size, remaining part is tree part
        id: str | None = None,
    ):
        
        if not bitstrings:
            raise ValueError("'bitstrings' cannot be empty.")
        m_set = {len(s) for s in bitstrings}
        if len(m_set) != 1:
            raise ValueError("All bitstrings must have equal length.")
        m = next(iter(m_set))
        bitstrings = list(bitstrings)

        qram_wires = Wires(qram_wires)
        target_wires = Wires(target_wires)
        work_wires = Wires(work_wires)

        #test wires
        n_total = len(qram_wires)
        if n_total == 0:
            raise ValueError("len(qram_wires) must be > 0.")

        if not (0 <= k < n_total):
            raise ValueError("k must satisfy 0 <= k < len(qram_wires).")

        if len(target_wires) != m:
            raise ValueError("len(target_wires) must equal bitstring length.")

        if len(bitstrings) != (1 << n_total):
            raise ValueError("len(bitstrings) must be 2^(len(qram_wires)).")

        # Split qram_wires into select and tree parts
        select_wires = Wires(qram_wires[:k])
        tree_qram_wires = Wires(qram_wires[k:])
        n_tree = len(tree_qram_wires)


        # work_wires = [ signal, bus, dir..., portL..., portR... ] for tree depth n_tree
        signal_wire = Wires(work_wires[0])

        if n_tree > 0:
            expected_nodes = (1 << n_tree) - 1
            expected_len = 1 + 1 + 3 * expected_nodes  # signal + bus + 3 per node
            if len(work_wires) != expected_len:
                raise ValueError(
                    f"work_wires must have length {expected_len} "
                    f"for k={k} and len(qram_wires)={n_total}."
                )

            bus_wire = Wires(work_wires[1])
            divider = len(work_wires[2:]) // 3
            dir_wires = Wires(work_wires[2 : 2 + divider])
            portL_wires = Wires(work_wires[2 + divider : 2 + 2 * divider])
            portR_wires = Wires(work_wires[2 + 2 * divider : 2 + 3 * divider])
        else:
            # k = n_total-1 ensures n_tree >= 1, so we should never hit this if k < len(qram_wires), this is actually select-only qram
            bus_wire = Wires([])
            dir_wires = Wires([])
            portL_wires = Wires([])
            portR_wires = Wires([])

        tree_wire_manager = _QRAMWires(
            tree_qram_wires, target_wires, bus_wire, dir_wires, portL_wires, portR_wires
        )

        all_wires = list(qram_wires) + list(target_wires) + list(work_wires)

        Operation.__init__(self, wires=all_wires, id=id)

        self._hyperparameters = {
            "bitstrings": bitstrings,
            "qram_wires": qram_wires,
            "select_wires": select_wires,
            "tree_qram_wires": tree_qram_wires,
            "target_wires": target_wires,
            "signal_wire": signal_wire,
            "tree_wire_manager": tree_wire_manager,
            "k": k,
            "n_total": n_total,
            "n_tree": n_tree,
            "m": m,
        }

    # ---------- Helpers ----------
    @staticmethod
    def _bits(value: int, length: int) -> list[int]:
        """Return `length` bits of `value` (MSB first)."""
        return [(value >> (length - 1 - i)) & 1 for i in range(length)]

    # Tree helpers with signal control
    def _tree_mark_routers_via_bus_ctrl(self) -> List[Operator]:
        """Address loading for the tree (n_tree bits), controlled on signal."""
        ops: List[Operator] = []
        wm = self.hyperparameters["tree_wire_manager"]
        n_tree = self.hyperparameters["n_tree"]
        signal = self.hyperparameters["signal_wire"][0]

        for level in range(n_tree):
            # SWAP(tree_qram_wires[level], bus) controlled on signal
            origin = wm.qram_wires[level]
            target = wm.bus_wire[0]
            base_swap = SWAP(wires=[origin, target])
            ops.append(ctrl(base_swap, control=[signal], control_values=[1]))

            # route down qram wires for current levels
            ops += self._tree_route_bus_down_first_k_levels_ctrl(level)

            # deposit into dir[level, *] along active path
            if level == 0:
                base_swap = SWAP(wires=[wm.bus_wire[0], wm.router(0, 0)])
                ops.append(ctrl(base_swap, control=[signal], control_values=[1]))
            else:
                for p in range(1 << level):
                    parent = _node_index(level - 1, p >> 1)
                    if p % 2 == 0:
                        origin = wm.portL_wires[parent]
                    else:
                        origin = wm.portR_wires[parent]
                    target = wm.router(level, p)
                    base_swap = SWAP(wires=[origin, target])
                    ops.append(ctrl(base_swap, control=[signal], control_values=[1]))

        return ops

    def _tree_unmark_routers_via_bus_ctrl(self) -> List[Operator]:
        """Inverse of `_tree_mark_routers_via_bus_ctrl`."""
        return list(reversed(self._tree_mark_routers_via_bus_ctrl()))

    def _tree_route_bus_down_first_k_levels_ctrl(self, k_levels: int) -> List[Operator]:
        """Tree routing down for first `k_levels` levels, controlled on signal."""
        ops: List[Operator] = []
        wm = self.hyperparameters["tree_wire_manager"]
        signal = self.hyperparameters["signal_wire"][0]

        for ell in range(k_levels):
            for p in range(1 << ell):
                in_w = wm.node_in_wire(ell, p)
                L = wm.portL(ell, p)
                R = wm.portR(ell, p)
                d = wm.router(ell, p)

                # dir==1: CSWAP(d, in_w, R) — additionally controlled on signal
                base_cswap = CSWAP(wires=[d, in_w, R])
                ops.append(ctrl(base_cswap, control=[signal], control_values=[1]))

                # dir==0: SWAP(in_w, L) controlled on (d == 0) and signal == 1
                base_swap = SWAP(wires=[in_w, L])
                op = ctrl(base_swap, control=[d], control_values=[0])
                ops.append(ctrl(op, control=[signal], control_values=[1]))

        return ops

    def _tree_route_bus_up_first_k_levels_ctrl(self, k_levels: int) -> List[Operator]:
        """Inverse of `_tree_route_bus_down_first_k_levels_ctrl`."""
        return list(reversed(self._tree_route_bus_down_first_k_levels_ctrl(k_levels)))

    def _tree_leaf_ops_for_bit_block_ctrl(self, j: int, block_index: int) -> List[Operator]:
        """Leaf write for target bit j, for a given select prefix block, controlled on signal."""
        ops: List[Operator] = []
        wm = self.hyperparameters["tree_wire_manager"]
        n_tree = self.hyperparameters["n_tree"]
        bitstrings = self.hyperparameters["bitstrings"]
        signal = self.hyperparameters["signal_wire"][0]

        # For each leaf index p of the tree (n_tree bits)
        for p in range(1 << n_tree):
            # physical leaf wire (same pattern as BBQRAM)
            if p % 2 == 0:
                target = wm.portL(n_tree - 1, p >> 1)
            else:
                target = wm.portR(n_tree - 1, p >> 1)

            # Global address index: (block_index << n_tree) + p
            addr = (block_index << n_tree) + p
            bit = bitstrings[addr][j]
            if bit == "1":
                base_z = PauliZ(wires=target)
                ops.append(ctrl(base_z, control=[signal], control_values=[1]))

        return ops

    def _block_tree_query_ops(self, block_index: int) -> List[Operator]:
        """One BBQRAM-style query of the (n_tree)-depth tree for a fixed select prefix."""
        ops: List[Operator] = []
        wm = self.hyperparameters["tree_wire_manager"]
        n_tree = self.hyperparameters["n_tree"]
        signal = self.hyperparameters["signal_wire"][0]

        if n_tree == 0:
            # Degenerate case: no tree; nothing to do here
            return ops

        # 1) address loading for the tree (controlled on signal)
        ops += self._tree_mark_routers_via_bus_ctrl()

        # 2) per-target data phase, controlled on signal
        for j, tw in enumerate(wm.target_wires):
            # H on target
            base_h = Hadamard(wires=[tw])
            ops.append(ctrl(base_h, control=[signal], control_values=[1]))

            # Swap target <-> bus
            base_swap1 = SWAP(wires=[tw, wm.bus_wire[0]])
            ops.append(ctrl(base_swap1, control=[signal], control_values=[1]))

            # Route down tree
            ops += self._tree_route_bus_down_first_k_levels_ctrl(n_tree)

            # Leaf Z ops for this block and bit index j
            ops += self._tree_leaf_ops_for_bit_block_ctrl(j, block_index)

            # Route back up
            ops += self._tree_route_bus_up_first_k_levels_ctrl(n_tree)

            # Swap back bus -> target
            base_swap2 = SWAP(wires=[tw, wm.bus_wire[0]])
            ops.append(ctrl(base_swap2, control=[signal], control_values=[1]))

            # Final H on target
            base_h2 = Hadamard(wires=[tw])
            ops.append(ctrl(base_h2, control=[signal], control_values=[1]))

        # 3) address unloading for the tree (controlled on signal)
        ops += self._tree_unmark_routers_via_bus_ctrl()

        return ops

    # decomposition
    def decomposition(self) -> List[Operator]:
        bitstrings = self.hyperparameters["bitstrings"]
        k = self.hyperparameters["k"]
        n_tree = self.hyperparameters["n_tree"]
        m = self.hyperparameters["m"]

        select_wires = list(self.hyperparameters["select_wires"])
        signal = self.hyperparameters["signal_wire"][0]

        if len(bitstrings) != (1 << (k + n_tree)):
            # Should not happen if __init__ checks passed
            raise ValueError("Inconsistent bitstrings length for hybrid QRAM.")

        ops: List[Operator] = []

        num_blocks = 1 << k if k > 0 else 1

        for block_index in range(num_blocks):
            # Multi-controlled X to turn signal on when select bits == block_index
            x_op = PauliX(wires=signal)

            if k > 0:
                sel_pattern = self._bits(block_index, k)
                ops.append(ctrl(x_op, control=select_wires, control_values=sel_pattern))
            else:
                # No select bits: just flip signal for all addresses
                ops.append(x_op)

            # Perform one tree query, driven by lower n_tree bits, controlled on signal
            ops += self._block_tree_query_ops(block_index)

            # Uncompute signal
            if k > 0:
                ops.append(ctrl(x_op, control=select_wires, control_values=sel_pattern))
            else:
                ops.append(x_op)

        return ops
