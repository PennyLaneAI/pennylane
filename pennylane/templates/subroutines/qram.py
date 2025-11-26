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
from functools import reduce
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

        if not self.hyperparameters or not "k" in self.hyperparameters:
            n_k = len(qram_wires)
            if (1 << n_k) != len(bitstrings):
                raise ValueError("len(bitstrings) must be 2^(len(qram_wires)).")
        else:
            n_k = self.hyperparameters["n_k"]

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

        self._hyperparameters.update(
            {"wire_manager": wire_manager, "n_k": n_k, "bitstrings": bitstrings, "m": m}
        )

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

    # ---------- Decompositions ----------
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
    wires, wire_manager, bitstrings, n_k, m
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


class SelectOnlyQRAM(Operator):
    """Select-only QRAM implemented as multi-controlled X on target wires,
    controlled on all address wires (select_wires + qram_wires).

    Args:
        work_wires (WiresLike): ignored, as select-only qram doesn't require any work qubits
        select_wires (WiresLike, optional): actually also not used, but kept for API consistency with hybrid QRAM
        select_value (int or None, optional): if provided, only entries whose select bits match this value are loaded
        id (str or None): optional name for the operation
    """

    grad_method = None

    resource_keys = {
        "bitstrings",
        "select_value",
        "num_qram_wires",
        "num_select_wires",
        "k",
        "m",
        "n_total",
    }

    def __init__(
        self,
        bitstrings,
        qram_wires: WiresLike,
        target_wires: WiresLike,
        id: str | None = None,
    ):
        # Convert to Wires
        qram_wires = Wires(qram_wires)
        target_wires = Wires(target_wires)

        # ---- Validate bitstrings ----
        n_k = len(qram_wires)
        n_total = n_k

        if (1 << n_total) != len(bitstrings):
            raise ValueError("len(bitstrings) must be 2^(len(qram_wires)).")

        self._hyperparameters = {
            "bitstrings": bitstrings,
            "qram_wires": qram_wires,
            "target_wires": target_wires,
            "n_k": n_k,
            "n_total": n_total,
        }

        super().__init__(wires=list(qram_wires) + list(target_wires), id=id)

    # ---------- Helpers ----------
    @staticmethod
    def _address_bits(addr: int, n: int) -> list[int]:
        """Return the n-bit pattern (MSB first) for integer `addr`."""
        return [(addr >> (n - 1 - i)) & 1 for i in range(n)]

    @property
    def resource_params(self) -> dict:
        return {
            "bitstrings": self.hyperparameters["bitstrings"],
            "num_qram_wires": len(self.hyperparameters["qram_wires"]),
            "n_total": self.hyperparameters["n_total"],
        }

    # ---------- Decomposition ----------
    def decomposition(self) -> List[Operator]:
        bitstrings = self.hyperparameters["bitstrings"]
        qram_wires = list(self.hyperparameters["qram_wires"])
        target_wires = list(self.hyperparameters["target_wires"])

        n_total = self.hyperparameters["n_total"]

        # All controls = qram bits (LSBs)
        controls = qram_wires

        ops: List[Operator] = []

        # Loop over all addresses (0 .. 2^(k+n_k)-1)
        for addr, bits in enumerate(bitstrings):
            control_values = self._address_bits(addr, n_total)

            # For each bit position in the data
            for j in range(len(bitstrings[0])):
                if bits[j] != "1":
                    continue

                # Multi-controlled X on target_wires[j],
                # controlled on controls matching `control_values`.
                base_op = PauliX(wires=target_wires[j])
                ops.append(
                    ctrl(
                        base_op,
                        control=controls,
                        control_values=control_values,
                    )
                )
        return ops


def _select_only_qram_resources(bitstrings, num_qram_wires, n_total):
    resources = defaultdict(int)
    num_controls = num_qram_wires

    for addr, bits in enumerate(bitstrings):
        resources[
            controlled_resource_rep(
                base_class=PauliX,
                base_params={},
                num_control_wires=num_controls,
                num_zero_control_values=reduce(
                    lambda acc, nxt: acc + (nxt == 0),
                    [(addr >> (n_total - 1 - i)) & 1 for i in range(n_total)],
                    0,
                ),
            )
        ] += sum([1 if bits[j] == "1" else 0 for j in range(len(bitstrings[0]))])

    return resources


@register_resources(_select_only_qram_resources)
def _select_only_qram_decomposition(
    wires, bitstrings, qram_wires, target_wires, n_total, **_
):  # pylint: disable=unused-argument
    controls = qram_wires

    # Loop over all addresses (0 .. 2^(k+n_k)-1)
    for addr, bits in enumerate(bitstrings):

        control_values = [(addr >> (n_total - 1 - i)) & 1 for i in range(n_total)]

        # For each bit position in the data
        for j in range(len(bitstrings[0])):
            if bits[j] != "1":
                continue

            # Multi-controlled X on target_wires[j],
            # controlled on controls matching `control_values`.
            ctrl(
                PauliX(wires=target_wires[j]),
                control=controls,
                control_values=control_values,
            )


add_decomps(SelectOnlyQRAM, _select_only_qram_decomposition)
