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
from typing import List, Sequence

from dataclasses import dataclass

from pennylane.operation import Operation, Operator
from pennylane.ops import CSWAP, SWAP, Hadamard, PauliZ, ctrl
from pennylane.wires import Wires


# -----------------------------
# Wires Data Structure
# -----------------------------
@dataclass
class _QRAMWires:

    qram_wires: Sequence[Wires]
    target_wires: Sequence[Wires]
    bus_wires: Sequence[Wires]
    dir_wires: Sequence[Wires]
    portL_wires: Sequence[Wires]
    portR_wires: Sequence[Wires]

    def __init__(self, qram_wires, target_wires, bus_wire, dir_wires, portL_wires, portR_wires):
        self.qram_wires = qram_wires
        self.target_wires = target_wires
        self.bus_wire = bus_wire
        self.dir_wires = dir_wires
        self.portL_wires = portL_wires
        self.portR_wires = portR_wires

    # ---------- Tree helpers ----------
    def node_in_wire(self, level: int, prefix: int):
        """The input wire of node (level, prefix): root input is `bus`, else parent's L/R port."""
        if level == 0:
            return self.bus_wire[0]
        parent = _node_index(level - 1, prefix >> 1)
        return (
            self.portL_wires[parent]
            if (prefix % 2 == 0)
            else self.portR_wires[parent]
        )

    def router(self, level: int, prefix: int):
        return self.dir_wires[_node_index(level, prefix)]

    def portL(self, level: int, prefix: int):
        return self.portL_wires[_node_index(level, prefix)]

    def portR(self, level: int, prefix: int):
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
        bitstrings (Sequence[int]): the classical data as a sequence of bitstrings
        qram_wires (Sequence[int]): stores the index for the entry of the classical data we want to access
        target_wires (Sequence[int]): where the classical data gets loaded
        work_wires (Sequence[int]): the bus, direction, left port and right port wires in that order. Each node in the
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

    def __init__(
        self,
        bitstrings: Sequence[str],
        qram_wires: Sequence[int],
        target_wires: Sequence[int],
        work_wires: Sequence[int],
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

        wire_manager = _QRAMWires(qram_wires, target_wires, bus_wire, dir_wires, portL_wires, portR_wires)

        self._hyperparameters = {
            "wire_manager": wire_manager,
            "m": m,
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
            ops.append(
                SWAP(
                    wires=[
                        wire_manager.qram_wires[k],
                        wire_manager.bus_wire[0],
                    ]
                )
            )
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
                        ops.append(
                            SWAP(
                                wires=[
                                    wire_manager.portL_wires[parent],
                                    wire_manager.router(k, p),
                                ]
                            )
                        )
                    else:
                        ops.append(
                            SWAP(
                                wires=[
                                    wire_manager.portR_wires[parent],
                                    wire_manager.router(k, p),
                                ]
                            )
                        )
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
