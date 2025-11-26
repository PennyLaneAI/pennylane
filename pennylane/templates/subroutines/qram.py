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
from typing import Sequence

from pennylane import math
from pennylane.decomposition import (
    add_decomps,
    controlled_resource_rep,
    register_resources,
    resource_rep,
)
from pennylane.operation import Operation
from pennylane.ops import CSWAP, SWAP, Hadamard, PauliZ, adjoint, ctrl
from pennylane.wires import Wires, WiresLike

# pylint: disable=consider-using-generator


# -----------------------------
# Wires Data Structure
# -----------------------------
@dataclass
class _QRAMWires:

    qram_wires: Wires
    target_wires: Wires
    bus_wire: Wires
    dir_wires: Wires
    portL_wires: Wires
    portR_wires: Wires

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
    r"""Bucket-brigade QRAM with **explicit bus routing** using 3 qubits per node.

    Bucket-brigade QRAM achieves an :math:`O(\log N)` complexity instead of the typical :math:`N`,
    where :math:`N` is the number of memory cells addressed. It does this by reducing the number of
    nodes that need to be visited in a tree which converts our binary address into a unary address
    at the leaves. In the end, the target wires' state corresponds to the data at the desired
    address. For more theoretical details on how this algorithm works, please consult
    `arXiv:0708.1879 <https://arxiv.org/pdf/0708.1879>`__.

    Args:
        bitstrings (Sequence[str]):
            The classical data as a sequence of bitstrings. The size of the classical data must be
            :math:`2^{\texttt{len(qram_wires)}}`.
        qram_wires (WiresLike):
            The register that stores the index for the entry of the classical data we want to
            access.
        target_wires (WiresLike):
            The register in which the classical data gets loaded. The size of this register must
            equal each bitstring length in ``bitstrings``.
        work_wires (WiresLike):
            The additional wires required to funnel the desired entry of ``bitstrings`` into the
            target register. The size of the ``work_wires`` register must be
            :math:`1 + 3 ((1 << \texttt{len(qram_wires)}) - 1)`. More specifically, the
            ``work_wires`` register includes the bus, direction, left port and right port wires in
            that order. Each node in the tree contains one address (direction), one left port and
            one right port wire. The single bus wire is used for address loading and data routing.

    Raises:
        ValueError: if the ``bitstrings`` are not provided, the ``bitstrings`` are of the wrong
            length, the ``target_wires`` are of the size of the ``work_wires`` register is not exactly
            equal to :math:`1 + 3 ((1 << \texttt{len(qram_wires)}) - 1)`.


    **Example:**

    Consider the following example, where the classical data is a list of four bitstrings (each of
    length 3):

    .. code-block:: python

        bitstrings = ["010", "111", "110", "000"]
        bitstring_size = 3

    The number of wires needed to store a length-4 array is 2, which means that the ``qram_wires``
    register must contain 2 wires. Additionally, this lets us specify the number of work wires
    needed.

    .. code-block:: python

        num_qram_wires = 2 # len(bistrings) = 4 = 2**2
        num_work_wires = 1 + 3 * ((1 << num_qram_wires) - 1) # 10

    Now, we can define all three registers concretely and demonstrate ``BBQRAM`` in practice. In the
    following circuit, we prepare the state :math:`\vert 2 \rangle = \vert 10 \rangle` on the
    ``qram_wires``, which indicates that we would like to access the second (zero-indexed) entry of
    ``bitstrings`` (which is ``"110"``). The ``target_wires`` register should therefore store this
    state after ``BBQRAM`` is applied.

    .. code-block:: python

        import pennylane as qml
        reg = qml.registers(
            {
                "qram": num_qram_wires,
                "target": bitstring_size,
                "work_wires": num_work_wires
            }
        )

        dev = qml.device("default.qubit")
        @qml.qnode(dev)
        def bb_quantum():
            # prepare an address, e.g., |10> (index 2)
            qml.BasisEmbedding(2, wires=reg["qram"])

            qml.BBQRAM(
                bitstrings,
                qram_wires=reg["qram"],
                target_wires=reg["target"],
                work_wires=reg["work_wires"],
            )
            return qml.probs(wires=reg["target"])

    >>> import numpy as np
    >>> print(np.round(bb_quantum()))  # doctest: +SKIP
    [0. 0. 0. 0. 0. 0. 1. 0.]

    Note that ``"110"`` in binary is equal to 6 in decimal, which is the only non-zero entry in
    the ``target_wires`` register.
    """

    grad_method = None

    resource_keys = {"bitstrings"}

    @property
    def resource_params(self) -> dict:
        return {
            "bitstrings": self.hyperparameters["bitstrings"],
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

        expected_nodes = (1 << n_k) - 1 if n_k > 0 else 0

        if len(work_wires) != 1 + 3 * expected_nodes:
            raise ValueError(f"work_wires must have length {1 + 3 * expected_nodes}.")

        bus_wire = Wires(work_wires[0])
        divider = len(work_wires[1:]) // 3
        dir_wires = Wires(work_wires[1 : 1 + divider])
        portL_wires = Wires(work_wires[1 + divider : 1 + divider * 2])
        portR_wires = Wires(work_wires[1 + divider * 2 : 1 + divider * 3])

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
            "bitstrings": bitstrings,
        }

        super().__init__(wires=all_wires, id=id)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)


def _bucket_brigade_qram_resources(bitstrings):
    num_target_wires = len(bitstrings[0])
    num_qram_wires = int(math.log2(len(bitstrings)))
    n_k = num_qram_wires
    resources = defaultdict(int)
    resources[resource_rep(SWAP)] = ((1 << n_k) - 1 + n_k) * 2 + num_target_wires * 2
    resources[resource_rep(CSWAP)] = ((1 << num_qram_wires) - 1) * num_target_wires * 2 + (
        ((1 << n_k) - 1 - n_k) * 2
    )
    resources[
        controlled_resource_rep(
            base_class=SWAP, base_params={}, num_control_wires=1, num_zero_control_values=1
        )
    ] = ((1 << num_qram_wires) - 1) * num_target_wires * 2 + (((1 << n_k) - 1 - n_k) * 2)
    resources[resource_rep(Hadamard)] += num_target_wires * 2
    for j in range(num_target_wires):
        for p in range(1 << n_k):
            resources[resource_rep(PauliZ)] += 1 if int(bitstrings[p][j]) else 0
    return resources


def _mark_routers_via_bus(wire_manager, n_k):
    """Write low-order address bits into router directions **layer-by-layer** via the bus.

    For each low bit a_k (k = 0..n_k-1):
      1) SWAP(qram_wires[k], bus)
      2) Route bus down k levels (CSWAPs controlled by routers at levels < k)
      3) At node (k, path-prefix), SWAP(bus, dir[k, path-prefix])
    """
    SWAP([wire_manager.qram_wires[0], wire_manager.bus_wires[0]])
    SWAP([wire_manager.bus_wire[0], wire_manager.router(0, 0)])
    for k in range(1, n_k):
        # 1) load a_k into the bus
        origin = wire_manager.qram_wires[k]
        target = wire_manager.bus_wire[0]
        SWAP(wires=[origin, target])
        # 2) route down k levels
        _route_bus_down_first_k_levels(wire_manager, k)
        # 3) deposit at level-k node on the active path
        for p in range(1 << k):
            # change to  in_wire later
            parent = _node_index(k - 1, p >> 1)
            origin = (
                wire_manager.portL_wires[parent] if p % 2 == 0 else wire_manager.portR_wires[parent]
            )
            target = wire_manager.router(k, p)
            SWAP(wires=[origin, target])


def _route_bus_down_first_k_levels(wire_manager, k_levels):
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


def _leaf_ops_for_bit(wire_manager, bitstrings, n_k, j):
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
    wires, wire_manager, bitstrings
):  # pylint: disable=unused-argument
    bus_wire = wire_manager.bus_wire
    qram_wires = wire_manager.qram_wires
    n_k = len(qram_wires)
    # 1) address loading
    _mark_routers_via_bus(wire_manager, n_k)
    # 2) For each target bit: load→route down→leaf op→route up→restore (reuse the route bus function)
    for j, tw in enumerate(wire_manager.target_wires):
        Hadamard(wires=[tw])
        SWAP(wires=[tw, bus_wire[0]])
        _route_bus_down_first_k_levels(wire_manager, len(qram_wires))
        _leaf_ops_for_bit(wire_manager, bitstrings, n_k, j)
        adjoint(_route_bus_down_first_k_levels, lazy=False)(wire_manager, len(qram_wires))
        SWAP(wires=[tw, bus_wire[0]])
        Hadamard(wires=[tw])
    # 3) address unloading
    adjoint(_mark_routers_via_bus, lazy=False)(wire_manager, n_k)


add_decomps(BBQRAM, _bucket_brigade_qram_decomposition)
