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
- Bucket-brigade QRAM LSBs (``control_wires``) using 3-qubits-per-node (dir, portL, portR)

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
from pennylane.ops import CNOT, CSWAP, SWAP, Controlled, Hadamard, PauliX, PauliZ, adjoint, ctrl
from pennylane.wires import Wires, WiresLike

# pylint: disable=consider-using-generator


# -----------------------------
# Wires Data Structure
# -----------------------------
@dataclass
class _QRAMWires:

    control_wires: Wires
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
            :math:`2^{\texttt{len(control_wires)}}`.
        control_wires (WiresLike):
            The register that stores the index for the entry of the classical data we want to
            access.
        target_wires (WiresLike):
            The register in which the classical data gets loaded. The size of this register must
            equal each bitstring length in ``bitstrings``.
        work_wires (WiresLike):
            The additional wires required to funnel the desired entry of ``bitstrings`` into the
            target register. The size of the ``work_wires`` register must be
            :math:`1 + 3 ((2^\texttt{len(control_wires)}) - 1)`. More specifically, the
            ``work_wires`` register includes the bus, direction, left port and right port wires in
            that order. Each node in the tree contains one address (direction), one left port and
            one right port wire. The single bus wire is used for address loading and data routing.

    Raises:
        ValueError: if the ``bitstrings`` are not provided, the ``bitstrings`` are of the wrong
            length, the ``target_wires`` are of the wrong size, or the ``work_wires`` register size is not exactly
            equal to :math:`1 + 3 ((2^\texttt{len(control_wires)}) - 1)`.

    .. seealso:: :class:`~.QROM`, :class:`~.QROMStatePreparation`

    .. note::

        QRAM and QROM, though similar, have different applications and purposes. QRAM is intended
        for read-and-write capabilities, where the stored data can be loaded and changed. QROM is
        designed to only load stored data into a quantum register.

    **Example:**

    Consider the following example, where the classical data is a list of four bitstrings (each of
    length 3):

    .. code-block:: python

        bitstrings = ["010", "111", "110", "000"]
        bitstring_size = 3

    The number of wires needed to store a length-4 array is 2, which means that the ``control_wires``
    register must contain 2 wires. Additionally, this lets us specify the number of work wires
    needed.

    .. code-block:: python

        num_control_wires = 2 # len(bistrings) = 4 = 2**2
        num_work_wires = 1 + 3 * ((1 << num_control_wires) - 1) # 10

    Now, we can define all three registers concretely and demonstrate ``BBQRAM`` in practice. In the
    following circuit, we prepare the state :math:`\vert 2 \rangle = \vert 10 \rangle` on the
    ``control_wires``, which indicates that we would like to access the second (zero-indexed) entry of
    ``bitstrings`` (which is ``"110"``). The ``target_wires`` register should therefore store this
    state after ``BBQRAM`` is applied.

    .. code-block:: python

        import pennylane as qml
        reg = qml.registers(
            {
                "control": num_control_wires,
                "target": bitstring_size,
                "work_wires": num_work_wires
            }
        )

        dev = qml.device("default.qubit")
        @qml.qnode(dev)
        def bb_quantum():
            # prepare an address, e.g., |10> (index 2)
            qml.BasisEmbedding(2, wires=reg["control"])

            qml.BBQRAM(
                bitstrings,
                control_wires=reg["control"],
                target_wires=reg["target"],
                work_wires=reg["work_wires"],
            )
            return qml.probs(wires=reg["target"])

    >>> import numpy as np
    >>> print(np.round(bb_quantum()))  # doctest: +SKIP
    [0. 0. 0. 0. 0. 0. 1. 0.]

    Note that ``"110"`` in binary is equal to 6 in decimal, which is the position of the only
    non-zero entry in the ``target_wires`` register.
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
        control_wires: WiresLike,
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
        control_wires = Wires(control_wires)

        n_k = len(control_wires)
        if (1 << n_k) != len(bitstrings):
            raise ValueError("len(bitstrings) must be 2^(len(control_wires)).")

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
            list(control_wires)
            + list(target_wires)
            + list(bus_wire)
            + list(dir_wires)
            + list(portL_wires)
            + list(portR_wires)
        )

        wire_manager = _QRAMWires(
            control_wires, target_wires, bus_wire, dir_wires, portL_wires, portR_wires
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
    n_k = int(math.log2(len(bitstrings)))
    resources = defaultdict(int)
    resources[resource_rep(SWAP)] = ((1 << n_k) - 1 + n_k) * 2 + num_target_wires * 2
    resources[resource_rep(CSWAP)] = ((1 << n_k) - 1) * num_target_wires * 2 + (
        ((1 << n_k) - 1 - n_k) * 2
    )
    resources[
        controlled_resource_rep(
            base_class=SWAP, base_params={}, num_control_wires=1, num_zero_control_values=1
        )
    ] = ((1 << n_k) - 1) * num_target_wires * 2 + (((1 << n_k) - 1 - n_k) * 2)
    resources[resource_rep(Hadamard)] += num_target_wires * 2
    for j in range(num_target_wires):
        for p in range(1 << n_k):
            resources[resource_rep(PauliZ)] += 1 if int(bitstrings[p][j]) else 0
    return resources


def _mark_routers_via_bus(wire_manager, n_k):
    """Write low-order address bits into router directions **layer-by-layer** via the bus.

    For each low bit a_k (k = 0..n_k-1):
      1) SWAP(control_wires[k], bus)
      2) Route bus down k levels (CSWAPs controlled by routers at levels < k)
      3) At node (k, path-prefix), SWAP(bus, dir[k, path-prefix])
    """
    SWAP([wire_manager.control_wires[0], wire_manager.bus_wire[0]])
    SWAP([wire_manager.bus_wire[0], wire_manager.router(0, 0)])
    for k in range(1, n_k):
        # 1) load a_k into the bus
        origin = wire_manager.control_wires[k]
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
    control_wires = wire_manager.control_wires
    n_k = len(control_wires)
    # 1) address loading
    _mark_routers_via_bus(wire_manager, n_k)
    # 2) For each target bit: load→route down→leaf op→route up→restore (reuse the route bus function)
    for j, tw in enumerate(wire_manager.target_wires):
        Hadamard(wires=[tw])
        SWAP(wires=[tw, bus_wire[0]])
        _route_bus_down_first_k_levels(wire_manager, len(control_wires))
        _leaf_ops_for_bit(wire_manager, bitstrings, n_k, j)
        adjoint(_route_bus_down_first_k_levels, lazy=False)(wire_manager, len(control_wires))
        SWAP(wires=[tw, bus_wire[0]])
        Hadamard(wires=[tw])
    # 3) address unloading
    adjoint(_mark_routers_via_bus, lazy=False)(wire_manager, n_k)


add_decomps(BBQRAM, _bucket_brigade_qram_decomposition)


class HybridQRAM(Operation):
    r"""Hybrid QRAM combining select-only and bucket-brigade behavior.

    This operator encodes bitstrings associated with indexes:

    .. math::
        \text{HybridQRAM}|i\rangle|0\rangle = |i\rangle |b_i\rangle,

    where :math:`b_i` is the bitstring associated with index :math:`i`.

    This hybrid QRAM implements a space–time tradeoff:

    - Total memory address bits: ``n = len(control_wires)``
    - Choose an integer :math:`k` with :math:`0 ≤ k < n`.
        - The first :math:`k` address bits (high-order) are "select" bits.
        - The remaining :math:`n-k` bits (low-order) are routed through a bucket-brigade tree.

    Instead of a full-depth tree of size :math:`2^n` leaves, we build a smaller tree of depth :math:`n-k` (:math:`2^{n-k}`
    leaves) and reuse it :math:`2^k` times:

    For each prefix :math:`s \in {0, …, 2^k - 1}`:

    - Perofrm a multi-controlled-X on a "signal" auxiliary, controlled by the :math:`k` select bits being equal to :math:`s`.
    - Conditioned on ``signal==1``, perform a BBQRAM query using only the lower :math:`n-k` address bits and the sub-table of bitstrings whose prefix is :math:`s`.
    - Uncompute the signal with the same multi-controlled-X.

    In the end, for any full address ``a = (prefix, suffix)``, the target wires are loaded with
    ``bitstrings[a]``.

    Wire layout:
    ``control_wires``: [ :math:`sel_0`, ..., :math:`sel_{k-1}`, :math:`tree_0`, ..., :math:`tree_{n-k-1}` ]
    ``work_wires``: :math:`[ signal, bus, dir..., portL..., portR... ]` (tree auxiliaries)

    Args:
        bitstrings (Sequence[str]): classical data table; must have length :math:`2^n` where ``n = len(control_wires)``
        control_wires (WiresLike): full address register (length ``n``)
        target_wires (WiresLike): :math:`m` target qubits; :math:`m` must equal bitstring length
        work_wires (WiresLike): auxiliaries: :math:`[signal, bus, dir..., portL..., portR...]` for a tree of depth :math:`(n-k)`
        k (int): number of "select" bits taken from the MSB of ``control_wires``

    Raises:
        ValueError: if the ``bitstrings`` are not provided, the ``bitstrings`` are of the wrong length, there are
            no ``control_wires``, ``k >= len(control_wires)``, the ``target_wires`` are of the wrong length, or the
            ``work_wires`` are of the wrong length.

    .. seealso:: :class:`~.QROM`, :class:`~.QROMStatePreparation`, :class:`~.BBQRAM`

    .. note::

        QRAM and QROM, though similar, have different applications and purposes. QRAM is intended
        for read-and-write capabilities, where the stored data can be loaded and changed. QROM is
        designed to only load stored data into a quantum register.
    """

    grad_method = None

    resource_keys = {
        "bitstrings",
        "num_target_wires",
        "num_select_wires",
        "num_tree_control_wires",
    }

    def __init__(
        self,
        bitstrings: Sequence[str],
        control_wires: WiresLike,
        target_wires: WiresLike,
        work_wires: WiresLike,
        k: int,  # define the select part size, remaining part is tree part
        id: str | None = None,
    ):  # pylint: disable=too-many-arguments

        if not bitstrings:
            raise ValueError("'bitstrings' cannot be empty.")
        m_set = {len(s) for s in bitstrings}
        if len(m_set) != 1:
            raise ValueError("All bitstrings must have equal length.")
        m = next(iter(m_set))
        bitstrings = list(bitstrings)

        control_wires = Wires(control_wires)
        target_wires = Wires(target_wires)
        work_wires = Wires(work_wires)

        # test wires
        n_total = len(control_wires)
        if n_total == 0:
            raise ValueError("len(control_wires) must be > 0.")

        if not 0 <= k < n_total:
            raise ValueError("k must satisfy 0 <= k < len(control_wires).")

        if len(target_wires) != m:
            raise ValueError("len(target_wires) must equal bitstring length.")

        if len(bitstrings) != (1 << n_total):
            raise ValueError("len(bitstrings) must be 2^(len(control_wires)).")

        # Split control_wires into select and tree parts
        select_wires = Wires(control_wires[:k])
        tree_control_wires = Wires(control_wires[k:])
        n_tree = len(tree_control_wires)

        # work_wires = [ signal, bus, dir..., portL..., portR... ] for tree depth n_tree
        signal_wire = Wires(work_wires[0])

        expected_nodes = (1 << n_tree) - 1
        expected_len = 1 + 1 + 3 * expected_nodes  # signal + bus + 3 per node
        if len(work_wires) != expected_len:
            raise ValueError(
                f"work_wires must have length {expected_len} "
                f"for k={k} and len(control_wires)={n_total}."
            )

        bus_wire = Wires(work_wires[1])
        divider = len(work_wires[2:]) // 3
        dir_wires = Wires(work_wires[2 : 2 + divider])
        portL_wires = Wires(work_wires[2 + divider : 2 + 2 * divider])
        portR_wires = Wires(work_wires[2 + 2 * divider : 2 + 3 * divider])

        tree_wire_manager = _QRAMWires(
            control_wires, target_wires, bus_wire, dir_wires, portL_wires, portR_wires
        )

        all_wires = list(control_wires) + list(target_wires) + list(work_wires)

        super().__init__(wires=all_wires, id=id)

        self._hyperparameters = {
            "bitstrings": bitstrings,
            "select_wires": select_wires,
            "signal_wire": signal_wire,
            "tree_wire_manager": tree_wire_manager,
        }

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @property
    def resource_params(self) -> dict:
        wire_manager = self.hyperparameters["tree_wire_manager"]
        k = len(self.hyperparameters["select_wires"])
        return {
            "bitstrings": self.hyperparameters["bitstrings"],
            "num_target_wires": len(wire_manager.target_wires),
            "num_select_wires": k,
            "num_tree_control_wires": len(wire_manager.control_wires[k:]),
        }


def _hybrid_qram_resources(bitstrings, num_target_wires, num_select_wires, num_tree_control_wires):
    resources = defaultdict(int)
    num_blocks = 1 << num_select_wires

    resources[resource_rep(PauliX)] += (num_select_wires <= 0) * num_blocks * 2

    resources[
        controlled_resource_rep(
            base_class=SWAP,
            base_params={},
            num_control_wires=1,
            num_zero_control_values=0,
        )
    ] += (
        (num_tree_control_wires + (1 << num_tree_control_wires) - 1) * 2 + 2 * num_target_wires
    ) * num_blocks

    ccswap_count = (
        (
            ((1 << num_tree_control_wires) - 1 - num_tree_control_wires)
            + ((1 << num_tree_control_wires) - 1) * num_target_wires
        )
        * num_blocks
        * 2
    )

    resources[
        controlled_resource_rep(
            base_class=Controlled,
            base_params={
                "base_class": SWAP,
                "base_params": {},
                "num_control_wires": 1,
                "num_zero_control_values": 0,
                "num_work_wires": 0,
                "work_wire_type": "borrowed",
            },
            num_control_wires=1,
            num_zero_control_values=0,
        )
    ] += ccswap_count

    resources[
        controlled_resource_rep(
            base_class=Controlled,
            base_params={
                "base_class": SWAP,
                "base_params": {},
                "num_control_wires": 1,
                "num_zero_control_values": 1,
                "num_work_wires": 0,
                "work_wire_type": "borrowed",
            },
            num_control_wires=1,
            num_zero_control_values=0,
        )
    ] += ccswap_count

    resources[
        controlled_resource_rep(
            base_class=Hadamard,
            base_params={},
            num_control_wires=1,
            num_zero_control_values=0,
        )
    ] += (
        num_target_wires * num_blocks * 2
    )

    for block_index in range(num_blocks):
        zero_control_values = [
            (block_index >> (num_select_wires - 1 - i)) & 1 for i in range(num_select_wires)
        ].count(0)
        if zero_control_values == 0:
            resources[resource_rep(CNOT)] += (num_select_wires > 0) * 2
        else:
            resources[
                controlled_resource_rep(
                    base_class=PauliX,
                    base_params={},
                    num_control_wires=num_select_wires,
                    num_zero_control_values=zero_control_values,
                )
            ] += (num_select_wires > 0) * 2

        resources[
            controlled_resource_rep(
                base_class=PauliZ,
                base_params={},
                num_control_wires=1,
                num_zero_control_values=0,
            )
        ] += sum(
            [
                bitstrings[(block_index << num_tree_control_wires) + p][j] == "1"
                for j in range(num_target_wires)
                for p in range(1 << num_tree_control_wires)
            ]
        )
    return resources


def _bits(value: int, length: int) -> list[int]:
    """Return `length` bits of `value` (MSB first)."""
    return [(value >> (length - 1 - i)) & 1 for i in range(length)]


def _tree_leaf_ops_for_bit_block_ctrl(
    bitstrings, j, block_index, tree_wire_manager, n_tree, signal
):  # pylint: disable=too-many-arguments
    """Leaf write for target bit j, for a given select prefix block, controlled on signal."""

    # For each leaf index p of the tree (n_tree bits)
    for p in range(1 << n_tree):
        # physical leaf wire (same pattern as BBQRAM)
        if p % 2 == 0:
            target = tree_wire_manager.portL(n_tree - 1, p >> 1)
        else:
            target = tree_wire_manager.portR(n_tree - 1, p >> 1)

        # Global address index: (block_index << n_tree) + p
        addr = (block_index << n_tree) + p
        bit = bitstrings[addr][j]
        if bit == "1":
            ctrl(PauliZ(wires=target), control=[signal], control_values=[1])


def _tree_route_bus_down_first_k_levels_ctrl(k_levels, tree_wire_manager, signal):
    """Tree routing down for first `k_levels` levels, controlled on signal."""

    for ell in range(k_levels):
        for p in range(1 << ell):
            in_w = tree_wire_manager.node_in_wire(ell, p)
            L = tree_wire_manager.portL(ell, p)
            R = tree_wire_manager.portR(ell, p)
            d = tree_wire_manager.router(ell, p)

            # dir==1: CSWAP(d, in_w, R) — additionally controlled on signal
            ctrl(CSWAP(wires=[d, in_w, R]), control=[signal], control_values=[1])

            # dir==0: SWAP(in_w, L) controlled on (d == 0) and signal == 1
            ctrl(
                ctrl(SWAP(wires=[in_w, L]), control=[d], control_values=[0]),
                control=[signal],
                control_values=[1],
            )


def _swap_controlled_on_signal(tree_wire_manager, signal, level, k):
    origin = tree_wire_manager.control_wires[k:][level]
    target = tree_wire_manager.bus_wire[0]
    ctrl(SWAP(wires=[origin, target]), control=[signal], control_values=[1])


def _tree_mark_routers_via_bus_ctrl(tree_wire_manager, n_tree, k, signal):
    """Address loading for the tree (n_tree bits), controlled on signal."""

    # SWAP(tree_control_wires[0], bus) controlled on signal
    _swap_controlled_on_signal(tree_wire_manager, signal, 0, k)

    # route down qram wires for level 0
    _tree_route_bus_down_first_k_levels_ctrl(0, tree_wire_manager, signal)

    # deposit into dir[0, *] along active path
    ctrl(
        SWAP(wires=[tree_wire_manager.bus_wire[0], tree_wire_manager.router(0, 0)]),
        control=[signal],
        control_values=[1],
    )

    for level in range(1, n_tree):
        # SWAP(tree_control_wires[level], bus) controlled on signal
        _swap_controlled_on_signal(tree_wire_manager, signal, level, k)

        # route down qram wires for current levels
        _tree_route_bus_down_first_k_levels_ctrl(level, tree_wire_manager, signal)

        # deposit into dir[level, *] along active path
        for p in range(1 << level):
            parent = _node_index(level - 1, p >> 1)
            if p % 2 == 0:
                origin = tree_wire_manager.portL_wires[parent]
            else:
                origin = tree_wire_manager.portR_wires[parent]
            target = tree_wire_manager.router(level, p)
            ctrl(SWAP(wires=[origin, target]), control=[signal], control_values=[1])


def _block_tree_query_ops(
    bitstrings, block_index, tree_wire_manager, n_tree, k, signal
):  # pylint: disable=too-many-arguments
    """One BBQRAM-style query of the (n_tree)-depth tree for a fixed select prefix."""

    # 1) address loading for the tree (controlled on signal)
    _tree_mark_routers_via_bus_ctrl(tree_wire_manager, n_tree, k, signal)

    # 2) per-target data phase, controlled on signal
    for j, tw in enumerate(tree_wire_manager.target_wires):
        # H on target
        ctrl(Hadamard(wires=[tw]), control=[signal], control_values=[1])

        # Swap target <-> bus
        ctrl(SWAP(wires=[tw, tree_wire_manager.bus_wire[0]]), control=[signal], control_values=[1])

        # Route down tree
        _tree_route_bus_down_first_k_levels_ctrl(n_tree, tree_wire_manager, signal)

        # Leaf Z ops for this block and bit index j
        _tree_leaf_ops_for_bit_block_ctrl(
            bitstrings, j, block_index, tree_wire_manager, n_tree, signal
        )

        # Route back up
        adjoint(_tree_route_bus_down_first_k_levels_ctrl, lazy=False)(
            n_tree, tree_wire_manager, signal
        )

        # Swap back bus -> target
        ctrl(SWAP(wires=[tw, tree_wire_manager.bus_wire[0]]), control=[signal], control_values=[1])

        # Final H on target
        ctrl(Hadamard(wires=[tw]), control=[signal], control_values=[1])

    # 3) address unloading for the tree (controlled on signal)
    adjoint(_tree_mark_routers_via_bus_ctrl, lazy=False)(tree_wire_manager, n_tree, k, signal)


@register_resources(_hybrid_qram_resources)
def _hybrid_qram_decomposition(
    wires, bitstrings, tree_wire_manager, select_wires, signal_wire, **_
):  # pylint: disable=unused-argument, too-many-arguments
    k = len(select_wires)

    signal = signal_wire[0]
    num_blocks = 1 << k if k > 0 else 1

    for block_index in range(num_blocks):
        # Multi-controlled X to turn signal on when select bits == block_index
        if k > 0:
            sel_pattern = _bits(block_index, k)
            ctrl(PauliX(wires=signal), control=select_wires, control_values=sel_pattern)
        else:
            # No select bits: just flip signal for all addresses
            PauliX(wires=signal)

        # Perform one tree query, driven by lower n_tree bits, controlled on signal
        _block_tree_query_ops(
            bitstrings,
            block_index,
            tree_wire_manager,
            len(tree_wire_manager.control_wires[k:]),
            k,
            signal,
        )

        # Uncompute signal
        if k > 0:
            ctrl(PauliX(wires=signal), control=select_wires, control_values=sel_pattern)
        else:
            PauliX(wires=signal)


add_decomps(HybridQRAM, _hybrid_qram_decomposition)
