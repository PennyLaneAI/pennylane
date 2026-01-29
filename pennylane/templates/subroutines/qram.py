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
"""Contains three different implementations of QRAM: BBQRAM, HybridQRAM, and SelectOnlyQRAM."""
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from pennylane import math
from pennylane.decomposition import (
    add_decomps,
    controlled_resource_rep,
    register_resources,
    resource_rep,
)
from pennylane.operation import Operation
from pennylane.ops import (
    CNOT,
    CSWAP,
    SWAP,
    Controlled,
    Hadamard,
    PauliX,
    PauliZ,
    adjoint,
    cond,
    ctrl,
)
from pennylane.templates import BasisEmbedding
from pennylane.typing import TensorLike
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
    r"""Bucket-brigade QRAM with explicit bus routing using 3 wires per node. Bucket-brigade QRAM
    achieves an :math:`O(\log N)` complexity instead of the typical :math:`N`, where :math:`N` is
    the size of the classical data register being queried. For more theoretical details on how this
    algorithm works, please consult `arXiv:0708.1879 <https://arxiv.org/pdf/0708.1879>`__.

    ``BBQRAM`` encodes bitstrings, :math:`b_i`, corresponding to a given entry, :math:`i`, in a
    data set:

    .. math::
        \text{BBQRAM}|i\rangle|0\rangle = |i\rangle |b_i\rangle.

    Args:
            The classical data as a sequence of bitstrings. The size of the classical data must be
        data (TensorLike):
            The classical data as an array.  The shape must be ``(num_data, size_data)``, where ``num_data`` is
            :math:`2^{\texttt{len(control_wires)}}` and ``size_data = len(target_wires)``.
        control_wires (WiresLike):
            The register that stores the index for the entry of the classical data we want to
            access.
        target_wires (WiresLike):
            The register in which the classical data gets loaded. The size of this register must
            equal each bitstring length in ``data``.
        work_wires (WiresLike):
            The additional wires required to funnel the desired entry of ``data`` into the
            target register. The size of the ``work_wires`` register must be
            :math:`1 + 3 ((2^\texttt{len(control_wires)}) - 1)`. More specifically, the
            ``work_wires`` register includes the bus, direction, left port and right port wires in
            that order. Each node in the tree contains one address (direction), one left port and
            one right port wire. The single bus wire is used for address loading and data routing.
            For more information, consult `arXiv:0708.1879 <https://arxiv.org/pdf/0708.1879>`__.

    Raises:
        ValueError: if the ``data`` are not provided, the ``data`` are of the wrong
            length, the ``target_wires`` are of the wrong size, or the ``work_wires`` register size is not exactly
            equal to :math:`1 + 3 ((2^\texttt{len(control_wires)}) - 1)`.

    .. seealso::
        :class:`~.SelectOnlyQRAM`, :class:`~.HybridQRAM`, :class:`~.QROM`, :class:`~.QROMStatePreparation`

    .. note::

        QRAM and QROM, though similar, have different applications and purposes. QRAM is intended
        for read-and-write capabilities, where the stored data can be loaded and changed. QROM is
        designed to only load stored data into a quantum register.

    **Example:**

    Consider the following example, where the classical data is a list of four bitstrings (each of
    length 3):

    .. code-block:: python

        data = [[0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0]]
        bitstring_size = 3

    The number of wires needed to store a length-4 array is 2, which means that the
    ``control_wires`` register must contain 2 wires. Additionally, this lets us specify the number
    of work wires needed.

    .. code-block:: python

        num_control_wires = 2 # len(bistrings) = 4 = 2**2
        num_work_wires = 1 + 3 * ((1 << num_control_wires) - 1) # 10

    Now, we can define all three registers concretely and demonstrate ``BBQRAM`` in practice. In the
    following circuit, we prepare the state :math:`\vert 2 \rangle = \vert 10 \rangle` on the
    ``control_wires``, which indicates that we would like to access the second (zero-indexed) entry of
    ``data`` (which is ``[1, 1, 0]``). The ``target_wires`` register should therefore store this
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
                data,
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

    resource_keys = {"num_controls", "num_target_wires"}

    @property
    def resource_params(self) -> dict:
        manager = self.hyperparameters["wire_manager"]
        return {
            "num_controls": len(manager.control_wires),
            "num_target_wires": len(manager.target_wires),
        }

    def __init__(
        self,
        data: TensorLike | Sequence[str],
        control_wires: WiresLike,
        target_wires: WiresLike,
        work_wires: WiresLike,
        id: str | None = None,
    ):  # pylint: disable=too-many-arguments
        control_wires = Wires(control_wires)

        if isinstance(data, (list, tuple)):
            data = math.array(data)

        if data.shape[0] == 0:
            raise ValueError("'data' cannot be empty.")

        if isinstance(data[0], str):
            data = math.array(list(map(lambda bitstring: [int(bit) for bit in bitstring], data)))

        m = data.shape[1]
        n_k = len(control_wires)
        if (1 << n_k) != data.shape[0]:
            raise ValueError("data.shape[0] must be 2^(len(control_wires)).")

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
        }

        super().__init__(data, wires=all_wires, id=id)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)


def _bucket_brigade_qram_resources(num_controls, num_target_wires):
    """
    Calculates the resources, assuming the worst case where data is all ones.
    """
    n_k = int(math.log2(2**num_controls))
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
    resources[resource_rep(Hadamard)] = num_target_wires * 2
    resources[resource_rep(PauliZ)] = (1 << n_k) * num_target_wires
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


def _leaf_ops_for_bit(wire_manager, data, n_k, j):
    """Apply the leaf write for target bit index j."""
    ops = []
    for p in range(1 << n_k):
        if p % 2 == 0:
            target = wire_manager.portL(n_k - 1, p >> 1)
        else:
            target = wire_manager.portR(n_k - 1, p >> 1)
        bit = data[p, j]
        cond(bit, PauliZ)(wires=target)
    return ops


@register_resources(_bucket_brigade_qram_resources, exact=False)
def _bucket_brigade_qram_decomposition(data, wire_manager, **__):  # pylint: disable=unused-argument
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
        _leaf_ops_for_bit(wire_manager, data, n_k, j)
        adjoint(_route_bus_down_first_k_levels, lazy=False)(wire_manager, len(control_wires))
        SWAP(wires=[tw, bus_wire[0]])
        Hadamard(wires=[tw])
    # 3) address unloading
    adjoint(_mark_routers_via_bus, lazy=False)(wire_manager, n_k)


add_decomps(BBQRAM, _bucket_brigade_qram_decomposition)


class HybridQRAM(Operation):
    r"""A QRAM implementation that provides a width-depth tradeoff by combining behaviour from
    :class:`~.SelectOnlyQRAM` and :class:`~.BBQRAM`. For more theoretical information, consult
    `section 3 of arXiv:2306.03242 <https://arxiv.org/abs/2306.03242>`__.

    ``HybridQRAM`` encodes bitstrings, :math:`b_i`, corresponding to a given entry, :math:`i`, in a
    data set:

    .. math::
        \text{HybridQRAM}|i\rangle|0\rangle = |i\rangle |b_i\rangle.

    With ``HybridQRAM``, an integer :math:`k` with :math:`0 ≤ k < n` must be chosen, where
    :math:`N = 2^n` is the size of the classical data register being queried. The first :math:`k`
    address bits are used in a procedure akin to what's involved in :class:`~.SelectOnlyQRAM`. The
    remaining :math:`n-k` bits are used in a procedure akin to what's in :class:`~.BBQRAM`; instead
    of a full-depth tree of size :math:`N` leaves, ``HybridQRAM`` builds a smaller tree of depth
    :math:`n-k` (:math:`2^{n-k}` leaves) and reuses it :math:`2^k` times.

    Args:
        data (TensorLike):
            The classical data as a sequence of bitstrings. The size of the classical data must be
            :math:`2^{\texttt{len(control_wires)}}`.
        control_wires (WiresLike):
            The register that stores the index for the entry of the classical data we want to
            access.
        target_wires (WiresLike):
            The register in which the classical data gets loaded. The size of this register must
            equal each bitstring length in ``data``.
        work_wires (WiresLike):
            The additional wires required to funnel the desired entry of ``data`` into the
            ``target_wires`` register. The ``work_wires`` register includes the signal, bus,
            direction, left port and right port wires in that order for a tree of depth
            :math:`(n-k)`. For more details, consult
            `section 3 of arXiv:2306.03242 <https://arxiv.org/abs/2306.03242>`__.
        k (int):
            The number of "select" bits taken from ``control_wires``.

    Raises:
        ValueError: if the ``data`` are not provided, the ``data`` are of the wrong length, there are
            no ``control_wires``, ``k >= len(control_wires)``, the ``target_wires`` are of the wrong length, or the
            ``work_wires`` are of the wrong length.

    .. seealso::
        :class:`~.SelectOnlyQRAM`, :class:`~.BBQRAM`, :class:`~.QROM`, :class:`~.QROMStatePreparation`

    .. note::

        QRAM and QROM, though similar, have different applications and purposes. QRAM is intended
        for read-and-write capabilities, where the stored data can be loaded and changed. QROM is
        designed to only load stored data into a quantum register.

    **Example:**

    Consider the following example, where the classical data is a list of bitstrings (each of
    length 3):

    .. code-block:: python

        data = [[0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0], [0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0]]
        bitstring_size = 3

    The ``control_wires`` are split via the value of :math:`k`, which allows us to leverage
    :class:`~.SelectOnly` and :class:`~.BBQRAM` behaviour.

    .. code-block:: python

        k = 2
        num_control_wires = 3
        num_work_wires = 1 + 1 + 3 * (1 << (num_control_wires - k) - 1)

        import pennylane as qml
        reg = qml.registers(
            {
                "control": num_control_wires,
                "target": bitstring_size,
                "work": num_work_wires
            }
        )

    In the following circuit, we prepare the state :math:`\vert 2 \rangle = \vert 010 \rangle`
    on the ``control_wires``, which indicates that we would like to access the second
    (zero-indexed) entry of ``bitstrings`` (which is ``"110"``). The ``target_wires`` register
    should therefore store this state after ``HybridQRAM`` is applied.

    .. code-block:: python

        dev = qml.device("default.qubit")
        @qml.qnode(dev)
        def hybrid_qram():
            # prepare an address, e.g., |010> (index 2)
            qml.BasisEmbedding(2, wires=reg["control"])

            qml.HybridQRAM(
                data,
                control_wires=reg["control"],
                target_wires=reg["target"],
                work_wires=reg["work"],
                k=k
            )
            return qml.probs(wires=reg["target"])

    >>> import numpy as np
    >>> print(np.round(hybrid_qram()))
    [0. 0. 0. 0. 0. 0. 1. 0.]

    Note that ``"110"`` in binary is equal to 6 in decimal, which is the position of the only
    non-zero entry in the ``target_wires`` register.
    """

    grad_method = None

    resource_keys = {
        "num_target_wires",
        "num_select_wires",
        "num_tree_control_wires",
    }

    def __init__(
        self,
        data: TensorLike | Sequence[str],
        control_wires: WiresLike,
        target_wires: WiresLike,
        work_wires: WiresLike,
        k: int,  # define the select part size, remaining part is tree part
        id: str | None = None,
    ):  # pylint: disable=too-many-arguments

        if isinstance(data, (list, tuple)):
            data = math.array(data)

        if data.shape[0] == 0:
            raise ValueError("'data' cannot be empty.")

        if isinstance(data[0], str):
            data = math.array(list(map(lambda bitstring: [int(bit) for bit in bitstring], data)))

        m = data.shape[1]

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

        if data.shape[0] != (1 << n_total):
            raise ValueError("data.shape[0] must be 2^(len(control_wires)).")

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

        super().__init__(data, wires=all_wires, id=id)

        self._hyperparameters = {
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
            "num_target_wires": len(wire_manager.target_wires),
            "num_select_wires": k,
            "num_tree_control_wires": len(wire_manager.control_wires[k:]),
        }


def _hybrid_qram_resources(num_target_wires, num_select_wires, num_tree_control_wires):
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
        ] += (1 << num_tree_control_wires) * num_target_wires

    return resources


def _bits(value: int, length: int) -> list[int]:
    """Return `length` bits of `value` (MSB first)."""
    return [(value >> (length - 1 - i)) & 1 for i in range(length)]


def _tree_leaf_ops_for_bit_block_ctrl(
    data, j, block_index, tree_wire_manager, n_tree, signal
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
        bit = data[addr][j]
        # pylint: disable=cell-var-from-loop
        cond(bit, lambda: ctrl(PauliZ(wires=target), control=[signal], control_values=[1]))()


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
    data, block_index, tree_wire_manager, n_tree, k, signal
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
        _tree_leaf_ops_for_bit_block_ctrl(data, j, block_index, tree_wire_manager, n_tree, signal)

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


@register_resources(_hybrid_qram_resources, exact=False)
def _hybrid_qram_decomposition(
    data, tree_wire_manager, select_wires, signal_wire, **_
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
            data,
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


class SelectOnlyQRAM(Operation):
    r"""A QRAM implementation comprising :class:`~.MultiControlledX` gates on target (bus) wires,
    controlled on all address wires. This implementation of QRAM requires :math:`O(\log N)` wires,
    where :math:`N` is the size of the classical data register being queried. For more theoretical
    information, consult `Figure 8 of arXiv:2012.05340 <https://arxiv.org/abs/2012.05340>`__.

    ``SelectOnlyQRAM`` encodes bitstrings, :math:`b_i`, corresponding to a given entry, :math:`i`,
    in a data set:

    .. math::
        \text{SelectOnlyQRAM}|i\rangle|0\rangle = |i\rangle |b_i\rangle.

    Args:
        data (TensorLike):
            The classical data as a sequence of bitstrings. The size of the classical data must be
            :math:`2^{\texttt{len(select_wires)}+\texttt{len(control_wires)}}`.
        control_wires (WiresLike):
            The register that stores the index for the entry of the classical data we want to
            access.
        target_wires (WiresLike):
            The register in which the classical data gets loaded. The size of this register must
            equal each bitstring length in ``data``.
        select_wires (WiresLike, optional):
            Wires used to perform the selection.
        select_value (int or None, optional):
            If provided, only entries whose select bits match this value are loaded.
            The ``select_value`` must be an integer in :math:`[0, 2^{\texttt{len(select_wires)}}]`,
            and cannot be used if no ``select_wires`` are provided.
        id (str or None):
            Optional name for the operation.

    Raises:
        ValueError: if the ``data`` are of the wrong length, a ``select_value`` is provided without
             ``select_wires``, or the ``select_value`` is greater than [0, (:math:`2^{\texttt{len(select_wires)}}`) - 1].

    .. seealso::

        :class:`~.BBQRAM`, :class:`~.HybridQRAM`, :class:`~.QROM`, :class:`~.QROMStatePreparation`

    .. note::

        QRAM and QROM, though similar, have different applications and purposes. QRAM is intended
        for read-and-write capabilities, where the stored data can be loaded and changed. QROM is
        designed to only load stored data into a quantum register.

    **Example:**

    Consider the following example, where the classical data is a list of bitstrings (each of length
    3):

    .. code-block:: python

        data = [[0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0], [0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0]]
        bitstring_size = 3

    Given the number of bitstrings, the values of ``control_wires`` and ``select_wires`` can be
    inferred. We can also provide a ``select_value`` to apply a filter such that only entries whose
    select bits match this value are loaded. The full address that is accessed by the algorithm is
    then the ``select_value`` prepended to the initial state of the control wires.

    .. code-block:: python

        num_control_wires = 2
        num_select_wires = 1
        select_value = 0

        import pennylane as qml
        reg = qml.registers(
            {
                "control": num_control_wires,
                "target": bitstring_size,
                "select": num_select_wires
            }
        )

    In the following circuit, we prepare the state :math:`\vert 2 \rangle = \vert 010 \rangle`
    on the ``control_wires``, which indicates that we would like to access the second
    (zero-indexed) entry of ``bitstrings`` (which is ``"110"``). The ``target_wires`` register
    should therefore store this state after ``SelectOnlyQRAM`` is applied.

    .. code-block:: python

        dev = qml.device("default.qubit")
        @qml.qnode(dev)
        def select_only_qram():
            # prepare an address, e.g., |010> (index 2)
            qml.BasisEmbedding(2, wires=reg["control"])

            qml.SelectOnlyQRAM(
                data,
                control_wires=reg["control"],
                target_wires=reg["target"],
                select_wires=reg["select"],
                select_value=select_value,
            )
            return qml.probs(wires=reg["target"])

    >>> import numpy as np
    >>> print(np.round(select_only_qram()))
    [0. 0. 0. 0. 0. 0. 1. 0.]

    Note that ``"110"`` in binary is equal to 6 in decimal, which is the position of the only
    non-zero entry in the ``target_wires`` register.
    """

    grad_method = None

    resource_keys = {
        "select_value",
        "num_control_wires",
        "num_select_wires",
        "num_target_wires",
    }

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        data: TensorLike | Sequence[str],
        control_wires: WiresLike,
        target_wires: WiresLike,
        select_wires: WiresLike | None = None,
        select_value: int | None = None,
        id: str | None = None,
    ):

        if isinstance(data, (list, tuple)):
            data = math.array(data)

        if data.shape[0] == 0:
            raise ValueError("'data' cannot be empty.")

        if isinstance(data[0], str):
            data = math.array(list(map(lambda bitstring: [int(bit) for bit in bitstring], data)))

        target_wires = Wires(target_wires)
        m = data.shape[1]
        if m != len(target_wires):
            raise ValueError("len(target_wires) must equal bitstring length.")

        # Convert to Wires
        control_wires = Wires(control_wires)
        target_wires = Wires(target_wires)
        select_wires = Wires(select_wires) if select_wires is not None else Wires([])

        # ---- Validate data ----
        num_select = len(select_wires)
        num_controls = len(control_wires)
        n_total = num_select + num_controls

        if (1 << n_total) != data.shape[0]:
            raise ValueError("data.shape[0] must be 2^(len(select_wires)+len(control_wires)).")

            # Validate select_value (if provided)
        if select_value is not None:
            if num_select == 0:
                raise ValueError("select_value cannot be used when len(select_wires) == 0.")
            max_sel = 1 << num_select
            if not 0 <= select_value < max_sel:
                raise ValueError(f"select_value must be an integer in [0, {max_sel - 1}].")

        self._hyperparameters = {
            "control_wires": control_wires,
            "target_wires": target_wires,
            "select_wires": select_wires,
            "select_value": select_value,
        }

        super().__init__(
            data, wires=list(control_wires) + list(target_wires) + list(select_wires), id=id
        )

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @property
    def resource_params(self) -> dict:
        return {
            "num_control_wires": len(self.hyperparameters["control_wires"]),
            "select_value": self.hyperparameters["select_value"],
            "num_select_wires": len(self.hyperparameters["select_wires"]),
            "num_target_wires": len(self.hyperparameters["target_wires"]),
        }


def _select_only_qram_resources(
    select_value, num_control_wires, num_select_wires, num_target_wires
):
    resources = defaultdict(int)
    n_total = num_control_wires + num_select_wires

    if select_value is not None and num_select_wires > 0:
        resources[resource_rep(BasisEmbedding, num_wires=num_select_wires)] += 1

    for addr in range(2 ** (num_select_wires + num_control_wires)):
        if (
            select_value is not None
            and num_select_wires > 0
            and (addr >> num_control_wires) != select_value
        ):
            continue

        control_values = [(addr >> (n_total - 1 - i)) & 1 for i in range(n_total)]

        resources[resource_rep(PauliX)] += control_values.count(0) * 2

        resources[
            controlled_resource_rep(
                base_class=PauliX,
                base_params={},
                num_control_wires=n_total,
                num_zero_control_values=0,
            )
        ] += num_target_wires

    return resources


def _flip_controls(control_wires, control_vals):
    for i, control_value in enumerate(control_vals):
        if control_value == 0:
            PauliX(control_wires[i])


@register_resources(_select_only_qram_resources, exact=False)
def _select_only_qram_decomposition(
    data, select_value, select_wires, control_wires, target_wires, **_
):  # pylint: disable=unused-argument, too-many-arguments
    controls = select_wires + control_wires
    num_select = len(select_wires)
    n_total = num_select + len(control_wires)

    if select_value is not None and len(select_wires) > 0:
        BasisEmbedding(select_value, wires=select_wires)

    # Loop over all addresses (0 .. 2^(num_select+num_controls)-1)
    for addr, bits in enumerate(data):
        # If select_value is specified, only implement entries whose
        # high num_select bits (select part) match that value.
        if select_value is not None and num_select > 0:
            sel_part = addr >> (n_total - num_select)
            if sel_part != select_value:
                continue

        control_values = [(addr >> (n_total - 1 - i)) & 1 for i in range(n_total)]

        _flip_controls(controls, control_values)

        # For each bit position in the data
        for j in range(data[0].shape[0]):
            # Multi-controlled X on target_wires[j],
            # controlled on controls matching `control_values`.

            # pylint: disable=cell-var-from-loop
            cond(bits[j], lambda: ctrl(PauliX(wires=target_wires[j]), control=controls))()

        _flip_controls(controls, control_values)


add_decomps(SelectOnlyQRAM, _select_only_qram_decomposition)
