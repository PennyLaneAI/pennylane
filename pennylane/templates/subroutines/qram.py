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
"""Contains four different implementations of QRAM: BBQRAM, HybridQRAM, SelectOnlyQRAM, and FFQRAM."""

from collections.abc import Sequence
from dataclasses import dataclass

from pennylane import math
from pennylane.core.operator import Operation
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
        bitstrings (TensorLike | Sequence[str]):
            The classical data as a 2-D array.  The shape must be ``(num_bitstrings, size_bitstrings)``,
            where ``num_bitstrings`` is :math:`2^{\texttt{len(control_wires)}}` and
            ``size_bitstrings = len(target_wires)``.
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
            For more information, consult `arXiv:0708.1879 <https://arxiv.org/pdf/0708.1879>`__.

    Raises:
        ValueError: if the ``bitstrings`` are not provided, the ``bitstrings`` are of the wrong
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

        bitstrings = [[0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0]]
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
    ``bitstrings`` (which is ``[1, 1, 0]``). The ``target_wires`` register should therefore store this
    state after ``BBQRAM`` is applied.

    .. code-block:: python

        import pennylane as qp
        reg = qp.registers(
            {
                "control": num_control_wires,
                "target": bitstring_size,
                "work_wires": num_work_wires
            }
        )

        dev = qp.device("default.qubit")
        @qp.qnode(dev)
        def bb_quantum():
            # prepare an address, e.g., |10> (index 2)
            two_bin = qp.math.int_to_binary(2, len(reg["control"]))
            qp.BasisState(two_bin, wires=reg["control"])

            qp.BBQRAM(
                bitstrings,
                control_wires=reg["control"],
                target_wires=reg["target"],
                work_wires=reg["work_wires"],
            )
            return qp.probs(wires=reg["target"])

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
        bitstrings: TensorLike | Sequence[str],
        control_wires: WiresLike,
        target_wires: WiresLike,
        work_wires: WiresLike,
    ):  # pylint: disable=too-many-arguments
        control_wires = Wires(control_wires)

        if isinstance(bitstrings, (list, tuple)):
            bitstrings = math.array(bitstrings)

        if bitstrings.shape[0] == 0:
            raise ValueError("'bitstrings' cannot be empty.")

        if isinstance(bitstrings[0], str):
            bitstrings = math.array(
                list(map(lambda bitstring: [int(bit) for bit in bitstring], bitstrings))
            )

        m = bitstrings.shape[1]
        n_k = len(control_wires)
        if (1 << n_k) != bitstrings.shape[0]:
            raise ValueError("bitstrings.shape[0] must be 2^(len(control_wires)).")

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

        super().__init__(bitstrings, wires=all_wires)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)


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
        bitstrings (TensorLike):
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
            ``target_wires`` register. The ``work_wires`` register includes the signal, bus,
            direction, left port and right port wires in that order for a tree of depth
            :math:`(n-k)`. For more details, consult
            `section 3 of arXiv:2306.03242 <https://arxiv.org/abs/2306.03242>`__.
        k (int):
            The number of "select" bits taken from ``control_wires``.

    Raises:
        ValueError: if the ``bitstrings`` are not provided, the ``bitstrings`` are of the wrong length, there are
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

        bitstrings = [[0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0], [0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0]]
        bitstring_size = 3

    The ``control_wires`` are split via the value of :math:`k`, which allows us to leverage
    :class:`~.SelectOnly` and :class:`~.BBQRAM` behaviour.

    .. code-block:: python

        k = 2
        num_control_wires = 3
        num_work_wires = 1 + 1 + 3 * (1 << (num_control_wires - k) - 1)

        import pennylane as qp
        reg = qp.registers(
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

        dev = qp.device("default.qubit")
        @qp.qnode(dev)
        def hybrid_qram():
            # prepare an address, e.g., |010> (index 2)
            two_bin = qp.math.int_to_binary(2, len(reg["control"]))
            qp.BasisState(two_bin, wires=reg["control"])

            qp.HybridQRAM(
                bitstrings,
                control_wires=reg["control"],
                target_wires=reg["target"],
                work_wires=reg["work"],
                k=k
            )
            return qp.probs(wires=reg["target"])

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
        bitstrings: TensorLike | Sequence[str],
        control_wires: WiresLike,
        target_wires: WiresLike,
        work_wires: WiresLike,
        k: int,  # define the select part size, remaining part is tree part
    ):  # pylint: disable=too-many-arguments

        if isinstance(bitstrings, (list, tuple)):
            bitstrings = math.array(bitstrings)

        if bitstrings.shape[0] == 0:
            raise ValueError("'bitstrings' cannot be empty.")

        if isinstance(bitstrings[0], str):
            bitstrings = math.array(
                list(map(lambda bitstring: [int(bit) for bit in bitstring], bitstrings))
            )

        m = bitstrings.shape[1]

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

        if bitstrings.shape[0] != (1 << n_total):
            raise ValueError("bitstrings.shape[0] must be 2^(len(control_wires)).")

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

        super().__init__(bitstrings, wires=all_wires)

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
        bitstrings (TensorLike | Sequence[str]):
            The classical data as a sequence of bitstrings. The size of the classical data must be
            :math:`2^{\texttt{len(select_wires)}+\texttt{len(control_wires)}}`.
        control_wires (WiresLike):
            The register that stores the index for the entry of the classical data we want to
            access.
        target_wires (WiresLike):
            The register in which the classical data gets loaded. The size of this register must
            equal each bitstring length in ``bitstrings``.
        select_wires (WiresLike, optional):
            Wires used to perform the selection.
        select_value (int or None, optional):
            If provided, only entries whose select bits match this value are loaded.
            The ``select_value`` must be an integer in :math:`[0, 2^{\texttt{len(select_wires)}}]`,
            and cannot be used if no ``select_wires`` are provided.

    Raises:
        ValueError: if the ``bitstrings`` are of the wrong length, a ``select_value`` is provided without
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

        bitstrings = [[0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0], [0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0]]
        bitstring_size = 3

    Given the number of bitstrings, the values of ``control_wires`` and ``select_wires`` can be
    inferred. We can also provide a ``select_value`` to apply a filter such that only entries whose
    select bits match this value are loaded. The full address that is accessed by the algorithm is
    then the ``select_value`` prepended to the initial state of the control wires.

    .. code-block:: python

        num_control_wires = 2
        num_select_wires = 1
        select_value = 0

        import pennylane as qp
        reg = qp.registers(
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

        dev = qp.device("default.qubit")
        @qp.qnode(dev)
        def select_only_qram():
            # prepare an address, e.g., |010> (index 2)
            two_bin = qp.math.int_to_binary(2, len(reg["control"]))
            qp.BasisState(two_bin, wires=reg["control"])

            qp.SelectOnlyQRAM(
                bitstrings,
                control_wires=reg["control"],
                target_wires=reg["target"],
                select_wires=reg["select"],
                select_value=select_value,
            )
            return qp.probs(wires=reg["target"])

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
        bitstrings: TensorLike | Sequence[str],
        control_wires: WiresLike,
        target_wires: WiresLike,
        select_wires: WiresLike | None = None,
        select_value: int | None = None,
    ):

        if isinstance(bitstrings, (list, tuple)):
            bitstrings = math.array(bitstrings)

        if bitstrings.shape[0] == 0:
            raise ValueError("'bitstrings' cannot be empty.")

        if isinstance(bitstrings[0], str):
            bitstrings = math.array(
                list(map(lambda bitstring: [int(bit) for bit in bitstring], bitstrings))
            )

        target_wires = Wires(target_wires)
        m = bitstrings.shape[1]
        if m != len(target_wires):
            raise ValueError("len(target_wires) must equal bitstring length.")

        # Convert to Wires
        control_wires = Wires(control_wires)
        target_wires = Wires(target_wires)
        select_wires = Wires(select_wires) if select_wires is not None else Wires([])

        # ---- Validate bitstrings ----
        num_select = len(select_wires)
        num_controls = len(control_wires)
        n_total = num_select + num_controls

        if (1 << n_total) != bitstrings.shape[0]:
            raise ValueError(
                "bitstrings.shape[0] must be 2^(len(select_wires)+len(control_wires))."
            )

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
            bitstrings, wires=list(control_wires) + list(target_wires) + list(select_wires)
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


class FFQRAM(Operation):
    r"""Flip-flop QRAM (FF-QRAM) with an even-superposition initialization. FF-QRAM
    is a probabilistic protocol that encodes classical bitstrings and real amplitudes into
    a quantum circuit. For more theoretical details on how this algorithm works, please
    consult `Park et al. (2019) <https://www.nature.com/articles/s41598-019-40439-3>`__
    and `de Veras et al. (2021) <https://ieeexplore.ieee.org/abstract/document/9259210>`__.

    ``FFQRAM`` encodes :math:`L` distinct :math:`m`-bit addresses :math:`d_l` and
    real amplitudes :math:`b_l`, where :math:`L \leq N = 2^m`, into the state

    .. math::

        \sum_{l=0}^{L - 1} \tilde{b}_l |d_l\rangle,

    where :math:`\tilde{b}_l` is the normalized amplitude

    .. math::

        \tilde{b}_l = \frac{b_l}{\sqrt{\sum_k b_k^2}}.

    The preparation is successful only when the register qubit is in state
    :math:`|1\rangle`, and the success probability is
    :math:`P(\mathrm{reg}=1)=1 / 2^m`.

    Args:
        amplitudes (TensorLike):
            The real continuous amplitudes. The shape must be ``(L,)`` or
            ``(batch_size, L)``, where ``L <= 2**m``.
        wires (WiresLike):
            The wires the template acts on. The first ``m`` wires are address wires and
            the last wire is the register qubit used for post-selection.
        address (TensorLike | Sequence[str]):
            The classical address bitstrings as a 2-D array with shape ``(L, m)`` or as
            a sequence of unique bitstrings.

    Raises:
        ValueError: if the number of entries indicated by ``amplitudes`` and ``address``
            do not agree, if the number of entries exceeds ``2**m``, or if the addresses
            are not unique.

    .. seealso::
        :class:`~.BBQRAM`, :class:`~.HybridQRAM`, :class:`~.SelectOnlyQRAM`,
        :class:`~.QROM`, :class:`~.QROMStatePreparation`, :class:`~.AngleEmbedding`,
        :class:`~.AmplitudeEmbedding`

    **Example:**

    The following example is adapted from Section 4 of
    `de Veras et al. (2021) <https://ieeexplore.ieee.org/abstract/document/9259210>`__. The
    data to be registered has two entries:

    .. math::

        \left\{(\sqrt{0.3}, |000\rangle), (\sqrt{0.7}, |001\rangle)\right\}.

    To store this data we need three address wires (``m = 3``) and one register wire.

    .. code-block:: python

        import pennylane as qp
        import numpy as np

        addrs = ["000", "001"]
        amps = np.array([np.sqrt(0.3), np.sqrt(0.7)])
        wires = qp.registers({"address": 3, "register": 1})

        shots = 1000

        @qp.set_shots(shots)
        @qp.qnode(qp.device("default.qubit", seed=42))
        def circuit():
            qp.FFQRAM(
                amplitudes=amps,
                wires=wires["address"] + wires["register"],
                address=addrs,
            )
            # Postselect on the register qubit being in the |1> state
            qp.measure(wires["register"], postselect=1)
            return qp.probs(wires=wires["address"])

        results = circuit()

        # Theoretical post-selected probabilities: b_l^2 / sum_k b_k^2.
        expected = amps**2 / np.sum(amps**2)

        n = len(wires["address"])
        expected_full = np.zeros(2**n)
        for addr, prob in zip(addrs, expected, strict=True):
            expected_full[int(addr, 2)] = prob

        basis_states = [f"{i:0{n}b}" for i in range(2**n)]

    Because the protocol is sampled with a finite number of shots, the observed
    distribution approximates the theoretical values:

    >>> rows = [f"{'address':>8}{'observed':>11}{'expected':>11}"]
    >>> for state, obs, exp in zip(basis_states, results, expected_full, strict=True):
    ...     if obs > 1e-9 or exp > 1e-9:
    ...         rows.append(f"{state:>8}{obs:>11.3f}{exp:>11.3f}")
    >>> print("\n".join(rows))
     address   observed   expected
         000      0.238      0.300
         001      0.762      0.700
    >>> print(qp.draw(circuit, level=2)())
    0: ──H──X─╭●─────────X──X─╭●─────────X────┤ ╭Probs
    1: ──H──X─├●─────────X──X─├●─────────X────┤ ├Probs
    2: ──H──X─├●─────────X────├●──────────────┤ ╰Probs
    3: ───────╰RY(1.16)───────╰RY(1.98)──┤↗₁├─┤

    .. details::
        :title: Usage Details

        ``FFQRAM`` prepares a real-amplitude quantum state over a sparse list of
        computational basis states using the flip-register-flop construction. Given
        :math:`L` distinct :math:`m`-bit addresses :math:`d_l` and real amplitudes
        :math:`b_l`, where :math:`L \leq N = 2^m`, the amplitudes are normalized
        internally as

        .. math::

            \tilde{b}_l = \frac{b_l}{\sqrt{\sum_k b_k^2}}.

        This template splits the input ``wires`` into two sets. The first :math:`m` wires are used as address wires and are initialized to
        :math:`|+\rangle^{\otimes m}`. The last wire is a register qubit. For each
        address :math:`d_l`, the circuit flips the zero bits of :math:`d_l`, applies a
        multi-controlled :class:`~.RY` rotation with angle
        :math:`2\arcsin(\tilde{b}_l)` to the register qubit, and then uncomputes the
        flips.

        Before post-selection, the output state is

        .. math::

            \frac{1}{\sqrt{2^m}}
            \left[
            \sum_{x \notin D}|x\rangle|0\rangle
            +
            \sum_l |d_l\rangle
            \left(
            \sqrt{1 - \tilde{b}_l^2}|0\rangle
            +
            \tilde{b}_l|1\rangle
            \right)
            \right],

        where :math:`D` is the set of supplied addresses. After post-selecting the
        register qubit in :math:`|1\rangle`, the address register is prepared in

        .. math::

            \sum_l \tilde{b}_l |d_l\rangle.

        The post-selection success probability is :math:`P(\mathrm{reg}=1)=1 / 2^m`.
    """

    grad_method = None

    # num of zero bits in address -> num of X flips
    resource_keys = {"num_zero_bits", "num_address_wires", "num_entries"}

    def __init__(
        self, amplitudes: TensorLike, wires: WiresLike, address: TensorLike | Sequence[str]
    ):

        wires = Wires(wires)

        # use same input format as QROM
        if isinstance(address[0], str):
            address = math.array(
                list(map(lambda bitstring: [int(bit) for bit in bitstring], address))
            )

        if isinstance(address, (list, tuple)):
            address = math.array(address)

        num_address_wires = len(wires) - 1

        num_entries = math.shape(amplitudes)[-1]
        if num_entries != math.shape(address)[0]:
            raise ValueError("The number of amplitudes must equal the number of addresses.")
        if num_address_wires != math.shape(address)[1]:
            raise ValueError("Address bitstring length must equal the number of address wires.")
        if num_entries > 2**num_address_wires:
            raise ValueError("The number of entries cannot exceed 2 ** num_address_wires.")
        if len(math.unique(address, axis=0)) != len(address):
            raise ValueError("Addresses must be unique.")

        # hyperparameters should be hashable
        self._hyperparameters = {
            "address": tuple(tuple(int(bit) for bit in addr_bits) for addr_bits in address),
        }

        super().__init__(amplitudes, wires=wires)

    @property
    def resource_params(self):
        address = math.array(self._hyperparameters["address"])
        return {
            "num_zero_bits": int((1 - address).sum()),
            "num_address_wires": len(self.wires) - 1,
            "num_entries": address.shape[0],
        }

    @property
    def num_params(self):
        return 1  # amplitudes is the only trainable parameter

    @property
    def ndim_params(self):
        return (1,)  # 1d amplitude (without broadcasting)
