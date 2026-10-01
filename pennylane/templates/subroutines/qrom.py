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
This submodule contains the template for QROM.
"""

from collections.abc import Sequence
from functools import partial

import numpy as np

from pennylane import capture, compiler, math
from pennylane import ops as qp_ops
from pennylane.control_flow import for_loop
from pennylane.core.operator import Operator2
from pennylane.decomposition import (
    add_decomps,
    register_condition,
    register_resources,
)
from pennylane.math import ceil_log2
from pennylane.ops import CNOT, CZ, X, cond, ctrl, pauli_measure
from pennylane.ops.mid_measure.pauli_measure import PauliMeasure
from pennylane.typing import AbstractArray, Bool, Int, TensorLike, Wire
from pennylane.wires import Wires, WiresLike, validate_no_wire_overlaps

from .arithmetic import TemporaryAND
from .multix import MultiX


def _select_ops(
    bitstrings, depth, target_wires, swap_wires, select_control_wires, select_work_wires
):  # pylint:disable=too-many-arguments
    num_targets = len(target_wires)
    num_bitstrings = bitstrings.shape[0]
    num_columns = int(np.ceil(num_bitstrings / depth))
    num_missing = (-num_bitstrings) % depth

    if num_missing > 0:
        bitstrings = math.vstack([bitstrings, math.zeros((num_missing, num_targets), dtype=int)])

    num_targets_select = depth * num_targets
    new_bitstrings = bitstrings.reshape((num_columns, num_targets_select))
    QROM(
        new_bitstrings,
        control_wires=select_control_wires,
        target_wires=swap_wires[:num_targets_select],
        work_wires=select_work_wires,
        clean=False,
    )


def _swap_ops(swap_control_wires, swap_wires, target_wires, cswap_work_wires):
    num_targets = len(target_wires)

    if capture.enabled() or compiler.active():
        swap_wires = math.array(swap_wires, like="jax").reshape((-1, num_targets))
        swap_control_wires = math.array(swap_control_wires, like="jax")
    else:
        # Need to work with nested list here in order to not coerce (deprecated) string wire labels
        # to object-dtyped np.array. This forces manual "reshape" here and indexing via [j][k]
        # instead of [j, k] below.
        num_columns = len(swap_wires) // num_targets
        swap_wires = [
            swap_wires[i * num_targets : (i + 1) * num_targets] for i in range(num_columns)
        ]

    @for_loop(len(swap_control_wires) - 1, -1, -1)
    def outer(i):

        @for_loop(2**i - 1, -1, -1)
        def inner(j):

            @for_loop(num_targets)
            def swap_layer(k):
                ctrl(
                    qp_ops.SWAP([swap_wires[j][k], swap_wires[j + 2**i][k]]),
                    control=swap_control_wires[-i - 1],
                    work_wires=cswap_work_wires,
                    work_wire_type="zeroed",
                )

            swap_layer()  # pylint: disable=no-value-for-parameter

        inner()  # pylint: disable=no-value-for-parameter

    outer()  # pylint: disable=no-value-for-parameter


class QROM(Operator2):
    r"""Applies the QROM operator.

    This operator encodes bitstrings associated with indexes:

    .. math::
        \text{QROM}|i\rangle|0\rangle = |i\rangle |b_i\rangle,

    where :math:`b_i` is the bitstring associated with index :math:`i`.

    Args:
        bitstrings (TensorLike): the data to be encoded
        control_wires (WiresLike):
            The register that stores the index for the entry of the classical data we want to
            read.
        target_wires (Sequence[int]): the wires where the bitstring is loaded
        work_wires (Sequence[int]): the auxiliary wires used for the computation
        clean (bool): if True, the work wires are not altered by operator, default is ``True``

    .. seealso:: :class:`~.BBQRAM`, :class:`~.QROMStatePreparation`

    .. note::
        QRAM and QROM, though similar, have different applications and purposes. QRAM is intended
        for read-and-write capabilities, where the stored data can be loaded and changed. QROM is
        designed to only load stored data into a quantum register.

    **Example**

    In this example, the QROM operator is applied to encode the third bitstring, associated with index 2, in the target wires.

    .. code-block:: python

        # a list of bitstrings is defined
        bitstrings = [[0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0]]

        dev = qp.device("default.qubit")

        @qp.qnode(dev, shots=1)
        def circuit():

            # the third index is encoded in the control wires [0, 1]
            qp.BasisState([1, 0], wires = [0,1])

            qp.QROM(bitstrings = bitstrings,
                    control_wires = [0,1],
                    target_wires = [2,3,4],
                    work_wires = [5,6,7])

            return qp.sample(wires = [2,3,4])

    >>> print(circuit())
    [[1 1 0]]


    .. details::
        :title: Usage Details

        This template takes as input three different sets of wires. The first one is ``control_wires`` which is used
        to encode the desired index. Therefore, if we have :math:`m` bitstrings, we need
        at least :math:`\lceil \log_2(m)\rceil` control wires.

        The second set of wires is ``target_wires`` which stores the bitstrings.
        For instance, if the bitstring is ``[0, 1, 1, 0]``, we will need four target wires. Internally,
        the bitstrings are encoded using the :class:`~.MultiX` template.


        The ``work_wires`` are auxiliary qubits used to reduce the gate complexity of the
        operator. These wires are dynamically partitioned into two sets: one for the
        :class:`~.Select` block and another to facilitate parallel data loading via a
        `SWAP network <https://pennylane.ai/compilation/swap-network>`__.

        The template determines the depth, :math:`\lambda` (a power of 2),
        based on the available ``work_wires``. Let :math:`b` be the length of the bitstrings.
        The number of wires allocated to the SWAP network is :math:`k_{swap} = b \cdot (\lambda - 1)`.
        The remaining wires, :math:`k_{select}`, are assigned to the :class:`~.Select` block.

        To ensure the decomposition is valid, the template guarantees that
        :math:`k_{select} \geq c - \log_2(\lambda) - 1`, where :math:`c` is the number of
        control wires, updating the depth if needed.

        The QROM template has two variants. The first one (``clean = False``) is based on [`arXiv:1812.00954 <https://arxiv.org/abs/1812.00954>`__] that alternates the state in the ``work_wires``.
        The second one (``clean = True``), based on [`arXiv:1902.02134 <https://arxiv.org/abs/1902.02134>`__], solves that issue by
        returning ``work_wires`` to their initial state. This technique can be applied when the ``work_wires`` are not
        initialized to zero.

        .. note::

            More ``control_wires`` than the minimum :math:`\lceil \log_2(m) \rceil` may be
            provided. The extra wires are treated as the most-significant address bits: the data
            is loaded only when they are all in :math:`|0\rangle`, and the operation acts as the
            identity otherwise. This turns ``QROM`` into a *controlled* load gated by those extra
            wires.

    """

    dynamic_argnames = ("bitstrings",)
    wire_argnames = ("control_wires", "target_wires", "work_wires")
    compilable_argnames = ("clean",)

    arg_specs = {
        "bitstrings": Int[-1, -1],
        "control_wires": Wire[-1],
        "target_wires": Wire[-1],
        "work_wires": Wire[-1],
    }

    def __init__(
        self,
        bitstrings: TensorLike | Sequence[str],
        control_wires: WiresLike,
        target_wires: WiresLike,
        work_wires: WiresLike,
        clean=True,
    ):  # pylint: disable=too-many-arguments,disable=too-many-positional-arguments

        control_wires = Wires(control_wires)
        target_wires = Wires(target_wires)
        work_wires = Wires(() if work_wires is None else work_wires)

        if not isinstance(bitstrings, AbstractArray):
            if isinstance(bitstrings[0], str):
                bitstrings = [[int(bit) for bit in bitstring] for bitstring in bitstrings]

            if isinstance(bitstrings, (list, tuple)):
                bitstrings = math.array(bitstrings, dtype=int)

            else:
                bitstrings = bitstrings.astype(int)

        wire_args = {
            "control_wires": control_wires,
            "target_wires": target_wires,
            "work_wires": work_wires,
        }
        validate_no_wire_overlaps(wire_args)

        if 2 ** len(control_wires) < bitstrings.shape[0]:
            raise ValueError(
                f"Not enough control wires ({len(control_wires)}) for the desired number of "
                f"bitstrings ({bitstrings.shape[0]}). At least {ceil_log2(bitstrings.shape[0])} "
                "control wires are required."
            )

        if bitstrings.shape[1] != len(target_wires):
            raise ValueError("Bitstring length must match the number of target wires.")

        super().__init__(bitstrings, control_wires, target_wires, work_wires, clean)

    @property
    def wires(self):
        """All wires involved in the operation."""
        return self.control_wires + self.target_wires + self.work_wires


def _calculate_select_swap_sizes(
    num_bitstrings, num_control_wires, num_targets, num_work_wires, **_
):
    """Calculates the register sizes for the Select-SWAP decomposition.

    This utility function determines how many auxiliary wires from the total pool
    should be allocated to the Select operation versus the SWAP network.

    Args:
        num_bitstrings (int): number of bitstrings/entries in the data
        num_control_wires (int): number of control wires
        num_targets (int): number of target wires (bitstring length)
        num_work_wires (int): total number of available work wires

    Returns:
        tuple[int]: ``(num_control_wires_select, num_work_wires_select, num_work_wires_swap,
        num_work_wires_cswap, depth)`` — control and work wires assigned to the Select component,
        work wires assigned to the SWAP network, and the number of bitstrings loaded in parallel.
    """

    if num_work_wires < num_control_wires - 1:
        return num_control_wires, num_work_wires, 0, 0, 1

    # Initialize available swap space using total work wires
    num_work_wires_swap = num_work_wires
    num_wires_swap = num_targets + num_work_wires_swap

    # Calculate depth: how many bitstrings we can load in parallel (power of 2)
    depth = num_wires_swap // num_targets
    depth = int(2 ** math.floor(math.log2(min(depth, num_bitstrings))))

    # Recalculate actual wires used by SWAP and the remaining for Select
    num_work_wires_swap = num_targets * depth - num_targets
    num_work_wires_select = num_work_wires - num_work_wires_swap

    # Adjust depth if Select doesn't have enough work wires for the required control logic
    num_control_wires_select = num_control_wires - int(math.floor(math.log2(depth)))
    while num_work_wires_select < num_control_wires_select - 1:
        depth = depth // 2
        num_work_wires_swap = num_targets * (depth - 1)
        num_work_wires_select = num_work_wires - num_work_wires_swap
        num_control_wires_select = num_control_wires - int(math.floor(math.log2(depth)))

    # As soon as there is an excess work wire for Select, reroute it to the CSWAPs themselves.
    num_work_wires_cswap = int(num_work_wires_select - max(0, num_control_wires_select - 1) >= 1)

    return (
        num_control_wires_select,
        num_work_wires_select,
        num_work_wires_swap,
        num_work_wires_cswap,
        depth,
    )


def _select_swap_condition(bitstrings, control_wires, target_wires, work_wires, clean):
    # pylint: disable=unused-argument
    """We use Select-SWAP only when there are enough work wires for unary iteration on the
    nested select QROM and an actual SWAP network is used on top, i.e. for depth > 1."""
    num_control_wires = len(control_wires)
    num_work_wires = len(work_wires)

    if num_control_wires == 0 or num_work_wires < num_control_wires - 1:
        return False

    *_, depth = _calculate_select_swap_sizes(
        len(bitstrings), num_control_wires, len(target_wires), num_work_wires
    )
    return depth > 1


def _select_swap_resources(
    bitstrings, control_wires, target_wires, work_wires, clean
):  # pylint: disable=too-many-branches
    """Assumes depth computed below satisfies depth > 1, as guaranteed by _select_swap_condition."""
    num_bitstrings = len(bitstrings)
    num_control_wires = len(control_wires)
    num_targets = len(target_wires)
    num_work_wires = len(work_wires)

    num_control_wires_select, num_work_wires_select, _, num_work_wires_cswap, depth = (
        _calculate_select_swap_sizes(num_bitstrings, num_control_wires, num_targets, num_work_wires)
    )

    num_columns = int(np.ceil(num_bitstrings / depth))
    # Select block (implemented as a nested QROM over concatenated columns)
    num_targets_select = depth * num_targets
    bigger_qrom = QROM(
        Int[num_columns, num_targets_select],
        Wire[num_control_wires_select],
        Wire[num_targets_select],
        Wire[num_work_wires_select - num_work_wires_cswap],
        False,
    )

    # Swap block
    num_control_wires_swap = num_control_wires - num_control_wires_select
    num_cswaps_per_block = num_targets * (2**num_control_wires_swap - 1)

    cswap_rep = ctrl(
        qp_ops.SWAP(Wire[2]),
        control=Wire[1],
        work_wires=Wire[num_work_wires_cswap],
        work_wire_type="zeroed",
    )

    if not clean:
        return {bigger_qrom: 1, cswap_rep: num_cswaps_per_block}

    return {bigger_qrom: 2, cswap_rep: 4 * num_cswaps_per_block, qp_ops.H: 2 * num_targets}


@register_condition(_select_swap_condition)
@register_resources(_select_swap_resources)
def _select_swap(
    bitstrings, control_wires, target_wires, work_wires, clean
):  # pylint: disable=unused-argument, too-many-arguments
    if len(control_wires) == 0:
        MultiX(bitstrings[0, :], wires=target_wires)
        return

    num_control_wires_select, _, num_work_wires_swap, num_work_wires_cswap, depth = (
        _calculate_select_swap_sizes(
            len(bitstrings), len(control_wires), len(target_wires), len(work_wires)
        )
    )

    swap_work_wires = work_wires[:num_work_wires_swap]
    select_work_wires = work_wires[num_work_wires_swap : len(work_wires) - num_work_wires_cswap]
    cswap_work_wires = work_wires[len(work_wires) - num_work_wires_cswap :]
    swap_wires = Wires(target_wires) + Wires(swap_work_wires)

    select_control_wires = control_wires[:num_control_wires_select]
    swap_control_wires = control_wires[num_control_wires_select:]

    if not clean:
        _select_ops(
            bitstrings, depth, target_wires, swap_wires, select_control_wires, select_work_wires
        )
        _swap_ops(swap_control_wires, swap_wires, target_wires, cswap_work_wires)
        return

    if capture.enabled() or compiler.active():
        target_wires = math.array(target_wires, like="jax")

    @for_loop(2)
    def _select_swap_loop(i):

        @for_loop(len(target_wires))
        def apply_h(i):
            qp_ops.H(target_wires[i])

        apply_h()  # pylint: disable=no-value-for-parameter

        cswaps = partial(_swap_ops, swap_control_wires, swap_wires, target_wires, cswap_work_wires)

        qp_ops.adjoint(cswaps, lazy=False)()
        _select_ops(
            bitstrings, depth, target_wires, swap_wires, select_control_wires, select_work_wires
        )
        cswaps()

    _select_swap_loop()  # pylint: disable=no-value-for-parameter


def _measurement_uncompute(work_wire, ctrl_wires, targets, product):
    """Measurement-based uncomputation from Fig 18a) https://arxiv.org/abs/2211.15465

    Args:
        work_wire: the AND output wire to uncompute. Third wire on the figure.
        ctrl_wires: [ctrl0, ctrl1] -- the two AND control wires (for CZ correction). First and second qubit on the figure.
        targets: target register wires.
        product: bitstring indicating the X positions in the target register.
    """
    x_wires = [targets[i] for i, bit in enumerate(product) if bit == 1]

    m1 = pauli_measure("X" + "X" * len(x_wires), [work_wire, *x_wires])

    cond(m1 == 1, CZ)(wires=ctrl_wires)

    m2 = pauli_measure("Z", [work_wire])
    cond(m2 == 1, X)(wires=work_wire)
    cond(m2 == 1, MultiX)(product, wires=targets)


def _measurement_qrom_inner(controls, targets, bitstrings):
    """Inner binary recursion with measurement-based uncomputation.

    Each level opens a TemporaryAND, recurses into left/right halves,
    then uncomputes via measurement. The XOR product between subtree
    bases is absorbed into the measurement.

    Args:
        controls: interleaved [flag, sel, work, sel2, work2, ...]
        targets: target register wires
        bitstrings: The set of k strings to be loaded in the decomposition. They do not necessarily match the QROM input values.

    """

    k = len(bitstrings)
    if k <= 1:
        return

    num_bits = ceil_log2(k)
    needed = 2 * num_bits + 1
    controls = list(controls[:1]) + list(controls[-(needed - 1) :])

    flag, sel, work = controls[0], controls[1], controls[2]
    child_controls = controls[2:]

    k_left = 2 ** (num_bits - 1)

    if k > 2:
        TemporaryAND([flag, sel, work], control_values=[1, 0])
        _measurement_qrom_inner(child_controls, targets, bitstrings[:k_left])
        CNOT(wires=[flag, work])
        _measurement_qrom_inner(child_controls, targets, bitstrings[k_left:])
    else:
        TemporaryAND([flag, sel, work], control_values=[1, 1])

    product = math.bitwise_xor(bitstrings[0], bitstrings[k_left])
    _measurement_uncompute(work, [flag, sel], targets, product)


def _measurement_qrom_outer(controls, targets, bitstrings, k):
    """Outer 4-quarter split with measurement-based uncomputation.

    Splits k items into quarters [Q0, Q1 | Q2, Q3] and processes each.
    Base corrections absorbed into measurements where possible (CLOSE).
    Remaining corrections (diff_q1, diff_q2) are explicit CNOTs.

    ``k`` is always a power of two (the caller pads the data up to the next
    power of two), so the middle split reduces to merging the close+open of
    the two halves into two CNOTs.
    """
    a = ceil_log2(k)
    controls = list(controls[: 2 * a - 1])

    and_wires = controls[:3]
    child_controls = controls[2:]

    k01 = 2 ** (a - 1)
    k0 = k1 = 2 ** (a - 2)
    l = k - k01
    k2 = 2 ** (ceil_log2(l) - 1)
    k3 = k - k01 - k2

    # --- OPEN ---
    TemporaryAND(and_wires, control_values=[0, 0])

    # --- Q0 ---
    _measurement_qrom_inner(child_controls, targets, bitstrings[:k0])

    # --- Q0 -> Q1 transition ---
    ctrl(X(controls[2]), control=controls[0], control_values=[0])
    diff_q1 = math.bitwise_xor(bitstrings[0], bitstrings[k0])

    # --- Q1 ---
    if k1 > 1:
        _measurement_qrom_inner(child_controls, targets, bitstrings[k0:k01])

    # --- MIDDLE: merge close+open into 2 CNOTs (no measurement here) ---
    for i, bit in enumerate(diff_q1):
        if bit == 1:
            CNOT(wires=[controls[2], targets[i]])
    CNOT(wires=[and_wires[0], and_wires[2]])
    CNOT(wires=[and_wires[1], and_wires[2]])
    sec_wires = and_wires
    sec_child = child_controls

    # --- Q2 base correction (explicit, no measurement available here) ---
    diff_q2 = math.bitwise_xor(bitstrings[0], bitstrings[k01])
    for i, bit in enumerate(diff_q2):
        if bit == 1:
            CNOT(wires=[sec_wires[2], targets[i]])

    # --- Q2 ---
    if k2 > 1:
        _measurement_qrom_inner(sec_child, targets, bitstrings[k01 : k01 + k2])

    # --- Q2 -> Q3 transition ---
    CNOT(wires=[sec_wires[0], sec_wires[2]])

    # --- Q3 ---
    diff_q3 = math.bitwise_xor(bitstrings[0], bitstrings[k01 + k2])
    if k3 > 1:
        _measurement_qrom_inner(sec_child, targets, bitstrings[k01 + k2 :])

    # --- CLOSE: absorb diff_q3 into measurement ---
    _measurement_uncompute(sec_wires[2], [sec_wires[0], sec_wires[1]], targets, diff_q3)


def _count_tempAND_in_measurement_qrom(k):
    """Count TemporaryAND gates for the measurement-based decomposition."""

    if k < 3:
        return 0
    if k > 3 / 4 * 2 ** ceil_log2(k):
        return k - 3
    return k - 2


def _qrom_measurement_resources(  # pylint: disable=too-many-arguments,unused-argument
    bitstrings=None, control_wires=None, target_wires=None, work_wires=None, clean=None, base=None
):
    """Resource estimate for the measurement-based QROM decomposition.

    Each TemporaryAND is uncomputed via _measurement_uncompute which produces:
      - 2 PauliMeasure (one X-type joint measurement, one Z measurement)
      - 1 CZ (phase correction conditioned on X measurement)
      - conditional X gates on work + targets
    """
    # When called for Adjoint(QROM), extract params from the base parameters
    if base is not None:
        num_bitstrings = len(base.bitstrings)
        num_targets = len(base.target_wires)
        num_control_wires = len(base.control_wires)
    else:
        num_bitstrings = len(bitstrings)
        num_targets = len(target_wires)
        num_control_wires = len(control_wires)

    num_control_wires_extra = (
        0 if num_control_wires is None else num_control_wires - ceil_log2(num_bitstrings)
    )
    # L = num_bitstrings
    # TODO: allowing partial QROM will reduce this term
    L = 2 ** ceil_log2(num_bitstrings)

    if L <= 1 and num_control_wires_extra == 0:
        return {MultiX(Bool[num_targets], Wire[num_targets]): 1}

    if L == 2 and num_control_wires_extra == 0:
        return {
            MultiX(Bool[num_targets], Wire[num_targets]): 1,
            ctrl(MultiX(Bool[num_targets], Wire[num_targets]), Wire[1]): 1,
        }

    # Without extra wires the load uses the cheaper 4-quarter outer iterator; with extra wires
    # it uses the flag-gated binary inner iterator, which needs ``L - 1`` AND gates.
    num_ands = L - 1 if num_control_wires_extra > 0 else _count_tempAND_in_measurement_qrom(L)
    num_cz = num_ands  # CZ correction per uncomputation

    # TemporaryAND counts are exact
    # CNOTs, PauliX gates and MultiX ops are an approximation
    flag = _flag_resources(num_control_wires_extra, num_targets)
    resources = {
        TemporaryAND: num_ands + flag.get(TemporaryAND, 0),
        # Each of the ``num_ands`` uncomputations performs one Z measurement on the work wire and
        # one X-type joint measurement on the work wire plus the target wires flipped by that
        # bitstring. The joint measurement's size (``1 + len(x_wires)``) varies per bitstring, so
        # the worst case (all ``num_targets`` flipped) is used for this approximate estimate.
        PauliMeasure("Z", wires=Wire[1]): num_ands,
        PauliMeasure("X" * (num_targets + 1), wires=Wire[num_targets + 1]): num_ands,
        CZ: num_cz,
        CNOT: L - 1,
        MultiX(Bool[num_targets], Wire[num_targets]): L,
        X: L + flag.get(X, 0),
        ctrl(X(Wire[1]), control=Wire[1], control_values=Bool[1]): 1,
    }
    # Merge the remaining flag-only resource types (controlled-X load, adjoint ANDs).
    for rep, count in flag.items():
        if rep not in resources:
            resources[rep] = count
    return resources


def _flag_resources(num_control_wires_extra, num_targets):
    """Return the resources for the flag that gates the load on extra control wires.

    A single extra wire uses two X gates; two or more use a ladder of ``num_control_wires_extra - 1`` AND gates,
    all later uncomputed by the same number of adjoints. In both cases the base load is gated,
    adding up to ``num_targets`` controlled-X gates.
    """
    if num_control_wires_extra < 1:
        return {}
    resources = {ctrl(X(Wire[1]), control=Wire[1]): num_targets}
    if num_control_wires_extra == 1:
        resources[X] = 2
        return resources
    resources[TemporaryAND] = num_control_wires_extra - 1
    resources[qp_ops.adjoint(TemporaryAND(Wire[3]))] = num_control_wires_extra - 1
    return resources


def _qrom_measurement_condition(
    bitstrings=None, control_wires=None, target_wires=None, work_wires=None, clean=None, base=None
):  # pylint: disable=too-many-arguments,unused-argument

    if base is not None:
        num_bitstrings = len(base.bitstrings)
        num_work_wires = len(base.work_wires)
        num_control_wires = len(base.control_wires)
    else:
        num_bitstrings = len(bitstrings)
        num_work_wires = len(work_wires)
        num_control_wires = len(control_wires)

    if not compiler.active():
        return False

    num_control_wires = (
        num_control_wires if num_control_wires is not None else max(1, ceil_log2(num_bitstrings))
    )
    if num_bitstrings <= 2 and num_control_wires <= 1:
        return True
    return num_work_wires >= num_control_wires - 1


def _interleave_controls(sel_wires, work_wires, head=None):
    """Build the interleaved control list consumed by the measurement iterators.

    The iterators expect ``[head, sel0, work0, sel1, work1, ...]`` where ``head`` is either the
    first selection wire (outer iterator, no flag) or the flag wire (flag-gated inner iterator).
    When ``head`` is ``None`` the first selection wire is used as the head and is not repeated.
    """
    if head is None:
        controls = [sel_wires[0]]
        sel_wires = sel_wires[1:]
    else:
        controls = [head]
    for sel, work in zip(sel_wires, work_wires):
        controls.append(sel)
        controls.append(work)
    return controls


def _build_flag(extra_wires, work_wires):
    """Build a flag wire that is 1 iff all extra control wires are 0.

    A single extra wire is flipped in place so that ``flag == 1`` iff it was 0; two or more are
    folded with a ladder of ``AND`` gates into an ancilla work wire. Returns ``(flag, core_work)``,
    where ``core_work`` are the work wires left to drive the inner unary iterator.
    """
    num_control_wires_extra = len(extra_wires)
    if num_control_wires_extra == 1:
        X(extra_wires[0])
        return extra_wires[0], work_wires

    anc_work, core_work = (
        work_wires[: num_control_wires_extra - 1],
        work_wires[num_control_wires_extra - 1 :],
    )

    # Each node is ``(wire, sat_value)``: the subtree rooted at ``wire`` reports "all extra wires
    # zero" when ``wire == sat_value``. Raw extra wires are satisfied at 0; ancillas written by an
    # ``AND`` are satisfied at 1. Combine nodes pairwise, level by level, into a balanced tree.
    nodes = [(w, 0) for w in extra_wires]
    anc_iter = iter(anc_work)
    while len(nodes) > 1:
        next_nodes = []
        for i in range(0, len(nodes) - 1, 2):
            (w0, v0), (w1, v1) = nodes[i], nodes[i + 1]
            anc = next(anc_iter)
            TemporaryAND([w0, w1, anc], control_values=[v0, v1])
            next_nodes.append((anc, 1))
        if len(nodes) % 2:  # carry the unpaired node up to the next level
            next_nodes.append(nodes[-1])
        nodes = next_nodes

    return nodes[0][0], core_work


@register_condition(_qrom_measurement_condition)
@register_resources(_qrom_measurement_resources, exact=False)
def _qrom_measurement_decomposition(
    bitstrings=None, control_wires=None, target_wires=None, work_wires=None, clean=None, base=None
):  # pylint: disable=too-many-arguments,too-many-branches,unused-argument
    """QROM decomposition using measurement-based uncomputation.

    Uses L-3 (or L-2) TemporaryAND gates. All uncomputation is done via
    PauliMeasure + conditional corrections instead of adjoint(TemporaryAND).
    Work wires are always left clean (via measurement-based uncomputation).
    Decomposition is based on Fig 18. https://arxiv.org/abs/2211.15465

    Requires: len(work_wires) >= len(control_wires) - 1.
    """
    # When called for Adjoint(QROM), extract params from the base operator
    if base is not None:
        bitstrings = base.bitstrings
        control_wires = base.control_wires
        target_wires = base.target_wires
        work_wires = base.work_wires

    # Bitstrings are manipulated with integer bitwise operations (math.bitwise_xor)
    # below, but callers may pass float data (e.g. QROM(np.eye(b), ...)). Cast to
    # int so the XOR-relative encoding works regardless of the input dtype.
    bitstrings = math.cast(bitstrings, int)

    L = len(bitstrings)
    num_control_wires = len(control_wires)

    # Extra control wires beyond ceil_log2(L) are the most-significant address bits: the data
    # is loaded only when they are all zero, otherwise the operation is the identity (matching
    # the non-partial ``Select``). We build a flag qubit that is 1 iff every extra wire is 0
    # and control the whole load on it, reusing the unary iterator ``_measurement_qrom_inner``
    # over the real 2**num_control_wires_active table.
    #
    # ``num_control_wires_extra == 0`` is intentionally handled by the branches below (the 4-quarter outer
    # iterator), which is cheaper than the flag-gated inner iterator used here.
    num_control_wires_active = ceil_log2(L)
    num_control_wires_extra = num_control_wires - num_control_wires_active
    if num_control_wires_extra > 0:
        extra_wires, active_wires = (
            control_wires[:num_control_wires_extra],
            control_wires[num_control_wires_extra:],
        )

        # Fold the extra wires into a flag that is 1 iff all of them are 0, then run the whole
        # load conditioned on that flag; the flag is uncomputed afterwards so work wires stay clean.
        flag, core_work = _build_flag(extra_wires, work_wires)

        # Gated base load, then the flag-gated unary iterator over the padded 2**num_control_wires_active table.
        padded = math.zeros((2**num_control_wires_active, len(bitstrings[0])), dtype=int)
        padded[:L] = bitstrings
        base = padded[0]
        # Fanout the base bitstring onto the target register, controlled on the flag.
        ctrl(MultiX(base, wires=target_wires), control=flag)
        bitstrings = math.bitwise_xor(padded, base)
        controls = _interleave_controls(
            active_wires[:num_control_wires_active], core_work, head=flag
        )
        _measurement_qrom_inner(controls, list(target_wires), bitstrings)

        # Uncompute the flag by inverting the exact gate sequence queued by ``_build_flag``.
        qp_ops.adjoint(_build_flag)(extra_wires, work_wires)
        return

    # TODO: allowing partial qrom will remove this padding
    # Pad data up to the next power of 2 with all-zero bitstrings
    next_pow2 = 1 << ceil_log2(L)
    if L < next_pow2:
        width = len(bitstrings[0])
        bitstrings = math.concatenate([bitstrings, math.zeros((next_pow2 - L, width), dtype=int)])
        L = next_pow2

    if L == 1:
        MultiX(bitstrings[0], target_wires)
        return

    if L == 2:
        MultiX(bitstrings[0], target_wires)
        diff = math.bitwise_xor(bitstrings[0], bitstrings[1])
        ctrl(MultiX(diff, wires=target_wires), control=control_wires[0])
        return

    # Load base bitstring
    MultiX(bitstrings[0], target_wires)

    # Build interleaved controls: [in[0], in[1], work[0], in[2], work[1], ...]
    controls = _interleave_controls(control_wires, work_wires)

    # XOR-relative encoding: bitstrings[i] = bitstrings[i] XOR bitstrings[0]
    bitstrings = math.bitwise_xor(bitstrings, bitstrings[0])

    _measurement_qrom_outer(controls, list(target_wires), bitstrings, L)


def _unary_iteration_split(num_control_wires, num_work_wires):
    """Split the control register between the unary iteration tree and the data loads.

    The tree spans the most-significant control wires and consumes one work wire per level, so
    at most ``num_work_wires + 1`` control wires can be absorbed into it. The remaining,
    least-significant control wires are attached as additional controls to each load.

    Returns:
        tuple[int]: ``(num_control_wires_tree, num_control_wires_extra)`` — control wires spanned
        by the tree, and leftover control wires attached to each load.
    """
    num_control_wires_tree = min(num_control_wires, num_work_wires + 1)
    return num_control_wires_tree, num_control_wires - num_control_wires_tree


def _qrom_unary_iteration_resources(
    bitstrings,
    control_wires,
    target_wires,
    work_wires,
    clean=True,
):  # pylint: disable=unused-argument,too-many-arguments
    c = len(control_wires)
    K = len(bitstrings)
    num_targets = len(target_wires)

    basis_rep = MultiX(Bool[num_targets], Wire[num_targets])
    if c == 0:
        return {basis_rep: 1}
    if c == 1:
        cbasis_rep = ctrl(basis_rep, control=Wire[1])
        if K == 1:
            return {cbasis_rep: 1}
        return {cbasis_rep: 1, basis_rep: 1}

    num_control_wires_tree, num_control_wires_extra = _unary_iteration_split(c, len(work_wires))
    # Each load is controlled on the ``num_control_wires_extra`` leftover control wires, plus the flag wire of
    # the iteration tree if there is one.
    load_rep = ctrl(basis_rep, control=Wire[1 + num_control_wires_extra])
    if num_control_wires_tree == 1:
        # No work wires, so there is no tree and every load is controlled on all control wires.
        return {load_rep: K}

    # The tree iterates over blocks of ``2**num_control_wires_extra`` bitstrings rather than over single
    # bitstrings, so the elbow count below is in terms of the number of blocks.
    num_blocks = -(-K // (1 << num_control_wires_extra))

    # The number of elbows required for non-partial unary iteration over K slots with c control
    # nodes is given by
    # N(c, K) = c + K - 2 - ‖K-1‖_H - int(K>2^{c-1}),
    # where ‖.‖_H denotes the Hamming weight, or bit count.
    # To see this, note that adding a control node to a given unary iteration is done by using the
    # given iteration, and replacing each "slot" (controlled unitary) by a construction that
    # yields two new "slots" and requires one elbow. Consequently, the addition of a control
    # node uses the given iteration with ⌈K/2⌉ slots, and ⌈K/2⌉ additional elbows, leading to the
    # recursion relation
    # N(c+1, K) = N(c, ⌈K/2⌉) + ⌈K/2⌉
    # In addition, we know that for two control nodes, just a single elbow is required:
    # N(2, K) = 1
    # The formula at the top is the solution to this recursion relation. An alternative expression
    # for the same is
    # N(c,K)=1+∑_{j=1}^{c−2} ⌈K⋅2^{−j}⌉
    more_than_half = int(num_blocks > 2 ** (num_control_wires_tree - 1))
    num_elbows = (
        num_control_wires_tree + num_blocks - 2 - (num_blocks - 1).bit_count() - more_than_half
    )
    return {
        TemporaryAND: num_elbows,
        qp_ops.adjoint(TemporaryAND(Wire[3])): num_elbows,
        CNOT: num_blocks - 1 + more_than_half,
        X: 2 * int(num_blocks > 2 ** (num_control_wires_tree - 2)),
        load_rep: K,
    }


def _load_block(block, address_wires, target_wires, flag=None):
    """Load one block of bitstrings into the target register.

    The ``s``-th entry of ``block`` is loaded when the ``address_wires`` hold the value ``s``,
    read most-significant bit first. If given, ``flag`` is an additional control wire that must
    be in state 1, which is how the unary iteration selects the current block.
    """
    num_address = len(address_wires)
    controls = list(address_wires) if flag is None else [flag, *address_wires]
    address_start = 0 if flag is None else (1 << num_address)

    @for_loop(len(block))
    def ctrl_sequence(s):
        control_values = math.int_to_binary(s + address_start, len(controls))
        ctrl(MultiX(block[s], target_wires), control=controls, control_values=control_values)

    ctrl_sequence()  # pylint: disable=no-value-for-parameter


def flip_iteration_bit(a, triples, top_not_flipped):
    """Update the unary iteration tree by flipping the most significant bit that differs between
    subsequent addresses (the central remains of merging a temporary AND uncomputation
    ladder and the next temporary AND computation ladder).

    Args:
        a (int): MSB-first index of least-significant 0 bit of k, the last loaded address
        triples (TensorLike): wire triples for unary iteration, sliced from the interleaved
            control and auxiliary wires.
        top_not_flipped (bool): whether the last loaded address is in the first half of
            the total capacity given by the number of control wires.

    """
    flip_first_control = math.logical_and(a == 1, top_not_flipped)

    # Once resource hints are merged, use those estimates:
    # cond(flip_first_control, X, estimated_probability=quarter_prob)(triples[0][0])
    # cond(a > 0, CNOT, estimated_probability=1 - mid_prob)(triples[a - 1][::2])
    # cond(flip_first_control, X, estimated_probability=quarter_prob)(triples[0][0])
    # cond(a == 0, CNOT, estimated_probability=mid_prob)(triples[0][::2])
    # cond(a == 0, CNOT, estimated_probability=mid_prob)(triples[0][1:])
    cond(flip_first_control, X)(triples[0][0])
    cond(a > 0, CNOT)(triples[a - 1][::2])
    cond(flip_first_control, X)(triples[0][0])
    cond(a == 0, CNOT)(triples[0][::2])
    cond(a == 0, CNOT)(triples[0][1:])


def _main_unary_loop_monolithic(bitstrings, triples, target_wires, extra_control_wires):
    c = len(triples) + 1
    assert c >= 2
    block_size = 1 << len(extra_control_wires)
    # The iteration runs over blocks of bitstrings, all of which are full except for the last.
    num_blocks = -(-len(bitstrings) // block_size)
    # An explicit trailing dimension is required because ``num_blocks - 1`` may be zero.
    blocks = math.reshape(
        bitstrings[: (num_blocks - 1) * block_size], (num_blocks - 1, block_size, len(target_wires))
    )
    last_block = bitstrings[(num_blocks - 1) * block_size :]
    # last work wire in use acts as the flag qubit for data loading.
    flag = triples[-1][2]

    concrete_load = partial(
        _load_block, address_wires=extra_control_wires, target_wires=target_wires, flag=flag
    )

    TemporaryAND(triples[0], (0, 0))
    for i in range(1, len(triples)):
        TemporaryAND(triples[i], (1, 0))

    # [dwierichs] todo: Once resource hints are merged, use those estimates:
    # [sc-129626] [sc-129627]
    # quarter_prob = int(num_blocks > (1 << (c - 2))) / (num_blocks - 1)
    # mid_prob = int(num_blocks > (1 << (c - 1))) / (num_blocks - 1)
    # est_ladder_len = float(
    # np.mean([math.bitwise_count(math.bitwise_xor(k, k + 1)) - 1 for k in range(num_blocks - 1)])
    # )

    # Loop over all blocks but the last one. Skip entirely when there is only one block:
    # ``blocks`` then has shape ``(0, ...)``, and Catalyst's ``for_loop(0)`` still traces the
    # body, which would index into that empty axis.
    if num_blocks > 1:

        def loop(k):
            # 1. load the k-th block, controlled on the flag circuit
            concrete_load(blocks[k])

            # 2. transition address k -> k+1
            # a is the MSB-first index of least-significant 0 bit of k
            a = c - math.bitwise_count(math.bitwise_xor(k, k + 1)).astype(int)

            # 2a. right-elbow ladder: uncompute levels c-2 .. max(a,1) (top-down)
            lower_bound = math.max(math.array([a, 1], like=a))

            @for_loop(c - 2, lower_bound - 1, -1)
            # Once resource hints are merged, use those estimates:
            # @for_loop(c - 2, max(a - 1, 0), -1, estimated_iterations=est_ladder_len)
            def uncompute(i):
                qp_ops.adjoint(TemporaryAND)(wires=triples[i])

            uncompute()  # pylint: disable=no-value-for-parameter

            # 2b. merge gate(s) at the boundary
            # Whether we are in the first half of the iteration, so that the top bit
            # has not been flipped yet
            top_not_flipped = k < (1 << (c - 1))
            flip_iteration_bit(a, triples, top_not_flipped)

            # 2c. left-elbow ladder: recompute levels max(a,1) .. c-2 (bottom-up)
            # Once resource hints are merged, use those estimates:
            @for_loop(lower_bound, c - 1)
            # @for_loop(max(a, 1), c - 1, estimated_iterations=est_ladder_len)
            def recompute(i):
                TemporaryAND(triples[i], (1, 0))

            recompute()  # pylint: disable=no-value-for-parameter

        for_loop(num_blocks - 1)(loop)()  # pylint: disable=no-value-for-parameter

    # Load the last block, which may be partially filled
    concrete_load(last_block)

    # closing ladder of right elbows for address num_blocks-1; control values depend on the bits of num_blocks-1
    closing_bits = [(num_blocks - 1 >> (c - 1 - b)) & 1 for b in range(c)]
    # levels i=c-2 .. 1 close with cvals (1, closing_bits[i+1]); level 0 closes with
    # cvals closing_bits[:2]
    for i in range(len(triples) - 1, 0, -1):
        qp_ops.adjoint(TemporaryAND(wires=triples[i], control_values=(1, closing_bits[i + 1])))
    qp_ops.adjoint(TemporaryAND(wires=triples[0], control_values=tuple(closing_bits[:2])))


@register_resources(_qrom_unary_iteration_resources)
def _qrom_unary_iteration(
    bitstrings, control_wires, target_wires, work_wires, clean, **__
):  # pylint: disable=unused-argument, too-many-arguments
    """Unary iteration decomposition of QROM.

    The unary iteration tree is grown over the most-significant control wires, one level per
    available work wire. Any control wire that is left over becomes an additional control of the
    ``MultiX`` loads, so that each slot of the unary iteration tree loads a block of bitstrings.
    With no work wires at all this reduces to one multi-controlled load per bitstring, with
    at least ``len(control_wires)-1`` work wires, we obtain classic unary iteration.
    """
    num_control_wires = len(control_wires)

    if num_control_wires == 0:
        # Simply load unique bit string
        MultiX(bitstrings[0], target_wires)
        return

    if num_control_wires == 1:
        if len(bitstrings) == 1:
            # One bit string to be applied
            ctrl(MultiX(bitstrings[0], target_wires), control=control_wires, control_values=[0])
            return
        # Two bit strings to be applied. Load the first unconditionally and control-load the diff
        MultiX(bitstrings[0], target_wires)
        ctrl(MultiX(bitstrings[0] ^ bitstrings[1], target_wires), control=control_wires)
        return

    num_control_wires_tree, num_control_wires_extra = _unary_iteration_split(
        num_control_wires, len(work_wires)
    )

    if num_control_wires_tree == 1:
        # Without work wires there is no unary iteration tree to build and the full control
        # register controls each bitstring to be loaded.
        _load_block(bitstrings, control_wires, target_wires)
        return

    # We are guaranteed num_control_wires_tree > 1 from here on, because num_control_wires_tree >= 1 initially.
    tree_wires, extra_control_wires = (
        control_wires[:num_control_wires_tree],
        control_wires[num_control_wires_tree:],
    )

    # Compute unary iteration wires
    interleaved = _interleave_controls(tree_wires, work_wires, head=None)
    triples = [interleaved[2 * i : 2 * i + 3] for i in range(num_control_wires_tree - 1)]

    if compiler.active() or capture.enabled():
        bitstrings = math.array(bitstrings, like="jax")
        triples = math.array(triples, like="jax")
        if num_control_wires_extra > 0:
            extra_control_wires = math.array(extra_control_wires, like="jax")

    _main_unary_loop_monolithic(bitstrings, triples, target_wires, extra_control_wires)


add_decomps(
    QROM,
    _select_swap,
    _qrom_unary_iteration,
    _qrom_measurement_decomposition,
)
add_decomps("Adjoint(QROM)", _qrom_measurement_decomposition)
