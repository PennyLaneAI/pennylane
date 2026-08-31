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

from collections import Counter
from collections.abc import Sequence
from functools import partial, reduce

import numpy as np

from pennylane import capture, compiler, math
from pennylane import ops as qp_ops
from pennylane.control_flow import for_loop
from pennylane.core.operator import Operator2
from pennylane.core.queuing import QueuingManager
from pennylane.decomposition import (
    add_decomps,
    register_condition,
    register_resources,
    resource_rep,
)
from pennylane.math import ceil_log2
from pennylane.ops import CNOT, CZ, X, cond, ctrl, pauli_measure
from pennylane.ops.mid_measure.pauli_measure import PauliMeasure
from pennylane.ops.op_math.adjoint2 import _adjoint_abstract
from pennylane.typing import AbstractArray, Bool, Int, TensorLike, Wire
from pennylane.wires import Wires, WiresLike

from .arithmetic import TemporaryAND
from .multix import MultiX
from .select import Select


def _select_ops(
    control_wires, depth, target_wires, swap_wires, bitstrings, select_work_wires
):  # pylint:disable=too-many-arguments
    capacity = 1 << len(control_wires)
    n_control_select_wires = ceil_log2(capacity / depth)
    control_select_wires = control_wires[:n_control_select_wires]

    with QueuingManager.stop_recording():
        with capture.pause():
            ops_new = [MultiX(bits, wires=target_wires) for bits in bitstrings]
            ops_identity_new = ops_new + [qp_ops.I(target_wires)] * (capacity - len(ops_new))

    n_columns = int(np.ceil(bitstrings.shape[0] / depth))
    num_targets = len(target_wires)
    wire_maps = [
        dict(zip(target_wires, swap_wires[j*num_targets: (j+1)*num_targets], strict=True))
        for j in range(depth)
    ]

    new_ops = []
    for i in range(n_columns):
        column_ops = [
            ops_identity_new[i * depth + j].map_wires(wire_maps[j])
            for j in range(depth)
        ]
        new_ops.append(qp_ops.prod(*column_ops))

    if control_select_wires:
        Select(new_ops, control=control_select_wires, work_wires=select_work_wires)


def _multi_swap(wires1, wires2):
    """Apply a series of SWAP gates between two sets of wires."""
    for wire1, wire2 in zip(wires1, wires2, strict=True):
        qp_ops.SWAP(wires=[wire1, wire2])


def _swap_ops(control_wires, depth, swap_wires, target_wires):
    n_control_select_wires = ceil_log2(2 ** len(control_wires) / depth)
    control_swap_wires = control_wires[n_control_select_wires:]
    num_targets = len(target_wires)
    for i in range(len(control_swap_wires) - 1, -1, -1):
        for j in range(2**i - 1, -1, -1):
            _wires0 = swap_wires[j * num_targets : (j + 1) * num_targets]
            _wires1 = swap_wires[(j + 2**i) * num_targets : (j + 2**i + 1) * num_targets]
            ctrl(_multi_swap, control=control_swap_wires[-i - 1])(_wires0, _wires1)


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

        if isinstance(bitstrings[0], str):
            bitstrings = np.array(
                list(map(lambda bitstring: [int(bit) for bit in bitstring], bitstrings))
            )

        elif isinstance(bitstrings, (list, tuple)):
            bitstrings = math.array(bitstrings, dtype=int)

        else:
            bitstrings = bitstrings.astype(int)

        work_wires = Wires(() if work_wires is None else work_wires)

        _wires_are_traced = any(
            math.is_abstract(w) for ws in (control_wires, target_wires, work_wires) for w in ws
        )

        # Wire overlap validation must be skipped when wires are JAX tracers,
        # as their concrete values are not available during tracing.
        if not _wires_are_traced:
            if len(work_wires) != 0:
                if any(wire in work_wires for wire in control_wires):
                    raise ValueError("Control wires should be different from work wires.")

                if any(wire in work_wires for wire in target_wires):
                    raise ValueError("Target wires should be different from work wires.")

            if any(wire in control_wires for wire in target_wires):
                raise ValueError("Target wires should be different from control wires.")

        if 2 ** len(control_wires) < bitstrings.shape[0]:
            raise ValueError(
                f"Not enough control wires ({len(control_wires)}) for the desired number of "
                f"bitstrings ({bitstrings.shape[0]}). At least {ceil_log2(bitstrings.shape[0])} "
                "control wires are required."
            )

        if bitstrings[0].shape[0] != len(target_wires):
            raise ValueError("Bitstring length must match the number of target wires.")

        super().__init__(bitstrings, control_wires, target_wires, work_wires, clean)

    # pylint: disable-next=arguments-differ, too-many-arguments
    def __abstract_init__(
        self,
        bitstrings: AbstractArray | TensorLike | Sequence[str],
        control_wires: AbstractArray | WiresLike,
        target_wires: AbstractArray | WiresLike,
        work_wires: AbstractArray | WiresLike,
        clean=True,
    ):
        if isinstance(bitstrings, Sequence) and isinstance(bitstrings[0], str):
            bitstrings = AbstractArray(shape=(len(bitstrings), len(bitstrings[0])), dtype=np.int64)
        super().__abstract_init__(
            bitstrings,
            control_wires=Wire[len(control_wires)],
            target_wires=Wire[len(target_wires)],
            work_wires=Wire[len(work_wires)],
            clean=clean,
        )

    @property
    def wires(self):
        """All wires involved in the operation."""
        return self.control_wires + self.target_wires + self.work_wires


def _calculate_n_select_work_wires(terms, num_control_wires, num_target_wires, num_work_wires, **_):
    """Calculates the number of work wires passes to the select block.

    This utility function determines how many auxiliary wires from the total pool
    should be allocated to the Select operation versus the SWAP network.

    Args:
        terms (int): number of bitstrings/entries in the data
        num_control_wires (int): number of control wires
        num_target_wires (int): number of target wires (bitstring length)
        num_work_wires (int): total number of available work wires

    Returns:
        int: The number of work wires assigned to the Select component.
    """

    if num_work_wires < num_control_wires - 1:
        return num_work_wires

    # Initialize available swap space using total work wires
    n_swap_work_wires = num_work_wires
    n_swap_wires = num_target_wires + n_swap_work_wires

    # Calculate depth: how many bitstrings we can load in parallel (power of 2)
    depth = n_swap_wires // num_target_wires
    depth = 1 << math.floor_log2(min(depth, terms))

    # Recalculate actual wires used by SWAP and the remaining for Select
    n_swap_work_wires = num_target_wires * depth - num_target_wires
    n_select_work_wires = num_work_wires - n_swap_work_wires

    # Adjust depth if Select doesn't have enough work wires for the required control logic
    n_select_control_wires = num_control_wires - math.floor_log2(depth)
    while n_select_work_wires < n_select_control_wires - 1:
        depth = depth // 2
        n_swap_work_wires = num_target_wires * depth - num_target_wires
        n_select_work_wires = num_work_wires - n_swap_work_wires
        n_select_control_wires = num_control_wires - math.floor_log2(depth)

    return n_select_work_wires


def _qrom_decomposition_resources(
    bitstrings, control_wires, target_wires, work_wires, clean
):  # pylint: disable=too-many-branches

    num_bitstrings = len(bitstrings)
    num_control_wires = len(control_wires)
    num_target_wires = len(target_wires)
    num_work_wires = len(work_wires)

    num_work_wires_select = _calculate_n_select_work_wires(
        num_bitstrings, num_control_wires, num_target_wires, num_work_wires
    )

    num_work_wires_swap = num_work_wires - num_work_wires_select

    if num_control_wires == 0:
        return {MultiX(Bool[num_target_wires], Wire[num_target_wires]): num_bitstrings}

    num_swap_wires = num_target_wires + num_work_wires_swap

    # number of operators we store per column (power of 2)
    depth = num_swap_wires // num_target_wires
    depth = 1 << math.floor_log2(depth)
    depth = min(depth, num_bitstrings)

    ops = [MultiX(Bool[num_target_wires], Wire[num_target_wires]) for _ in range(num_bitstrings)]
    ops_identity = ops + [resource_rep(qp_ops.I)] * int(2**num_control_wires - num_bitstrings)

    n_columns = (
        num_bitstrings // depth if num_bitstrings % depth == 0 else num_bitstrings // depth + 1
    )
    # Select block
    num_control_select_wires = ceil_log2(2**num_control_wires / depth)

    # New ops block
    new_ops = Counter()
    for i in range(n_columns):
        column_ops = Counter()
        for j in range(depth):
            column_ops[ops_identity[i * depth + j]] += 1
        if len(column_ops) == 1 and list(column_ops.values())[0] == 1:
            new_ops[list(column_ops.keys())[0]] += 1
        else:
            new_ops[resource_rep(qp_ops.op_math.Prod, resources=dict(column_ops))] += 1

    new_ops_reps = reduce(
        lambda acc, lst: acc + lst, [[key for _ in range(val)] for key, val in new_ops.items()]
    )

    if num_control_select_wires > 0:
        select_ops = {
            resource_rep(
                Select,
                num_control_wires=num_control_select_wires,
                op_reps=tuple(new_ops_reps),
                partial=False,
                num_work_wires=num_work_wires_select,
            ): 1
        }
    else:
        select_ops = new_ops

    # Swap block
    num_control_swap_wires = num_control_wires - num_control_select_wires
    swap_resources = Counter()
    for ind in range(num_control_swap_wires):
        for j in range(2**ind):
            num_swaps = min(
                (j + 1) * num_target_wires - (j) * num_target_wires,
                (j + 2 ** (ind + 1)) * num_target_wires - (j + 2**ind) * num_target_wires,
            )
            if num_swaps > 1:
                swap_resources[qp_ops.CSWAP] += num_swaps
            else:
                swap_resources[qp_ops.CSWAP] += 1

    if not clean or depth == 1:
        resources = swap_resources
        resources.update(select_ops)
        return resources

    resources = {}

    hadamard_ops = {qp_ops.Hadamard: num_target_wires}

    for key, val in swap_resources.items():
        swap_resources[key] = val * 2

    resources.update(hadamard_ops)
    resources.update(swap_resources)
    resources.update(select_ops)

    for key, val in resources.items():
        resources[key] = val * 2

    return resources


@register_resources(_qrom_decomposition_resources)
def _qrom_decomposition(
    bitstrings, control_wires, target_wires, work_wires, clean
):  # pylint: disable=unused-argument, too-many-arguments
    if len(control_wires) == 0:
        MultiX(bitstrings[0, :], wires=target_wires)
        return

    n_select_work_wires = _calculate_n_select_work_wires(
        len(bitstrings), len(control_wires), len(target_wires), len(work_wires)
    )

    n_swap_work_wires = len(work_wires) - n_select_work_wires
    swap_work_wires = work_wires[:n_swap_work_wires]
    select_work_wires = work_wires[n_swap_work_wires:]
    swap_wires = target_wires + swap_work_wires

    # number of operators we store per column (power of 2)
    depth = len(swap_wires) // len(target_wires)
    depth = min(1 << math.floor_log2(depth), bitstrings.shape[0])

    if not clean or depth == 1:
        _select_ops(control_wires, depth, target_wires, swap_wires, bitstrings, select_work_wires)
        if not clean:
            _swap_ops(control_wires, depth, swap_wires, target_wires)

    else:
        for _ in range(2):
            for w in target_wires:
                qp_ops.Hadamard(wires=w)
            qp_ops.adjoint(
                partial(_swap_ops, control_wires, depth, swap_wires, target_wires), lazy=False
            )()
            _select_ops(
                control_wires, depth, target_wires, swap_wires, bitstrings, select_work_wires
            )
            _swap_ops(control_wires, depth, swap_wires, target_wires)


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

    product = np.bitwise_xor(bitstrings[0], bitstrings[k_left])
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
    diff_q1 = np.bitwise_xor(bitstrings[0], bitstrings[k0])

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
    diff_q2 = np.bitwise_xor(bitstrings[0], bitstrings[k01])
    for i, bit in enumerate(diff_q2):
        if bit == 1:
            CNOT(wires=[sec_wires[2], targets[i]])

    # --- Q2 ---
    if k2 > 1:
        _measurement_qrom_inner(sec_child, targets, bitstrings[k01 : k01 + k2])

    # --- Q2 -> Q3 transition ---
    CNOT(wires=[sec_wires[0], sec_wires[2]])

    # --- Q3 ---
    diff_q3 = np.bitwise_xor(bitstrings[0], bitstrings[k01 + k2])
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
        num_target_wires = len(base.target_wires)
        num_control_wires = len(base.control_wires)
    else:
        num_bitstrings = len(bitstrings)
        num_target_wires = len(target_wires)
        num_control_wires = len(control_wires)

    n_extra = 0 if num_control_wires is None else num_control_wires - ceil_log2(num_bitstrings)
    # L = num_bitstrings
    # TODO: allowing partial QROM will reduce this term
    L = 2 ** ceil_log2(num_bitstrings)

    if L <= 1 and n_extra == 0:
        return {MultiX(Bool[num_target_wires], Wire[num_target_wires]): 1}

    if L == 2 and n_extra == 0:
        return {
            MultiX(Bool[num_target_wires], Wire[num_target_wires]): 1,
            ctrl(MultiX(Bool[num_target_wires], Wire[num_target_wires]), Wire[1]): 1,
        }

    # Without extra wires the load uses the cheaper 4-quarter outer iterator; with extra wires
    # it uses the flag-gated binary inner iterator, which needs ``L - 1`` AND gates.
    num_ands = L - 1 if n_extra > 0 else _count_tempAND_in_measurement_qrom(L)
    num_cz = num_ands  # CZ correction per uncomputation

    # TemporaryAND counts are exact
    # CNOTs, PauliX gates and MultiX ops are an approximation
    flag = _flag_resources(n_extra, num_target_wires)
    resources = {
        TemporaryAND: num_ands + flag.get(TemporaryAND, 0),
        # Each of the ``num_ands`` uncomputations performs one Z measurement on the work wire and
        # one X-type joint measurement on the work wire plus the target wires flipped by that
        # bitstring. The joint measurement's size (``1 + len(x_wires)``) varies per bitstring, so
        # the worst case (all ``num_target_wires`` flipped) is used for this approximate estimate.
        PauliMeasure("Z", wires=Wire[1]): num_ands,
        PauliMeasure("X" * (num_target_wires + 1), wires=Wire[num_target_wires + 1]): num_ands,
        CZ: num_cz,
        CNOT: L - 1,
        MultiX(Bool[num_target_wires], Wire[num_target_wires]): L,
        X: L + flag.get(X, 0),
        ctrl(X(Wire[1]), control=Wire[1], control_values=Bool[1]): 1,
    }
    # Merge the remaining flag-only resource types (controlled-X load, adjoint ANDs).
    for rep, count in flag.items():
        if rep not in resources:
            resources[rep] = count
    return resources


def _flag_resources(n_extra, num_target_wires):
    """Return the resources for the flag that gates the load on extra control wires.

    A single extra wire uses two X gates; two or more use a ladder of ``n_extra - 1`` AND gates,
    all later uncomputed by the same number of adjoints. In both cases the base load is gated,
    adding up to ``num_target_wires`` controlled-X gates.
    """
    if n_extra < 1:
        return {}
    resources = {ctrl(X(Wire[1]), control=Wire[1]): num_target_wires}
    if n_extra == 1:
        resources[X] = 2
        return resources
    resources[TemporaryAND] = n_extra - 1
    resources[_adjoint_abstract(TemporaryAND)] = n_extra - 1
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

    n_input = (
        num_control_wires if num_control_wires is not None else max(1, ceil_log2(num_bitstrings))
    )
    if num_bitstrings <= 2 and n_input <= 1:
        return True
    return num_work_wires >= n_input - 1


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
    n_extra = len(extra_wires)
    if n_extra == 1:
        X(extra_wires[0])
        return extra_wires[0], work_wires

    anc_work, core_work = work_wires[: n_extra - 1], work_wires[n_extra - 1 :]

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

    # Bitstrings are manipulated with integer bitwise operations (np.bitwise_xor)
    # below, but callers may pass float data (e.g. QROM(np.eye(b), ...)). Cast to
    # int so the XOR-relative encoding works regardless of the input dtype.
    bitstrings = np.asarray(bitstrings).astype(int)

    L = len(bitstrings)
    n_input = len(control_wires)

    # Extra control wires beyond ceil_log2(L) are the most-significant address bits: the data
    # is loaded only when they are all zero, otherwise the operation is the identity (matching
    # the non-partial ``Select``). We build a flag qubit that is 1 iff every extra wire is 0
    # and control the whole load on it, reusing the unary iterator ``_measurement_qrom_inner``
    # over the real 2**n_active table.
    #
    # ``n_extra == 0`` is intentionally handled by the branches below (the 4-quarter outer
    # iterator), which is cheaper than the flag-gated inner iterator used here.
    n_active = ceil_log2(L)
    n_extra = n_input - n_active
    if n_extra > 0:
        extra_wires, active_wires = control_wires[:n_extra], control_wires[n_extra:]

        # Fold the extra wires into a flag that is 1 iff all of them are 0, then run the whole
        # load conditioned on that flag; the flag is uncomputed afterwards so work wires stay clean.
        flag, core_work = _build_flag(extra_wires, work_wires)

        # Gated base load, then the flag-gated unary iterator over the padded 2**n_active table.
        padded = np.zeros((2**n_active, len(bitstrings[0])), dtype=int)
        padded[:L] = bitstrings
        base = padded[0]
        # Fanout the base bitstring onto the target register, controlled on the flag.
        ctrl(MultiX(base, wires=target_wires), control=flag)
        bitstrings = np.bitwise_xor(padded, base)
        controls = _interleave_controls(active_wires[:n_active], core_work, head=flag)
        _measurement_qrom_inner(controls, list(target_wires), bitstrings)

        # Uncompute the flag by inverting the exact gate sequence queued by ``_build_flag``.
        qp_ops.adjoint(_build_flag)(extra_wires, work_wires)
        return

    # TODO: allowing partial qrom will remove this padding
    # Pad data up to the next power of 2 with all-zero bitstrings
    next_pow2 = 1 << ceil_log2(L)
    if L < next_pow2:
        width = len(bitstrings[0])
        bitstrings = np.concatenate([bitstrings, np.zeros((next_pow2 - L, width), dtype=int)])
        L = next_pow2

    if L == 1:
        MultiX(bitstrings[0], target_wires)
        return

    if L == 2:
        MultiX(bitstrings[0], target_wires)
        diff = np.bitwise_xor(bitstrings[0], bitstrings[1])
        ctrl(MultiX(diff, wires=target_wires), control=control_wires[0])
        return

    # Load base bitstring
    MultiX(bitstrings[0], target_wires)

    # Build interleaved controls: [in[0], in[1], work[0], in[2], work[1], ...]
    controls = _interleave_controls(control_wires, work_wires)

    # XOR-relative encoding: bitstrings[i] = bitstrings[i] XOR bitstrings[0]
    bitstrings = np.bitwise_xor(bitstrings, bitstrings[0])

    _measurement_qrom_outer(controls, list(target_wires), bitstrings, L)


def _popcount(x, nbits=40):
    pc = np.int64(0)
    for j in range(nbits):
        pc = pc + ((x >> j) & 1)
    return pc


def _qrom_unary_iteration_condition(
    bitstrings=None, control_wires=None, target_wires=None, work_wires=None, clean=None, base=None
):  # pylint: disable=unused-argument,too-many-arguments
    return len(work_wires) >= len(control_wires) - 1


def _qrom_unary_iteration_resources(
    bitstrings=None, control_wires=None, target_wires=None, work_wires=None, clean=None, base=None
):  # pylint: disable=unused-argument,too-many-arguments
    c = len(control_wires)
    K = len(bitstrings)
    num_target_wires = len(target_wires)

    basis_rep = MultiX(Bool[num_target_wires], Wire[num_target_wires])
    cbasis_rep = ctrl(basis_rep, control=Wire[1])
    if c == 0:
        return {basis_rep: 1}
    if c == 1:
        if K == 1:
            return {cbasis_rep: 1}
        return {cbasis_rep: 1, basis_rep: 1}

    # The number of elbows required for non-partial unary iteration is given by
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
    more_than_half = int(K > 2 ** (c - 1))
    num_elbows = c + K - 2 - (K - 1).bit_count() - more_than_half
    return {
        TemporaryAND: num_elbows,
        _adjoint_abstract(TemporaryAND): num_elbows,
        CNOT: K - 1 + more_than_half,
        X: 2 * int(K > 2 ** (c - 2)),
        cbasis_rep: K,
    }


def _main_unary_loop_monolithic(bitstrings, triples, target_wires):
    K = len(bitstrings)
    c = len(triples) + 1
    # last work wire in use acts as the flag qubit for data loading.
    flag = triples[-1][2]
    assert c >= 2

    TemporaryAND(triples[0], (0, 0))
    for i in range(1, len(triples)):
        TemporaryAND(triples[i], (1, 0))

    # Once resource hints are merged, use those estimates:
    # quarter_prob = int(K > (1 << (c - 2))) / (K - 1)
    # mid_prob = int(K > (1 << (c - 1))) / (K - 1)
    # est_ladder_len = float(
    # np.mean([_popcount(math.bitwise_xor(k, k + 1)) - 1 for k in range(K - 1)])
    # )

    # Loop over all bitstrings but the last one
    @for_loop(K - 1)
    def loop(k):
        # 1. load bitstrings[k], controlled on the flag circuit
        ctrl(MultiX(bitstrings[k], target_wires), control=[flag])

        # 2. transition address k -> k+1
        # a is the MSB-first index of least-significant 0 bit of k
        a = c - _popcount(math.bitwise_xor(k, k + 1))

        # Whether we are in the first half of the iteration, so that the top bit
        # has not been flipped yet
        top_not_flipped = k < (1 << (c - 1))

        # 2a. right-elbow ladder: uncompute levels c-2 .. max(a,1) (top-down)
        # Once resource hints are merged, use those estimates:
        lower_bound = math.max(math.array([a, 1], like=a))

        @for_loop(c - 2, lower_bound - 1, -1)
        # @for_loop(c - 2, max(a - 1, 0), -1, estimated_iterations=est_ladder_len)
        def uncompute(i):
            qp_ops.adjoint(TemporaryAND)(wires=triples[i])

        uncompute()  # pylint: disable=no-value-for-parameter

        # 2b. merge gate(s) at the boundary
        # Once resource hints are merged, use those estimates:
        # cond(math.logical_and(a == 1, top_not_flipped), X, estimated_probability=quarter_prob)(
        #    triples[0][0]
        # )
        cond(math.logical_and(a == 1, top_not_flipped), X)(triples[0][0])
        # cond(a > 0, CNOT, estimated_probability=1 - mid_prob)(triples[a - 1][::2])
        cond(a > 0, CNOT)(triples[a - 1][::2])
        # cond(math.logical_and(a == 1, top_not_flipped), X, estimated_probability=quarter_prob)(
        #    triples[0][0]
        # )
        cond(math.logical_and(a == 1, top_not_flipped), X)(triples[0][0])

        # Once resource hints are merged, use those estimates:
        # cond(a == 0, CNOT, estimated_probability=mid_prob)(triples[0][::2])
        cond(a == 0, CNOT)(triples[0][::2])
        # cond(a == 0, CNOT, estimated_probability=mid_prob)(triples[0][1:])
        cond(a == 0, CNOT)(triples[0][1:])

        # 2c. left-elbow ladder: recompute levels max(a,1) .. c-2 (bottom-up)
        # Once resource hints are merged, use those estimates:
        @for_loop(lower_bound, c - 1)
        # @for_loop(max(a, 1), c - 1, estimated_iterations=est_ladder_len)
        def recompute(i):
            TemporaryAND(triples[i], (1, 0))

        recompute()  # pylint: disable=no-value-for-parameter

    loop()  # pylint: disable=no-value-for-parameter

    # Load last bit string
    ctrl(MultiX(bitstrings[K - 1], target_wires), control=[flag])

    # closing ladder of right elbows for address K-1; control values depend on the bits of K-1
    closing_bits = [(K - 1 >> (c - 1 - b)) & 1 for b in range(c)]
    # levels i=c-2 .. 1 close with cvals (1, closing_bits[i+1]); level 0 closes with
    # cvals closing_bits[:2]
    for i in range(len(triples) - 1, 0, -1):
        qp_ops.adjoint(TemporaryAND(wires=triples[i], control_values=(1, closing_bits[i + 1])))
    qp_ops.adjoint(TemporaryAND(wires=triples[0], control_values=tuple(closing_bits[:2])))


@register_condition(_qrom_unary_iteration_condition)
@register_resources(_qrom_unary_iteration_resources)
def _qrom_unary_iteration(
    bitstrings, control_wires, target_wires, work_wires, clean, **__
):  # pylint: disable=unused-argument, too-many-arguments
    """Unary iteration decomposition of QROM."""
    num_controls = len(control_wires)

    if num_controls == 0:
        # Simply load unique bit string
        MultiX(bitstrings[0], target_wires)
        return

    if num_controls == 1:
        if len(bitstrings) == 1:
            # One bit string to be applied
            ctrl(MultiX(bitstrings[0], target_wires), control=control_wires, control_values=[0])
            return
        # Two bit strings to be applied. Load the first unconditionally and control-load the diff
        MultiX(bitstrings[0], target_wires)
        ctrl(MultiX((bitstrings[0] + bitstrings[1]) % 2, target_wires), control=control_wires)
        return

    # Compute unary iteration wires
    interleaved = _interleave_controls(control_wires, work_wires, None)
    triples = [interleaved[2 * i : 2 * i + 3] for i in range(num_controls - 1)]

    if compiler.active() or capture.enabled():
        bitstrings = math.array(bitstrings, like="jax")
        triples = math.array(triples, like="jax")

    _main_unary_loop_monolithic(bitstrings, triples, target_wires)


add_decomps(QROM, _qrom_decomposition, _qrom_unary_iteration, _qrom_measurement_decomposition)
add_decomps("Adjoint(QROM)", _qrom_measurement_decomposition)
