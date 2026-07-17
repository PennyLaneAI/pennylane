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
from functools import reduce

import numpy as np

from pennylane import capture, compiler, math
from pennylane import ops as qp_ops
from pennylane.control_flow import for_loop
from pennylane.core.operator import Operation
from pennylane.core.queuing import QueuingManager, apply
from pennylane.decomposition import (
    add_decomps,
    controlled_resource_rep,
    register_condition,
    register_resources,
    resource_rep,
)
from pennylane.math import ceil_log2
from pennylane.ops import CNOT, CZ, BasisState, X, cond, ctrl, pauli_measure
from pennylane.ops.mid_measure.pauli_measure import PauliMeasure
from pennylane.ops.op_math.adjoint2 import _adjoint_abstract
from pennylane.templates.embeddings import BasisEmbedding
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike

from .arithmetic import TemporaryAND
from .select import Select


def _multi_swap(wires1, wires2):
    """Apply a series of SWAP gates between two sets of wires."""
    for wire1, wire2 in zip(wires1, wires2, strict=True):
        qp_ops.SWAP(wires=[wire1, wire2])


def _new_ops(depth, target_wires, capacity, swap_wires, data):

    with QueuingManager.stop_recording():
        ops_new = [BasisEmbedding(bits, wires=target_wires) for bits in data]
        ops_identity_new = ops_new + [qp_ops.I(target_wires)] * (capacity - len(ops_new))

    n_columns = data.shape[0] // depth if data.shape[0] % depth == 0 else data.shape[0] // depth + 1
    new_ops = []
    for i in range(n_columns):
        column_ops = []
        for j in range(depth):
            dic_map = {
                target_wires[l]: swap_wires[j * len(target_wires) + l]
                for l in range(len(target_wires))
            }
            column_ops.append(ops_identity_new[i * depth + j].map_wires(dic_map))
        new_ops.append(qp_ops.prod(*column_ops))
    return new_ops


def _new_data(depth, capacity, swap_wires, data):
    num_data, num_bits = data.shape
    # Pad with zeros to fill up to max-capacity bitstrings
    data = math.concatenate([data, np.zeros((capacity - num_data, num_bits), dtype=data.dtype)])
    num_data = capacity
    assert len(data) == num_data  # TMP

    # depth is min(a power of two, num_bits), so it can happen that num_data%depth != 0
    new_num_data = num_data // depth + int(num_data % depth > 0)
    new_num_bits = len(swap_wires)
    assert new_num_bits == num_bits * depth  # TMP
    new_data = math.zeros((new_num_data, new_num_bits), dtype=data.dtype)
    # TODO: vectorize
    for i in range(new_num_data):
        for j in range(depth):
            new_data[i, j * num_bits : (j + 1) * num_bits] = data[i * depth + j]
    return new_data


def _select_ops(
    control_wires, depth, target_wires, swap_wires, data, select_work_wires
):  # pylint:disable=too-many-arguments
    capacity = 1 << len(control_wires)
    n_control_select_wires = ceil_log2(capacity / depth)
    control_select_wires = control_wires[:n_control_select_wires]
    print(f"{capacity=}")
    print(f"{len(control_wires)=}")
    print(f"{depth=}")
    print(f"{n_control_select_wires=}")

    if control_select_wires:
        if len(select_work_wires) < n_control_select_wires - 1:
            Select(
                _new_ops(depth, target_wires, capacity, swap_wires, data),
                control=control_select_wires,
                work_wires=select_work_wires,
            )
        else:
            print(f"{data.shape=}")
            print(f"{_new_data(depth, capacity, swap_wires, data).shape}")
            QROM(
                _new_data(depth, capacity, swap_wires, data),
                control_wires=control_select_wires,
                target_wires=swap_wires,
                work_wires=select_work_wires,
            )

    else:
        _new_ops(depth, target_wires, capacity, swap_wires, data)


def _swap_ops(control_wires, depth, swap_wires, target_wires):
    n_control_select_wires = ceil_log2(2 ** len(control_wires) / depth)
    control_swap_wires = control_wires[n_control_select_wires:]
    num_targets = len(target_wires)
    for i in range(len(control_swap_wires) - 1, -1, -1):
        for j in range(2**i - 1, -1, -1):
            _wires0 = swap_wires[j * num_targets : (j + 1) * num_targets]
            _wires1 = swap_wires[(j + 2**i) * num_targets : (j + 2**i + 1) * num_targets]
            qp_ops.ctrl(_multi_swap, control=control_swap_wires[-i - 1])(_wires0, _wires1)


class QROM(Operation):
    r"""Applies the QROM operator.

    This operator encodes bitstrings associated with indexes:

    .. math::
        \text{QROM}|i\rangle|0\rangle = |i\rangle |b_i\rangle,

    where :math:`b_i` is the bitstring associated with index :math:`i`.

    Args:
        data (TensorLike): the data to be encoded
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
        data = [[0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0]]

        dev = qp.device("default.qubit")

        @qp.qnode(dev, shots=1)
        def circuit():

            # the third index is encoded in the control wires [0, 1]
            qp.BasisEmbedding(2, wires = [0,1])

            qp.QROM(data = data,
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
        For instance, if the data is ``[0, 1, 1, 0]``, we will need four target wires. Internally,
        the bitstrings are encoded using the :class:`~.BasisEmbedding` template.


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

    """

    resource_keys = {
        "num_bitstrings",
        "num_control_wires",
        "num_target_wires",
        "num_work_wires",
        "clean",
    }

    def __init__(
        self,
        data: TensorLike | Sequence[str],
        control_wires: WiresLike,
        target_wires: WiresLike,
        work_wires: WiresLike,
        clean=True,
    ):  # pylint: disable=too-many-arguments,disable=too-many-positional-arguments

        control_wires = Wires(control_wires)
        target_wires = Wires(target_wires)

        if isinstance(data[0], str):
            data = np.array(list(map(lambda bitstring: [int(bit) for bit in bitstring], data)))

        if isinstance(data, (list, tuple)):
            data = math.array(data)

        work_wires = Wires(() if work_wires is None else work_wires)

        self.hyperparameters["control_wires"] = control_wires
        self.hyperparameters["target_wires"] = target_wires
        self.hyperparameters["work_wires"] = work_wires
        self.hyperparameters["clean"] = clean

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

        if 2 ** len(control_wires) < data.shape[0]:
            raise ValueError(
                f"Not enough control wires ({len(control_wires)}) for the desired number of "
                + f"data ({data.shape[0]}). At least {ceil_log2(data.shape[0])} control "
                + "wires are required."
            )

        if data[0].shape[0] != len(target_wires):
            raise ValueError("Bitstring length must match the number of target wires.")

        all_wires = target_wires + control_wires + work_wires
        super().__init__(data, wires=all_wires)

    def _flatten(self):
        metadata = tuple((key, value) for key, value in self.hyperparameters.items())
        return tuple(self.data), metadata

    @property
    def resource_params(self) -> dict:
        return {
            "num_bitstrings": self.data[0].shape[0],
            "num_control_wires": len(self.hyperparameters["control_wires"]),
            "num_target_wires": len(self.hyperparameters["target_wires"]),
            "num_work_wires": len(self.hyperparameters["work_wires"]),
            "clean": self.hyperparameters["clean"],
        }

    @classmethod
    def _unflatten(cls, data, metadata):
        hyperparams_dict = dict(metadata)
        return cls(*data, **hyperparams_dict)

    def __repr__(self):
        return f"QROM(control_wires={self.control_wires}, target_wires={self.target_wires},  work_wires={self.work_wires}, clean={self.clean})"

    def map_wires(self, wire_map: dict):
        new_dict = {
            key: [wire_map.get(w, w) for w in self.hyperparameters[key]]
            for key in ["target_wires", "control_wires", "work_wires"]
        }

        return QROM(
            self.data[0],
            new_dict["control_wires"],
            new_dict["target_wires"],
            new_dict["work_wires"],
            self.clean,
        )

    def __copy__(self):
        """Copy this op"""
        cls = self.__class__
        copied_op = cls.__new__(cls)

        for attr, value in vars(self).items():
            setattr(copied_op, attr, value)

        return copied_op

    def decomposition(self):

        return self.compute_decomposition(
            self.data[0],
            control_wires=self.control_wires,
            target_wires=self.target_wires,
            work_wires=self.work_wires,
            clean=self.clean,
        )

    @staticmethod
    def compute_decomposition(
        data, control_wires, target_wires, work_wires, clean
    ):  # pylint: disable=arguments-differ

        if len(control_wires) == 0:
            return [BasisEmbedding(bits, wires=target_wires) for bits in data]

        with QueuingManager.stop_recording():
            n_select_work_wires = _calculate_n_select_work_wires(
                len(data), len(control_wires), len(target_wires), len(work_wires)
            )
            n_swap_work_wires = len(work_wires) - n_select_work_wires
            swap_work_wires = work_wires[:n_swap_work_wires]
            select_work_wires = work_wires[n_swap_work_wires:]
            swap_wires = target_wires + swap_work_wires

            # number of operators we store per column (power of 2)
            depth = len(swap_wires) // len(target_wires)
            depth = 1 << math.floor_log2(depth)
            depth = min(depth, data.shape[0])

            ops = [BasisEmbedding(bits, wires=target_wires) for bits in data]
            ops_identity = ops + [qp_ops.I(target_wires)] * int(2 ** len(control_wires) - len(ops))

            n_columns = len(ops) // depth + int(bool(len(ops) % depth))
            new_ops = []
            for i in range(n_columns):
                column_ops = []
                for j in range(depth):
                    dic_map = {
                        ops_identity[i * depth + j].wires[l]: swap_wires[j * len(target_wires) + l]
                        for l in range(len(target_wires))
                    }
                    column_ops.append(ops_identity[i * depth + j].map_wires(dic_map))
                new_ops.append(qp_ops.prod(*column_ops))

            # Select block
            n_control_select_wires = ceil_log2(2 ** len(control_wires) / depth)
            control_select_wires = control_wires[:n_control_select_wires]

            select_ops = []
            if control_select_wires:
                select_ops += [
                    Select(new_ops, control=control_select_wires, work_wires=select_work_wires)
                ]
            else:
                select_ops = new_ops

            # Swap block
            control_swap_wires = control_wires[n_control_select_wires:]
            swap_ops = []
            num_targets = len(target_wires)
            for ind in range(len(control_swap_wires)):
                for j in range(2**ind):
                    _wires0 = swap_wires[j * num_targets : (j + 1) * num_targets]
                    _wires1 = swap_wires[
                        (j + 2**ind) * num_targets : (j + 2**ind + 1) * num_targets
                    ]
                    new_op = qp_ops.prod(_multi_swap)(_wires0, _wires1)
                    swap_ops.insert(0, qp_ops.ctrl(new_op, control=control_swap_wires[-ind - 1]))

            if not clean or depth == 1:
                # Based on this paper (Fig 1.c): https://arxiv.org/abs/1812.00954
                decomp_ops = select_ops + swap_ops

            else:
                # Based on this paper (Fig 4): https://arxiv.org/abs/1902.02134
                adjoint_swap_ops = swap_ops[::-1]
                hadamard_ops = [qp_ops.Hadamard(wires=w) for w in target_wires]

                decomp_ops = 2 * (hadamard_ops + adjoint_swap_ops + select_ops + swap_ops)

        if QueuingManager.recording():
            for op in decomp_ops:
                apply(op)

        return decomp_ops

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @property
    def control_wires(self):
        """The control wires."""
        return self.hyperparameters["control_wires"]

    @property
    def target_wires(self):
        """The wires where the bitstring is loaded."""
        return self.hyperparameters["target_wires"]

    @property
    def work_wires(self):
        """The wires where the index is specified."""
        return self.hyperparameters["work_wires"]

    @property
    def wires(self):
        """All wires involved in the operation."""
        return (
            self.hyperparameters["control_wires"]
            + self.hyperparameters["target_wires"]
            + self.hyperparameters["work_wires"]
        )

    @property
    def clean(self):
        """Boolean to select the version of QROM."""
        return self.hyperparameters["clean"]


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
    num_bitstrings, num_control_wires, num_target_wires, num_work_wires, clean
):  # pylint: disable=too-many-branches

    num_work_wires_select = _calculate_n_select_work_wires(
        num_bitstrings, num_control_wires, num_target_wires, num_work_wires
    )

    num_work_wires_swap = num_work_wires - num_work_wires_select

    if num_control_wires == 0:
        return {resource_rep(BasisEmbedding, num_wires=num_target_wires): num_bitstrings}

    num_swap_wires = num_target_wires + num_work_wires_swap

    # number of operators we store per column (power of 2)
    depth = num_swap_wires // num_target_wires
    depth = 1 << math.floor_log2(depth)
    depth = min(depth, num_bitstrings)

    ops = [resource_rep(BasisEmbedding, num_wires=num_target_wires) for _ in range(num_bitstrings)]
    ops_identity = ops + [qp_ops.I] * int(2**num_control_wires - num_bitstrings)

    n_columns = (
        num_bitstrings // depth if num_bitstrings % depth == 0 else num_bitstrings // depth + 1
    )
    # Select block
    num_control_select_wires = ceil_log2(2**num_control_wires / depth)

    if num_control_select_wires > 0 and num_work_wires_select >= num_control_select_wires - 1:
        select_ops = {
            resource_rep(
                QROM,
                num_control_wires=num_control_select_wires,
                num_target_wires=num_swap_wires,
                num_work_wires=num_work_wires_select,
                num_bitstrings=n_columns,
                clean=True,
            ): 1
        }
    else:
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
    data, control_wires, target_wires, work_wires, clean, **__
):  # pylint: disable=unused-argument, too-many-arguments
    if len(control_wires) == 0:
        BasisEmbedding(data[0, :], wires=target_wires)

    n_select_work_wires = _calculate_n_select_work_wires(
        len(data), len(control_wires), len(target_wires), len(work_wires)
    )

    select_work_wires = work_wires[:n_select_work_wires]
    swap_work_wires = work_wires[n_select_work_wires:]
    swap_wires = target_wires + swap_work_wires

    # number of operators we store per column (power of 2)
    depth = len(swap_wires) // len(target_wires)
    depth = 1 << math.floor_log2(depth)
    depth = min(depth, data.shape[0])

    if not clean or depth == 1:
        _select_ops(control_wires, depth, target_wires, swap_wires, data, select_work_wires)
        if not clean:
            _swap_ops(control_wires, depth, swap_wires, target_wires)

    else:
        for _ in range(2):
            for w in target_wires:
                qp_ops.Hadamard(wires=w)
            qp_ops.adjoint(_swap_ops, lazy=False)(control_wires, depth, swap_wires, target_wires)
            _select_ops(control_wires, depth, target_wires, swap_wires, data, select_work_wires)
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
    cond(m2 == 1, BasisState)(state=product, wires=targets)


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


def _qrom_measurement_resources(num_bitstrings=None, num_target_wires=None, base_params=None, **_):
    """Resource estimate for the measurement-based QROM decomposition.

    Each TemporaryAND is uncomputed via _measurement_uncompute which produces:
      - 2 PauliMeasure (one X-type joint measurement, one Z measurement)
      - 1 CZ (phase correction conditioned on X measurement)
      - conditional X gates on work + targets
    """
    # When called for Adjoint(QROM), extract params from the base parameters
    if base_params is not None:
        num_bitstrings = base_params["num_bitstrings"]
        num_target_wires = base_params["num_target_wires"]

    # L = num_bitstrings
    # TODO: allowing partial QROM will reduce this term
    L = 2 ** ceil_log2(num_bitstrings)

    if L <= 1:
        return {resource_rep(BasisState, num_wires=num_target_wires): 1}

    if L == 2:
        return {
            resource_rep(BasisState, num_wires=num_target_wires): 1,
            resource_rep(CNOT): num_target_wires,
        }

    num_ands = _count_tempAND_in_measurement_qrom(L)
    num_measurements = 2 * num_ands  # X-type + Z per uncomputation
    num_cz = num_ands  # CZ correction per uncomputation

    # TemporaryAND counts are exact
    # CNOTs, PauliX gates and BasisState ops are an approximation
    return {
        resource_rep(TemporaryAND): num_ands,
        resource_rep(PauliMeasure): num_measurements,
        resource_rep(CZ): num_cz,
        resource_rep(CNOT): L - 1,
        resource_rep(BasisState, num_wires=num_target_wires): L,
        resource_rep(X): L,
        controlled_resource_rep(X, {}, num_control_wires=1, num_zero_control_values=1): 1,
    }


def _qrom_measurement_condition(
    num_bitstrings=None, num_control_wires=None, num_work_wires=None, base_params=None, **_
):  # pylint: disable=unused-argument

    if not compiler.active():
        return False
    if base_params is not None:
        num_bitstrings = base_params["num_bitstrings"]
        num_work_wires = base_params["num_work_wires"]

    n_input = max(1, ceil_log2(num_bitstrings))
    if num_bitstrings <= 2:
        return True
    return num_work_wires >= n_input - 1


@register_condition(_qrom_measurement_condition)
@register_resources(_qrom_measurement_resources, exact=False)
def _qrom_measurement_decomposition(  # pylint: disable=too-many-arguments
    data=None, control_wires=None, target_wires=None, work_wires=None, base=None, **_
):
    """QROM decomposition using measurement-based uncomputation.

    Uses L-3 (or L-2) TemporaryAND gates. All uncomputation is done via
    PauliMeasure + conditional corrections instead of adjoint(TemporaryAND).
    Work wires are always left clean (via measurement-based uncomputation).
    Decomposition is based on Fig 18. https://arxiv.org/abs/2211.15465

    Requires: len(work_wires) >= ceil_log2(L) - 1.
    """
    # When called for Adjoint(QROM), extract params from the base operator
    if base is not None:
        data = base.data[0]
        control_wires = base.hyperparameters["control_wires"]
        target_wires = base.hyperparameters["target_wires"]
        work_wires = base.hyperparameters["work_wires"]

    # Bitstrings are manipulated with integer bitwise operations (np.bitwise_xor)
    # below, but callers may pass float data (e.g. QROM(np.eye(b), ...)). Cast to
    # int so the XOR-relative encoding works regardless of the input dtype.
    data = np.asarray(data).astype(int)

    L = len(data)
    n_input = len(control_wires)

    # TODO: allowing partial qrom will remove this padding
    # Pad data up to the next power of 2 with all-zero bitstrings
    next_pow2 = 1 << ceil_log2(L)
    if L < next_pow2:
        width = len(data[0])
        data = np.concatenate([data, np.zeros((next_pow2 - L, width), dtype=int)])
        L = next_pow2

    if L == 1:
        BasisState(data[0], target_wires)
        return

    if L == 2:
        BasisState(data[0], target_wires)
        diff = np.bitwise_xor(data[0], data[1])
        for i, bit in enumerate(diff):
            if bit == 1:
                CNOT(wires=[control_wires[0], target_wires[i]])
        return

    # Load base bitstring
    BasisState(data[0], target_wires)

    # Build interleaved controls: [in[0], in[1], work[0], in[2], work[1], ...]
    controls = [control_wires[0]]
    for i in range(n_input - 1):
        controls.append(control_wires[i + 1])
        controls.append(work_wires[i])

    # XOR-relative encoding: bitstrings[i] = data[i] XOR data[0]
    base = list(data[0])
    bitstrings = [np.bitwise_xor(row, base) for row in data]

    _measurement_qrom_outer(controls, list(target_wires), bitstrings, L)


def _temporary_and_triples(control: WiresLike, work: WiresLike) -> list[list]:
    """Create a list of wire triples as used by temporary AND ladders,
    by interleaving control and work wires and slicing them into triples that overlap on one wire.

    Args:
        control (WiresLike): Control wire register containing ``c`` wires.
        work (WiresLike): Work wire register containing at least ``c-1`` wires.

    Returns:
        list[list]: List with ``c-1`` elements, where each element is a three-element list.

    """
    aux = [control[0]] + [
        _wires[i] for i in range(len(control) - 1) for _wires in [control[1:], work]
    ]
    return [aux[2 * i : 2 * i + 3] for i in range(len(control) - 1)]


def _popcount(x, nbits=40):
    pc = np.int64(0)
    for j in range(nbits):
        pc = pc + ((x >> j) & 1)
    return pc


def _qrom_unary_iteration_condition(num_control_wires=None, num_work_wires=None, **_):
    return num_work_wires >= num_control_wires - 1


def _qrom_unary_iteration_resources(
    num_bitstrings=None, num_control_wires=None, num_target_wires=None, **_
):
    c = num_control_wires
    K = num_bitstrings

    basis_rep = resource_rep(BasisState, num_wires=num_target_wires)
    cbasis_rep = controlled_resource_rep(
        BasisState, base_params={"num_wires": num_target_wires}, num_control_wires=1
    )
    if c == 0:
        return {basis_rep: 1}
    if c == 1:
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
    num_elbows = c + K - 2 - (K - 1).bit_count() - int(K > 2 ** (c - 1))
    return {
        TemporaryAND: num_elbows,
        _adjoint_abstract(TemporaryAND): num_elbows,
        CNOT: K - 1 + int(K > 2 ** (c - 1)),
        X: 2 * int(K > 2 ** (c - 2)),
        cbasis_rep: K,
    }


def _main_unary_loop(data, triples, target_wires):
    K = len(data)
    c = len(triples) + 1
    # last work wire in use acts as the flag qubit for data loading.
    flag = triples[-1][2]

    # Loop over all data but the last one
    @for_loop(K - 1)
    def loop(k):
        # 1. load data[k], controlled on the flag circuit
        qp_ops.ctrl(BasisState(data[k], target_wires), control=[flag])

        # 2. transition address k -> k+1
        # a is the MSB-first index of least-significant 0 bit of k
        a = c - _popcount(math.bitwise_xor(k, k + 1))

        # Whether we are in the first half of the iteration, so that the top bit
        # has not been flipped yet
        top_not_flipped = k < (1 << (c - 1))

        # 2a. right-elbow ladder: uncompute levels c-2 .. max(a,1) (top-down)
        for i in range(c - 2, 0, -1):
            qp_ops.adjoint(cond(i >= a, TemporaryAND))(wires=triples[i])

        # 2b. merge gate(s) at the boundary
        cond(math.logical_and(a == 1, top_not_flipped), X)(triples[0][0])
        cond(a > 0, CNOT)(triples[a - 1][::2])
        cond(math.logical_and(a == 1, top_not_flipped), X)(triples[0][0])

        cond(a == 0, CNOT)(triples[0][::2])
        cond(a == 0, CNOT)(triples[0][1:])
        # 2c. left-elbow ladder: recompute levels max(a,1) .. c-2 (bottom-up)
        for i in range(1, c - 1):
            cond(i >= a, TemporaryAND)(triples[i], (1, 0))

    loop()  # pylint: disable=no-value-for-parameter

    # Load last bit string
    qp_ops.ctrl(BasisState(data[K - 1], target_wires), control=[flag])


@register_condition(_qrom_unary_iteration_condition)
@register_resources(_qrom_unary_iteration_resources)
def _qrom_unary_iteration(
    data, control_wires, target_wires, work_wires, clean, **__
):  # pylint: disable=unused-argument, too-many-arguments
    """Unary iteration decomposition of QROM."""
    K = len(data)
    c = len(control_wires)

    if c == 0:
        # Simply load unique bit string
        BasisState(data[0], target_wires)
        return

    if c == 1:
        # Two bit strings to be applies. Load the first unconditionally and control-load the diff
        BasisState(data[0], target_wires)
        qp_ops.ctrl(BasisState((data[0] + data[1]) % 2, target_wires), control=control_wires)
        return

    # Compute unary iteration wires
    triples = _temporary_and_triples(control_wires, work_wires)

    if compiler.active() or capture.enabled():
        data = math.array(data, like="jax")
        triples = math.array(triples, like="jax")

    # initial ladder of left elbows (open at level 0 with cvals (0,0))
    TemporaryAND(triples[0], (0, 0))
    for i in range(1, c - 1):
        TemporaryAND(triples[i], (1, 0))

    _main_unary_loop(data, triples, target_wires)

    # closing ladder of right elbows for address K-1; control values depend on the bits of K-1
    closing_bits = [(K - 1 >> (c - 1 - b)) & 1 for b in range(c)]
    # levels i=c-2 .. 1 close with cvals (1, closing_bits[i+1]); level 0 closes with
    # cvals closing_bits[:2]
    for i in range(c - 2, 0, -1):
        qp_ops.adjoint(TemporaryAND(wires=triples[i], control_values=(1, closing_bits[i + 1])))
    qp_ops.adjoint(TemporaryAND(wires=triples[0], control_values=tuple(closing_bits[:2])))


add_decomps(QROM, _qrom_unary_iteration, _qrom_decomposition, _qrom_measurement_decomposition)
add_decomps("Adjoint(QROM)", _qrom_measurement_decomposition)
