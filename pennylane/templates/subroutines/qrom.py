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

import math
from collections import Counter
from functools import reduce

import numpy as np

from pennylane import ops as qml_ops
from pennylane.decomposition import (
    add_decomps,
    controlled_resource_rep,
    register_resources,
    resource_rep,
)
from pennylane.operation import Operation
from pennylane.queuing import QueuingManager, apply
from pennylane.templates.embeddings import BasisEmbedding
from pennylane.wires import Wires, WiresLike

from .select import Select


def _multi_swap(wires1, wires2):
    """Apply a series of SWAP gates between two sets of wires."""
    for wire1, wire2 in zip(wires1, wires2):
        qml_ops.SWAP(wires=[wire1, wire2])


def _new_ops(depth, target_wires, control_wires, swap_wires, bitstrings):

    with QueuingManager.stop_recording():
        ops_new = [BasisEmbedding(int(bits, 2), wires=target_wires) for bits in bitstrings]
        ops_identity_new = ops_new + [qml_ops.I(target_wires)] * int(
            2 ** len(control_wires) - len(ops_new)
        )

    n_columns = (
        len(bitstrings) // depth if len(bitstrings) % depth == 0 else len(bitstrings) // depth + 1
    )
    new_ops = []
    for i in range(n_columns):
        column_ops = []
        for j in range(depth):
            dic_map = {
                ops_identity_new[i * depth + j].wires[l]: swap_wires[j * len(target_wires) + l]
                for l in range(len(target_wires))
            }
            column_ops.append(ops_identity_new[i * depth + j].map_wires(dic_map))
        new_ops.append(qml_ops.prod(*column_ops))
    return new_ops


def _select_ops(control_wires, depth, target_wires, swap_wires, bitstrings):
    n_control_select_wires = int(math.ceil(math.log2(2 ** len(control_wires) / depth)))
    control_select_wires = control_wires[:n_control_select_wires]

    if control_select_wires:
        Select(
            _new_ops(depth, target_wires, control_wires, swap_wires, bitstrings),
            control=control_select_wires,
        )
    else:
        _new_ops(depth, target_wires, control_wires, swap_wires, bitstrings)


def _swap_ops(control_wires, depth, swap_wires, target_wires):
    n_control_select_wires = int(math.ceil(math.log2(2 ** len(control_wires) / depth)))
    control_swap_wires = control_wires[n_control_select_wires:]
    for ind in range(len(control_swap_wires)):
        for j in range(2**ind):
            new_op = qml_ops.prod(_multi_swap)(
                swap_wires[(j) * len(target_wires) : (j + 1) * len(target_wires)],
                swap_wires[
                    (j + 2**ind) * len(target_wires) : (j + 2 ** (ind + 1)) * len(target_wires)
                ],
            )
            qml_ops.ctrl(new_op, control=control_swap_wires[-ind - 1])


class QROM(Operation):
    r"""Applies the QROM operator.

    This operator encodes bitstrings associated with indexes:

    .. math::
        \text{QROM}|i\rangle|0\rangle = |i\rangle |b_i\rangle,

    where :math:`b_i` is the bitstring associated with index :math:`i`.

    Args:
        bitstrings (list[str]): the bitstrings to be encoded
        control_wires (Sequence[int]): the wires where the indexes are specified
        target_wires (Sequence[int]): the wires where the bitstring is loaded
        work_wires (Sequence[int]): the auxiliary wires used for the computation
        clean (bool): if True, the work wires are not altered by operator, default is ``True``

    **Example**

    In this example, the QROM operator is applied to encode the third bitstring, associated with index 2, in the target wires.

    .. code-block:: python

        # a list of bitstrings is defined
        bitstrings = ["010", "111", "110", "000"]

        dev = qml.device("default.qubit")

        @qml.qnode(dev, shots=1)
        def circuit():

            # the third index is encoded in the control wires [0, 1]
            qml.BasisEmbedding(2, wires = [0,1])

            qml.QROM(bitstrings = bitstrings,
                    control_wires = [0,1],
                    target_wires = [2,3,4],
                    work_wires = [5,6,7])

            return qml.sample(wires = [2,3,4])

    >>> print(circuit())
    [[1 1 0]]


    .. details::
        :title: Usage Details

        This template takes as input three different sets of wires. The first one is ``control_wires`` which is used
        to encode the desired index. Therefore, if we have :math:`m` bitstrings, we need
        at least :math:`\lceil \log_2(m)\rceil` control wires.

        The second set of wires is ``target_wires`` which stores the bitstrings.
        For instance, if the bitstring is "0110", we will need four target wires. Internally, the bitstrings are
        encoded using the :class:`~.BasisEmbedding` template.


        The ``work_wires`` are the auxiliary qubits used by the template to reduce the number of gates required.
        Let :math:`k` be the number of work wires. If :math:`k = 0`, the template is equivalent to executing :class:`~.Select`.
        Following the idea in [`arXiv:1812.00954 <https://arxiv.org/abs/1812.00954>`__], auxiliary qubits can be used to
        load more than one bitstring in parallel. Let :math:`\lambda` be
        the number of bitstrings we want to store in parallel, assumed to be a power of :math:`2`.
        Then, :math:`k = l \cdot (\lambda-1)` work wires are needed,
        where :math:`l` is the length of the bitstrings.

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
        bitstrings,
        control_wires: WiresLike,
        target_wires: WiresLike,
        work_wires: WiresLike,
        clean=True,
        id=None,
    ):  # pylint: disable=too-many-arguments,disable=too-many-positional-arguments

        control_wires = Wires(control_wires)
        target_wires = Wires(target_wires)

        work_wires = Wires(() if work_wires is None else work_wires)

        self.hyperparameters["bitstrings"] = tuple(bitstrings)
        self.hyperparameters["control_wires"] = control_wires
        self.hyperparameters["target_wires"] = target_wires
        self.hyperparameters["work_wires"] = work_wires
        self.hyperparameters["clean"] = clean

        if len(work_wires) != 0:
            if any(wire in work_wires for wire in control_wires):
                raise ValueError("Control wires should be different from work wires.")

            if any(wire in work_wires for wire in target_wires):
                raise ValueError("Target wires should be different from work wires.")

        if any(wire in control_wires for wire in target_wires):
            raise ValueError("Target wires should be different from control wires.")

        if 2 ** len(control_wires) < len(bitstrings):
            raise ValueError(
                f"Not enough control wires ({len(control_wires)}) for the desired number of "
                + f"bitstrings ({len(bitstrings)}). At least {int(math.ceil(math.log2(len(bitstrings))))} control "
                + "wires are required."
            )

        if len(bitstrings[0]) != len(target_wires):
            raise ValueError("Bitstring length must match the number of target wires.")

        all_wires = target_wires + control_wires + work_wires
        super().__init__(wires=all_wires, id=id)

    def _flatten(self):
        metadata = tuple((key, value) for key, value in self.hyperparameters.items())
        return tuple(), metadata

    @property
    def resource_params(self) -> dict:
        return {
            "num_bitstrings": len(self.hyperparameters["bitstrings"]),
            "num_control_wires": len(self.hyperparameters["control_wires"]),
            "num_target_wires": len(self.hyperparameters["target_wires"]),
            "num_work_wires": len(self.hyperparameters["work_wires"]),
            "clean": self.hyperparameters["clean"],
        }

    @classmethod
    def _unflatten(cls, data, metadata):
        hyperparams_dict = dict(metadata)
        return cls(**hyperparams_dict)

    def __repr__(self):
        return f"QROM(control_wires={self.control_wires}, target_wires={self.target_wires},  work_wires={self.work_wires}, clean={self.clean})"

    def map_wires(self, wire_map: dict):
        new_dict = {
            key: [wire_map.get(w, w) for w in self.hyperparameters[key]]
            for key in ["target_wires", "control_wires", "work_wires"]
        }

        return QROM(
            self.bitstrings,
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
            self.bitstrings,
            control_wires=self.control_wires,
            target_wires=self.target_wires,
            work_wires=self.work_wires,
            clean=self.clean,
        )

    @staticmethod
    def compute_decomposition(
        bitstrings, control_wires, target_wires, work_wires, clean
    ):  # pylint: disable=arguments-differ

        if len(control_wires) == 0:
            return [BasisEmbedding(int(bits, 2), wires=target_wires) for bits in bitstrings]

        with QueuingManager.stop_recording():

            swap_wires = target_wires + work_wires

            # number of operators we store per column (power of 2)
            depth = len(swap_wires) // len(target_wires)
            depth = int(2 ** np.floor(np.log2(depth)))
            depth = min(depth, len(bitstrings))

            ops = [BasisEmbedding(int(bits, 2), wires=target_wires) for bits in bitstrings]
            ops_identity = ops + [qml_ops.I(target_wires)] * int(2 ** len(control_wires) - len(ops))

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
                new_ops.append(qml_ops.prod(*column_ops))

            # Select block
            n_control_select_wires = int(math.ceil(math.log2(2 ** len(control_wires) / depth)))
            control_select_wires = control_wires[:n_control_select_wires]

            select_ops = []
            if control_select_wires:
                select_ops += [Select(new_ops, control=control_select_wires)]
            else:
                select_ops = new_ops

            # Swap block
            control_swap_wires = control_wires[n_control_select_wires:]
            swap_ops = []
            for ind in range(len(control_swap_wires)):
                for j in range(2**ind):
                    new_op = qml_ops.prod(_multi_swap)(
                        swap_wires[(j) * len(target_wires) : (j + 1) * len(target_wires)],
                        swap_wires[
                            (j + 2**ind)
                            * len(target_wires) : (j + 2 ** (ind + 1))
                            * len(target_wires)
                        ],
                    )
                    swap_ops.insert(0, qml_ops.ctrl(new_op, control=control_swap_wires[-ind - 1]))

            if not clean or depth == 1:
                # Based on this paper (Fig 1.c): https://arxiv.org/abs/1812.00954
                decomp_ops = select_ops + swap_ops

            else:
                # Based on this paper (Fig 4): https://arxiv.org/abs/1902.02134
                adjoint_swap_ops = swap_ops[::-1]
                hadamard_ops = [qml_ops.Hadamard(wires=w) for w in target_wires]

                decomp_ops = 2 * (hadamard_ops + adjoint_swap_ops + select_ops + swap_ops)

        if QueuingManager.recording():
            for op in decomp_ops:
                apply(op)

        return decomp_ops

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @property
    def bitstrings(self):
        """bitstrings to be added."""
        return self.hyperparameters["bitstrings"]

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


def _qrom_decomposition_resources(
    num_bitstrings, num_control_wires, num_target_wires, num_work_wires, clean
):  # pylint: disable=too-many-branches
    if num_control_wires == 0:
        return {resource_rep(BasisEmbedding, num_wires=num_target_wires): num_bitstrings}

    num_swap_wires = num_target_wires + num_work_wires

    # number of operators we store per column (power of 2)
    depth = num_swap_wires // num_target_wires
    depth = int(2 ** np.floor(np.log2(depth)))
    depth = min(depth, num_bitstrings)

    ops = [resource_rep(BasisEmbedding, num_wires=num_target_wires) for _ in range(num_bitstrings)]
    ops_identity = ops + [qml_ops.I] * int(2**num_control_wires - num_bitstrings)

    n_columns = (
        num_bitstrings // depth if num_bitstrings % depth == 0 else num_bitstrings // depth + 1
    )

    # New ops block
    new_ops = Counter()
    for i in range(n_columns):
        column_ops = Counter()
        for j in range(depth):
            column_ops[ops_identity[i * depth + j]] += 1
        if len(column_ops) == 1 and list(column_ops.values())[0] == 1:
            new_ops[list(column_ops.keys())[0]] += 1
        else:
            new_ops[resource_rep(qml_ops.op_math.Prod, resources=dict(column_ops))] += 1

    # Select block
    num_control_select_wires = int(math.ceil(math.log2(2**num_control_wires / depth)))

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
                num_work_wires=0,
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
                swaps = {resource_rep(qml_ops.SWAP): num_swaps}
                swap_resources[
                    controlled_resource_rep(
                        base_class=qml_ops.op_math.Prod,
                        base_params={"resources": swaps},
                        num_control_wires=1,
                    )
                ] += 1
            else:
                swap_resources[
                    controlled_resource_rep(
                        base_class=qml_ops.SWAP,
                        base_params={},
                        num_control_wires=1,
                    )
                ] += 1

    if not clean or depth == 1:
        resources = swap_resources
        resources.update(select_ops)
        return resources

    resources = {}

    hadamard_ops = {qml_ops.Hadamard: num_target_wires}

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
    wires, bitstrings, control_wires, target_wires, work_wires, clean
):  # pylint: disable=unused-argument, too-many-arguments
    if len(control_wires) == 0:
        for bits in bitstrings:
            BasisEmbedding(int(bits, 2), wires=target_wires)
        return

    swap_wires = target_wires + work_wires

    # number of operators we store per column (power of 2)
    depth = len(swap_wires) // len(target_wires)
    depth = int(2 ** np.floor(np.log2(depth)))
    depth = min(depth, len(bitstrings))

    if not clean or depth == 1:
        _select_ops(control_wires, depth, target_wires, swap_wires, bitstrings)
        _swap_ops(control_wires, depth, swap_wires, target_wires)

    else:
        for _ in range(2):
            for w in target_wires:
                qml_ops.Hadamard(wires=w)
            _swap_ops(control_wires, depth, swap_wires, target_wires)
            _select_ops(control_wires, depth, target_wires, swap_wires, bitstrings)
            _swap_ops(control_wires, depth, swap_wires, target_wires)


add_decomps(QROM, _qrom_decomposition)
