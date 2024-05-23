# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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


import copy
import math

import numpy as np

import pennylane as qml
from pennylane.operation import Operation


class QROM(Operation):
    r"""Applies the QROM operator.

    This operator encodes bitstrings associated with indexes:

    .. math::
        \text{QROM}|i\rangle|0\rangle = |i\rangle |b_i\rangle,

    where :math:`b_i` is the bitstring associated with index :math:`i`.

    Args:
        bitstrings (list[str]): the bitstrings to be encoded
        target_wires (Sequence[int]): the wires where the bitstring is loaded
        control_wires (Sequence[int]): the wires where the indexes are specified
        work_wires (Sequence[int]): the auxiliar wires used for the computation
        clean (bool): if True, the work wires are not altered by operator, default is ``True``

    .. note::

        The position of the bitstrings in the list determines which index store that bitstring.

    **Example**

    In this example the QROM is applied and the target wires are measured to get the third bitstring.

    .. code-block::

        bitstrings = ["010", "111", "110", "000"]

        dev = qml.device("default.qubit", shots = 1)

        @qml.qnode(dev)
        def circuit():

            # third index
            qml.BasisEmbedding(2, wires = [0,1])

            qml.QROM(bitstrings = bitstrings,
                    control_wires = [0,1],
                    target_wires = [2,3,4],
                    work_wires = [5,6,7])

          return qml.sample(wires = [2,3,4])

    .. code-block:: pycon

        >>> print(circuit())
        [1 1 0]


    .. details::
        :title: Usage Details

        The ``work_wires`` are the auxiliary qubits used by the template. If :math:`m` is the number of target wires,
        :math:`m \cdot 2^{\lambda-1}` work wires can be used, where :math:`\lambda` is the number of bitstrings
        we want to store per column. In case it is not a power of two, this template uses the closest approximation.
        See `arXiv:1812.00954 <https://arxiv.org/abs/1812.00954>`__ for more information.

        The version applied by setting ``clean = True`` is able to work with ``work_wires`` that are not initialized to zero.
        In `arXiv:1902.02134 <https://arxiv.org/abs/1902.02134>`__ you could find more details.
    """

    def _flatten(self):
        data = (self.hyperparameters["bitstrings"],)
        metadata = tuple((key, value) for key, value in self.hyperparameters.items() if key != "bitstrings")
        return data, metadata

    @classmethod
    def _unflatten(cls, data, metadata):
        bitstrings = data[0]
        hyperparams_dict = dict(metadata)
        return cls(bitstrings, **hyperparams_dict)

    def __repr__(self):
        return f"QROM(target_wires={self.target_wires}, control_wires={self.control_wires},  work_wires={self.work_wires})"

    def __init__(
        self, bitstrings, target_wires, control_wires, work_wires, clean=True, id=None
    ):  # pylint: disable=too-many-arguments

        control_wires = qml.wires.Wires(control_wires)
        target_wires = qml.wires.Wires(target_wires)

        if work_wires:
            work_wires = qml.wires.Wires(work_wires)

        self.hyperparameters["bitstrings"] = bitstrings
        self.hyperparameters["target_wires"] = target_wires
        self.hyperparameters["control_wires"] = control_wires
        self.hyperparameters["work_wires"] = work_wires
        self.hyperparameters["clean"] = clean

        if work_wires:
            if any(wire in work_wires for wire in control_wires):
                raise ValueError("Control wires should be different from work wires.")

            if any(wire in work_wires for wire in target_wires):
                raise ValueError("Target wires should be different from work wires.")

        if any(wire in control_wires for wire in target_wires):
            raise ValueError("Target wires should be different from control wires.")

        all_wires = target_wires + control_wires + work_wires
        super().__init__(wires=all_wires, id=id)

    def map_wires(self, wire_map: dict):
        new_target_wires = [
            wire_map.get(wire, wire) for wire in self.hyperparameters["target_wires"]
        ]
        new_control_wires = [
            wire_map.get(wire, wire) for wire in self.hyperparameters["control_wires"]
        ]

        new_work_wires = []
        if self.hyperparameters["work_wires"]:
            new_work_wires = [
                wire_map.get(wire, wire) for wire in self.hyperparameters["work_wires"]
            ]

        return QROM(self.bitstrings, new_target_wires, new_control_wires, new_work_wires, self.clean)

    @staticmethod
    def multi_swap(wires1, wires2):
        """Apply a series of SWAP gates between two sets of wires."""
        for wire1, wire2 in zip(wires1, wires2):
            qml.SWAP(wires=[wire1, wire2])

    def __copy__(self):
        """Copy this op"""
        cls = self.__class__
        copied_op = cls.__new__(cls)

        new_data = copy.copy(self.data)

        for attr, value in vars(self).items():
            if attr != "data":
                setattr(copied_op, attr, value)

        copied_op.data = new_data

        return copied_op

    def decomposition(self):  # pylint: disable=arguments-differ

        return self.compute_decomposition(
            self.bitstrings,
            control_wires=self.control_wires,
            work_wires=self.work_wires,
            target_wires=self.target_wires,
            clean=self.clean,
        )

    @staticmethod
    def compute_decomposition(
        bitstrings, target_wires, control_wires, work_wires, clean
    ):  # pylint: disable=arguments-differ
        with qml.QueuingManager.stop_recording():

            if work_wires:
                swap_wires = target_wires + work_wires  # wires available to apply the operators
            else:
                swap_wires = target_wires

            # number of operators we store per column (power of 2)
            depth = len(swap_wires) // len(target_wires)
            depth = int(2 ** np.floor(np.log2(depth)))

            # control wires used in the Select block
            c_sel_wires = control_wires[
                : int(math.ceil(math.log2(2 ** len(control_wires) / depth)))
            ]

            # control wires used in the Swap block
            c_swap_wires = control_wires[len(c_sel_wires) :]

            ops = [qml.BasisEmbedding(int(bits, 2), wires=target_wires) for bits in bitstrings]
            ops_I = ops + [qml.I(target_wires)] * int(2 ** len(control_wires) - len(ops))

            # number of new operators after grouping
            length = len(ops) // depth if len(ops) % depth == 0 else len(ops) // depth + 1

            # operators to be included in the Select block
            s_ops = []

            for i in range(length):

                # map the wires to put more than one bitstring per column.
                s_ops.append(
                    qml.prod(
                        *[
                            qml.map_wires(
                                ops_I[i * depth + j],
                                {
                                    ops_I[i * depth + j].wires[l]: swap_wires[
                                        j * len(target_wires) + l
                                    ]
                                    for l in range(len(target_wires))
                                },
                            )
                            for j in range(depth)
                        ]
                    )
                )

            # Select block
            sel_ops = []
            if c_sel_wires:
                sel_ops += [qml.Select(s_ops, control=c_sel_wires)]

            # Swap block
            swap_ops = []
            for ind, wire in enumerate(c_swap_wires):
                for j in range(2**ind):
                    new_op = qml.prod(QROM.multi_swap)(
                        swap_wires[(j) * len(target_wires) : (j + 1) * len(target_wires)],
                        swap_wires[
                            (j + 2**ind)
                            * len(target_wires) : (j + 2 ** (ind + 1))
                            * len(target_wires)
                        ],
                    )
                    swap_ops.append(qml.ctrl(new_op, control=wire))

            if not clean:
                # Based on this paper: https://arxiv.org/abs/1812.00954
                decomp_ops = sel_ops + swap_ops

            else:
                # Based on this paper: https://arxiv.org/abs/1902.02134

                # Adjoint Swap block
                adjoint_swap_ops = swap_ops[::-1]

                decomp_ops = []

                for _ in range(2):

                    for w in target_wires:
                        decomp_ops.append(qml.Hadamard(wires=w))

                    decomp_ops += swap_ops + sel_ops + adjoint_swap_ops

        if qml.QueuingManager.recording():
            for op in decomp_ops:
                qml.apply(op)

        return decomp_ops

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
        """Boolean that choose the version ussed."""
        return self.hyperparameters["clean"]
