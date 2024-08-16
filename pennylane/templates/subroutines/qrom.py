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

import math

import numpy as np

import pennylane as qml
from pennylane.operation import Operation


def _multi_swap(wires1, wires2):
    """Apply a series of SWAP gates between two sets of wires."""
    for wire1, wire2 in zip(wires1, wires2):
        qml.SWAP(wires=[wire1, wire2])


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

    .. code-block::

        # a list of bitstrings is defined
        bitstrings = ["010", "111", "110", "000"]

        dev = qml.device("default.qubit", shots = 1)

        @qml.qnode(dev)
        def circuit():

            # the third index is encoded in the control wires [0, 1]
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

        This template takes as input three different sets of wires. The first one is ``control_wires`` which is used
        to encode the desired index. Therefore, if we have :math:`m` bitstrings, we need
        at least :math:`\lceil \log_2(m)\rceil` control wires.

        The second set of wires is ``target_wires`` which stores the bitstrings.
        For instance, if the bitstring is "0110", we will need four target wires. Internally, the bitstrings are
        encoded using the :class:`~.BasisEmbedding` template.


        The ``work_wires`` are the auxiliary qubits used by the template to reduce the number of gates required.
        Let :math:`k` be the number of work wires. If :math:`k = 0`, the template is equivalent to executing :class:`~.Select`.
        Following the idea in [`arXiv:1812.00954 <https://arxiv.org/abs/1812.00954>`__], auxiliary qubits can be used to
        load more than one bitstring in parallel . Let :math:`\lambda` be
        the number of bitstrings we want to store in parallel, assumed to be a power of :math:`2`.
        Then, :math:`k = l \cdot (\lambda-1)` work wires are needed,
        where :math:`l` is the length of the bitstrings.

        The QROM template has two variants. The first one (``clean = False``) is based on [`arXiv:1812.00954 <https://arxiv.org/abs/1812.00954>`__] that alterates the state in the ``work_wires``.
        The second one (``clean = True``), based on [`arXiv:1902.02134 <https://arxiv.org/abs/1902.02134>`__], solves that issue by
        returning ``work_wires`` to their initial state. This technique can be applied when the ``work_wires`` are not
        initialized to zero.

    """

    def __init__(
        self, bitstrings, control_wires, target_wires, work_wires, clean=True, id=None
    ):  # pylint: disable=too-many-arguments

        control_wires = qml.wires.Wires(control_wires)
        target_wires = qml.wires.Wires(target_wires)

        work_wires = qml.wires.Wires(work_wires) if work_wires else qml.wires.Wires([])

        self.hyperparameters["bitstrings"] = tuple(bitstrings)
        self.hyperparameters["control_wires"] = control_wires
        self.hyperparameters["target_wires"] = target_wires
        self.hyperparameters["work_wires"] = work_wires
        self.hyperparameters["clean"] = clean

        if work_wires:
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

    def decomposition(self):  # pylint: disable=arguments-differ

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
        with qml.QueuingManager.stop_recording():

            swap_wires = target_wires + work_wires

            # number of operators we store per column (power of 2)
            depth = len(swap_wires) // len(target_wires)
            depth = int(2 ** np.floor(np.log2(depth)))

            ops = [qml.BasisEmbedding(int(bits, 2), wires=target_wires) for bits in bitstrings]
            ops_identity = ops + [qml.I(target_wires)] * int(2 ** len(control_wires) - len(ops))

            n_columns = len(ops) // depth if len(ops) % depth == 0 else len(ops) // depth + 1
            new_ops = []
            for i in range(n_columns):
                column_ops = []
                for j in range(depth):
                    dic_map = {
                        ops_identity[i * depth + j].wires[l]: swap_wires[j * len(target_wires) + l]
                        for l in range(len(target_wires))
                    }
                    column_ops.append(qml.map_wires(ops_identity[i * depth + j], dic_map))
                new_ops.append(qml.prod(*column_ops))

            # Select block
            n_control_select_wires = int(math.ceil(math.log2(2 ** len(control_wires) / depth)))
            control_select_wires = control_wires[:n_control_select_wires]

            select_ops = []
            if control_select_wires:
                select_ops += [qml.Select(new_ops, control=control_select_wires)]
            else:
                select_ops = new_ops

            # Swap block
            control_swap_wires = control_wires[n_control_select_wires:]
            swap_ops = []
            for ind in range(len(control_swap_wires)):
                for j in range(2**ind):
                    new_op = qml.prod(_multi_swap)(
                        swap_wires[(j) * len(target_wires) : (j + 1) * len(target_wires)],
                        swap_wires[
                            (j + 2**ind)
                            * len(target_wires) : (j + 2 ** (ind + 1))
                            * len(target_wires)
                        ],
                    )
                    swap_ops.insert(0, qml.ctrl(new_op, control=control_swap_wires[-ind - 1]))

            if not clean:
                # Based on this paper (Fig 1.c): https://arxiv.org/abs/1812.00954
                decomp_ops = select_ops + swap_ops

            else:
                # Based on this paper (Fig 4): https://arxiv.org/abs/1902.02134
                adjoint_swap_ops = swap_ops[::-1]
                hadamard_ops = [qml.Hadamard(wires=w) for w in target_wires]

                decomp_ops = 2 * (hadamard_ops + adjoint_swap_ops + select_ops + swap_ops)

        if qml.QueuingManager.recording():
            for op in decomp_ops:
                qml.apply(op)

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
