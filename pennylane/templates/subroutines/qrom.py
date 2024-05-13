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

import pennylane as qml
from pennylane.operation import Operation
import numpy as np
import itertools
import math
import copy


class QROM(Operation):

    def _flatten(self):
        return (self.ops), (self.control)

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(data, metadata)

    def __repr__(self):
        return f"QROM(ops={self.ops}, control={self.control}, work_wires={self.work_wires})"

    def __init__(self, bitstrings, target_wires, control_wires, work_wires, clean=True, id=None):

        control_wires = qml.wires.Wires(control_wires)
        target_wires = qml.wires.Wires(target_wires)
        work_wires = qml.wires.Wires(work_wires)

        self.hyperparameters["bitstrings"] = bitstrings
        self.hyperparameters["target_wires"] = target_wires
        self.hyperparameters["control_wires"] = control_wires
        self.hyperparameters["work_wires"] = work_wires
        self.hyperparameters["clean"] = clean

        if 2 ** len(control_wires) < len(bitstrings):
            raise ValueError(
                f"Not enough control wires ({len(control_wires)}) for the desired number of "
                + f"operations ({len(bitstrings)}). At least {int(math.ceil(math.log2(len(bitstrings))))} control "
                + "wires required."
            )

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
        new_work_wires = [wire_map.get(wire, wire) for wire in self.hyperparameters["work_wires"]]
        return QROM(
            self.bitstrings, new_target_wires, new_control_wires, new_work_wires, self.clean
        )

    @staticmethod
    def multi_swap(wires1, wires2):
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

    def decomposition(self):

        return self.compute_decomposition(
            self.bitstrings,
            control_wires=self.control_wires,
            work_wires=self.work_wires,
            target_wires=self.target_wires,
            clean=self.clean,
        )

    @staticmethod
    def compute_decomposition(bitstrings, target_wires, control_wires, work_wires, clean):
        # BasisEmbedding applied to embed the bitstrings
        ops = [qml.BasisEmbedding(int(bits, 2), wires=target_wires) for bits in bitstrings]

        swap_wires = target_wires + work_wires  # wires available to apply the operators
        depth = len(swap_wires) // len(target_wires)  # number of operators we store per column (power of 2)
        depth = int(2 ** np.floor(np.log2(depth)))

        c_sel_wires = control_wires[
            : int(math.ceil(math.log2(2 ** len(control_wires) / depth)))
        ]  # control wires used in the Select block
        c_swap_wires = control_wires[len(c_sel_wires) :]  # control wires used in the Swap block

        # with qml.QueuingManager.stop_recording():
        ops_I = ops + [qml.I(target_wires)] * int(2 ** len(control_wires) - len(ops))

        decomp_ops = []

        # Select block
        sel_ops = []
        bin_indx = list(itertools.product([0, 1], repeat=len(c_sel_wires)))
        length = len(ops) // depth if len(ops) % depth == 0 else len(ops) // depth + 1
        for i, bin in enumerate(bin_indx[:length]):
            new_op = qml.prod(
                *[
                    qml.map_wires(
                        ops_I[i * depth + j],
                        {
                            ops_I[i * depth + j].wires[l]: swap_wires[j * len(target_wires) + l]
                            for l in range(len(target_wires))
                        },
                    )
                    for j in range(depth)
                ]
            )
            sel_ops.append(
                qml.ctrl(new_op, control=control_wires[: len(c_sel_wires)], control_values=bin)
            )

        # Swap block
        # Todo: This should be optimized
        swap_ops = []
        bin_indx = list(itertools.product([0, 1], repeat=len(c_swap_wires)))
        for i, bin in enumerate(bin_indx[1:depth]):
            new_op = qml.prod(QROM.multi_swap)(
                target_wires, swap_wires[(i + 1) * len(target_wires) : (i + 2) * len(target_wires)]
            )
            swap_ops.append(qml.ctrl(new_op, control=c_swap_wires, control_values=bin))

        if not clean:
            # Based on this paper: https://arxiv.org/pdf/1812.00954
            decomp_ops += sel_ops + swap_ops

        else:
            # Based on this paper: https://arxiv.org/abs/1902.02134

            for _ in range(2):

                for w in target_wires:
                    decomp_ops.append(qml.Hadamard(wires=w))

                # TODO: It should be adjoint the last one
                decomp_ops += swap_ops + sel_ops + swap_ops

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
