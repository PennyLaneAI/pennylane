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
        data = (self.hyperparameters["b"],)
        metadata = tuple((key, value) for key, value in self.hyperparameters.items() if key != "b")
        return data, metadata

    @classmethod
    def _unflatten(cls, data, metadata):
        b = data[0]
        hyperparams_dict = dict(metadata)
        return cls(b, **hyperparams_dict)

    def __repr__(self):
        return f"QROM(control_wires={self.control_wires}, target_wires={self.target_wires}  work_wires={self.work_wires})"

    def __init__(self, b, target_wires, control_wires, work_wires, clean=True, id=None):

        control_wires = qml.wires.Wires(control_wires)
        target_wires = qml.wires.Wires(target_wires)

        if work_wires:
            work_wires = qml.wires.Wires(work_wires)

        self.hyperparameters["b"] = b
        self.hyperparameters["target_wires"] = target_wires
        self.hyperparameters["control_wires"] = control_wires
        self.hyperparameters["work_wires"] = work_wires
        self.hyperparameters["clean"] = clean

        if 2 ** len(control_wires) < len(b):
            raise ValueError(
                f"Not enough control wires ({len(control_wires)}) for the desired number of "
                + f"operations ({len(b)}). At least {int(math.ceil(math.log2(len(b))))} control "
                + "wires required."
            )

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
        if self.hyperparameters["work_wires"]:
            new_work_wires = [
                wire_map.get(wire, wire) for wire in self.hyperparameters["work_wires"]
            ]
        else:
            new_work_wires = []
        return QROM(self.b, new_target_wires, new_control_wires, new_work_wires, self.clean)

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
            self.b,
            control_wires=self.control_wires,
            work_wires=self.work_wires,
            target_wires=self.target_wires,
            clean=self.clean,
        )

    @staticmethod
    def compute_decomposition(b, target_wires, control_wires, work_wires, clean):
        # BasisEmbedding applied to embed the bitstrings
        with qml.QueuingManager.stop_recording():
            ops = [qml.BasisEmbedding(int(bits, 2), wires=target_wires) for bits in b]

            if work_wires:
                swap_wires = target_wires + work_wires  # wires available to apply the operators
            else:
                swap_wires = target_wires
            depth = len(swap_wires) // len(
                target_wires
            )  # number of operators we store per column (power of 2)
            depth = int(2 ** np.floor(np.log2(depth)))

            c_sel_wires = control_wires[
                : int(math.ceil(math.log2(2 ** len(control_wires) / depth)))
            ]  # control wires used in the Select block
            c_swap_wires = control_wires[len(c_sel_wires) :]  # control wires used in the Swap block

            ops_I = ops + [qml.I(target_wires)] * int(2 ** len(control_wires) - len(ops))

            decomp_ops = []

            length = len(ops) // depth if len(ops) % depth == 0 else len(ops) // depth + 1

            s_ops = []
            for i in range(length):
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

            sel_ops = [qml.Select(s_ops, control=control_wires[: len(c_sel_wires)])]

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

            adjoint_swap_ops = swap_ops[::-1]

            if not clean:
                # Based on this paper: https://arxiv.org/pdf/1812.00954
                decomp_ops += sel_ops + swap_ops

            else:
                # Based on this paper: https://arxiv.org/abs/1902.02134

                for _ in range(2):

                    for w in target_wires:
                        decomp_ops.append(qml.Hadamard(wires=w))

                    decomp_ops += swap_ops + sel_ops + adjoint_swap_ops

        if qml.QueuingManager.recording():
            for op in decomp_ops:
                qml.apply(op)

        return decomp_ops

    @property
    def b(self):
        """bitstrings to be added."""
        return self.hyperparameters["b"]

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
