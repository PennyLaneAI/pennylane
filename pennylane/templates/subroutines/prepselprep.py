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
Contains the PrepSelPrep template.
This template contains a decomposition for performing a block-encoding on a
linear combination of unitaries using the Prepare, Select, Prepare method.
"""
# pylint: disable=arguments-differ,import-outside-toplevel,too-many-arguments
import copy

import pennylane as qml
from pennylane.operation import Operation


def _get_new_terms(lcu):
    """Compute a new sum of unitaries with positive coefficients"""

    new_coeffs = []
    new_ops = []

    for coeff, op in zip(*lcu.terms()):

        angle = qml.math.angle(coeff)
        new_coeffs.append(qml.math.abs(coeff))

        new_op = op @ qml.GlobalPhase(-angle, wires=op.wires)
        new_ops.append(new_op)

    interface = qml.math.get_interface(lcu.terms()[0])
    new_coeffs = qml.math.array(new_coeffs, like=interface)

    return new_coeffs, new_ops


class PrepSelPrep(Operation):
    """This class implements a block-encoding of a linear combination of unitaries
    using the Prepare, Select, Prepare method"""

    def __init__(self, lcu, control=None, id=None):
        coeffs, ops = lcu.terms()
        control = qml.wires.Wires(control)
        self.hyperparameters["lcu"] = lcu
        self.hyperparameters["coeffs"] = coeffs
        self.hyperparameters["ops"] = ops
        self.hyperparameters["control"] = control

        if any(
            control_wire in qml.wires.Wires.all_wires([op.wires for op in ops])
            for control_wire in control
        ):
            raise ValueError("Control wires should be different from operation wires.")

        target_wires = qml.wires.Wires.all_wires([op.wires for op in ops])
        self.hyperparameters["target_wires"] = target_wires

        all_wires = target_wires + control
        super().__init__(*self.data, wires=all_wires, id=id)

    def _flatten(self):
        return tuple(self.lcu), (self.control)

    @classmethod
    def _unflatten(cls, data, metadata) -> "PrepSelPrep":
        coeffs = [term.terms()[0][0] for term in data]
        ops = [term.terms()[1][0] for term in data]
        lcu = qml.ops.LinearCombination(coeffs, ops)
        return cls(lcu, metadata)

    def __repr__(self):
        return f"PrepSelPrep(coeffs={tuple(self.coeffs)}, ops={tuple(self.ops)}, control={self.control})"

    def map_wires(self, wire_map: dict) -> "PrepSelPrep":
        new_ops = [o.map_wires(wire_map) for o in self.hyperparameters["ops"]]
        new_control = [wire_map.get(wire, wire) for wire in self.hyperparameters["control"]]
        new_lcu = qml.ops.LinearCombination(self.hyperparameters["coeffs"], new_ops)
        return PrepSelPrep(new_lcu, new_control)

    def decomposition(self):
        return self.compute_decomposition(self.lcu, self.control)

    @staticmethod
    def compute_decomposition(lcu, control):
        coeffs, ops = _get_new_terms(lcu)

        decomp_ops = []
        decomp_ops.append(
            qml.AmplitudeEmbedding(qml.math.sqrt(coeffs), normalize=True, pad_with=0, wires=control)
        )
        decomp_ops.append(qml.Select(ops, control))
        decomp_ops.append(
            qml.adjoint(
                qml.AmplitudeEmbedding(
                    qml.math.sqrt(coeffs), normalize=True, pad_with=0, wires=control
                )
            )
        )

        return decomp_ops

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

    @property
    def data(self):
        """Create data property"""
        return self.lcu.data

    @data.setter
    def data(self, new_data):
        """Set the data property"""
        self.hyperparameters["lcu"].data = new_data

    @property
    def coeffs(self):
        """The coefficients of the LCU."""
        return self.hyperparameters["coeffs"]

    @property
    def ops(self):
        """The operations of the LCU."""
        return self.hyperparameters["ops"]

    @property
    def lcu(self):
        """The LCU to be block-encoded."""
        return self.hyperparameters["lcu"]

    @property
    def control(self):
        """The control wires."""
        return self.hyperparameters["control"]

    @property
    def target_wires(self):
        """The wires of the input operators."""
        return self.hyperparameters["target_wires"]

    @property
    def wires(self):
        """All wires involved in the operation."""
        return self.hyperparameters["control"] + self.hyperparameters["target_wires"]
