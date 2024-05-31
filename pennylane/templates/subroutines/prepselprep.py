# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
# pylint: disable=arguments-differ
import copy
import itertools
import math

import pennylane as qml
from pennylane.operation import Operation


class PrepSelPrep(Operation):
    """This class implements a block-encoding of a linear combination of unitaries
    using the Prepare, Select, Prepare method"""

    def __init__(self, lcu, control, id=None):
        coeffs, ops = lcu.terms()
        control = qml.wires.Wires(control)
        self.hyperparameters["lcu"] = lcu
        self.hyperparameters["coeffs"] = coeffs
        self.hyperparameters["ops"] = tuple(ops)
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
        return f"PrepSelPrep(coeffs={tuple(self.coeffs)}, ops={self.ops}, control={self.control})"

    def map_wires(self, wire_map: dict) -> "PrepSelPrep":
        new_ops = [o.map_wires(wire_map) for o in self.hyperparameters["ops"]]
        new_control = [wire_map.get(wire, wire) for wire in self.hyperparameters["control"]]
        new_lcu = qml.dot(self.hyperparameters["coeffs"], new_ops)
        return PrepSelPrep(new_lcu, new_control)

    def decomposition(self):
        return self.compute_decomposition(self.lcu, self.control)

    @staticmethod
    def compute_decomposition(lcu, control):
        new_coeffs = []
        new_ops = []
        for coeff, op in zip(*lcu.terms()):
            if qml.math.iscomplex(coeff):
                real = qml.math.real(coeff)
                if real < 0:
                    new_coeffs.append((-1)*real)
                    new_ops.append((-1)*op)
                if real > 0:
                    new_coeffs.append(real)
                    new_ops.append(op)

                imag = qml.math.imag(coeff)
                if imag < 0:
                    new_coeffs.append((-1)*imag)
                    new_ops.append((-1j)*op)
                if imag > 0:
                    new_coeffs.append(imag)
                    new_ops.append((1j)*op)
            else:
                if coeff < 0:
                    new_coeffs.append((-1)*coeff)
                    new_ops.append((-1)*op)
                else:
                    new_coeffs.append(coeff)
                    new_ops.append(op)

        if (len(new_coeffs) & (len(new_coeffs)-1) == 0) and len(new_coeffs) != 0:
            pow2 = len(new_coeffs)
        else:
            pow2 = 2**math.ceil(math.log2(len(new_coeffs)))

        pad_zeros = list(itertools.repeat(0, pow2 - len(new_coeffs)))

        new_coeffs = new_coeffs + pad_zeros
        normalized_coeffs = qml.math.sqrt(new_coeffs) / qml.math.norm(qml.math.sqrt(new_coeffs))

        with qml.QueuingManager.stop_recording():
            prep_ops = qml.StatePrep.compute_decomposition(normalized_coeffs, control)
            select_ops = qml.Select.compute_decomposition(new_ops, control)
            adjoint_prep_ops = qml.adjoint(
                qml.StatePrep(normalized_coeffs, control)
            ).decomposition()

        ops = prep_ops + select_ops + adjoint_prep_ops

        for op in ops:
            if qml.QueuingManager.recording():
                qml.apply(op)

        return ops

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
        return tuple(d for op in self.ops for d in op.data)

    @data.setter
    def data(self, new_data):
        """Set the data property"""
        for op in self.ops:
            op_num_params = op.num_params
            if op_num_params > 0:
                op.data = new_data[:op_num_params]
                new_data = new_data[op_num_params:]

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
