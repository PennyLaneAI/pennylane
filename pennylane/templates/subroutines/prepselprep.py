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
# pylint: disable=arguments-differ,import-outside-toplevel,too-many-arguments
import copy

import pennylane as qml
from pennylane.operation import Operation


def get_new_terms(lcu):
    """Compute a new sum of unitaries with positive coefficients"""

    coeffs, _ = lcu.terms()

    if qml.math.iscomplexobj(coeffs):
        new_coeffs, new_ops = new_terms_is_complex(lcu)
    else:
        new_coeffs, new_ops = new_terms_is_real(lcu)

    interface = qml.math.get_interface(lcu.terms()[0])
    new_coeffs = qml.math.array(new_coeffs, like=interface)

    return new_coeffs, new_ops


def new_terms_is_complex(lcu):
    """Computes new terms when the coefficients are complex.
    This doubles the number of terms."""

    new_coeffs = []
    new_ops = []
    for coeff, op in zip(*lcu.terms()):
        real = qml.math.real(coeff)
        imag = qml.math.imag(coeff)

        sign = qml.math.sign(real)
        new_coeffs.append(sign * real)
        new_ops.append(qml.ops.LinearCombination([sign], [op]))

        sign = qml.math.sign(imag)
        new_coeffs.append(sign * imag)
        new_ops.append(qml.ops.LinearCombination([1j * sign], [op]))

    return new_coeffs, new_ops


def new_terms_is_real(lcu):
    """Computes new terms when the coefficients are real.
    This preserves the number of terms."""

    new_coeffs = []
    new_ops = []
    for coeff, op in zip(*lcu.terms()):
        sign = qml.math.sign(coeff)
        new_coeffs.append(sign * coeff)
        new_ops.append(qml.ops.LinearCombination([sign], [op]))

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
    def normalization_factor(lcu):
        """Return the normalization factor lambda such that
        A/lambda is in the upper left of the block encoding"""

        new_coeffs, _ = get_new_terms(lcu)
        return qml.math.sum(new_coeffs)

    @staticmethod
    def preprocess_lcu(lcu):
        """Convert LCU into an equivalent form with positive real coefficients"""

        new_coeffs, new_ops = get_new_terms(lcu)

        new_unitaries = []
        for op in new_ops:
            if len(op.wires) == 0:
                unitary = op
            else:
                unitary = qml.QubitUnitary(qml.matrix(op), wires=op.wires)

            new_unitaries.append(unitary)

        return new_coeffs, new_unitaries

    @staticmethod
    def compute_decomposition(lcu, control):
        coeffs, ops = get_new_terms(lcu)

        decomp_ops = []
        decomp_ops.append(qml.AmplitudeEmbedding(coeffs, normalize=True, pad_with=0, wires=control))
        decomp_ops.append(qml.Select(ops, control))
        decomp_ops.append(
            qml.adjoint(qml.AmplitudeEmbedding(coeffs, normalize=True, pad_with=0, wires=control))
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
