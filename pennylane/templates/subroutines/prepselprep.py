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
# pylint: disable=arguments-differ,import-outside-toplevel
import copy
import itertools
import math

import pennylane as qml
from pennylane.operation import Operation

def is_pow2(n):
    """Returns true if n is a power of 2, false otherwise"""
    return ((n & (n - 1) == 0) and n != 0)

def normalize(n):
    return qml.math.sqrt(n) / qml.math.norm(qml.math.sqrt(n))

class PrepSelPrep(Operation):
    """This class implements a block-encoding of a linear combination of unitaries
    using the Prepare, Select, Prepare method"""

    def __init__(self, lcu, control=None, jit=False, id=None):
        coeffs, ops = lcu.terms()
        control = qml.wires.Wires(control)
        self.hyperparameters["lcu"] = lcu
        self.hyperparameters["coeffs"] = coeffs
        self.hyperparameters["ops"] = ops
        self.hyperparameters["control"] = control
        self.hyperparameters["jit"] = jit

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
        new_lcu = qml.dot(self.hyperparameters["coeffs"], new_ops)
        return PrepSelPrep(new_lcu, new_control)

    def decomposition(self):
        return self.compute_decomposition(self.lcu, self.control, self.jit)

    @staticmethod
    def get_new_terms(lcu):
        """Compute a new sum of unitaries with positive coefficients"""

        new_coeffs = []
        new_ops = []

        for coeff, op in zip(*lcu.terms()):
            if qml.math.allclose(coeff, 0):
                new_coeffs.append(coeff)
                new_ops.append(op)
                continue

            real = qml.math.real(coeff)
            if not qml.math.allclose(real, 0):
                sign = qml.math.sign(real)
                new_coeffs.append(sign*real)
                new_ops.append(sign*op)

            imag = qml.math.imag(coeff)
            if not qml.math.allclose(imag, 0):
                sign = qml.math.sign(imag)
                new_coeffs.append(sign*imag)
                new_ops.append(1j*sign*op)

        return new_coeffs, new_ops

    @staticmethod
    def normalization_factor(lcu):
        """Return the normalization factor lambda such that
        A/lambda is in the upper left of the block encoding"""

        new_coeffs, _ = PrepSelPrep.get_new_terms(lcu)
        return qml.math.sum(new_coeffs)

    @staticmethod
    def preprocess_lcu(lcu):
        """Convert LCU into an equivalent form with positive real coefficients,
        and a power of 2 number of terms"""

        new_coeffs, new_ops = PrepSelPrep.get_new_terms(lcu)

        all_op_wires = qml.wires.Wires.all_wires([op.wires for op in new_ops])
        final_ops = []
        for op in new_ops:
            if len(op.wires) == 0:
                unitary = op
            else:
                unitary = qml.QubitUnitary(qml.matrix(op), wires=op.wires)

            final_ops.append(unitary)

        if is_pow2(len(new_coeffs)):
            pow2 = len(new_coeffs)
        else:
            pow2 = 2 ** math.ceil(math.log2(len(new_coeffs)))

        pad_zeros = list(itertools.repeat(0, pow2 - len(new_coeffs)))
        with qml.QueuingManager.stop_recording():
            pad_ident = list(itertools.repeat(qml.Identity(all_op_wires), pow2-len(new_ops)))

        interface_coeffs = qml.math.get_interface(lcu.terms()[0])
        final_coeffs = qml.math.array(new_coeffs + pad_zeros, like=interface_coeffs)
        final_ops = final_ops + pad_ident

        if not qml.math.allclose(qml.math.norm(final_coeffs), 1):
            final_coeffs = qml.math.sqrt(final_coeffs) / qml.math.norm(qml.math.sqrt(final_coeffs))

        return final_coeffs, final_ops

    @staticmethod
    def compute_decomposition(lcu, control, jit):
        if jit:
            import jax

            def raiseIfNegative(x):
                if x < 0:
                    raise ValueError("Coefficients must be positive real numbers.")

            def raiseIfNotOne(x):
                if not qml.math.allclose(x, 1):
                    raise ValueError("Coefficients must have norm 1.")

            coeffs, ops = lcu.terms()

            if not is_pow2(len(coeffs)):
                raise ValueError("Number of terms must be a power of 2.")

            if jax.numpy.iscomplexobj(coeffs):
                raise ValueError("Coefficients must be positive real numbers.")

            smallest = jax.numpy.min(coeffs)
            jax.debug.callback(raiseIfNegative, smallest)
            jax.debug.callback(raiseIfNotOne, qml.math.norm(coeffs))

        else:
            coeffs, ops = PrepSelPrep.preprocess_lcu(lcu)

        with qml.QueuingManager.stop_recording():
            prep_ops = qml.StatePrep.compute_decomposition(coeffs, control)
            select_ops = qml.Select.compute_decomposition(ops, control)
            adjoint_prep_ops = qml.adjoint(
                qml.StatePrep(coeffs, control)
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
        return self.lcu.data

    @data.setter
    def data(self, new_data):
        """Set the data property"""
        self.hyperparameters['lcu'].data = new_data

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

    @property
    def jit(self):
        """Prevent preprocessing to enable Jax Jit"""
        return self.hyperparameters["jit"]
