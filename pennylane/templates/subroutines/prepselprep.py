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

def is_pow2(n):
    """Returns true if n is a power of 2, false otherwise"""
    return ((n & (n - 1) == 0) and n != 0)

class PrepSelPrep(Operation):
    """This class implements a block-encoding of a linear combination of unitaries
    using the Prepare, Select, Prepare method"""

    def __init__(self, lcu, control, jit=False, id=None):
        coeffs, ops = lcu.terms()
        control = qml.wires.Wires(control)
        self.hyperparameters["lcu"] = lcu
        self.hyperparameters["coeffs"] = coeffs
        self.hyperparameters["ops"] = tuple(ops)
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
        return f"PrepSelPrep(coeffs={tuple(self.coeffs)}, ops={self.ops}, control={self.control})"

    def map_wires(self, wire_map: dict) -> "PrepSelPrep":
        new_ops = [o.map_wires(wire_map) for o in self.hyperparameters["ops"]]
        new_control = [wire_map.get(wire, wire) for wire in self.hyperparameters["control"]]
        new_lcu = qml.dot(self.hyperparameters["coeffs"], new_ops)
        return PrepSelPrep(new_lcu, new_control)

    def decomposition(self):
        return self.compute_decomposition(self.lcu, self.control, self.jit)

    @staticmethod
    def preprocess_lcu(lcu):
        """Convert LCU into an equivalent form with positive real coefficients,
        and a power of 2 number of terms"""
        new_coeffs = []
        new_ops = []

        for coeff, op in zip(*lcu.terms()):
            real = qml.math.real(coeff)
            sign = qml.math.sign(real)
            new_coeffs.append(sign * real)
            new_op = qml.ops.LinearCombination([sign], [op])
            new_ops.append(new_op)

            imag = qml.math.imag(coeff)
            sign = qml.math.sign(imag)
            new_coeffs.append(sign * imag)
            new_op = qml.ops.LinearCombination([1j * sign], [op])
            new_ops.append(new_op)

        keep_coeffs = []
        keep_ops = []

        for coeff, op in zip(new_coeffs, new_ops):
            if not qml.math.allclose(op.terms()[0], 0):
                keep_coeffs.append(coeff)
                keep_ops.append(op)

        final_ops = []
        for op in keep_ops:
            if len(op.wires) == 0:
                continue

            unitary = qml.QubitUnitary(qml.matrix(op), wires=op.wires)
            final_ops.append(unitary)

        if is_pow2(len(keep_coeffs)):
            pow2 = len(keep_coeffs)
        else:
            pow2 = 2 ** math.ceil(math.log2(len(keep_coeffs)))

        pad_zeros = list(itertools.repeat(0, pow2 - len(keep_coeffs)))

        interface_coeffs = qml.math.get_interface(lcu.terms()[0])
        final_coeffs = qml.math.array(keep_coeffs + pad_zeros, like=interface_coeffs)

        return final_coeffs, final_ops

    @staticmethod
    def compute_decomposition(lcu, control, jit):
        if jit:
            import jax

            def raiseIfNegative(x):
                if x < 0:
                    raise ValueError("Coefficients must be positive real numbers.")

            coeffs, ops = lcu.terms()

            if not is_pow2(len(coeffs)):
                raise ValueError("Number of terms must be a power of 2.")

            if jax.numpy.iscomplexobj(coeffs):
                raise ValueError("Coefficients must be positive real numbers.")

            smallest = jax.numpy.min(coeffs)
            jax.debug.callback(raiseIfNegative, smallest)


        else:
            coeffs, ops = PrepSelPrep.preprocess_lcu(lcu)

        normalized_coeffs = qml.math.sqrt(coeffs) / qml.math.norm(qml.math.sqrt(coeffs))

        with qml.QueuingManager.stop_recording():
            prep_ops = qml.StatePrep.compute_decomposition(normalized_coeffs, control)
            select_ops = qml.Select.compute_decomposition(ops, control)
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

    @property
    def jit(self):
        """Prevent preprocessing to enable Jax Jit"""
        return self.hyperparameters["jit"]
