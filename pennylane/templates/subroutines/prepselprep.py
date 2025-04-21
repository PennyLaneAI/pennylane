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
"""
# pylint: disable=arguments-differ,import-outside-toplevel,too-many-arguments
import copy

import pennylane as qml
from pennylane.operation import Operation


def _get_new_terms(lcu):
    """Compute a new sum of unitaries with positive coefficients"""
    coeffs, ops = lcu.terms()
    coeffs = qml.math.stack(coeffs)
    angles = qml.math.angle(coeffs)
    new_ops = []

    for angle, op in zip(angles, ops):
        new_op = op @ qml.GlobalPhase(-angle, wires=op.wires)
        new_ops.append(new_op)

    return qml.math.abs(coeffs), new_ops


class PrepSelPrep(Operation):
    """Implements a block-encoding of a linear combination of unitaries.

    .. warning::
        Derivatives of this operator are not always guaranteed to exist.

    Args:
        lcu (Union[.Hamiltonian, .Sum, .Prod, .SProd, .LinearCombination]): The operator
            written as a linear combination of unitaries.
        control (Iterable[Any], Wires): The control qubits for the PrepSelPrep operator.

    **Example**

    We define an operator and a block-encoding circuit as:

    >>> lcu = qml.dot([0.3, -0.1], [qml.X(2), qml.Z(2)])
    >>> control = [0, 1]
    >>> @qml.qnode(qml.device("default.qubit"))
    ... def circuit(lcu, control):
    ...     qml.PrepSelPrep(lcu, control)
    ...     return qml.state()

    We can see that the operator matrix, up to a normalization constant, is block encoded in the
    circuit matrix:

    >>> matrix_psp = qml.matrix(circuit, wire_order = [0, 1, 2])(lcu, control = control)
    >>> print(matrix_psp.real[0:2, 0:2])
    [[-0.25  0.75]
     [ 0.75  0.25]]

    >>> matrix_lcu = qml.matrix(lcu)
    >>> print(qml.matrix(lcu).real / sum(abs(np.array(lcu.terms()[0]))))
    [[-0.25  0.75]
     [ 0.75  0.25]]
    """

    grad_method = None

    def __init__(self, lcu, control=None, id=None):

        coeffs, ops = lcu.terms()
        control = qml.wires.Wires(control)
        self.hyperparameters["lcu"] = qml.ops.LinearCombination(coeffs, ops)
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
        return (self.lcu,), (self.control,)

    @classmethod
    def _unflatten(cls, data, metadata) -> "PrepSelPrep":
        return cls(data[0], metadata[0])

    def __repr__(self):
        return f"PrepSelPrep(coeffs={tuple(self.coeffs)}, ops={tuple(self.ops)}, control={self.control})"

    def map_wires(self, wire_map: dict) -> "PrepSelPrep":
        new_ops = [o.map_wires(wire_map) for o in self.hyperparameters["ops"]]
        new_control = [wire_map.get(wire, wire) for wire in self.hyperparameters["control"]]
        new_lcu = qml.ops.LinearCombination(self.hyperparameters["coeffs"], new_ops)
        return PrepSelPrep(new_lcu, new_control)

    def decomposition(self):
        return self.compute_decomposition(self.lcu, self.control)

    def label(self, decimals=None, base_label=None, cache=None) -> str:
        op_label = base_label or self.__class__.__name__
        if cache is None or not isinstance(cache.get("matrices", None), list):
            return op_label if self._id is None else f'{op_label}("{self._id}")'

        coeffs = qml.math.array(self.coeffs)
        shape = qml.math.shape(coeffs)
        for i, mat in enumerate(cache["matrices"]):
            if shape == qml.math.shape(mat) and qml.math.allclose(coeffs, mat):
                str_wo_id = f"{op_label}(M{i})"
                break
        else:
            mat_num = len(cache["matrices"])
            cache["matrices"].append(coeffs)
            str_wo_id = f"{op_label}(M{mat_num})"

        return str_wo_id if self._id is None else f'{str_wo_id[:-1]},"{self._id}")'

    @staticmethod
    def compute_decomposition(lcu, control):
        coeffs, ops = _get_new_terms(lcu)

        decomp_ops = [
            qml.AmplitudeEmbedding(
                qml.math.sqrt(coeffs), normalize=True, pad_with=0, wires=control
            ),
            qml.Select(ops, control),
            qml.adjoint(
                qml.AmplitudeEmbedding(
                    qml.math.sqrt(coeffs), normalize=True, pad_with=0, wires=control
                )
            ),
        ]

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
