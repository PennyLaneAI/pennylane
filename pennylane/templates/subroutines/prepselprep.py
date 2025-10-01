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
# pylint: disable=arguments-differ
import copy

from pennylane import math
from pennylane.decomposition import (
    add_decomps,
    change_op_basis_resource_rep,
    register_resources,
    resource_rep,
)
from pennylane.operation import Operation
from pennylane.ops import GlobalPhase, Prod, StatePrep, change_op_basis, prod
from pennylane.queuing import QueuingManager
from pennylane.templates.embeddings import AmplitudeEmbedding
from pennylane.wires import Wires

from .select import Select


def _get_new_terms(lcu):
    """Compute a new sum of unitaries with positive coefficients"""
    coeffs, ops = lcu.terms()
    coeffs = math.stack(coeffs)
    angles = math.angle(coeffs)
    # The following will produce a nested `Prod` object for a `Prod` object in`ops`
    new_ops = [prod(op, GlobalPhase(-angle, wires=op.wires)) for angle, op in zip(angles, ops)]

    return math.abs(coeffs), new_ops


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

    resource_keys = frozenset({"num_control", "op_reps"})

    @property
    def resource_params(self):
        ops = self.lcu.terms()[1]
        op_reps = tuple(resource_rep(type(op), **op.resource_params) for op in ops)
        return {"op_reps": op_reps, "num_control": len(self.control)}

    grad_method = None

    def __init__(self, lcu, control=None, id=None):

        control = Wires(control)
        self.hyperparameters["lcu"] = lcu
        self.hyperparameters["control"] = control
        target_wires = lcu.wires

        if any(control_wire in target_wires for control_wire in control):
            raise ValueError("Control wires should be different from operation wires.")

        self.hyperparameters["target_wires"] = target_wires

        all_wires = target_wires + control
        super().__init__(*self.data, wires=all_wires, id=id)

    def _flatten(self):
        return (self.lcu,), (self.control,)

    @classmethod
    def _primitive_bind_call(cls, lcu, control, **kwargs):
        return super()._primitive_bind_call(lcu, wires=control, **kwargs)

    @classmethod
    def _unflatten(cls, data, metadata) -> "PrepSelPrep":
        return cls(data[0], metadata[0])

    def __repr__(self):
        return f"PrepSelPrep(lcu={self.lcu}, control={self.control})"

    def map_wires(self, wire_map: dict) -> "PrepSelPrep":
        new_control = [wire_map.get(wire, wire) for wire in self.hyperparameters["control"]]
        new_lcu = self.lcu.map_wires(wire_map)
        return PrepSelPrep(new_lcu, new_control)

    def decomposition(self):
        return self.compute_decomposition(self.lcu, self.control)

    def label(self, decimals=None, base_label=None, cache=None) -> str:
        op_label = base_label or self.__class__.__name__
        if cache is None or not isinstance(cache.get("matrices", None), list):
            return op_label if self._id is None else f'{op_label}("{self._id}")'

        coeffs = math.array(self.coeffs)
        for i, mat in enumerate(cache["matrices"]):
            if math.shape(coeffs) == math.shape(mat) and math.allclose(coeffs, mat):
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

        return [
            change_op_basis(
                AmplitudeEmbedding(math.sqrt(coeffs), normalize=True, pad_with=0, wires=control),
                Select(ops, control, partial=True),
            ),
        ]

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
    def lcu(self):
        """The LCU to be block-encoded."""
        return self.hyperparameters["lcu"]

    @property
    def coeffs(self):
        """The coefficients of the LCU to be block-encoded."""
        return self.lcu.terms()[0]

    @property
    def ops(self):
        """The operators of the LCU to be block-encoded."""
        return self.lcu.terms()[1]

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

    def queue(self, context: QueuingManager = QueuingManager):
        """Append the operator to the Operator queue."""
        context.remove(self.lcu)
        context.append(self)
        return self


def _prepselprep_resources(op_reps, num_control):
    prod_reps = tuple(
        resource_rep(Prod, resources={resource_rep(GlobalPhase): 1, rep: 1}) for rep in op_reps
    )
    return {
        change_op_basis_resource_rep(
            resource_rep(StatePrep, num_wires=num_control),
            resource_rep(
                Select,
                op_reps=prod_reps,
                num_control_wires=num_control,
                partial=True,
                num_work_wires=0,
            ),
        ): 1,
    }


# pylint: disable=unused-argument
@register_resources(_prepselprep_resources)
def _prepselprep_decomp(*_, wires, lcu, control, target_wires):
    coeffs, ops = _get_new_terms(lcu)
    sqrt_coeffs = math.sqrt(coeffs)
    change_op_basis(
        StatePrep(sqrt_coeffs, normalize=True, pad_with=0, wires=control),
        Select(ops, control, partial=True),
    )


add_decomps(PrepSelPrep, _prepselprep_decomp)

# pylint: disable=protected-access
if PrepSelPrep._primitive is not None:

    @PrepSelPrep._primitive.def_impl
    def _(*args, n_wires, **kwargs):
        (lcu,), control = args[:-n_wires], args[-n_wires:]
        return type.__call__(PrepSelPrep, lcu, control, **kwargs)
