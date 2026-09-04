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
from pennylane import math
from pennylane import ops as qp_ops
from pennylane.core.operator import Operation, abstractify
from pennylane.core.queuing import QueuingManager
from pennylane.decomposition import (
    add_decomps,
    register_resources,
    resource_rep,
)
from pennylane.ops import GlobalPhase, StatePrep, change_op_basis
from pennylane.ops.op_math.adjoint2 import _adjoint_abstract
from pennylane.ops.op_math.change_op_basis2 import _change_op_basis_abstract
from pennylane.ops.op_math.composite import CompositeOp
from pennylane.ops.op_math.prod import prod
from pennylane.ops.op_math.symbolicop import SymbolicOp
from pennylane.templates.embeddings import AmplitudeEmbedding
from pennylane.typing import Wire
from pennylane.wires import Wires, WiresLike

from .select import Select


def _get_new_terms(lcu):
    """Compute the positive coefficients, LCU unitaries, and their global phases."""
    coeffs, ops = lcu.terms()
    coeffs = math.stack(coeffs)
    angles = math.angle(coeffs)
    phases = [GlobalPhase(-angle) for angle in angles]

    return math.abs(coeffs), ops, phases


class PrepSelPrep(Operation):
    """Implements a block-encoding of a linear combination of unitaries.

    .. warning::
        Derivatives of this operator are not always guaranteed to exist.

    Args:
        lcu (Union[.Hamiltonian, .Sum, .Prod, .Prod2, .SProd, .LinearCombination]): The
            operator written as a linear combination of unitaries.
        control (WiresLike): The control qubits for the PrepSelPrep operator.

    **Example**

    We define an operator and a block-encoding circuit as:

    >>> lcu = qp.dot([0.3, -0.1], [qp.X(2), qp.Z(2)])
    >>> control = [0, 1]
    >>> @qp.qnode(qp.device("default.qubit"))
    ... def circuit(lcu, control):
    ...     qp.PrepSelPrep(lcu, control)
    ...     return qp.state()

    We can see that the operator matrix, up to a normalization constant, is block encoded in the
    circuit matrix:

    >>> matrix_psp = qp.matrix(circuit, wire_order = [0, 1, 2])(lcu, control = control)
    >>> print(matrix_psp.real[0:2, 0:2])
    [[-0.25  0.75]
     [ 0.75  0.25]]

    >>> matrix_lcu = qp.matrix(lcu)
    >>> print(qp.matrix(lcu).real / sum(abs(np.array(lcu.terms()[0]))))
    [[-0.25  0.75]
     [ 0.75  0.25]]
    """

    resource_keys = frozenset({"num_control", "op_reps"})

    @property
    def resource_params(self):
        ops = self.lcu.terms()[1]
        op_reps = tuple(abstractify(op) for op in ops)
        return {"op_reps": op_reps, "num_control": len(self.control)}

    grad_method = None

    def __init__(self, lcu: SymbolicOp | CompositeOp, control: WiresLike) -> None:
        control = Wires(control)
        target_wires = lcu.wires

        if any(control_wire in target_wires for control_wire in control):
            raise ValueError("Control wires should be different from operation wires.")

        self.hyperparameters["lcu"] = lcu
        self.hyperparameters["control"] = control
        self.hyperparameters["target_wires"] = target_wires

        all_wires = target_wires + control
        super().__init__(*self.data, wires=all_wires)

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
            return op_label

        coeffs = math.array(self.coeffs)
        for i, mat in enumerate(cache["matrices"]):
            if math.shape(coeffs) == math.shape(mat) and math.allclose(coeffs, mat):
                str_wo_id = f"{op_label}(M{i})"
                break
        else:
            mat_num = len(cache["matrices"])
            cache["matrices"].append(coeffs)
            str_wo_id = f"{op_label}(M{mat_num})"

        return str_wo_id

    @staticmethod
    def compute_decomposition(lcu, control):
        coeffs, ops, phases = _get_new_terms(lcu)

        return [
            change_op_basis(
                AmplitudeEmbedding(math.sqrt(coeffs), normalize=True, pad_with=0, wires=control),
                prod(Select(ops, control, partial=True), Select(phases, control, partial=True)),
            ),
        ]

    def __copy__(self):
        """Copy this op"""
        with QueuingManager.stop_recording():
            return qp_ops.functions.bind_new_parameters(self, self.data)

    @property
    def data(self):
        """Create data property"""
        return self.lcu.data

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
    select_lcu = Select(list(op_reps), control=Wire[num_control], work_wires=Wire[0], partial=True)
    select_phases = Select(
        [abstractify(GlobalPhase)] * len(op_reps),
        control=Wire[num_control],
        work_wires=Wire[0],
        partial=True,
    )

    _compute_op = resource_rep(StatePrep, num_wires=num_control)
    _target_op = Prod2([select_lcu, select_phases])
    resources = {
        _change_op_basis_abstract(
            _compute_op, _target_op, _adjoint_abstract(_compute_op),
        ): 1,
    }
    return resources


# pylint: disable=unused-argument
@register_resources(_prepselprep_resources)
def _prepselprep_decomp(*_, wires, lcu, control, target_wires):
    coeffs, ops, phases = _get_new_terms(lcu)
    sqrt_coeffs = math.sqrt(coeffs)
    change_op_basis(
        StatePrep(sqrt_coeffs, normalize=True, pad_with=0, wires=control),
        prod(Select(ops, control, partial=True), Select(phases, control, partial=True)),
    )


add_decomps(PrepSelPrep, _prepselprep_decomp)

# pylint: disable=protected-access
if PrepSelPrep._primitive is not None:

    @PrepSelPrep._primitive.def_impl
    def _(*args, n_wires, **kwargs):
        (lcu,), control = args[:-n_wires], args[-n_wires:]
        return type.__call__(PrepSelPrep, lcu, control, **kwargs)
