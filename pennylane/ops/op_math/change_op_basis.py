# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
This submodule defines a class for compute, uncompute patterns.
"""
from collections import Counter, defaultdict
from functools import reduce

from pennylane import math, queuing
from pennylane.decomposition import (
    add_decomps,
    controlled_resource_rep,
    register_resources,
    resource_rep,
)
from pennylane.operation import (
    DiagGatesUndefinedError,
    EigvalsUndefinedError,
    MatrixUndefinedError,
    Operator,
    SparseMatrixUndefinedError,
)
from pennylane.ops.op_math import adjoint, ctrl

from .composite import CompositeOp, handle_recursion_error


def change_op_basis(compute_op, target_op, uncompute_op=None):
    """Construct an operator which represents the product of the
    operators provided.

    Args:
        compute_op (:class:`~.operation.Operator`): A single operator or product that applies quantum operations.
        target_op (:class:`~.operation.Operator`): A single operator or a product that applies quantum operations.
        uncompute_op (:class:`~.operation.Operator`): A single operator or a product that applies quantum operations. Default is uncompute_op=qml.adjoint(compute_op).

    Returns:
        ~ops.op_math.ChangeOpBasis: the operator representing the compute, uncompute pattern.
    """

    return ChangeOpBasis(compute_op, target_op, uncompute_op)


class ChangeOpBasis(CompositeOp):
    """
    Composite operator representing a compute, uncompute pattern of operators.

    Args:
        compute_op (:class:`~.operation.Operator`): A single operator or product that applies quantum operations.
        target_op (:class:`~.operation.Operator`): A single operator or a product that applies quantum operations.
        uncompute_op (:class:`~.operation.Operator`): A single operator or a product that applies quantum operations. Default is uncompute_op=qml.adjoint(compute_op).

    Returns:
        (Operator): Returns an Operator which is the change_op_basis of the provided Operators: compute_op, target_op, compute_op†.
    """

    def __init__(self, compute_op, target_op, uncompute_op=None):
        if uncompute_op is None:
            uncompute_op = adjoint(compute_op)
        super().__init__(compute_op, target_op, uncompute_op)

    resource_keys = frozenset({"compute_op", "target_op", "uncompute_op"})

    has_matrix = False
    has_sparse_matrix = False

    _op_symbol = "@"
    _math_op = staticmethod(math.prod)

    def matrix(self, wire_order=None):
        raise MatrixUndefinedError

    def sparse_matrix(self, wire_order=None, format="csr"):
        raise SparseMatrixUndefinedError

    def diagonalizing_gates(self):
        raise DiagGatesUndefinedError

    def eigvals(self):
        raise EigvalsUndefinedError

    @property
    @handle_recursion_error
    def resource_params(self):
        resources = {}
        resources["compute_op"] = self[0]
        resources["target_op"] = self[1]
        resources["uncompute_op"] = self[2]
        return resources

    grad_method = None

    @classmethod
    def _sort(cls, op_list, wire_map: dict = None) -> list[Operator]:
        """
        We do not sort the ops. The order is guaranteed to matter since if the compute
        and the base operator commute, the pattern would simplify to just being the base operator.

        Args:
            op_list (List[.Operator]): list of operators to be sorted
            wire_map (dict): Dictionary containing the wire values as keys and its indexes as values.
                Defaults to None.

        Returns:
            List[.Operator]: sorted list of operators
        """
        return op_list

    @property
    def is_hermitian(self):
        """Check if the product operator is hermitian.

        Note, this check is not exhaustive. There can be hermitian operators for which this check
        yields false, which ARE hermitian. So a false result only implies a more explicit check
        must be performed.
        """
        return self[1].is_hermitian

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_decomposition(self):
        return True

    def decomposition(self):
        r"""Decomposition of the product operator is given by each of compute_op, target_op, compute_op† applied in succession."""
        if queuing.QueuingManager.recording():
            return [
                self[0]._unflatten(*self[0]._flatten()),  # pylint: disable=protected-access
                self[1]._unflatten(*self[1]._flatten()),  # pylint: disable=protected-access
                self[2]._unflatten(*self[2]._flatten()),  # pylint: disable=protected-access
            ]
        return list(self)

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_adjoint(self):
        return True

    def adjoint(self):
        return ChangeOpBasis(*(adjoint(factor, lazy=False) for factor in self[::-1]))

    def _build_pauli_rep(self):
        """PauliSentence representation of the Product of operations."""
        if all(operand_pauli_reps := [op.pauli_rep for op in self.operands]):
            return reduce(lambda a, b: a @ b, operand_pauli_reps) if operand_pauli_reps else None
        return None


def _change_op_basis_resources(compute_op, target_op, uncompute_op):
    resources = Counter()

    resources[resource_rep(type(compute_op), **compute_op.resource_params)] += 1
    resources[resource_rep(type(target_op), **target_op.resource_params)] += 1
    resources[resource_rep(type(uncompute_op), **uncompute_op.resource_params)] += 1

    return resources


def _controlled_change_op_basis_resources(
    *_,
    num_control_wires,
    num_zero_control_values,
    num_work_wires,
    work_wire_type,
    base_class,
    base_params,
    **__,
):  # pylint: disable=unused-argument, too-many-arguments
    resources = defaultdict(int)
    resources[
        resource_rep(type(base_params["compute_op"]), **base_params["compute_op"].resource_params)
    ] += 1
    resources[
        controlled_resource_rep(
            type(base_params["target_op"]),
            base_params["target_op"].resource_params,
            num_control_wires=num_control_wires,
            num_zero_control_values=num_zero_control_values,
            num_work_wires=num_work_wires,
            work_wire_type=work_wire_type,
        )
    ] += 1
    resources[
        resource_rep(
            type(base_params["uncompute_op"]), **base_params["uncompute_op"].resource_params
        )
    ] += 1
    return resources


@register_resources(_controlled_change_op_basis_resources)
def _controlled_change_op_basis_decomposition(
    *_,
    wires,
    control_wires,
    control_values,
    work_wires,
    work_wire_type,
    base,
    **__,
):  # pylint: disable=unused-argument, too-many-arguments
    base.resource_params["compute_op"]._unflatten(  # pylint: disable=protected-access
        *base.resource_params["compute_op"]._flatten()  # pylint: disable=protected-access
    )
    ctrl(
        base.resource_params["target_op"]._unflatten(  # pylint: disable=protected-access
            *base.resource_params["target_op"]._flatten()  # pylint: disable=protected-access
        ),
        control=control_wires,
        control_values=control_values,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )
    base.resource_params["uncompute_op"]._unflatten(  # pylint: disable=protected-access
        *base.resource_params["uncompute_op"]._flatten()  # pylint: disable=protected-access
    )


# pylint: disable=unused-argument
@register_resources(_change_op_basis_resources)
def _change_op_basis_decomp(*_, wires=None, operands):
    for op in operands:
        op._unflatten(*op._flatten())  # pylint: disable=protected-access


add_decomps(ChangeOpBasis, _change_op_basis_decomp)
add_decomps("C(ChangeOpBasis)", _controlled_change_op_basis_decomposition)
