# Copyright 2018-2026 Xanadu Quantum Technologies Inc.

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
This submodule defines a class for compute-uncompute patterns.
"""

from collections import Counter, defaultdict
from functools import reduce

from pennylane import math
from pennylane.core import queuing
from pennylane.core.operator import Operator, Operator2, abstractify
from pennylane.decomposition import add_decomps, register_resources
from pennylane.decomposition.resources import change_op_basis_resource_rep
from pennylane.exceptions import (
    DiagGatesUndefinedError,
    MatrixUndefinedError,
    SparseMatrixUndefinedError,
)
from pennylane.ops.op_math import adjoint, ctrl
from pennylane.ops.op_math.controlled2 import _ctrl_abstract, flip_zero_control

from .composite import handle_recursion_error
from .composite2 import CompositeOp2


class ChangeOpBasis2(CompositeOp2):
    """
    Composite operator representing a compute-uncompute pattern of operators, which constitutes changing the basis in
    which an operator is applied.

    Args:
        compute_op (:class:`~.Operator`): A single operator or product that applies quantum operations.
        target_op (:class:`~.Operator`): A single operator or a product that applies quantum operations.
        uncompute_op (:class:`~.Operator`): A single operator or a product that applies quantum operations.
            Default is uncompute_op=qp.adjoint(compute_op).

    Returns:
        (Operator): Returns an Operator which is the change_op_basis of the provided Operators: compute_op, target_op, uncompute_op.

    .. note::
        When a ``ChangeOpBasis`` operator is iterated over, its factors are iterated in the reverse order. This is to
        have a similar behaviour to ``Prod`` which applies its factors in reverse order.

    .. seealso:: :func:`~.change_op_basis`
    """

    def __init__(self, compute_op: Operator, target_op: Operator, uncompute_op: Operator = None):
        if uncompute_op is None:
            uncompute_op = adjoint(compute_op)
            self._init_args["uncompute_op"] = uncompute_op
        super().__init__((uncompute_op, target_op, compute_op))

    hybrid_argnames = ("compute_op", "target_op", "uncompute_op")

    _hash = None
    _has_overlapping_wires = None
    _overlapping_ops = None

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

    @property
    def operands(self):
        """The operators in matrix-product order."""
        return self.uncompute_op, self.target_op, self.compute_op

    def _check_batching(self):
        self._batch_size = None
        self._ndim_params = ()

    grad_method = None

    @classmethod
    def _sort(cls, op_list: list, wire_map: dict = None) -> list[Operator]:
        """
        We do not sort the ops. The order is guaranteed to matter since if the compute operator
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
    def is_verified_hermitian(self):
        """Check if the product operator is hermitian.

        Note, this check is not exhaustive. There can be hermitian operators for which this check
        yields false, which ARE hermitian. So a false result only implies that a more explicit check
        must be performed.
        """
        return self.target_op.is_verified_hermitian

    def adjoint(self):
        return ChangeOpBasis2(
            self.compute_op,
            adjoint(self.target_op, lazy=False),
            self.uncompute_op,
        )

    @handle_recursion_error
    def map_wires(self, wire_map: dict):
        # ``CompositeOp2.map_wires`` rebuilds via ``cls(operands, _init_pauli_rep)``, which does
        # not match this operator's three-argument constructor.
        return ChangeOpBasis2(
            self.compute_op.map_wires(wire_map=wire_map),
            self.target_op.map_wires(wire_map=wire_map),
            self.uncompute_op.map_wires(wire_map=wire_map),
        )

    def _build_pauli_rep(self):
        """PauliSentence representation of the Product of operations."""
        if all(operand_pauli_reps := [op.pauli_rep for op in self.operands[::-1]]):
            return reduce(lambda a, b: a @ b, operand_pauli_reps) if operand_pauli_reps else None
        return None


@abstractify.register
def _abstractify_change_op_basis(op: ChangeOpBasis2):
    """Create the abstract resource representation of a concrete COB."""
    if op.is_fully_abstract:
        return op
    return _change_op_basis_abstract(
        abstractify(op.compute_op), abstractify(op.target_op), abstractify(op.uncompute_op)
    )


def _change_op_basis_abstract(compute_op, target_op, uncompute_op):
    """Construct a native abstract COB across the legacy resource boundary.

    The operands are abstractified here so callers can pass concrete operators, abstract
    operators, operator types (e.g. those with a fixed signature), or ``CompressedResourceOp``
    representations interchangeably.
    """
    compute_op = abstractify(compute_op)
    target_op = abstractify(target_op)
    uncompute_op = abstractify(uncompute_op)
    if all(isinstance(op, Operator2) for op in (compute_op, target_op, uncompute_op)):
        return ChangeOpBasis2(compute_op, target_op, uncompute_op)
    return change_op_basis_resource_rep(compute_op, target_op, uncompute_op)


def _controlled_change_op_basis_resources(
    base,
    control_wires,
    control_values,
    work_wires,
    work_wire_type,
):  # pylint: disable=unused-argument
    resources = defaultdict(int)
    resources[base.compute_op] += 1
    resources[
        _ctrl_abstract(
            base.target_op,
            control_wires,
            work_wires,
            work_wire_type,
        )
    ] += 1
    resources[base.uncompute_op] += 1
    return resources


@register_resources(_controlled_change_op_basis_resources)
def _controlled_change_op_basis_decomposition(
    base,
    control_wires,
    control_values,
    work_wires,
    work_wire_type,
):
    queuing.apply(base.compute_op)
    ctrl(
        base.target_op,
        control=control_wires,
        control_values=control_values,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )
    queuing.apply(base.uncompute_op)


def _change_op_basis_resources(compute_op, target_op, uncompute_op):
    resources = Counter()

    resources[compute_op] += 1
    resources[target_op] += 1
    resources[uncompute_op] += 1

    return resources


@register_resources(_change_op_basis_resources)
def _change_op_basis_decomp(compute_op, target_op, uncompute_op):
    queuing.apply(compute_op)
    queuing.apply(target_op)
    queuing.apply(uncompute_op)


add_decomps(ChangeOpBasis2, _change_op_basis_decomp)
add_decomps("C(ChangeOpBasis2)", flip_zero_control(_controlled_change_op_basis_decomposition))
