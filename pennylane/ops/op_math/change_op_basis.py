from collections import Counter
from functools import reduce

from pennylane import apply, math, queuing
from pennylane.decomposition import add_decomps, register_resources, resource_rep
from pennylane.operation import (
    DiagGatesUndefinedError,
    EigvalsUndefinedError,
    MatrixUndefinedError,
    Operator,
    SparseMatrixUndefinedError,
)
from pennylane.ops.op_math import adjoint

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

    ops_simp = ChangeOpBasis(compute_op, target_op, uncompute_op)

    return ops_simp


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

    resource_keys = frozenset({"resources"})

    has_matrix = False
    has_sparse_matrix = False

    _op_symbol = "@"
    _math_op = staticmethod(math.prod)

    def matrix(self):
        raise MatrixUndefinedError

    def sparse_matrix(self):
        raise SparseMatrixUndefinedError

    def diagonalizing_gates(self):
        raise DiagGatesUndefinedError

    def eigvals(self):
        raise EigvalsUndefinedError

    @property
    @handle_recursion_error
    def resource_params(self):
        resources = dict(Counter(resource_rep(type(op), **op.resource_params) for op in self))
        return {"resources": resources}

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
            return [apply(op) for op in self]
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


def _change_op_basis_resources(resources):
    return resources


# pylint: disable=unused-argument
@register_resources(_change_op_basis_resources)
def _change_op_basis_decomp(*_, wires=None, operands):
    for op in reversed(operands):
        op._unflatten(*op._flatten())  # pylint: disable=protected-access


add_decomps(ChangeOpBasis, _change_op_basis_decomp)
