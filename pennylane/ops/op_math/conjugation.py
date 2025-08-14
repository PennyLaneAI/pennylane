from collections import Counter
from functools import reduce

from pennylane import apply, math, queuing
from pennylane.decomposition import add_decomps, register_resources, resource_rep
from pennylane.operation import Operator, SparseMatrixUndefinedError, DiagGatesUndefinedError, EigvalsUndefinedError
from pennylane.ops.op_math import adjoint

from .composite import CompositeOp, handle_recursion_error
from ...exceptions import MatrixUndefinedError


def conjugation(U, V, U_dag=None):
    """Construct an operator which represents the product of the
    operators provided.

    Args:
        U (:class:`~.operation.Operator`): A single operator or product that applies quantum operations.
        V (:class:`~.operation.Operator`): A single operator or a product that applies quantum operations.
        U_dag (:class:`~.operation.Operator`): A single operator or a product that applies quantum operations. Default is U_dag=qml.adjoint(U).

    Returns:
        ~ops.op_math.Conjugation: the operator representing the compute, uncompute pattern.
    """

    ops_simp = Conjugation(U, V, U_dag)

    return ops_simp


class Conjugation(CompositeOp):
    """
    Composite operator representing a compute, uncompute pattern of operators.

    Args:
        U (:class:`~.operation.Operator`): A single operator or product that applies quantum operations.
        V (:class:`~.operation.Operator`): A single operator or a product that applies quantum operations.
        U_dag (:class:`~.operation.Operator`): A single operator or a product that applies quantum operations. Default is U_dag=qml.adjoint(U).

    Returns:
        (Operator): Returns an Operator which is the conjugation of the provided Operators: U, V, U†.
    """

    def __init__(self, U, V, U_dag=None):
        if U_dag is None:
            U_dag = adjoint(U)
        super().__init__(U, V, U_dag)

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
        r"""Decomposition of the product operator is given by each of U, V, U† applied in succession."""
        if queuing.QueuingManager.recording():
            return [apply(op) for op in self]
        return list(self)

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_adjoint(self):
        return True

    def adjoint(self):
        return Conjugation(*(adjoint(factor) for factor in self[::-1]))

    def _build_pauli_rep(self):
        """PauliSentence representation of the Product of operations."""
        if all(operand_pauli_reps := [op.pauli_rep for op in self.operands]):
            return reduce(lambda a, b: a @ b, operand_pauli_reps) if operand_pauli_reps else None
        return None


def _conjugation_resources(resources):
    return resources


# pylint: disable=unused-argument
@register_resources(_conjugation_resources)
def _conjugation_decomp(*_, wires=None, operands):
    for op in reversed(operands):
        op._unflatten(*op._flatten())  # pylint: disable=protected-access


add_decomps(Conjugation, _conjugation_decomp)
