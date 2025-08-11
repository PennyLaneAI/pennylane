from collections import Counter
from functools import reduce
from itertools import combinations

from scipy.sparse import kron as sparse_kron

from pennylane import apply, math, queuing
from pennylane.decomposition import add_decomps, register_resources, resource_rep
from pennylane.ops.op_math import adjoint
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from pennylane.operation import Operator

from .composite import CompositeOp, handle_recursion_error

MAX_NUM_WIRES_KRON_PRODUCT = 9
"""The maximum number of wires up to which using ``math.kron`` is faster than ``math.dot`` for
computing the sparse matrix representation."""


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

    _op_symbol = "@"
    _math_op = staticmethod(math.prod)

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
        return self[1].is_hermitian()

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_decomposition(self):
        return True

    def decomposition(self):
        r"""Decomposition of the product operator is given by each of U, V, U† applied in succession."""
        if queuing.QueuingManager.recording():
            return [apply(op) for op in self]
        return list(self)

    @handle_recursion_error
    def matrix(self, wire_order=None):
        """Representation of the operator as a matrix in the computational basis."""
        mats: list[TensorLike] = []
        batched: list[bool] = []  # batched[i] tells if mats[i] is batched or not
        for ops in self.overlapping_ops:
            gen = ((op.matrix(), op.wires) for op in ops)

            reduced_mat, _ = math.reduce_matrices(gen, reduce_func=math.matmul)

            if self.batch_size is not None:
                batched.append(any(op.batch_size is not None for op in ops))
            else:
                batched.append(False)

            mats.append(reduced_mat)

        if self.batch_size is None:
            full_mat = reduce(math.kron, mats)
        else:
            full_mat = math.stack(
                [
                    reduce(math.kron, [m[i] if b else m for m, b in zip(mats, batched)])
                    for i in range(self.batch_size)
                ]
            )
        return math.expand_matrix(full_mat, self.wires, wire_order=wire_order)

    @handle_recursion_error
    def sparse_matrix(self, wire_order=None, format="csr"):
        if self.pauli_rep:  # Get the sparse matrix from the PauliSentence representation
            return self.pauli_rep.to_mat(wire_order=wire_order or self.wires, format=format)

        if self.has_overlapping_wires or self.num_wires > MAX_NUM_WIRES_KRON_PRODUCT:
            gen = ((op.sparse_matrix(), op.wires) for op in self)

            reduced_mat, prod_wires = math.reduce_matrices(gen, reduce_func=math.dot)

            wire_order = wire_order or self.wires

            return math.expand_matrix(reduced_mat, prod_wires, wire_order=wire_order).asformat(
                format
            )
        mats = (op.sparse_matrix() for op in self)
        full_mat = reduce(sparse_kron, mats)
        return math.expand_matrix(full_mat, self.wires, wire_order=wire_order).asformat(format)

    @property
    @handle_recursion_error
    def has_sparse_matrix(self):
        return self.pauli_rep is not None or all(op.has_sparse_matrix for op in self)

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
