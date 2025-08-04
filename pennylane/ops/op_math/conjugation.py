from collections import Counter
from functools import reduce

from scipy.sparse import kron as sparse_kron

from pennylane import math, queuing, apply
from pennylane.ops import adjoint
from pennylane.typing import TensorLike
from pennylane.decomposition import resource_rep, register_resources, add_decomps

from .composite import CompositeOp, handle_recursion_error

MAX_NUM_WIRES_KRON_PRODUCT = 9
"""The maximum number of wires up to which using ``math.kron`` is faster than ``math.dot`` for
computing the sparse matrix representation."""


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

    @property
    @handle_recursion_error
    def resource_params(self):
        resources = dict(Counter(resource_rep(type(op), **op.resource_params) for op in self))
        return {"resources": resources}

    grad_method = None

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
