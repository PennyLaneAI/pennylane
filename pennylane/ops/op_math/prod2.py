# Copyright 2026 Xanadu Quantum Technologies Inc.

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
This file contains the implementation of the ``Prod2`` class, an :class:`~.Operator2`-based
symbolic operator representing the product of operators.
"""

from collections import Counter
from functools import reduce
from typing import override

import pennylane as qp
from pennylane import capture, math
from pennylane.core.operator import Operator, Operator2, abstractify
from pennylane.core.queuing import apply
from pennylane.decomposition import add_decomps, register_resources
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from .composite import handle_recursion_error
from .composite2 import CompositeOp2
from .prod import MAX_NUM_WIRES_KRON_PRODUCT, _swappable_ops


def _is_abstract_operator(op) -> bool:
    """Return whether ``op`` is an operator-valued JAX tracer."""
    return math.is_abstract(op) and isinstance(op.aval, capture.AbstractOperator)


class Prod2(CompositeOp2):
    r"""Symbolic operator representing the product of operators, built on :class:`~.CompositeOp2`.

    Args:
        operands (Sequence[~.operation.Operator]): the operators to be multiplied together.

    .. seealso:: :class:`~.ops.op_math.Prod`

    **Example**

    >>> prod_op = qp.ops.Prod2([qp.X(0), qp.Z(1)])
    >>> prod_op
    X(0) @ Z(1)
    >>> prod_op.decomposition()
    [Z(1), X(0)]

    .. note::
        When a ``Prod2`` operator is applied in a circuit, its factors are applied in the reverse
        order (i.e ``Prod2([op1, op2])`` corresponds to :math:`\hat{op}_{1}\cdot\hat{op}_{2}`, which
        indicates first applying :math:`\hat{op}_{2}` then :math:`\hat{op}_{1}` in the circuit).
    """

    # ``operands`` and ``_init_pauli_rep`` are inherited as hybrid arguments from ``CompositeOp2``
    arg_specs = {}

    _op_symbol = "@"
    _math_op = staticmethod(math.prod)

    def __init__(self, operands, _init_pauli_rep=None):

        # pylint: disable=import-outside-toplevel
        from pennylane.ops.mid_measure import MidMeasure, PauliMeasure

        if any(isinstance(op, (MidMeasure, PauliMeasure)) for op in operands):
            raise ValueError("Composite operators of mid-circuit measurements are not supported.")

        if not all(isinstance(op, Operator2) for op in operands):
            raise TypeError(
                "Prod2 operands must be Operator2 instances. Legacy operators (subclasses of "
                "Operator) should be combined using Prod instead."
            )

        super().__init__(operands, _init_pauli_rep=_init_pauli_rep)

    @property
    @handle_recursion_error
    def data(self):
        """The trainable parameters of the product are those of its operands."""
        return tuple(d for op in self.operands for d in op.data)

    def _check_batching(self):
        batch_sizes = {op.batch_size for op in self if op.batch_size is not None}
        if len(batch_sizes) > 1:
            raise ValueError(
                "Broadcasting was attempted but the broadcasted dimensions "
                f"do not match: {batch_sizes}."
            )
        self._batch_size = batch_sizes.pop() if batch_sizes else None
        # ``Prod2`` has no dynamic arguments of its own; parameters live in the operands.
        self._ndim_params = ()

    @classmethod
    def _sort(cls, op_list, wire_map: dict = None) -> list[Operator]:
        """Insertion sort of product factors by wire indices, respecting commutativity.

        Sorting relies on concrete wires; for abstract or compressed operands (which appear as
        resource representations in the decomposition graph) the construction order is preserved.
        """
        op_list = list(op_list)

        if not all(isinstance(getattr(op, "wires", None), Wires) for op in op_list):
            return op_list

        for i in range(1, len(op_list)):
            key_op = op_list[i]
            j = i - 1
            while j >= 0 and _swappable_ops(op1=op_list[j], op2=key_op, wire_map=wire_map):
                op_list[j + 1] = op_list[j]
                j -= 1
            op_list[j + 1] = key_op

        return op_list

    @property
    @handle_recursion_error
    def has_matrix(self) -> bool:
        return self.pauli_rep is not None or all(op.has_matrix for op in self)

    @handle_recursion_error
    def matrix(self, wire_order=None):
        """Representation of the operator as a matrix in the computational basis."""
        if self.pauli_rep:
            return self.pauli_rep.to_mat(wire_order=wire_order or self.wires)

        mats: list[TensorLike] = []
        batched: list[bool] = []
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
                    reduce(
                        math.kron, [m[i] if b else m for m, b in zip(mats, batched, strict=True)]
                    )
                    for i in range(self.batch_size)
                ]
            )
        return math.expand_matrix(full_mat, self.wires, wire_order=wire_order)

    @property
    @handle_recursion_error
    def has_sparse_matrix(  # pylint: disable=arguments-differ,invalid-overridden-method
        self,
    ) -> bool:
        return self.pauli_rep is not None or all(op.has_sparse_matrix for op in self)

    @handle_recursion_error
    def sparse_matrix(self, wire_order=None, format="csr"):
        if self.pauli_rep:
            return self.pauli_rep.to_mat(wire_order=wire_order or self.wires, format=format)

        from scipy.sparse import kron as sparse_kron  # pylint: disable=import-outside-toplevel

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

    # ------------------------------------------------------------------------
    # ------------------------------ Properties ------------------------------
    # ------------------------------------------------------------------------

    @property
    @override
    def is_verified_hermitian(self) -> bool:
        """Non-exhaustive check for whether the product is Hermitian."""
        from itertools import combinations  # pylint: disable=import-outside-toplevel

        for o1, o2 in combinations(self.operands, r=2):
            if Wires.shared_wires([o1.wires, o2.wires]):
                return False
        return all(op.is_verified_hermitian for op in self)

    @override
    def adjoint(self) -> "Prod2":
        return Prod2([qp.adjoint(factor) for factor in self[::-1]])

    @override
    def map_wires(self, wire_map: dict) -> "Prod2":
        return Prod2([op.map_wires(wire_map) for op in self])

    def _build_pauli_rep(self):
        """PauliSentence representation of the product of operators."""
        if all(operand_pauli_reps := [op.pauli_rep for op in self.operands]):
            return reduce(lambda a, b: a @ b, operand_pauli_reps) if operand_pauli_reps else None
        return None


def _prod2_resources(operands, _init_pauli_rep=None):  # pylint: disable=unused-argument
    return dict(Counter(abstractify(op) for op in operands))


@register_resources(_prod2_resources)
def _prod2_decomp(operands, _init_pauli_rep=None):  # pylint: disable=unused-argument
    for op in reversed(operands):
        apply(op)


# The controlled version of ``Prod2`` is obtained automatically by the ``Controlled2`` machinery,
# which wraps this base decomposition rule (controlling each operand in turn).
add_decomps(Prod2, _prod2_decomp)
