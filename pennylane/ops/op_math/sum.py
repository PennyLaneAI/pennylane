# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
This file contains the implementation of the Sum class which contains logic for
computing the sum of operations.
"""
from copy import copy
from typing import List

import numpy as np

import pennylane as qml
from pennylane import math
from pennylane.operation import Operator

from .composite import CompositeOp


def op_sum(*summands, do_queue=True, id=None):
    r"""Construct an operator which is the sum of the given operators.

    Args:
        summands (tuple[~.operation.Operator]): the operators we want to sum together.

    Keyword Args:
        do_queue (bool): determines if the sum operator will be queued (currently not supported).
            Default is True.
        id (str or None): id for the Sum operator. Default is None.

    Returns:
        ~ops.op_math.Sum: The operator representing the sum of summands.

    .. seealso:: :class:`~.ops.op_math.Sum`

    **Example**

    >>> summed_op = op_sum(qml.PauliX(0), qml.PauliZ(0))
    >>> summed_op
    PauliX(wires=[0]) + PauliZ(wires=[0])
    >>> summed_op.matrix()
    array([[ 1,  1],
           [ 1, -1]])
    """
    return Sum(*summands, do_queue=do_queue, id=id)


class Sum(CompositeOp):
    r"""Symbolic operator representing the sum of operators.

    Args:
        summands (tuple[~.operation.Operator]): a tuple of operators which will be summed together.

    Keyword Args:
        do_queue (bool): determines if the sum operator will be queued. Default is True.
        id (str or None): id for the sum operator. Default is None.

    .. note::
        Currently this operator can not be queued in a circuit as an operation, only measured terminally.

    .. seealso:: :func:`~.ops.op_math.op_sum`

    **Example**

    >>> summed_op = Sum(qml.PauliX(0), qml.PauliZ(0))
    >>> summed_op
    PauliX(wires=[0]) + PauliZ(wires=[0])
    >>> qml.matrix(summed_op)
    array([[ 1,  1],
           [ 1, -1]])
    >>> summed_op.terms()
    ([1.0, 1.0], (PauliX(wires=[0]), PauliZ(wires=[0])))

    .. details::
        :title: Usage Details

        We can combine parameterized operators, and support sums between operators acting on
        different wires.

        >>> summed_op = Sum(qml.RZ(1.23, wires=0), qml.Identity(wires=1))
        >>> summed_op.matrix()
        array([[1.81677345-0.57695852j, 0.        +0.j        ,
                0.        +0.j        , 0.        +0.j        ],
               [0.        +0.j        , 1.81677345-0.57695852j,
                0.        +0.j        , 0.        +0.j        ],
               [0.        +0.j        , 0.        +0.j        ,
                1.81677345+0.57695852j, 0.        +0.j        ],
               [0.        +0.j        , 0.        +0.j        ,
                0.        +0.j        , 1.81677345+0.57695852j]])

        The Sum operation can also be measured inside a qnode as an observable.
        If the circuit is parameterized, then we can also differentiate through the
        sum observable.

        .. code-block:: python

            sum_op = Sum(qml.PauliX(0), qml.PauliZ(1))
            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(dev, grad_method="best")
            def circuit(weights):
                qml.RX(weights[0], wires=0)
                qml.RY(weights[1], wires=1)
                qml.CNOT(wires=[0, 1])
                qml.RX(weights[2], wires=1)
                return qml.expval(sum_op)

        >>> weights = qnp.array([0.1, 0.2, 0.3], requires_grad=True)
        >>> qml.grad(circuit)(weights)
        tensor([-0.09347337, -0.18884787, -0.28818254], requires_grad=True)
    """

    _op_symbol = "+"

    @property
    def is_hermitian(self):
        """If all of the terms in the sum are hermitian, then the Sum is hermitian."""
        return all(s.is_hermitian for s in self)

    @property
    def pauli_rep(self):
        """The reduced pauli representation of a sum of operators
        is just the sum of each representation."""
        final_pauli_sentence = qml.ops.PauliArithmetic.PauliSentence({})
        for index, term in enumerate(self.summands):
            if term.pauli_rep is None:
                return None
            if index == 0:
                final_pauli_sentence = copy(term.pauli_rep)
            else:
                final_pauli_sentence += term.pauli_rep
        return final_pauli_sentence

    def terms(self):
        r"""Representation of the operator as a linear combination of other operators.

        .. math:: O = \sum_i c_i O_i

        A ``TermsUndefinedError`` is raised if no representation by terms is defined.

        .. seealso:: :meth:`~.Operator.compute_terms`

        Returns:
            tuple[list[tensor_like or float], list[.Operation]]: list of coefficients :math:`c_i`
            and list of operations :math:`O_i`
        """
        return [1.0] * len(self), list(self)

    def eigvals(self):
        r"""Return the eigenvalues of the specified operator.

        This method uses pre-stored eigenvalues for standard observables where
        possible and stores the corresponding eigenvectors from the eigendecomposition.

        Returns:
            array: array containing the eigenvalues of the operator
        """
        if self.has_overlapping_wires:
            return self.eigendecomposition["eigval"]
        eigvals = [
            qml.utils.expand_vector(summand.eigvals(), list(summand.wires), list(self.wires))
            for summand in self
        ]
        return qml.math.sum(eigvals, axis=0)

    def matrix(self, wire_order=None):
        r"""Representation of the operator as a matrix in the computational basis.

        If ``wire_order`` is provided, the numerical representation considers the position of the
        operator's wires in the global wire order. Otherwise, the wire order defaults to the
        operator's wires.

        If the matrix depends on trainable parameters, the result
        will be cast in the same autodifferentiation framework as the parameters.

        A ``MatrixUndefinedError`` is raised if the matrix representation has not been defined.

        .. seealso:: :meth:`~.Operator.compute_matrix`

        Args:
            wire_order (Iterable): global wire order, must contain all wire labels from the
            operator's wires

        Returns:
            tensor_like: matrix representation
        """
        mats_and_wires_gen = (
            (qml.matrix(op) if isinstance(op, qml.Hamiltonian) else op.matrix(), op.wires)
            for op in self
        )

        reduced_mat, sum_wires = math.reduce_matrices(
            mats_and_wires_gen=mats_and_wires_gen, reduce_func=math.add
        )

        wire_order = wire_order or self.wires

        return math.expand_matrix(reduced_mat, sum_wires, wire_order=wire_order)

    def sparse_matrix(self, wire_order=None):
        """Compute the sparse matrix representation of the Sum op in csr representation."""
        mats_and_wires_gen = ((op.sparse_matrix(), op.wires) for op in self)

        reduced_mat, sum_wires = math.reduce_matrices(
            mats_and_wires_gen=mats_and_wires_gen, reduce_func=math.add
        )

        wire_order = wire_order or self.wires

        return math.expand_matrix(reduced_mat, sum_wires, wire_order=wire_order)

    @property
    def _queue_category(self):  # don't queue Sum instances because it may not be unitary!
        """Used for sorting objects into their respective lists in `QuantumTape` objects.
        This property is a temporary solution that should not exist long-term and should not be
        used outside of ``QuantumTape._process_queue``.

        Returns: None
        """
        return None

    def adjoint(self):
        return Sum(*(qml.adjoint(summand) for summand in self))

    @classmethod
    def _simplify_summands(cls, summands: List[Operator]):
        """Reduces the depth of nested summands and groups equal terms together.

        Args:
            summands (List[~.operation.Operator]): summands list to simplify

        Returns:
            .SumSummandsGrouping: Class containing the simplified and grouped summands.
        """
        new_summands = _SumSummandsGrouping()
        for summand in summands:
            # This code block is not needed but it speeds things up when having a lot of  stacked Sums
            if isinstance(summand, Sum):
                sum_summands = cls._simplify_summands(summands=summand.operands)
                for op_hash, [coeff, sum_summand] in sum_summands.queue.items():
                    new_summands.add(summand=sum_summand, coeff=coeff, op_hash=op_hash)
                continue

            simplified_summand = summand.simplify()
            if isinstance(simplified_summand, Sum):
                sum_summands = cls._simplify_summands(summands=simplified_summand.operands)
                for op_hash, [coeff, sum_summand] in sum_summands.queue.items():
                    new_summands.add(summand=sum_summand, coeff=coeff, op_hash=op_hash)
            else:
                new_summands.add(summand=simplified_summand)

        return new_summands

    def simplify(self, cutoff=1.0e-12) -> "Sum":  # pylint: disable=arguments-differ
        new_summands = self._simplify_summands(summands=self.operands).get_summands(cutoff=cutoff)
        if new_summands:
            return Sum(*new_summands) if len(new_summands) > 1 else new_summands[0]
        return qml.s_prod(
            0,
            qml.prod(*(qml.Identity(w) for w in self.wires))
            if len(self.wires) > 1
            else qml.Identity(self.wires[0]),
        )

    @classmethod
    def _sort(cls, op_list, wire_map: dict = None) -> List[Operator]:
        """Sort algorithm that sorts a list of sum summands by their wire indices.

        Args:
            op_list (List[.Operator]): list of operators to be sorted
            wire_map (dict): Dictionary containing the wire values as keys and its indexes as values.
                Defaults to None.

        Returns:
            List[.Operator]: sorted list of operators
        """

        if isinstance(op_list, tuple):
            op_list = list(op_list)

        def _sort_key(op) -> bool:
            """Sorting key.

            Args:
                op (.Operator): Operator.

            Returns:
                int: Minimum wire value.
            """
            wires = op.wires
            if wire_map is not None:
                wires = wires.map(wire_map)
            return np.min(wires), len(wires)

        return sorted(op_list, key=_sort_key)


class _SumSummandsGrouping:
    """Utils class used for grouping sum summands together."""

    def __init__(self):
        self.queue = {}  # {hash: [coeff, summand]}

    def add(self, summand: Operator, coeff=1, op_hash=None):
        """Add operator to the summands dictionary.

        If the operator hash is already in the dictionary, the coefficient is increased instead.

        Args:
            summand (Operator): operator to add to the summands dictionary
            coeff (int, optional): Coefficient of the operator. Defaults to 1.
            op_hash (int, optional): Hash of the operator. Defaults to None.
        """
        if isinstance(summand, qml.ops.SProd):  # pylint: disable=no-member
            coeff = summand.scalar if coeff == 1 else summand.scalar * coeff
            self.add(summand=summand.base, coeff=coeff)
        else:
            op_hash = summand.hash if op_hash is None else op_hash
            if op_hash in self.queue:
                self.queue[op_hash][0] += coeff
            else:
                self.queue[op_hash] = [copy(coeff), summand]

    def get_summands(self, cutoff=1.0e-12):
        """Get summands list.

        All summands with a coefficient less than cutoff are ignored.

        Args:
            cutoff (float, optional): Cutoff value. Defaults to 1.0e-12.
        """
        new_summands = []
        for coeff, summand in self.queue.values():
            if coeff == 1:
                new_summands.append(summand)
            elif abs(coeff) > cutoff:
                new_summands.append(qml.s_prod(coeff, summand))

        return new_summands
