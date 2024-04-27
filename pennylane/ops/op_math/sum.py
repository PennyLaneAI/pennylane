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
# pylint: disable=too-many-arguments,too-many-instance-attributes,protected-access

import warnings
import itertools
from collections.abc import Iterable
from copy import copy
from typing import List

import pennylane as qml
from pennylane import math
from pennylane.operation import Operator, convert_to_opmath
from pennylane.queuing import QueuingManager

from .composite import CompositeOp


def sum(*summands, grouping_type=None, method="rlf", id=None, lazy=True):
    r"""Construct an operator which is the sum of the given operators.

    Args:
        *summands (tuple[~.operation.Operator]): the operators we want to sum together.

    Keyword Args:
        id (str or None): id for the Sum operator. Default is None.
        lazy=True (bool): If ``lazy=False``, a simplification will be performed such that when any
            of the operators is already a sum operator, its operands (summands) will be used instead.
        grouping_type (str): The type of binary relation between Pauli words used to compute
            the grouping. Can be ``'qwc'``, ``'commuting'``, or ``'anticommuting'``.
        method (str): The graph coloring heuristic to use in solving minimum clique cover for
            grouping, which can be ``'lf'`` (Largest First) or ``'rlf'`` (Recursive Largest
            First). This keyword argument is ignored if ``grouping_type`` is ``None``.

    Returns:
        ~ops.op_math.Sum: The operator representing the sum of summands.

    .. note::

        This operator supports batched operands:

        >>> op = qml.sum(qml.RX(np.array([1, 2, 3]), wires=0), qml.X(1))
        >>> op.matrix().shape
        (3, 4, 4)

        But it doesn't support batching of operators:

        >>> op = qml.sum(np.array([qml.RX(0.4, 0), qml.RZ(0.3, 0)]), qml.Z(0))
        AttributeError: 'numpy.ndarray' object has no attribute 'wires'

    .. note::

        If grouping is requested, the computed groupings are stored as a list of list of indices
        in ``Sum.grouping_indices``. The indices refer to the operators and coefficients returned
        by ``Sum.terms()``, not ``Sum.operands``, as these are not guaranteed to be equivalent.

    .. seealso:: :class:`~.ops.op_math.Sum`

    **Example**

    >>> summed_op = qml.sum(qml.X(0), qml.Z(0))
    >>> summed_op
    X(0) + Z(0)
    >>> summed_op.matrix()
    array([[ 1,  1],
           [ 1, -1]])

    .. details::
        :title: Grouping

        Grouping information can be collected during construction using the ``grouping_type`` and ``method``
        keyword arguments. For example:

        .. code-block:: python

            import pennylane as qml

            a = qml.s_prod(1.0, qml.X(0))
            b = qml.s_prod(2.0, qml.prod(qml.X(0), qml.X(1)))
            c = qml.s_prod(3.0, qml.Z(0))

            op = qml.sum(a, b, c, grouping_type="qwc")

        >>> op.grouping_indices
        ((2,), (0, 1))

        ``grouping_type`` can be ``"qwc"`` (qubit-wise commuting), ``"commuting"``, or ``"anticommuting"``, and
        ``method`` can be ``"rlf"`` or ``"lf"``. To see more details about how these affect grouping, see
        :ref:`Pauli Graph Colouring<graph_colouring>` and :func:`~pennylane.pauli.group_observables`.
    """
    summands = tuple(convert_to_opmath(op) for op in summands)
    if lazy:
        return Sum(*summands, grouping_type=grouping_type, method=method, id=id)

    summands_simp = Sum(
        *itertools.chain.from_iterable([op if isinstance(op, Sum) else [op] for op in summands]),
        grouping_type=grouping_type,
        method=method,
        id=id,
    )

    for op in summands:
        QueuingManager.remove(op)

    return summands_simp


class Sum(CompositeOp):
    r"""Symbolic operator representing the sum of operators.

    Args:
        *summands (tuple[~.operation.Operator]): a tuple of operators which will be summed together.

    Keyword Args:
        grouping_type (str): The type of binary relation between Pauli words used to compute
            the grouping. Can be ``'qwc'``, ``'commuting'``, or ``'anticommuting'``.
        method (str): The graph coloring heuristic to use in solving minimum clique cover for
            grouping, which can be ``'lf'`` (Largest First) or ``'rlf'`` (Recursive Largest
            First). This keyword argument is ignored if ``grouping_type`` is ``None``.
        id (str or None): id for the sum operator. Default is None.

    .. note::
        Currently this operator can not be queued in a circuit as an operation, only measured terminally.

    .. note::

        This operator supports batched operands:

        >>> op = qml.sum(qml.RX(np.array([1, 2, 3]), wires=0), qml.X(1))
        >>> op.matrix().shape
        (3, 4, 4)

        But it doesn't support batching of operators:
        >>> op = qml.sum(np.array([qml.RX(0.4, 0), qml.RZ(0.3, 0)]), qml.Z(0))
        AttributeError: 'numpy.ndarray' object has no attribute 'wires'

    .. note::

        If grouping is requested, the computed groupings are stored as a list of list of indices
        in ``Sum.grouping_indices``. The indices refer to the operators and coefficients returned
        by ``Sum.terms()``, not ``Sum.operands``, as these are not guaranteed to be equivalent.

    .. seealso:: :func:`~.ops.op_math.sum`

    **Example**

    >>> summed_op = Sum(qml.X(0), qml.Z(0))
    >>> summed_op
    X(0) + Z(0)
    >>> qml.matrix(summed_op)
    array([[ 1,  1],
           [ 1, -1]])
    >>> summed_op.terms()
    ([1.0, 1.0], (X(0), Z(0)))

    .. details::
        :title: Usage Details

        We can combine parameterized operators, and support sums between operators acting on
        different wires.

        >>> summed_op = Sum(qml.RZ(1.23, wires=0), qml.I(wires=1))
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

            sum_op = Sum(qml.X(0), qml.Z(1))
            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(dev, diff_method="best")
            def circuit(weights):
                qml.RX(weights[0], wires=0)
                qml.RY(weights[1], wires=1)
                qml.CNOT(wires=[0, 1])
                qml.RX(weights[2], wires=1)
                return qml.expval(sum_op)

        >>> weights = qnp.array([0.1, 0.2, 0.3], requires_grad=True)
        >>> qml.grad(circuit)(weights)
        array([-0.09347337, -0.18884787, -0.28818254])
    """

    _op_symbol = "+"
    _math_op = math.sum
    grad_method = "A"

    def _flatten(self):
        return tuple(self.operands), (self.grouping_indices,)

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(*data, _grouping_indices=metadata[0])

    def __init__(
        self,
        *operands: Operator,
        grouping_type=None,
        method="rlf",
        id=None,
        _grouping_indices=None,
        _pauli_rep=None,
    ):
        super().__init__(*operands, id=id, _pauli_rep=_pauli_rep)

        self._grouping_indices = _grouping_indices
        if _grouping_indices is not None and grouping_type is not None:
            raise ValueError(
                "_grouping_indices and grouping_type cannot be specified at the same time."
            )
        if grouping_type is not None:
            self.compute_grouping(grouping_type=grouping_type, method=method)

    @property
    def hash(self):
        # Since addition is always commutative, we do not need to sort
        return hash(("Sum", frozenset(o.hash for o in self.operands)))

    @property
    def grouping_indices(self):
        """Return the grouping indices attribute.

        Returns:
            list[list[int]]: indices needed to form groups of commuting observables
        """
        return self._grouping_indices

    @grouping_indices.setter
    def grouping_indices(self, value):
        """Set the grouping indices, if known without explicit computation, or if
        computation was done externally. The groups are not verified.

        Args:
            value (list[list[int]]): List of lists of indexes of the observables in ``self.ops``. Each sublist
                represents a group of commuting observables.
        """
        if value is None:
            return

        _, ops = self.terms()

        if (
            not isinstance(value, Iterable)
            or any(not isinstance(sublist, Iterable) for sublist in value)
            or any(i not in range(len(ops)) for sl in value for i in sl)
        ):
            raise ValueError(
                f"The grouped index value needs to be a tuple of tuples of integers between 0 and the "
                f"number of observables in the Sum; got {value}"
            )
        # make sure all tuples so can be hashable
        self._grouping_indices = tuple(tuple(sublist) for sublist in value)

    def __str__(self):
        """String representation of the Sum."""
        ops = self.operands
        return " + ".join(f"{str(op)}" if i == 0 else f"{str(op)}" for i, op in enumerate(ops))

    def __repr__(self):
        """Terminal representation for Sum"""
        # post-processing the flat str() representation
        # We have to do it like this due to the possible
        # nesting of Sums, e.g. X(0) + X(1) + X(2) is a sum(sum(X(0), X(1)), X(2))
        if len(main_string := str(self)) > 50:
            main_string = main_string.replace(" + ", "\n  + ")
            return f"(\n    {main_string}\n)"
        return main_string

    @property
    def is_hermitian(self):
        """If all of the terms in the sum are hermitian, then the Sum is hermitian."""
        if self.pauli_rep is not None:
            coeffs_list = list(self.pauli_rep.values())
            if len(coeffs_list) == 0:
                return True
            if not math.is_abstract(coeffs_list[0]):
                return not any(math.iscomplex(c) for c in coeffs_list)

        return all(s.is_hermitian for s in self)

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
        if self.pauli_rep:
            return self.pauli_rep.to_mat(wire_order=wire_order or self.wires)
        gen = (
            (qml.matrix(op) if isinstance(op, qml.ops.Hamiltonian) else op.matrix(), op.wires)
            for op in self
        )

        reduced_mat, sum_wires = math.reduce_matrices(gen, reduce_func=math.add)

        wire_order = wire_order or self.wires

        return math.expand_matrix(reduced_mat, sum_wires, wire_order=wire_order)

    def sparse_matrix(self, wire_order=None):
        if self.pauli_rep:  # Get the sparse matrix from the PauliSentence representation
            return self.pauli_rep.to_mat(wire_order=wire_order or self.wires, format="csr")

        gen = ((op.sparse_matrix(), op.wires) for op in self)

        reduced_mat, sum_wires = math.reduce_matrices(gen, reduce_func=math.add)

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

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_adjoint(self):
        return True

    def adjoint(self):
        return Sum(*(qml.adjoint(summand) for summand in self))

    def _build_pauli_rep(self):
        """PauliSentence representation of the Sum of operations."""

        if all(operand_pauli_reps := [op.pauli_rep for op in self.operands]):
            new_rep = qml.pauli.PauliSentence()
            for operand_rep in operand_pauli_reps:
                for pw, coeff in operand_rep.items():
                    new_rep[pw] += coeff
            return new_rep
        return None

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
        # try using pauli_rep:
        if pr := self.pauli_rep:
            pr.simplify()
            return pr.operation(wire_order=self.wires)

        new_summands = self._simplify_summands(summands=self.operands).get_summands(cutoff=cutoff)
        if new_summands:
            return Sum(*new_summands) if len(new_summands) > 1 else new_summands[0]
        return qml.s_prod(0, qml.Identity(self.wires))

    def terms(self):
        r"""Representation of the operator as a linear combination of other operators.

        .. math:: O = \sum_i c_i O_i

        A ``TermsUndefinedError`` is raised if no representation by terms is defined.

        Returns:
            tuple[list[tensor_like or float], list[.Operation]]: list of coefficients :math:`c_i`
            and list of operations :math:`O_i`

        **Example**

        >>> op = 0.5 * X(0) + 0.7 * X(1) + 1.5 * Y(0) @ Y(1)
        >>> op.terms()
        ([0.5, 0.7, 1.5],
         [X(0), X(1), Y(1) @ Y(0)])

        Note that this method disentangles nested structures of ``Sum`` instances like so.

        >>> op = 0.5 * X(0) + (2. * (X(1) + 3. * X(2)))
        >>> print(op)
        (0.5*(PauliX(wires=[0]))) + (2.0*((0.5*(PauliX(wires=[1]))) + (3.0*(PauliX(wires=[2])))))
        >>> print(op.terms())
        ([0.5, 1.0, 6.0], [PauliX(wires=[0]), PauliX(wires=[1]), PauliX(wires=[2])])

        """
        # try using pauli_rep:
        if pr := self.pauli_rep:
            with qml.QueuingManager.stop_recording():
                ops = [pauli.operation() for pauli in pr.keys()]
            return list(pr.values()), ops

        with qml.QueuingManager.stop_recording():
            new_summands = self._simplify_summands(summands=self.operands).get_summands()

        coeffs = []
        ops = []
        for factor in new_summands:
            if isinstance(factor, qml.ops.SProd):
                coeffs.append(factor.scalar)
                ops.append(factor.base)
            else:
                coeffs.append(1.0)
                ops.append(factor)
        return coeffs, ops

    def compute_grouping(self, grouping_type="qwc", method="rlf"):
        """
        Compute groups of operators and coefficients corresponding to commuting
        observables of this Sum.

        .. note::

            If grouping is requested, the computed groupings are stored as a list of list of indices
            in ``Sum.grouping_indices``. The indices refer to operators and coefficients returned
            by ``Sum.terms()``, not ``Sum.operands``, as these are not guaranteed to be equivalent.

        Args:
            grouping_type (str): The type of binary relation between Pauli words used to compute
                the grouping. Can be ``'qwc'``, ``'commuting'``, or ``'anticommuting'``.
            method (str): The graph coloring heuristic to use in solving minimum clique cover for
                grouping, which can be ``'lf'`` (Largest First) or ``'rlf'`` (Recursive Largest
                First).

        **Example**

        .. code-block:: python

            import pennylane as qml

            a = qml.X(0)
            b = qml.prod(qml.X(0), qml.X(1))
            c = qml.Z(0)
            obs = [a, b, c]
            coeffs = [1.0, 2.0, 3.0]

            op = qml.dot(coeffs, obs)

        >>> op.grouping_indices is None
        True
        >>> op.compute_grouping(grouping_type="qwc")
        >>> op.grouping_indices
        ((2,), (0, 1))
        """
        if not self.pauli_rep:
            raise ValueError("Cannot compute grouping for Sums containing non-Pauli operators.")

        _, ops = self.terms()

        with qml.QueuingManager.stop_recording():
            op_groups = qml.pauli.group_observables(ops, grouping_type=grouping_type, method=method)
        self._grouping_indices = tuple(tuple(ops.index(o) for o in group) for group in op_groups)

    @property
    def coeffs(self):
        r"""
        Scalar coefficients of the operator when flattened out.

        This is a deprecated attribute, please use :meth:`~Sum.terms` instead.

        .. seealso:: :attr:`~Sum.ops`, :class:`~Sum.pauli_rep`"""
        warnings.warn(
            "Sum.coeffs is deprecated and will be removed in future releases. You can access both "
            "(coeffs, ops) via op.terms(). Also consider op.operands.",
            qml.PennyLaneDeprecationWarning,
        )
        coeffs, _ = self.terms()
        return coeffs

    @property
    def ops(self):
        r"""
        Operator terms without scalar coefficients of the operator when flattened out.

        This is a deprecated attribute, please use :meth:`~Sum.terms` instead.

        .. seealso:: :attr:`~Sum.coeffs`, :class:`~Sum.pauli_rep`"""
        warnings.warn(
            "Sum.ops is deprecated and will be removed in future releases. You can access both "
            "(coeffs, ops) via op.terms(). Also consider op.operands.",
            qml.PennyLaneDeprecationWarning,
        )
        _, ops = self.terms()
        return ops

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

        def _sort_key(op: Operator) -> tuple:
            """Sorting key used in the `sorted` python built-in function.

            Args:
                op (.Operator): Operator.

            Returns:
                Tuple[int, int, str]: Tuple containing the minimum wire value, the number of wires
                    and the string of the operator. This tuple is used to compare different operators
                    in the sorting algorithm.
            """
            wires = op.wires
            if wire_map is not None:
                wires = wires.map(wire_map)
            if not op.wires:
                return ("", 0, str(op))
            sorted_wires = sorted(list(map(str, wires)))[0]
            return sorted_wires, len(wires), str(op)

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
