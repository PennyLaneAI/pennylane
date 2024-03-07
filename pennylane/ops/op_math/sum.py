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
import numbers
from collections.abc import Iterable
from copy import copy
from typing import List

import numpy as np
import pennylane as qml
from pennylane import math
from pennylane.operation import Observable, Tensor, Operator, convert_to_opmath
from pennylane.queuing import QueuingManager

from .composite import CompositeOp


def sum(*summands, id=None, lazy=True):
    r"""Construct an operator which is the sum of the given operators.

    Args:
        *summands (tuple[~.operation.Operator]): the operators we want to sum together.

    Keyword Args:
        id (str or None): id for the Sum operator. Default is None.
        lazy=True (bool): If ``lazy=False``, a simplification will be performed such that when any
            of the operators is already a sum operator, its operands (summands) will be used instead.

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

    .. seealso:: :class:`~.ops.op_math.Sum`

    **Example**

    >>> summed_op = qml.sum(qml.X(0), qml.Z(0))
    >>> summed_op
    X(0) + Z(0)
    >>> summed_op.matrix()
    array([[ 1,  1],
           [ 1, -1]])
    """
    summands = tuple(convert_to_opmath(op) for op in summands)
    if lazy:
        return Sum(*summands, id=id)

    summands_simp = Sum(
        *itertools.chain.from_iterable([op if isinstance(op, Sum) else [op] for op in summands]),
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

    @property
    def hash(self):
        # Since addition is always commutative, we do not need to sort
        return hash(("Sum", frozenset(o.hash for o in self.operands)))

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
        gen = (
            (
                (
                    qml.matrix(op)
                    if isinstance(op, (qml.Hamiltonian, LinearCombination))
                    else op.matrix()
                ),
                op.wires,
            )
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

        >>> qml.operation.enable_new_opmath()
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

    @property
    def coeffs(self):
        r"""
        Scalar coefficients of the operator when flattened out.

        This is a deprecated attribute, please use :meth:`~Sum.terms` instead.

        .. seealso:: :attr:`~Sum.ops`, :class:`~Sum.pauli_rep`"""
        warnings.warn(
            "Sum.coeffs is deprecated and will be removed in future releases. You can access both (coeffs, ops) via op.terms(). Also consider op.operands.",
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
            "Sum.ops is deprecated and will be removed in future releases. You can access both (coeffs, ops) via op.terms(). Also consider op.operands.",
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
            return sorted(list(map(str, wires)))[0], len(wires), str(op)

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


# LinearCombination class as special case of Sum

OBS_MAP = {"PauliX": "X", "PauliY": "Y", "PauliZ": "Z", "Hadamard": "H", "Identity": "I"}


def _compute_grouping_indices(observables, grouping_type="qwc", method="rlf"):
    # todo: directly compute the
    # indices, instead of extracting groups of observables first
    observable_groups = qml.pauli.group_observables(
        observables, coefficients=None, grouping_type=grouping_type, method=method
    )

    observables = copy(observables)

    indices = []
    available_indices = list(range(len(observables)))
    for partition in observable_groups:  # pylint:disable=too-many-nested-blocks
        indices_this_group = []
        for pauli_word in partition:
            # find index of this pauli word in remaining original observables,
            for ind, observable in enumerate(observables):
                if qml.pauli.are_identical_pauli_words(pauli_word, observable):
                    indices_this_group.append(available_indices[ind])
                    # delete this observable and its index, so it cannot be found again
                    observables.pop(ind)
                    available_indices.pop(ind)
                    break
        indices.append(tuple(indices_this_group))

    return tuple(indices)


class LinearCombination(Sum):
    r"""TODO
    """

    num_wires = qml.operation.AnyWires
    grad_method = "A"  # supports analytic gradients
    batch_size = None
    ndim_params = None  # could be (0,) * len(coeffs), but it is not needed. Define at class-level

    def _flatten(self):
        # note that we are unable to restore grouping type or method without creating new properties
        return (self._coeffs, self._ops, self.data), (self.grouping_indices,)

    @classmethod
    def _unflatten(cls, data, metadata):
        new_op = cls(data[0], data[1])
        new_op._grouping_indices = metadata[0]  # pylint: disable=protected-access
        new_op.data = data[2]
        return new_op

    def __init__(
        self,
        coeffs,
        observables: List[Operator],
        simplify=False,
        grouping_type=None,
        method="rlf",
        _pauli_rep=None,
        id=None,
    ):
        if qml.math.shape(coeffs)[0] != len(observables):
            raise ValueError(
                "Could not create valid LinearCombination; "
                "number of coefficients and operators does not match."
            )

        self._coeffs = coeffs

        self._ops = [convert_to_opmath(op) for op in observables]

        self._hyperparameters = {"ops": self._ops}

        self._grouping_indices = None

        with qml.QueuingManager().stop_recording():
            operands = [qml.s_prod(c, op) for c, op in zip(coeffs, observables)]

        # TODO use grouping functionality of Sum once https://github.com/PennyLaneAI/pennylane/pull/5179 is merged
        super().__init__(*operands, id=id, _pauli_rep=_pauli_rep)

        if simplify:
            # TODO clean up this logic, seems unnecesssarily complicated

            simplified_coeffs, simplified_ops, pr = self._simplify_coeffs_ops()

            self._coeffs = (
                simplified_coeffs  # Losing gradient in case of torch interface at this point
            )

            self._ops = simplified_ops
            with qml.QueuingManager().stop_recording():
                operands = [qml.s_prod(c, op) for c, op in zip(self._coeffs, self._ops)]

            super().__init__(*operands, id=id, _pauli_rep=pr)

        if grouping_type is not None:
            with qml.QueuingManager.stop_recording():
                self._grouping_indices = _compute_grouping_indices(
                    self.ops, grouping_type=grouping_type, method=method
                )

    def _check_batching(self):
        """Override for LinearCombination, batching is not yet supported."""

    def label(self, decimals=None, base_label=None, cache=None):
        decimals = None if (len(self.parameters) > 3) else decimals
        return super(CompositeOp, self).label(
            decimals=decimals, base_label=base_label or "𝓗", cache=cache
        )  # Skipping the label method of CompositeOp

    @property
    def coeffs(self):
        """Return the coefficients defining the LinearCombination.

        Returns:
            Iterable[float]): coefficients in the LinearCombination expression
        """
        return self._coeffs

    @property
    def ops(self):
        """Return the operators defining the LinearCombination.

        Returns:
            Iterable[Observable]): observables in the LinearCombination expression
        """
        return self._ops

    def terms(self):
        r"""TODO
        """
        return self.coeffs, self.ops

    @property
    def wires(self):
        r"""The sorted union of wires from all operators.

        Returns:
            (Wires): Combined wires present in all terms, sorted.
        """
        return self._wires

    @property
    def name(self):
        return "LinearCombination"

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

        **Example**

        Examples of valid groupings for the LinearCombination

        >>> H = qml.LinearCombination([qml.PauliX('a'), qml.PauliX('b'), qml.PauliY('b')])

        are

        >>> H.grouping_indices = [[0, 1], [2]]

        or

        >>> H.grouping_indices = [[0, 2], [1]]

        since both ``qml.PauliX('a'), qml.PauliX('b')`` and ``qml.PauliX('a'), qml.PauliY('b')`` commute.


        Args:
            value (list[list[int]]): List of lists of indexes of the observables in ``self.ops``. Each sublist
                represents a group of commuting observables.
        """
        if value is None:
            return

        if (
            not isinstance(value, Iterable)
            or any(not isinstance(sublist, Iterable) for sublist in value)
            or any(i not in range(len(self.ops)) for i in [i for sl in value for i in sl])
        ):
            raise ValueError(
                f"The grouped index value needs to be a tuple of tuples of integers between 0 and the "
                f"number of observables in the LinearCombination; got {value}"
            )
        # make sure all tuples so can be hashable
        self._grouping_indices = tuple(tuple(sublist) for sublist in value)

    def compute_grouping(self, grouping_type="qwc", method="rlf"):
        """
        Compute groups of indices corresponding to commuting observables of this
        LinearCombination, and store it in the ``grouping_indices`` attribute.

        Args:
            grouping_type (str): The type of binary relation between Pauli words used to compute the grouping.
                Can be ``'qwc'``, ``'commuting'``, or ``'anticommuting'``.
            method (str): The graph coloring heuristic to use in solving minimum clique cover for grouping, which
                can be ``'lf'`` (Largest First) or ``'rlf'`` (Recursive Largest First).
        """

        with qml.QueuingManager.stop_recording():
            self._grouping_indices = _compute_grouping_indices(
                self.ops, grouping_type=grouping_type, method=method
            )

    @qml.QueuingManager.stop_recording()
    def _simplify_coeffs_ops(self):
        """Simplify coeffs and ops
        
        Returns:
            coeffs, ops, pauli_rep"""

        if len(self.ops) == 0:
            return self

        # try using pauli_rep:
        if pr := self.pauli_rep:

            wire_order = self.wires
            if len(pr) == 0:
                return [], [], pr

            # collect coefficients and ops
            coeffs = []
            ops = []

            for pw, coeff in pr.items():
                pw_op = pw.operation(wire_order=wire_order)
                ops.append(pw_op)
                coeffs.append(coeff)

            return coeffs, ops, pr

        if len(self.ops) == 1:
            return self.coeffs, [self.ops[0].simplify()], pr

        op_as_sum = qml.sum(*self.operands)
        op_as_sum = op_as_sum.simplify()
        coeffs, ops = op_as_sum.terms()
        return coeffs, ops, None

    def simplify(self):
        r"""TODO
        """
        coeffs, ops, pr = self._simplify_coeffs_ops()
        return LinearCombination(coeffs, ops, _pauli_rep=pr)

    def _obs_data(self):
        r"""Extracts the data from a LinearCombination and serializes it in an order-independent fashion.

        This allows for comparison between LinearCombinations that are equivalent, but are defined with terms and tensors
        expressed in different orders. For example, `qml.PauliX(0) @ qml.PauliZ(1)` and
        `qml.PauliZ(1) @ qml.PauliX(0)` are equivalent observables with different orderings.

        .. Note::

            In order to store the data from each term of the LinearCombination in an order-independent serialization,
            we make use of sets. Note that all data contained within each term must be immutable, hence the use of
            strings and frozensets.

        **Example**

        >>> H = qml.LinearCombination([1, 1], [qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(0)])
        >>> print(H._obs_data())
        {(1, frozenset({('PauliX', <Wires = [1]>, ()), ('PauliX', <Wires = [0]>, ())})),
         (1, frozenset({('PauliZ', <Wires = [0]>, ())}))}
        """
        data = set()

        coeffs_arr = qml.math.toarray(self.coeffs)
        for co, op in zip(coeffs_arr, self.ops):
            obs = op.non_identity_obs if isinstance(op, Tensor) else [op]
            tensor = []
            for ob in obs:
                parameters = tuple(
                    str(param) for param in ob.parameters
                )  # Converts params into immutable type
                if isinstance(ob, qml.GellMann):
                    parameters += (ob.hyperparameters["index"],)
                tensor.append((ob.name, ob.wires, parameters))
            data.add((co, frozenset(tensor)))

        return data

    def compare(self, other):
        r"""Determines whether the operator is equivalent to another.

        Currently only supported for :class:`~LinearCombination`, :class:`~.Observable`, or :class:`~.Tensor`.
        LinearCombinations/observables are equivalent if they represent the same operator
        (their matrix representations are equal), and they are defined on the same wires.

        .. Warning::

            The compare method does **not** check if the matrix representation
            of a :class:`~.Hermitian` observable is equal to an equivalent
            observable expressed in terms of Pauli matrices, or as a
            linear combination of Hermitians.
            To do so would require the matrix form of LinearCombinations and Tensors
            be calculated, which would drastically increase runtime.

        Returns:
            (bool): True if equivalent.

        **Examples**

        >>> H = qml.LinearCombination(
        ...     [0.5, 0.5],
        ...     [qml.PauliZ(0) @ qml.PauliY(1), qml.PauliY(1) @ qml.PauliZ(0) @ qml.Identity("a")]
        ... )
        >>> obs = qml.PauliZ(0) @ qml.PauliY(1)
        >>> print(H.compare(obs))
        True

        >>> H1 = qml.LinearCombination([1, 1], [qml.PauliX(0), qml.PauliZ(1)])
        >>> H2 = qml.LinearCombination([1, 1], [qml.PauliZ(0), qml.PauliX(1)])
        >>> H1.compare(H2)
        False

        >>> ob1 = qml.LinearCombination([1], [qml.PauliX(0)])
        >>> ob2 = qml.Hermitian(np.array([[0, 1], [1, 0]]), 0)
        >>> ob1.compare(ob2)
        False
        """

        if (pr1 := self.pauli_rep) is not None and (pr2 := other.pauli_rep) is not None:
            pr1.simplify()
            pr2.simplify()
            return pr1 == pr2

        if isinstance(other, (LinearCombination, qml.Hamiltonian)):
            op1 = self.simplify()
            op2 = other.simplify()
            return op1._obs_data() == op2._obs_data()  # pylint: disable=protected-access

        if isinstance(other, (Tensor, Observable)):
            op1 = self.simplify()
            return op1._obs_data() == {
                (1, frozenset(other._obs_data()))  # pylint: disable=protected-access
            }

        raise ValueError(
            "Can only compare a LinearCombination, and a LinearCombination/Observable/Tensor."
        )

    def __matmul__(self, other):
        """The product operation between Operator objects."""
        if isinstance(other, LinearCombination):
            coeffs1 = copy(self.coeffs)
            ops1 = self.ops.copy()
            shared_wires = qml.wires.Wires.shared_wires([self.wires, other.wires])
            if len(shared_wires) > 0:
                raise ValueError(
                    "Hamiltonians can only be multiplied together if they act on "
                    "different sets of wires"
                )

            coeffs2 = other.coeffs
            ops2 = other.ops

            coeffs = qml.math.kron(coeffs1, coeffs2)
            ops_list = itertools.product(ops1, ops2)
            terms = [qml.prod(t[0], t[1], lazy=False) for t in ops_list]
            return qml.LinearCombination(coeffs, terms)

        if isinstance(other, Operator):
            if other.arithmetic_depth == 0:
                new_ops = [op @ other for op in self.ops]

                # build new pauli rep using old pauli rep
                if (pr1 := self.pauli_rep) is not None and (pr2 := other.pauli_rep) is not None:
                    new_pr = pr1 @ pr2
                else:
                    new_pr = None
                return LinearCombination(self.coeffs, new_ops, _pauli_rep=new_pr)
            return qml.prod(self, other)

        return NotImplemented

    def __add__(self, H):
        r"""The addition operation between a LinearCombination and a LinearCombination/Tensor/Observable."""
        ops = self.ops.copy()
        self_coeffs = copy(self.coeffs)

        if isinstance(H, numbers.Number) and H == 0:
            return self

        if isinstance(H, (LinearCombination, qml.Hamiltonian)):
            coeffs = qml.math.concatenate([self_coeffs, copy(H.coeffs)], axis=0)
            ops.extend(H.ops.copy())
            if (pr1 := self.pauli_rep) is not None and (pr2 := H.pauli_rep) is not None:
                _pauli_rep = pr1 + pr2
            else:
                _pauli_rep = None
            return qml.LinearCombination(coeffs, ops, _pauli_rep=_pauli_rep)

        if isinstance(H, (Tensor, Observable, qml.ops.Prod, qml.ops.SProd)):
            coeffs = qml.math.concatenate(
                [self_coeffs, qml.math.cast_like([1.0], self_coeffs)], axis=0
            )
            ops.append(H)

            return qml.LinearCombination(coeffs, ops)

        return NotImplemented

    __radd__ = __add__

    def __mul__(self, a):
        r"""The scalar multiplication operation between a scalar and a LinearCombination."""
        if isinstance(a, (int, float)):
            self_coeffs = copy(self.coeffs)
            coeffs = qml.math.multiply(a, self_coeffs)
            return qml.LinearCombination(coeffs, self.ops.copy())

        return NotImplemented

    __rmul__ = __mul__

    def __sub__(self, H):
        r"""The subtraction operation between a LinearCombination and a LinearCombination/Tensor/Observable."""
        if isinstance(H, (LinearCombination, qml.Hamiltonian, Tensor, Observable)):
            return self + qml.s_prod(-1.0, H, lazy=False)
        return NotImplemented

    def queue(self, context=qml.QueuingManager):
        """Queues a qml.LinearCombination instance"""
        for o in self.ops:
            context.remove(o)
        context.append(self)
        return self

    def map_wires(self, wire_map: dict):
        """Returns a copy of the current LinearCombination with its wires changed according to the given
        wire map.

        Args:
            wire_map (dict): dictionary containing the old wires as keys and the new wires as values

        Returns:
            .LinearCombination: new LinearCombination
        """
        new_ops = tuple(op.map_wires(wire_map) for op in self.ops)
        new_op = LinearCombination(self.data, new_ops)
        new_op.grouping_indices = self._grouping_indices
        return new_op
