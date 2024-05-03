# Copyright 2024 Xanadu Quantum Technologies Inc.

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
LinearCombination class
"""
# pylint: disable=too-many-arguments, protected-access, too-many-instance-attributes
import warnings
import itertools
import numbers
from copy import copy
from typing import List

import pennylane as qml
from pennylane.operation import Observable, Tensor, Operator, convert_to_opmath

from .sum import Sum


class LinearCombination(Sum):
    r"""Operator representing a linear combination of operators.

    The ``LinearCombination`` is represented as a linear combination of other operators, e.g.,
    :math:`\sum_{k=0}^{N-1} c_k O_k`, where the :math:`c_k` are trainable parameters.

    Args:
        coeffs (tensor_like): coefficients of the ``LinearCombination`` expression
        observables (Iterable[Observable]): observables in the ``LinearCombination`` expression, of same length as ``coeffs``
        simplify (bool): Specifies whether the ``LinearCombination`` is simplified upon initialization
                         (like-terms are combined). The default value is `False`. Note that ``coeffs`` cannot
                         be differentiated when using the ``'torch'`` interface and ``simplify=True``.
        grouping_type (str): If not ``None``, compute and store information on how to group commuting
            observables upon initialization. This information may be accessed when a :class:`~.QNode` containing this
            ``LinearCombination`` is executed on devices. The string refers to the type of binary relation between Pauli words.
            Can be ``'qwc'`` (qubit-wise commuting), ``'commuting'``, or ``'anticommuting'``.
        method (str): The graph coloring heuristic to use in solving minimum clique cover for grouping, which
            can be ``'lf'`` (Largest First) or ``'rlf'`` (Recursive Largest First). Ignored if ``grouping_type=None``.
        id (str): name to be assigned to this ``LinearCombination`` instance

    **Example:**

    A ``LinearCombination`` can be created by simply passing the list of coefficients
    as well as the list of observables:

    >>> coeffs = [0.2, -0.543]
    >>> obs = [qml.X(0) @ qml.Z(1), qml.Z(0) @ qml.Hadamard(2)]
    >>> H = qml.ops.LinearCombination(coeffs, obs)
    >>> print(H)
    0.2 * (X(0) @ Z(1)) + -0.543 * (Z(0) @ Hadamard(wires=[2]))


    The coefficients can be a trainable tensor, for example:

    >>> coeffs = tf.Variable([0.2, -0.543], dtype=tf.double)
    >>> obs = [qml.X(0) @ qml.Z(1), qml.Z(0) @ qml.Hadamard(2)]
    >>> H = qml.ops.LinearCombination(coeffs, obs)
    >>> print(H)
    0.2 * (X(0) @ Z(1)) + -0.543 * (Z(0) @ Hadamard(wires=[2]))


    A ``LinearCombination`` can store information on which commuting observables should be measured together in
    a circuit:

    >>> obs = [qml.X(0), qml.X(1), qml.Z(0)]
    >>> coeffs = np.array([1., 2., 3.])
    >>> H = qml.ops.LinearCombination(coeffs, obs, grouping_type='qwc')
    >>> H.grouping_indices
    ((0, 1), (2,))

    This attribute can be used to compute groups of coefficients and observables:

    >>> grouped_coeffs = [coeffs[list(indices)] for indices in H.grouping_indices]
    >>> grouped_obs = [[H.ops[i] for i in indices] for indices in H.grouping_indices]
    >>> grouped_coeffs
    [array([1., 2.]), array([3.])]
    >>> grouped_obs
    [[X(0), X(1)], [Z(0)]]

    Devices that evaluate a ``LinearCombination`` expectation by splitting it into its local observables can
    use this information to reduce the number of circuits evaluated.

    Note that one can compute the ``grouping_indices`` for an already initialized ``LinearCombination`` by
    using the :func:`compute_grouping <pennylane.ops.LinearCombination.compute_grouping>` method.
    """

    num_wires = qml.operation.AnyWires
    grad_method = "A"  # supports analytic gradients
    batch_size = None
    ndim_params = None  # could be (0,) * len(coeffs), but it is not needed. Define at class-level

    def _flatten(self):
        # note that we are unable to restore grouping type or method without creating new properties
        return self.terms(), (self.grouping_indices,)

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(data[0], data[1], _grouping_indices=metadata[0])

    def __init__(
        self,
        coeffs,
        observables: List[Operator],
        simplify=False,
        grouping_type=None,
        method="rlf",
        _grouping_indices=None,
        _pauli_rep=None,
        id=None,
    ):
        if isinstance(observables, Operator):
            raise ValueError(
                "observables must be an Iterable of Operator's, and not an Operator itself."
            )
        if qml.math.shape(coeffs)[0] != len(observables):
            raise ValueError(
                "Could not create valid LinearCombination; "
                "number of coefficients and operators does not match."
            )
        if _pauli_rep is None:
            _pauli_rep = self._build_pauli_rep_static(coeffs, observables)

        if simplify:
            # simplify upon initialization changes ops such that they wouldnt be removed in self.queue() anymore
            if qml.QueuingManager.recording():
                for o in observables:
                    qml.QueuingManager.remove(o)

            coeffs, observables, _pauli_rep = self._simplify_coeffs_ops(
                coeffs, observables, _pauli_rep
            )

        self._coeffs = coeffs

        self._ops = [convert_to_opmath(op) for op in observables]

        self._hyperparameters = {"ops": self._ops}

        with qml.QueuingManager.stop_recording():
            operands = tuple(qml.s_prod(c, op) for c, op in zip(coeffs, observables))

        super().__init__(
            *operands,
            grouping_type=grouping_type,
            method=method,
            id=id,
            _grouping_indices=_grouping_indices,
            _pauli_rep=_pauli_rep,
        )

    @staticmethod
    def _build_pauli_rep_static(coeffs, observables):
        """PauliSentence representation of the Sum of operations."""

        if all(pauli_reps := [op.pauli_rep for op in observables]):
            new_rep = qml.pauli.PauliSentence()
            for c, ps in zip(coeffs, pauli_reps):
                for pw, coeff in ps.items():
                    new_rep[pw] += coeff * c
            return new_rep
        return None

    def _check_batching(self):
        """Override for LinearCombination, batching is not yet supported."""

    def label(self, decimals=None, base_label=None, cache=None):
        decimals = None if (len(self.parameters) > 3) else decimals
        return Operator.label(self, decimals=decimals, base_label=base_label or "𝓗", cache=cache)

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
        r"""Retrieve the coefficients and operators of the ``LinearCombination``.

        Returns:
            tuple[list[tensor_like or float], list[.Operation]]: list of coefficients :math:`c_i`
            and list of operations :math:`O_i`

        **Example**

        >>> coeffs = [1., 2., 3.]
        >>> ops = [X(0), X(0) @ X(1), X(1) @ X(2)]
        >>> op = qml.ops.LinearCombination(coeffs, ops)
        >>> op.terms()
        ([1.0, 2.0, 3.0], [X(0), X(0) @ X(1), X(1) @ X(2)])

        """
        return self.coeffs, self.ops

    def compute_grouping(self, grouping_type="qwc", method="rlf"):
        """
        Compute groups of operators and coefficients corresponding to commuting
        observables of this ``LinearCombination``.

        .. note::

            If grouping is requested, the computed groupings are stored as a list of list of indices
            in ``LinearCombination.grouping_indices``.

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

            op = qml.ops.LinearCombination(coeffs, obs)

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

        ops = copy(ops)

        indices = []
        available_indices = list(range(len(ops)))
        for partition in op_groups:  # pylint:disable=too-many-nested-blocks
            indices_this_group = []
            for pauli_word in partition:
                # find index of this pauli word in remaining original observables,
                for ind, observable in enumerate(ops):
                    if qml.pauli.are_identical_pauli_words(pauli_word, observable):
                        indices_this_group.append(available_indices[ind])
                        # delete this observable and its index, so it cannot be found again
                        ops.pop(ind)
                        available_indices.pop(ind)
                        break
            indices.append(tuple(indices_this_group))

        self._grouping_indices = tuple(indices)

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

    @staticmethod
    @qml.QueuingManager.stop_recording()
    def _simplify_coeffs_ops(coeffs, ops, pr, cutoff=1.0e-12):
        """Simplify coeffs and ops

        Returns:
            coeffs, ops, pauli_rep"""

        if len(ops) == 0:
            return [], [], pr

        # try using pauli_rep:
        if pr is not None:
            if len(pr) == 0:
                return [], [], pr

            # collect coefficients and ops
            new_coeffs = []
            new_ops = []

            for pw, coeff in pr.items():
                pw_op = pw.operation(wire_order=pr.wires)
                new_ops.append(pw_op)
                new_coeffs.append(coeff)

            return new_coeffs, new_ops, pr

        if len(ops) == 1:
            return coeffs, [ops[0].simplify()], pr

        op_as_sum = qml.dot(coeffs, ops)
        op_as_sum = op_as_sum.simplify(cutoff)
        new_coeffs, new_ops = op_as_sum.terms()
        return new_coeffs, new_ops, pr

    def simplify(self, cutoff=1.0e-12):
        coeffs, ops, pr = self._simplify_coeffs_ops(self.coeffs, self.ops, self.pauli_rep, cutoff)
        return LinearCombination(coeffs, ops, _pauli_rep=pr)

    def compare(self, other):
        r"""Determines mathematical equivalence between operators

        ``LinearCombination`` and other operators are equivalent if they mathematically represent the same operator
        (their matrix representations are equal), acting on the same wires.

        .. Warning::

            This method does not compute explicit matrices but uses the underlyding operators and coefficients for comparisons. When both operators
            consist purely of Pauli operators, and therefore have a valid ``op.pauli_rep``, the comparison is cheap.
            When that is not the case (e.g. one of the operators contains a ``Hadamard`` gate), it can be more expensive as it involves mathematical simplification of both operators.

        Returns:
            (bool): True if equivalent.

        **Examples**

        >>> H = qml.ops.LinearCombination(
        ...     [0.5, 0.5],
        ...     [qml.PauliZ(0) @ qml.PauliY(1), qml.PauliY(1) @ qml.PauliZ(0) @ qml.Identity("a")]
        ... )
        >>> obs = qml.PauliZ(0) @ qml.PauliY(1)
        >>> print(H.compare(obs))
        True

        >>> H1 = qml.ops.LinearCombination([1, 1], [qml.PauliX(0), qml.PauliZ(1)])
        >>> H2 = qml.ops.LinearCombination([1, 1], [qml.PauliZ(0), qml.PauliX(1)])
        >>> H1.compare(H2)
        False

        >>> ob1 = qml.ops.LinearCombination([1], [qml.PauliX(0)])
        >>> ob2 = qml.Hermitian(np.array([[0, 1], [1, 0]]), 0)
        >>> ob1.compare(ob2)
        False
        """

        if isinstance(other, (Operator)):
            if (pr1 := self.pauli_rep) is not None and (pr2 := other.pauli_rep) is not None:
                pr1.simplify()
                pr2.simplify()
                return pr1 == pr2

            if isinstance(other, (qml.ops.Hamiltonian, Tensor)):
                warnings.warn(
                    f"Attempting to compare a legacy operator class instance {other} of type {type(other)} with {self} of type {type(self)}."
                    f"You are likely disabling/enabling new opmath in the same script or explicitly create legacy operator classes Tensor and ops.Hamiltonian."
                    f"Please visit https://docs.pennylane.ai/en/stable/news/new_opmath.html for more information and help troubleshooting.",
                    UserWarning,
                )
                op1 = self.simplify()
                op2 = other.simplify()

                op2 = qml.operation.convert_to_opmath(op2)
                op2 = qml.ops.LinearCombination(*op2.terms())

                return qml.equal(op1, op2)

            op1 = self.simplify()
            op2 = other.simplify()
            return qml.equal(op1, op2)

        raise ValueError(
            "Can only compare a LinearCombination, and a LinearCombination/Observable/Tensor."
        )

    def __matmul__(self, other):
        """The product operation between Operator objects."""
        if isinstance(other, LinearCombination):
            coeffs1 = self.coeffs
            ops1 = self.ops
            shared_wires = qml.wires.Wires.shared_wires([self.wires, other.wires])
            if len(shared_wires) > 0:
                raise ValueError(
                    "LinearCombinations can only be multiplied together if they act on "
                    "different sets of wires"
                )

            coeffs2 = other.coeffs
            ops2 = other.ops

            coeffs = qml.math.kron(coeffs1, coeffs2)
            ops_list = itertools.product(ops1, ops2)
            terms = [qml.prod(t[0], t[1], lazy=False) for t in ops_list]
            return qml.ops.LinearCombination(coeffs, terms)

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
        ops = copy(self.ops)
        self_coeffs = self.coeffs

        if isinstance(H, numbers.Number) and H == 0:
            return self

        if isinstance(H, (LinearCombination, qml.ops.Hamiltonian)):
            coeffs = qml.math.concatenate([self_coeffs, H.coeffs], axis=0)
            ops.extend(H.ops)
            if (pr1 := self.pauli_rep) is not None and (pr2 := H.pauli_rep) is not None:
                _pauli_rep = pr1 + pr2
            else:
                _pauli_rep = None
            return qml.ops.LinearCombination(coeffs, ops, _pauli_rep=_pauli_rep)

        if isinstance(H, qml.operation.Operator):
            coeffs = qml.math.concatenate(
                [self_coeffs, qml.math.cast_like([1.0], self_coeffs)], axis=0
            )
            ops.append(H)

            return qml.ops.LinearCombination(coeffs, ops)

        return NotImplemented

    __radd__ = __add__

    def __mul__(self, a):
        r"""The scalar multiplication operation between a scalar and a LinearCombination."""
        if isinstance(a, (int, float, complex)):
            self_coeffs = self.coeffs
            coeffs = qml.math.multiply(a, self_coeffs)
            return qml.ops.LinearCombination(coeffs, self.ops)

        return NotImplemented

    __rmul__ = __mul__

    def __sub__(self, H):
        r"""The subtraction operation between a LinearCombination and a LinearCombination/Tensor/Observable."""
        if isinstance(H, (LinearCombination, qml.ops.Hamiltonian, Tensor, Observable)):
            return self + qml.s_prod(-1.0, H, lazy=False)
        return NotImplemented

    def queue(self, context=qml.QueuingManager):
        """Queues a ``qml.ops.LinearCombination`` instance"""
        if qml.QueuingManager.recording():
            for o in self.ops:
                context.remove(o)
            context.append(self)
        return self

    def eigvals(self):
        """Return the eigenvalues of the specified operator.

        This method uses pre-stored eigenvalues for standard observables where
        possible and stores the corresponding eigenvectors from the eigendecomposition.

        Returns:
            array: array containing the eigenvalues of the operator
        """
        eigvals = []
        for ops in self.overlapping_ops:
            if len(ops) == 1:
                eigvals.append(
                    qml.utils.expand_vector(ops[0].eigvals(), list(ops[0].wires), list(self.wires))
                )
            else:
                tmp_composite = Sum(*ops)  # only change compared to CompositeOp.eigvals()
                eigvals.append(
                    qml.utils.expand_vector(
                        tmp_composite.eigendecomposition["eigval"],
                        list(tmp_composite.wires),
                        list(self.wires),
                    )
                )

        return self._math_op(
            qml.math.asarray(eigvals, like=qml.math.get_deep_interface(eigvals)), axis=0
        )

    def diagonalizing_gates(self):
        r"""Sequence of gates that diagonalize the operator in the computational basis.

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U^{\dagger}`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        A ``DiagGatesUndefinedError`` is raised if no representation by decomposition is defined.

        .. seealso:: :meth:`~.Operator.compute_diagonalizing_gates`.

        Returns:
            list[.Operator] or None: a list of operators
        """
        diag_gates = []
        for ops in self.overlapping_ops:
            if len(ops) == 1:
                diag_gates.extend(ops[0].diagonalizing_gates())
            else:
                tmp_sum = Sum(*ops)  # only change compared to CompositeOp.diagonalizing_gates()
                eigvecs = tmp_sum.eigendecomposition["eigvec"]
                diag_gates.append(
                    qml.QubitUnitary(
                        qml.math.transpose(qml.math.conj(eigvecs)), wires=tmp_sum.wires
                    )
                )
        return diag_gates

    def map_wires(self, wire_map: dict):
        """Returns a copy of the current ``LinearCombination`` with its wires changed according to the given
        wire map.

        Args:
            wire_map (dict): dictionary containing the old wires as keys and the new wires as values

        Returns:
            .LinearCombination: new ``LinearCombination``
        """
        coeffs, ops = self.terms()
        new_ops = tuple(op.map_wires(wire_map) for op in ops)
        new_op = LinearCombination(coeffs, new_ops)
        new_op.grouping_indices = self._grouping_indices
        return new_op
