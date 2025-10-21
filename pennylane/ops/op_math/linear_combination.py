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
import itertools
import numbers

# pylint: disable=too-many-arguments,protected-access
from copy import copy

import pennylane as qml
from pennylane.operation import Operator

from .sprod import SProd
from .sum import Sum


class LinearCombination(Sum):
    r"""Operator representing a linear combination of operators.

    The ``LinearCombination`` is represented as a linear combination of other operators, e.g.,
    :math:`\sum_{k=0}^{N-1} c_k O_k`, where the :math:`c_k` are trainable parameters.

    .. note::

        ``qml.Hamiltonian`` dispatches to :class:`~pennylane.ops.op_math.LinearCombination`.

    Args:
        coeffs (tensor_like): coefficients of the ``LinearCombination`` expression
        observables (Iterable[Operator]): observables in the ``LinearCombination`` expression, of same length as ``coeffs``
        grouping_type (str): If not ``None``, compute and store information on how to group commuting
            observables upon initialization. This information may be accessed when a :class:`~.QNode` containing this
            ``LinearCombination`` is executed on devices. The string refers to the type of binary relation between Pauli words.
            Can be ``'qwc'`` (qubit-wise commuting), ``'commuting'``, or ``'anticommuting'``.
        method (str): The graph colouring heuristic to use in solving minimum clique cover for grouping, which
            can be ``'lf'`` (Largest First), ``'rlf'`` (Recursive Largest First), ``'dsatur'`` (Degree of Saturation), or
            ``'gis'`` (IndependentSet). Defaults to ``'lf'``. Ignored if ``grouping_type=None``.
        id (str): name to be assigned to this ``LinearCombination`` instance

    .. seealso:: `rustworkx.ColoringStrategy <https://www.rustworkx.org/apiref/rustworkx.ColoringStrategy.html#coloringstrategy>`_
        for more information on the ``('lf', 'dsatur', 'gis')`` strategies.

    **Example:**

    A ``LinearCombination`` can be created by simply passing the list of coefficients
    as well as the list of observables:

    >>> coeffs = [0.2, -0.543]
    >>> obs = [qml.X(0) @ qml.Z(1), qml.Z(0) @ qml.Hadamard(2)]
    >>> H = qml.ops.LinearCombination(coeffs, obs)
    >>> print(H)
    0.2 * (X(0) @ Z(1)) + -0.543 * (Z(0) @ H(2))

    The same ``LinearCombination`` can be created using the ``qml.Hamiltonian`` alias:

    >>> H = qml.Hamiltonian(coeffs, obs)
    >>> print(H)
    0.2 * (X(0) @ Z(1)) + -0.543 * (Z(0) @ H(2))

    The coefficients can be a trainable tensor, for example:

    >>> coeffs = qml.numpy.array([0.2, -0.543], requires_grad=True)
    >>> obs = [qml.X(0) @ qml.Z(1), qml.Z(0) @ qml.Hadamard(2)]
    >>> H = qml.ops.LinearCombination(coeffs, obs)
    >>> print(H)
    0.2 * (X(0) @ Z(1)) + -0.543 * (Z(0) @ H(2))

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

    grad_method = "A"  # supports analytic gradients
    batch_size = None
    ndim_params = None  # could be (0,) * len(coeffs), but it is not needed. Define at class-level

    def _flatten(self):
        # note that we are unable to restore grouping type or method without creating new properties
        return self.terms(), (self.grouping_indices,)

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(data[0], data[1], _grouping_indices=metadata[0])

    # pylint: disable=arguments-differ
    @classmethod
    def _primitive_bind_call(cls, coeffs, observables, _pauli_rep=None, **kwargs):
        return cls._primitive.bind(*coeffs, *observables, **kwargs, n_obs=len(observables))

    def __init__(
        self,
        coeffs,
        observables: list[Operator],
        grouping_type=None,
        method="lf",
        *,
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

        self._coeffs = coeffs

        self._ops = list(observables)

        self._hyperparameters = {"ops": self._ops}

        with qml.QueuingManager.stop_recording():
            # type.__call__ valid when capture is enabled and creating an instance
            operands = tuple(type.__call__(SProd, c, op) for c, op in zip(coeffs, observables))

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
            Iterable[Operator]): observables in the LinearCombination expression
        """
        return self._ops

    def terms(self):
        r"""Retrieve the coefficients and operators of the ``LinearCombination``.

        Returns:
            tuple[list[tensor_like or float], list[.Operation]]: list of coefficients :math:`c_i`
            and list of operations :math:`O_i`

        **Example**

        >>> coeffs = [1., 2., 3.]
        >>> ops = [qml.X(0), qml.X(0) @ qml.X(1), qml.X(1) @ qml.X(2)]
        >>> op = qml.ops.LinearCombination(coeffs, ops)
        >>> op.terms()
        ([1.0, 2.0, 3.0], [X(0), X(0) @ X(1), X(1) @ X(2)])

        """
        return self.coeffs, self.ops

    def compute_grouping(self, grouping_type="qwc", method="lf"):
        """
        Compute groups of operators and coefficients corresponding to commuting
        observables of this ``LinearCombination``.

        .. note::

            If grouping is requested, the computed groupings are stored as a list of list of indices
            in ``LinearCombination.grouping_indices``.

        Args:
            grouping_type (str): The type of binary relation between Pauli words used to compute
                the grouping. Can be ``'qwc'``, ``'commuting'``, or ``'anticommuting'``.
                Defaults to ``'qwc'``.
            method (str): The graph colouring heuristic to use in solving minimum clique cover for
                grouping, which can be ``'lf'`` (Largest First), ``'rlf'`` (Recursive Largest First),
                ``'dsatur'`` (Degree of Saturation), or ``'gis'`` (Greedy Independent Set).

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
        ((0, 1), (2,))
        """
        if not self.pauli_rep:
            raise ValueError("Cannot compute grouping for Sums containing non-Pauli operators.")

        _, ops = self.terms()

        self._grouping_indices = qml.pauli.compute_partition_indices(
            ops, grouping_type=grouping_type, method=method
        )

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

    def __matmul__(self, other: Operator) -> Operator:
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

    def __add__(self, H: numbers.Number | Operator) -> Operator:
        r"""The addition operation between a LinearCombination and an Operator."""
        ops = copy(self.ops)
        self_coeffs = self.coeffs

        if isinstance(H, numbers.Number) and H == 0:
            return self

        if isinstance(H, LinearCombination):
            coeffs = qml.math.concatenate([self_coeffs, H.coeffs], axis=0)
            ops.extend(H.ops)
            if (pr1 := self.pauli_rep) is not None and (pr2 := H.pauli_rep) is not None:
                _pauli_rep = pr1 + pr2
            else:
                _pauli_rep = None
            return qml.ops.LinearCombination(coeffs, ops, _pauli_rep=_pauli_rep)

        if isinstance(H, Operator):
            coeffs = qml.math.concatenate(
                [self_coeffs, qml.math.cast_like([1.0], self_coeffs)], axis=0
            )
            ops.append(H)

            return qml.ops.LinearCombination(coeffs, ops)
        return NotImplemented

    def __sub__(self, H: Operator) -> Operator:
        r"""The subtraction operation between a LinearCombination and an Operator."""
        if isinstance(H, Operator):
            return self + qml.s_prod(-1.0, H, lazy=False)
        return NotImplemented

    __radd__ = __add__

    def __mul__(self, a: int | float | complex) -> "LinearCombination":
        r"""The scalar multiplication operation between a scalar and a LinearCombination."""
        if isinstance(a, (int, float, complex)):
            self_coeffs = self.coeffs
            coeffs = qml.math.multiply(a, self_coeffs)
            return qml.ops.LinearCombination(coeffs, self.ops)

        return NotImplemented

    __rmul__ = __mul__

    def queue(self, context: qml.QueuingManager | qml.queuing.AnnotatedQueue = qml.QueuingManager):
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
                    qml.math.expand_vector(ops[0].eigvals(), list(ops[0].wires), list(self.wires))
                )
            else:
                tmp_composite = Sum(*ops)  # only change compared to CompositeOp.eigvals()
                eigvals.append(
                    qml.math.expand_vector(
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


if LinearCombination._primitive is not None:

    @LinearCombination._primitive.def_impl
    def _(*args, n_obs, **kwargs):
        coeffs = args[:n_obs]
        observables = args[n_obs:]
        return type.__call__(LinearCombination, coeffs, observables, **kwargs)


# this just exists for the docs build for now, since we're waiting until the next PR to fix the docs
# pylint: disable=too-few-public-methods
class Hamiltonian:
    r"""Returns an operator representing a Hamiltonian.

    The Hamiltonian is represented as a linear combination of other operators, e.g.,
    :math:`\sum_{k=0}^{N-1} c_k O_k`, where the :math:`c_k` are trainable parameters.

    .. note::

        ``qml.Hamiltonian`` dispatches to :class:`~pennylane.ops.op_math.LinearCombination`.

    Args:
        coeffs (tensor_like): coefficients of the Hamiltonian expression
        observables (Iterable[Operator]): observables in the Hamiltonian expression, of same length as coeffs
        grouping_type (str): If not None, compute and store information on how to group commuting
            observables upon initialization. This information may be accessed when QNodes containing this
            Hamiltonian are executed on devices. The string refers to the type of binary relation between Pauli words.
            Can be ``'qwc'`` (qubit-wise commuting), ``'commuting'``, or ``'anticommuting'``.
        method (str): The graph colouring heuristic to use in solving minimum clique cover for grouping, which
            can be ``'lf'`` (Largest First), ``'rlf'`` (Recursive Largest First), ``'dsatur'`` (Degree of Saturation),
            or ``'gis'`` (Greedy Independent Set). Ignored if ``grouping_type=None``.
        id (str): name to be assigned to this Hamiltonian instance

    **Example:**

    ``qml.Hamiltonian`` takes in a list of coefficients and a list of operators.

    >>> coeffs = [0.2, -0.543]
    >>> obs = [qml.X(0) @ qml.Z(1), qml.Z(0) @ qml.Hadamard(2)]
    >>> H = qml.Hamiltonian(coeffs, obs)
    >>> print(H)
    0.2 * (X(0) @ Z(1)) + -0.543 * (Z(0) @ H(2))

    The coefficients can be a trainable tensor, for example:

    >>> coeffs = qml.numpy.array([0.2, -0.543], requires_grad=True)
    >>> obs = [qml.X(0) @ qml.Z(1), qml.Z(0) @ qml.Hadamard(2)]
    >>> H = qml.Hamiltonian(coeffs, obs)
    >>> print(H)
    0.2 * (X(0) @ Z(1)) + -0.543 * (Z(0) @ H(2))

    A ``qml.Hamiltonian`` stores information on which commuting observables should be measured
    together in a circuit:

    >>> obs = [qml.X(0), qml.X(1), qml.Z(0)]
    >>> coeffs = np.array([1., 2., 3.])
    >>> H = qml.Hamiltonian(coeffs, obs, grouping_type='qwc')
    >>> H.grouping_indices
    ((0, 1), (2,))

    This attribute can be used to compute groups of coefficients and observables:

    >>> grouped_coeffs = [coeffs[list(indices)] for indices in H.grouping_indices]
    >>> grouped_obs = [[H.ops[i] for i in indices] for indices in H.grouping_indices]
    >>> grouped_coeffs
    [array([1., 2.]), array([3.])]
    >>> grouped_obs
    [[X(0), X(1)], [Z(0)]]

    Devices that evaluate a ``qml.Hamiltonian`` expectation by splitting it into its local
    observables can use this information to reduce the number of circuits evaluated.

    Note that one can compute the ``grouping_indices`` for an already initialized ``qml.Hamiltonian``
    by using the :func:`compute_grouping <pennylane.ops.LinearCombination.compute_grouping>` method.

    """
