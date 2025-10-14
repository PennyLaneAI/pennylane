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
This file contains the implementation of the Prod class which contains logic for
computing the product between operations.
"""
import itertools
from collections import Counter
from copy import copy
from functools import reduce
from itertools import combinations
from typing import Union

from scipy.sparse import kron as sparse_kron

import pennylane as qml
from pennylane import math
from pennylane.capture.autograph import wraps
from pennylane.operation import Operator
from pennylane.ops.op_math.pow import Pow
from pennylane.ops.op_math.sprod import SProd
from pennylane.ops.op_math.sum import Sum
from pennylane.ops.qubit.non_parametric_ops import PauliX, PauliY, PauliZ
from pennylane.queuing import QueuingManager
from pennylane.typing import TensorLike

from .composite import CompositeOp, handle_recursion_error

MAX_NUM_WIRES_KRON_PRODUCT = 9
"""The maximum number of wires up to which using ``math.kron`` is faster than ``math.dot`` for
computing the sparse matrix representation."""


def prod(*ops, id=None, lazy=True):
    """Construct an operator which represents the generalized product of the
    operators provided.

    The generalized product operation represents both the tensor product as
    well as matrix composition. This can be resolved naturally from the wires
    that the given operators act on.

    Args:
        *ops (Union[tuple[~.operation.Operator], Callable]): The operators we would like to multiply.
            Alternatively, a single qfunc that queues operators can be passed to this function.

    Keyword Args:
        id (str or None): id for the product operator. Default is None.
        lazy=True (bool): If ``lazy=False``, a simplification will be performed such that when any
            of the operators is already a product operator, its operands will be used instead.

    Returns:
        ~ops.op_math.Prod: the operator representing the product.

    .. note::

        This operator supports batched operands:

        >>> op = qml.prod(qml.RX(np.array([1, 2, 3]), wires=0), qml.X(1))
        >>> op.matrix().shape
        (3, 4, 4)

        But it doesn't support batching of operators:

        >>> qml.prod(np.array([qml.RX(0.5, 0), qml.RZ(0.3, 0)]), qml.Z(0))
        Traceback (most recent call last):
            ...
        AttributeError: 'numpy.ndarray' object has no attribute 'wires'

    .. seealso:: :class:`~.ops.op_math.Prod`

    **Example**

    >>> prod_op = prod(qml.X(0), qml.Z(0))
    >>> prod_op
    X(0) @ Z(0)
    >>> prod_op.matrix()
    array([[ 0.+0.j, -1.+0.j],
           [ 1.+0.j,  0.+0.j]])
    >>> prod_op.simplify()
    -1j * Y(0)
    >>> prod_op.terms()
    ([-1j], [Y(0)])


    You can also create a prod operator by passing a qfunc to prod, like the following:

    >>> def qfunc(x):
    ...     qml.RX(x, 0)
    ...     qml.CNOT([0, 1])
    >>> prod_op = prod(qfunc)(1.1)
    >>> prod_op
    (CNOT(wires=[0, 1])) @ RX(1.1, wires=[0])


    Notice how the order in the output appears reversed. However, this is correct because the operators are applied from right to left.
    """
    if len(ops) == 1:
        if isinstance(ops[0], qml.operation.Operator):
            return ops[0]

        fn = ops[0]

        if not callable(fn):
            raise TypeError(f"Unexpected argument of type {type(fn).__name__} passed to qml.prod")

        @wraps(fn)
        def wrapper(*args, **kwargs):

            # dequeue operators passed as arguments to the quantum function
            leaves, _ = qml.pytrees.flatten((args, kwargs), lambda obj: isinstance(obj, Operator))
            for l in leaves:
                if isinstance(l, Operator):
                    qml.QueuingManager.remove(l)

            qs = qml.tape.make_qscript(fn)(*args, **kwargs)
            if len(qs.operations) == 1:
                op = qs[0]
                if qml.QueuingManager.recording():
                    op = qml.apply(op)
                return op
            return prod(*qs.operations[::-1], id=id, lazy=lazy)

        return wrapper

    if lazy:
        return Prod(*ops, id=id)

    ops_simp = Prod(
        *itertools.chain.from_iterable([op if isinstance(op, Prod) else [op] for op in ops]),
        id=id,
    )

    for op in ops:
        QueuingManager.remove(op)

    return ops_simp


class Prod(CompositeOp):
    r"""Symbolic operator representing the product of operators.

    Args:
        *factors (tuple[~.operation.Operator]): a tuple of operators which will be multiplied
            together.

    Keyword Args:
        id (str or None): id for the product operator. Default is None.

    .. seealso:: :func:`~.ops.op_math.prod`

    **Example**

    >>> prod_op = Prod(qml.X(0), qml.PauliZ(1))
    >>> prod_op
    X(0) @ Z(1)
    >>> qml.matrix(prod_op, wire_order=prod_op.wires)
    array([[ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j],
           [ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
           [ 0.+0.j, -1.+0.j,  0.+0.j,  0.+0.j]])
    >>> prod_op.terms()
    ([1.0], [X(0) @ Z(1)])

    .. note::
        When a Prod operator is applied in a circuit, its factors are applied in the reverse order.
        (i.e ``Prod(op1, op2)`` corresponds to :math:`\hat{op}_{1}\cdot\hat{op}_{2}` which indicates
        first applying :math:`\hat{op}_{2}` then :math:`\hat{op}_{1}` in the circuit). We can see this
        in the decomposition of the operator.

    >>> op = Prod(qml.X(0), qml.Z(1))
    >>> op.decomposition()
    [Z(1), X(0)]

    .. details::
        :title: Usage Details

        The Prod operator represents both matrix composition and tensor products
        between operators.

        >>> prod_op = Prod(qml.RZ(1.23, wires=0), qml.X(0), qml.Z(1))
        >>> prod_op.matrix()
        array([[ 0.        +0.j        ,  0.        +0.j        ,
                 0.816...-0.57...j,  0.        +0.j        ],
               [ 0.        +0.j        , -0.        +0.j        ,
                 0.        +0.j        , -0.816...+0.57...j],
               [ 0.816...+0.57...j,  0.        +0.j        ,
                 0.        +0.j        ,  0.        +0.j        ],
               [ 0.        +0.j        , -0.816...-0.57...j,
                 0.        +0.j        , -0.        +0.j        ]])

        The Prod operation can be used inside a `qnode` as an operation which,
        if parametrized, can be differentiated.

        .. code-block:: python

            dev = qml.device("default.qubit", wires=3)

            @qml.qnode(dev)
            def circuit(theta):
                qml.prod(qml.Z(0), qml.RX(theta, 1))
                return qml.expval(qml.Z(1))

        >>> par = qml.numpy.array(1.23, requires_grad=True)
        >>> circuit(par)
        tensor(0.334..., requires_grad=True)
        >>> qml.grad(circuit)(par)
        tensor(-0.942..., requires_grad=True)

        The Prod operation can also be measured as an observable.
        If the circuit is parametrized, then we can also differentiate through the
        product observable.

        .. code-block:: python

            prod_op = Prod(qml.Z(0), qml.Hadamard(wires=1))
            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(dev)
            def circuit(weights):
                qml.RX(weights[0], wires=0)
                return qml.expval(prod_op)

        >>> weights = qml.numpy.array([0.1], requires_grad=True)
        >>> qml.grad(circuit)(weights)
        array([-0.070...])

        Note that the :meth:`~Prod.terms` method always simplifies and flattens the operands.

        >>> op = qml.ops.Prod(qml.X(0), qml.sum(qml.Y(0), qml.Z(1)))
        >>> op.terms()
        ([1j, 1.0], [Z(0), X(0) @ Z(1)])

    """

    resource_keys = frozenset({"resources"})

    @property
    @handle_recursion_error
    def resource_params(self):
        resources = dict(Counter(qml.resource_rep(type(op), **op.resource_params) for op in self))
        return {"resources": resources}

    _op_symbol = "@"
    _math_op = staticmethod(math.prod)
    grad_method = None

    @property
    def is_hermitian(self):
        """Check if the product operator is hermitian.

        Note, this check is not exhaustive. There can be hermitian operators for which this check
        yields false, which ARE hermitian. So a false result only implies a more explicit check
        must be performed.
        """
        for o1, o2 in combinations(self.operands, r=2):
            if qml.wires.Wires.shared_wires([o1.wires, o2.wires]):
                return False
        return all(op.is_hermitian for op in self)

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_decomposition(self):
        return True

    def decomposition(self):
        r"""Decomposition of the product operator is given by each factor applied in succession.

        Note that the decomposition is the list of factors returned in reversed order. This is
        to support the intuition that when we write :math:`\hat{O} = \hat{A} \cdot \hat{B}` it is implied
        that :math:`\hat{B}` is applied to the state before :math:`\hat{A}` in the quantum circuit.
        """
        if qml.queuing.QueuingManager.recording():
            return [qml.apply(op) for op in self[::-1]]
        return list(self[::-1])

    @handle_recursion_error
    def matrix(self, wire_order=None):
        """Representation of the operator as a matrix in the computational basis."""
        if self.pauli_rep:
            return self.pauli_rep.to_mat(wire_order=wire_order or self.wires)

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
            full_mat = qml.math.stack(
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

    # pylint: disable=protected-access
    @property
    @handle_recursion_error
    def _queue_category(self):
        """Used for sorting objects into their respective lists in `QuantumTape` objects.
        This property is a temporary solution that should not exist long-term and should not be
        used outside of ``QuantumTape._process_queue``.

        Options are:
        * `"_ops"`
        * `"_measurements"`
        * `None`

        Returns (str or None): "_ops" if the _queue_catagory of all factors is "_ops", else None.
        """
        return "_ops" if all(op._queue_category == "_ops" for op in self) else None

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_adjoint(self):
        return True

    def adjoint(self):
        return Prod(*(qml.adjoint(factor) for factor in self[::-1]))

    def _build_pauli_rep(self):
        """PauliSentence representation of the Product of operations."""
        if all(operand_pauli_reps := [op.pauli_rep for op in self.operands]):
            return reduce(lambda a, b: a @ b, operand_pauli_reps) if operand_pauli_reps else None
        return None

    def _simplify_factors(self, factors: tuple[Operator]) -> tuple[complex, Operator]:
        """Reduces the depth of nested factors and groups identical factors.

        Returns:
            Tuple[complex, List[~.operation.Operator]: tuple containing the global phase and a list
            of the simplified factors
        """
        new_factors = _ProductFactorsGrouping()

        for factor in factors:
            simplified_factor = factor.simplify()
            new_factors.add(factor=simplified_factor)
        new_factors.remove_factors(wires=self.wires)
        return new_factors.global_phase, new_factors.factors

    @handle_recursion_error
    def simplify(self) -> Union["Prod", Sum]:
        r"""
        Transforms any nested Prod instance into the form :math:`\sum c_i O_i` where
        :math:`c_i` is a scalar coefficient and :math:`O_i` is a single PL operator
        or pure product of single PL operators.
        """
        # try using pauli_rep:
        if pr := self.pauli_rep:
            pr.simplify()
            return pr.operation(wire_order=self.wires)

        global_phase, factors = self._simplify_factors(factors=self.operands)

        factors = list(itertools.product(*factors))
        if len(factors) == 1:
            factor = factors[0]
            if len(factor) == 0:
                op = qml.Identity(self.wires)
            else:
                op = factor[0] if len(factor) == 1 else Prod(*factor)
            return op if global_phase == 1 else qml.s_prod(global_phase, op)

        factors = [Prod(*factor).simplify() if len(factor) > 1 else factor[0] for factor in factors]
        op = Sum(*factors).simplify()
        return op if global_phase == 1 else qml.s_prod(global_phase, op).simplify()

    @classmethod
    def _sort(cls, op_list, wire_map: dict = None) -> list[Operator]:
        """Insertion sort algorithm that sorts a list of product factors by their wire indices, taking
        into account the operator commutivity.

        Args:
            op_list (List[.Operator]): list of operators to be sorted
            wire_map (dict): Dictionary containing the wire values as keys and its indexes as values.
                Defaults to None.

        Returns:
            List[.Operator]: sorted list of operators
        """

        if isinstance(op_list, tuple):
            op_list = list(op_list)

        for i in range(1, len(op_list)):
            key_op = op_list[i]

            j = i - 1
            while j >= 0 and _swappable_ops(op1=op_list[j], op2=key_op, wire_map=wire_map):
                op_list[j + 1] = op_list[j]
                j -= 1
            op_list[j + 1] = key_op

        return op_list

    @handle_recursion_error
    def terms(self):
        r"""Representation of the operator as a linear combination of other operators.

        .. math:: O = \sum_i c_i O_i

        A ``TermsUndefinedError`` is raised if no representation by terms is defined.

        Returns:
            tuple[list[tensor_like or float], list[.Operation]]: list of coefficients :math:`c_i`
            and list of operations :math:`O_i`

        **Example**

        >>> op = qml.X(0) @ (0.5 * qml.X(1) + qml.X(2))
        >>> op.terms()
        ([np.float64(0.5), 1.0], [X(0) @ X(1), X(0) @ X(2)])

        """
        # try using pauli_rep:
        if pr := self.pauli_rep:
            with qml.QueuingManager.stop_recording():
                ops = [pauli.operation() for pauli in pr.keys()]
            return list(pr.values()), ops

        with qml.QueuingManager.stop_recording():
            global_phase, factors = self._simplify_factors(factors=self.operands)
            factors = list(itertools.product(*factors))

            factors = [
                Prod(*factor).simplify() if len(factor) > 1 else factor[0] for factor in factors
            ]

        # harvest coeffs and ops
        coeffs = []
        ops = []
        for factor in factors:
            if isinstance(factor, SProd):
                coeffs.append(global_phase * factor.scalar)
                ops.append(factor.base)
            else:
                coeffs.append(global_phase)
                ops.append(factor)
        return coeffs, ops


def _prod_resources(resources):
    return resources


# pylint: disable=unused-argument
@qml.register_resources(_prod_resources)
def _prod_decomp(*_, wires=None, operands):
    for op in reversed(operands):
        op._unflatten(*op._flatten())  # pylint: disable=protected-access


qml.add_decomps(Prod, _prod_decomp)


def _swappable_ops(op1, op2, wire_map: dict = None) -> bool:
    """Boolean expression that indicates if op1 and op2 don't have intersecting wires and if they
    should be swapped when sorting them by wire values.

    Args:
        op1 (.Operator): First operator.
        op2 (.Operator): Second operator.
        wire_map (dict): Dictionary containing the wire values as keys and its indexes as values.
            Defaults to None.

    Returns:
        bool: True if operators should be swapped, False otherwise.
    """
    # one is broadcasted onto all wires.
    if not op1.wires:
        return True
    if not op2.wires:
        return False
    wires1 = op1.wires
    wires2 = op2.wires
    if wire_map is not None:
        wires1 = wires1.map(wire_map)
        wires2 = wires2.map(wire_map)
    wires1 = set(wires1)
    wires2 = set(wires2)
    # compare strings of wire labels so that we can compare arbitrary wire labels like 0 and "a"
    return False if wires1 & wires2 else str(wires1.pop()) > str(wires2.pop())


class _ProductFactorsGrouping:
    """Utils class used for grouping identical product factors."""

    _identity_map = {
        "Identity": (1.0, "Identity"),
        "PauliX": (1.0, "PauliX"),
        "PauliY": (1.0, "PauliY"),
        "PauliZ": (1.0, "PauliZ"),
    }
    _x_map = {
        "Identity": (1.0, "PauliX"),
        "PauliX": (1.0, "Identity"),
        "PauliY": (1.0j, "PauliZ"),
        "PauliZ": (-1.0j, "PauliY"),
    }
    _y_map = {
        "Identity": (1.0, "PauliY"),
        "PauliX": (-1.0j, "PauliZ"),
        "PauliY": (1.0, "Identity"),
        "PauliZ": (1.0j, "PauliX"),
    }
    _z_map = {
        "Identity": (1.0, "PauliZ"),
        "PauliX": (1.0j, "PauliY"),
        "PauliY": (-1.0j, "PauliX"),
        "PauliZ": (1.0, "Identity"),
    }
    _pauli_mult = {"Identity": _identity_map, "PauliX": _x_map, "PauliY": _y_map, "PauliZ": _z_map}
    _paulis = {"PauliX": PauliX, "PauliY": PauliY, "PauliZ": PauliZ}

    def __init__(self):
        self._pauli_factors = {}  #  {wire: (pauli_coeff, pauli_word)}
        self._non_pauli_factors = {}  # {wires: [hash, exponent, operator]}
        self._factors = []
        self.global_phase = 1

    def add(self, factor: Operator):
        """Add factor.

        Args:
            factor (Operator): Factor to add.
        """
        wires = factor.wires
        if isinstance(factor, Prod):
            for prod_factor in factor:
                self.add(prod_factor)
        elif isinstance(factor, Sum):
            self._remove_pauli_factors(wires=wires)
            self._remove_non_pauli_factors(wires=wires)
            self._factors += (factor.operands,)
        elif not isinstance(factor, qml.Identity):
            if isinstance(factor, SProd):
                self.global_phase *= factor.scalar
                factor = factor.base
            if isinstance(factor, (qml.Identity, qml.X, qml.Y, qml.Z)):
                self._add_pauli_factor(factor=factor, wires=wires)
                self._remove_non_pauli_factors(wires=wires)
            else:
                self._add_non_pauli_factor(factor=factor, wires=wires)
                self._remove_pauli_factors(wires=wires)

    def _add_pauli_factor(self, factor: Operator, wires: list[int]):
        """Adds the given Pauli operator to the temporary ``self._pauli_factors`` dictionary. If
        there was another Pauli operator acting on the same wire, the two operators are grouped
        together using the ``self._pauli_mult`` dictionary.

        Args:
            factor (Operator): Factor to be added.
            wires (List[int]): Factor wires. This argument is added to avoid calling
                ``factor.wires`` several times.
        """
        wire = wires[0]
        op2_name = factor.name
        old_coeff, old_word = self._pauli_factors.get(wire, (1, "Identity"))
        coeff, new_word = self._pauli_mult[old_word][op2_name]
        self._pauli_factors[wire] = old_coeff * coeff, new_word

    def _add_non_pauli_factor(self, factor: Operator, wires: list[int]):
        """Adds the given non-Pauli factor to the temporary ``self._non_pauli_factors`` dictionary.
        If there alerady exists an identical operator in the dictionary, the two are grouped
        together.

        If there isn't an identical operator in the dictionary, all non Pauli factors that act on
        the same wires are removed and added to the ``self._factors`` tuple.

        Args:
            factor (Operator): Factor to be added.
            wires (List[int]): Factor wires. This argument is added to avoid calling
                ``factor.wires`` several times.
        """
        if isinstance(factor, Pow):
            exponent = factor.z
            factor = factor.base
        else:
            exponent = 1
        op_hash = factor.hash
        old_hash, old_exponent, old_op = self._non_pauli_factors.get(wires, [None, None, None])
        if isinstance(old_op, (qml.RX, qml.RY, qml.RZ)) and factor.name == old_op.name:
            self._non_pauli_factors[wires] = [
                op_hash,
                old_exponent,
                factor.__class__(factor.data[0] + old_op.data[0], wires).simplify(),
            ]
        elif op_hash == old_hash:
            self._non_pauli_factors[wires][1] += exponent
        else:
            self._remove_non_pauli_factors(wires=wires)
            self._non_pauli_factors[wires] = [op_hash, copy(exponent), factor]

    def _remove_non_pauli_factors(self, wires: list[int]):
        """Remove all factors from the ``self._non_pauli_factors`` dictionary that act on the given
        wires and add them to the ``self._factors`` tuple.

        Args:
            wires (List[int]): Wires of the operators to be removed.
        """
        if not self._non_pauli_factors:
            return
        for wire in wires:
            for key, (_, exponent, op) in list(self._non_pauli_factors.items()):
                if wire in key:
                    self._non_pauli_factors.pop(key)
                    if exponent == 0:
                        continue
                    if exponent != 1:
                        op = Pow(base=op, z=exponent).simplify()
                    if not isinstance(op, qml.Identity):
                        self._factors += ((op,),)

    def _remove_pauli_factors(self, wires: list[int]):
        """Remove all Pauli factors from the ``self._pauli_factors`` dictionary that act on the
        given wires and add them to the ``self._factors`` tuple.

        Args:
            wires (List[int]): Wires of the operators to be removed.
        """
        if not self._pauli_factors:
            return
        for wire in wires:
            pauli_coeff, pauli_word = self._pauli_factors.pop(wire, (1, "Identity"))
            if pauli_word != "Identity":
                pauli_op = self._paulis[pauli_word](wire)
                self._factors += ((pauli_op,),)
            self.global_phase *= pauli_coeff

    def remove_factors(self, wires: list[int]):
        """Remove all factors from the ``self._pauli_factors`` and ``self._non_pauli_factors``
        dictionaries that act on the given wires and add them to the ``self._factors`` tuple.

        Args:
            wires (List[int]): Wires of the operators to be removed.
        """
        self._remove_pauli_factors(wires=wires)
        self._remove_non_pauli_factors(wires=wires)

    @property
    def factors(self):
        """Grouped factors tuple.

        Returns:
            tuple: Tuple of grouped factors.
        """
        return tuple(self._factors)
