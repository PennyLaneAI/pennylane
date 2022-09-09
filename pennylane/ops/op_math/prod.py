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
from copy import copy
from itertools import combinations
from typing import List, Tuple, Union

import pennylane as qml
from pennylane import math
from pennylane.operation import Operator
from pennylane.ops.op_math.pow_class import Pow
from pennylane.ops.op_math.sprod import SProd
from pennylane.ops.op_math.sum import Sum
from pennylane.ops.qubit.non_parametric_ops import PauliX, PauliY, PauliZ

from .composite import CompositeOp


def prod(*ops, do_queue=True, id=None):
    """Construct an operator which represents the generalized product of the
    operators provided.

    The generalized product operation represents both the tensor product as
    well as matrix composition. This can be resolved naturally from the wires
    that the given operators act on.

    Args:
        ops (tuple[~.operation.Operator]): The operators we would like to multiply

    Keyword Args:
        do_queue (bool): determines if the product operator will be queued. Default is True.
        id (str or None): id for the product operator. Default is None.

    Returns:
        ~ops.op_math.Prod: the operator representing the product.

    .. seealso:: :class:`~.ops.op_math.Prod`

    **Example**

    >>> prod_op = prod(qml.PauliX(0), qml.PauliZ(0))
    >>> prod_op
    PauliX(wires=[0]) @ PauliZ(wires=[0])
    >>> prod_op.matrix()
    array([[ 0, -1],
           [ 1,  0]])
    """
    return Prod(*ops, do_queue=do_queue, id=id)


class Prod(CompositeOp):
    r"""Symbolic operator representing the product of operators.

    Args:
        factors (tuple[~.operation.Operator]): a tuple of operators which will be multiplied
        together.

    Keyword Args:
        do_queue (bool): determines if the product operator will be queued. Default is True.
        id (str or None): id for the product operator. Default is None.

    .. seealso:: :func:`~.ops.op_math.prod`

    **Example**

    >>> prop_op = Prod(qml.PauliX(wires=0), qml.PauliZ(wires=0))
    >>> prop_op
    PauliX(wires=[0]) @ PauliZ(wires=[0])
    >>> qml.matrix(prop_op)
    array([[ 0,  -1],
           [ 1,   0]])
    >>> prop_op.terms()
    ([1.0], [PauliX(wires=[0]) @ PauliZ(wires=[0])])

    .. note::
        When a Prod operator is applied in a circuit, its factors are applied in the reverse order.
        (i.e ``Prod(op1, op2)`` corresponds to :math:`\hat{op}_{1}\dot\hat{op}_{2}` which indicates
        first applying :math:`\hat{op}_{2}` then :math:`\hat{op}_{1}` in the circuit. We can see this
        in the decomposition of the operator.

    >>> op = Prod(qml.PauliX(wires=0), qml.PauliZ(wires=1))
    >>> op.decomposition()
    [PauliZ(wires=[1]), PauliX(wires=[0])]

    .. details::
        :title: Usage Details

        The Prod operator represents both matrix composition and tensor products
        between operators.

        >>> prod_op = Prod(qml.RZ(1.23, wires=0), qml.PauliX(wires=0), qml.PauliZ(wires=1))
        >>> prod_op.matrix()
        array([[ 0.        +0.j        ,  0.        +0.j        ,
                 0.81677345-0.57695852j,  0.        +0.j        ],
               [ 0.        +0.j        ,  0.        +0.j        ,
                 0.        +0.j        , -0.81677345+0.57695852j],
               [ 0.81677345+0.57695852j,  0.        +0.j        ,
                 0.        +0.j        ,  0.        +0.j        ],
               [ 0.        +0.j        , -0.81677345-0.57695852j,
                 0.        +0.j        ,  0.        +0.j        ]])

        The Prod operation can be used inside a `qnode` as an operation which,
        if parameterized, can be differentiated.

        .. code-block:: python

            dev = qml.device("default.qubit", wires=3)

            @qml.qnode(dev)
            def circuit(theta):
                qml.prod(qml.PauliZ(0), qml.RX(theta, 1))
                return qml.expval(qml.PauliZ(1))

        >>> par = np.array(1.23, requires_grad=True)
        >>> circuit(par)
        tensor(0.33423773, requires_grad=True)
        >>> qml.grad(circuit)(par)
        tensor(-0.9424888, requires_grad=True)

        The Prod operation can also be measured as an observable.
        If the circuit is parameterized, then we can also differentiate through the
        product observable.

        .. code-block:: python

            prod_op = Prod(qml.PauliZ(wires=0), qml.Hadamard(wires=1))
            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(dev)
            def circuit(weights):
                qml.RX(weights[0], wires=0)
                return qml.expval(prod_op)

        >>> weights = np.array([0.1], requires_grad=True)
        >>> qml.grad(circuit)(weights)
        array([-0.07059289])
    """

    _op_symbol = "@"

    def terms(self):  # is this method necessary for this class?
        return [1.0], [self]

    @property
    def batch_size(self):
        """Batch size of input parameters."""
        return next((op.batch_size for op in self if op.batch_size is not None), None)

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

    def decomposition(self):
        r"""Decomposition of the product operator is given by each factor applied in succession.

        Note that the decomposition is the list of factors returned in reversed order. This is
        to support the intuition that when we write $\hat{O} = \hat{A} \dot \hat{B}$ it is implied
        that $\hat{B}$ is applied to the state before $\hat{A}$ in the quantum circuit.
        """
        if qml.queuing.QueuingContext.recording():
            return [qml.apply(op) for op in self[::-1]]
        return list(self[::-1])

    def eigvals(self):
        """Return the eigenvalues of the specified operator.

        This method uses pre-stored eigenvalues for standard observables where
        possible and stores the corresponding eigenvectors from the eigendecomposition.

        Returns:
            array: array containing the eigenvalues of the operator
        """
        if self.has_overlapping_wires:
            return self.eigendecomposition["eigval"]
        eigvals = [
            qml.utils.expand_vector(factor.eigvals(), list(factor.wires), list(self.wires))
            for factor in self
        ]

        return qml.math.prod(eigvals, axis=0)

    def matrix(self, wire_order=None):
        """Representation of the operator as a matrix in the computational basis."""
        mats_and_wires_gen = (
            (qml.matrix(op) if isinstance(op, qml.Hamiltonian) else op.matrix(), op.wires)
            for op in self
        )

        reduced_mat, prod_wires = math.reduce_matrices(
            mats_and_wires_gen=mats_and_wires_gen, reduce_func=math.dot
        )

        wire_order = wire_order or self.wires

        return math.expand_matrix(reduced_mat, prod_wires, wire_order=wire_order)

    def sparse_matrix(self, wire_order=None):
        """Compute the sparse matrix representation of the Prod op in csr representation."""
        mats_and_wires_gen = ((op.sparse_matrix(), op.wires) for op in self)

        reduced_mat, prod_wires = math.reduce_matrices(
            mats_and_wires_gen=mats_and_wires_gen, reduce_func=math.dot
        )

        wire_order = wire_order or self.wires

        return math.expand_matrix(reduced_mat, prod_wires, wire_order=wire_order)

    # pylint: disable=protected-access
    @property
    def _queue_category(self):
        """Used for sorting objects into their respective lists in `QuantumTape` objects.
        This property is a temporary solution that should not exist long-term and should not be
        used outside of ``QuantumTape._process_queue``.

        Options are:
        * `"_prep"`
        * `"_ops"`
        * `"_measurements"`
        * `None`

        Returns (str or None): "_ops" if the _queue_catagory of all factors is "_ops", else None.
        """
        return "_ops" if all(op._queue_category == "_ops" for op in self) else None

    def adjoint(self):
        return Prod(*(qml.adjoint(factor) for factor in self[::-1]))

    @property
    def arithmetic_depth(self) -> int:
        return 1 + max(factor.arithmetic_depth for factor in self)

    def _simplify_factors(self, factors: Tuple[Operator]) -> Tuple[complex, Operator]:
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

    def simplify(self) -> Union["Prod", Sum]:
        global_phase, factors = self._simplify_factors(factors=self.operands)

        factors = list(itertools.product(*factors))
        if len(factors) == 1:
            factor = factors[0]
            if len(factor) == 0:
                op = (
                    Prod(*(qml.Identity(w) for w in self.wires))
                    if len(self.wires) > 1
                    else qml.Identity(self.wires[0])
                )
            else:
                op = factor[0] if len(factor) == 1 else Prod(*factor)
            return op if global_phase == 1 else qml.s_prod(global_phase, op)

        factors = [Prod(*factor).simplify() if len(factor) > 1 else factor[0] for factor in factors]
        op = Sum(*factors).simplify()
        return op if global_phase == 1 else qml.s_prod(global_phase, op).simplify()

    @classmethod
    def _sort(cls, op_list, wire_map: dict = None) -> List[Operator]:
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
    wires1 = op1.wires
    wires2 = op2.wires
    if wire_map is not None:
        wires1 = wires1.map(wire_map)
        wires2 = wires2.map(wire_map)
    wires1 = set(wires1)
    wires2 = set(wires2)
    return False if wires1 & wires2 else wires1.pop() > wires2.pop()


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
            if isinstance(factor, (qml.Identity, qml.PauliX, qml.PauliY, qml.PauliZ)):
                self._add_pauli_factor(factor=factor, wires=wires)
                self._remove_non_pauli_factors(wires=wires)
            else:
                self._add_non_pauli_factor(factor=factor, wires=wires)
                self._remove_pauli_factors(wires=wires)

    def _add_pauli_factor(self, factor: Operator, wires: List[int]):
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

    def _add_non_pauli_factor(self, factor: Operator, wires: List[int]):
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

    def _remove_non_pauli_factors(self, wires: List[int]):
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
                    if isinstance(op, Prod):
                        self._factors += tuple(
                            (factor,) for factor in op if not isinstance(factor, qml.Identity)
                        )
                    elif not isinstance(op, qml.Identity):
                        self._factors += ((op,),)

    def _remove_pauli_factors(self, wires: List[int]):
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

    def remove_factors(self, wires: List[int]):
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
