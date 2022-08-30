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
from functools import reduce
from itertools import combinations
from typing import List, Tuple, Union

import numpy as np

import pennylane as qml
from pennylane import math
from pennylane.operation import Operator, expand_matrix
from pennylane.ops.op_math.pow_class import Pow
from pennylane.ops.op_math.sprod import SProd
from pennylane.ops.op_math.sum import Sum
from pennylane.ops.qubit.non_parametric_ops import PauliX, PauliY, PauliZ


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


class Prod(Operator):
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
    _name = "Prod"
    _eigs = {}  # cache eigen vectors and values like in qml.Hermitian

    def __init__(
        self, *factors: Operator, do_queue=True, id=None
    ):  # pylint: disable=super-init-not-called
        """Initialize a Prod instance"""
        self._id = id
        self.queue_idx = None

        if len(factors) < 2:
            raise ValueError(f"Require at least two operators to multiply; got {len(factors)}")

        self.factors = factors
        self._wires = qml.wires.Wires.all_wires([f.wires for f in self.factors])
        self._hash = None

        if do_queue:
            self.queue()

    def __repr__(self):
        """Constructor-call-like representation."""
        return " @ ".join([f"{f}" for f in self.factors])

    def __copy__(self):
        cls = self.__class__
        copied_op = cls.__new__(cls)
        copied_op.factors = tuple(copy(f) for f in self.factors)

        for attr, value in vars(self).items():
            if attr not in {"factors"}:
                setattr(copied_op, attr, value)

        return copied_op

    def terms(self):  # is this method necessary for this class?
        return [1.0], [self]

    @property
    def data(self):
        """Create data property"""
        return [f.parameters for f in self.factors]

    @data.setter
    def data(self, new_data):
        """Set the data property"""
        for new_entry, op in zip(new_data, self.factors):
            op.data = new_entry

    @property
    def batch_size(self):
        """Batch size of input parameters."""
        return next((op.batch_size for op in self.factors if op.batch_size is not None), None)

    @property
    def num_params(self):
        return sum(op.num_params for op in self.factors)

    @property
    def num_wires(self):
        return len(self.wires)

    @property
    def is_hermitian(self):
        """Check if the product operator is hermitian.

        Note, this check is not exhaustive. There can be hermitian operators for which this check
        yields false, which ARE hermitian. So a false result only implies a more explicit check
        must be performed.
        """
        for o1, o2 in combinations(self.factors, r=2):
            if qml.wires.Wires.shared_wires([o1.wires, o2.wires]):
                return False
        return all(op.is_hermitian for op in self.factors)

    def decomposition(self):
        r"""Decomposition of the product operator is given by each factor applied in succession.

        Note that the decomposition is the list of factors returned in reversed order. This is
        to support the intuition that when we write $\hat{O} = \hat{A} \dot \hat{B}$ it is implied
        that $\hat{B}$ is applied to the state before $\hat{A}$ in the quantum circuit.
        """
        if qml.queuing.QueuingContext.recording():
            return [qml.apply(op) for op in self.factors[::-1]]
        return list(self.factors[::-1])

    @property
    def eigendecomposition(self):
        r"""Return the eigendecomposition of the matrix specified by the operator.

        This method uses pre-stored eigenvalues for standard observables where
        possible and stores the corresponding eigenvectors from the eigendecomposition.

        It transforms the input operator according to the wires specified.

        Returns:
            dict[str, array]: dictionary containing the eigenvalues and the
                eigenvectors of the operator.
        """
        Hmat = self.matrix()
        Hmat = math.to_numpy(Hmat)
        Hkey = tuple(Hmat.flatten().tolist())
        if Hkey not in self._eigs:
            w, U = np.linalg.eigh(Hmat)
            self._eigs[Hkey] = {"eigvec": U, "eigval": w}

        return self._eigs[Hkey]

    def diagonalizing_gates(self):
        r"""Sequence of gates that diagonalize the operator in the computational basis.

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        A ``DiagGatesUndefinedError`` is raised if no representation by decomposition is defined.

        .. seealso:: :meth:`~.Operator.compute_diagonalizing_gates`.

        Returns:
            list[.Operator] or None: a list of operators
        """

        eigen_vectors = self.eigendecomposition["eigvec"]
        return [qml.QubitUnitary(eigen_vectors.conj().T, wires=self.wires)]

    def eigvals(self):
        r"""Return the eigenvalues of the specified operator.

        This method uses pre-stored eigenvalues for standard observables where
        possible and stores the corresponding eigenvectors from the eigendecomposition.

        Returns:
            array: array containing the eigenvalues of the operator
        """
        return self.eigendecomposition["eigval"]

    def matrix(self, wire_order=None):
        """Representation of the operator as a matrix in the computational basis."""
        if wire_order is None:
            wire_order = self.wires
        mats = (
            expand_matrix(qml.matrix(op), op.wires, wire_order=wire_order)
            if isinstance(op, qml.Hamiltonian)
            else expand_matrix(op.matrix(), op.wires, wire_order=wire_order)
            for op in self.factors
        )

        return reduce(math.dot, mats)

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
        return "_ops" if all(op._queue_category == "_ops" for op in self.factors) else None

    def queue(self, context=qml.QueuingContext):
        """Updates each operator's owner to Prod, this ensures
        that the operators are not applied to the circuit repeatedly."""
        for op in self.factors:
            context.safe_update_info(op, owner=self)
        context.append(self, owns=self.factors)
        return self

    def adjoint(self):
        return Prod(*(qml.adjoint(factor) for factor in self.factors[::-1]))

    @property
    def arithmetic_depth(self) -> int:
        return 1 + max(factor.arithmetic_depth for factor in self.factors)

    def _simplify_factors(self, factors: Tuple[Operator]) -> Tuple[complex, Operator]:
        """Reduces the depth of nested factors and groups identical factors.

        Returns:
            Tuple[complex, List[~.operation.Operator]: tuple containing the global phase and a list
            of the simplified factors
        """
        new_factors = ProductFactorsGrouping()

        for factor in factors:
            simplified_factor = factor.simplify()
            new_factors.add(factor=simplified_factor)
        new_factors.queue_factors(wires=self.wires)
        return new_factors.global_phase, new_factors.queue

    def simplify(self) -> Union["Prod", Sum]:
        global_phase, factors = self._simplify_factors(factors=self.factors)

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

        return Sum(*factors) if global_phase == 1 else qml.s_prod(global_phase, Sum(*factors))

    @property
    def hash(self):
        if self._hash is None:
            self._hash = hash(
                (str(self.name), str([factor.hash for factor in _prod_sort(self.factors)]))
            )
        return self._hash


def _prod_sort(op_list, wire_map: dict = None):
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


class ProductFactorsGrouping:
    """Utils class used for grouping identical product factors."""

    _pauli_mult = {
        "II": (1.0, "I"),
        "IX": (1.0, "X"),
        "IY": (1.0, "Y"),
        "IZ": (1.0, "Z"),
        "XI": (1.0, "X"),
        "XX": (1.0, "I"),
        "XY": (1.0j, "Z"),
        "XZ": (-1.0j, "Y"),
        "YI": (1.0, "Y"),
        "YX": (-1.0j, "Z"),
        "YY": (1.0, "I"),
        "YZ": (1.0j, "X"),
        "ZI": (1.0, "Z"),
        "ZX": (1.0j, "Y"),
        "ZY": (-1.0j, "X"),
        "ZZ": (1.0, "I"),
    }
    _paulis = {"X": PauliX, "Y": PauliY, "Z": PauliZ}

    def __init__(self):
        self._pauli_factors = {}  #  {wire: (pauli_coeff, pauli_word)}
        self._non_pauli_factors = {}  # {wires: [hash, pow_coeff, operator]}
        self.queue = ()
        self.global_phase = 1

    def add(self, factor: Operator):
        """Add operator to the factors queue.

        Args:
            factor (Operator): Operator to add to the queue.
        """
        if isinstance(factor, Prod):
            for prod_factor in factor.factors:
                self.add(prod_factor)
        elif isinstance(factor, Sum):
            self.queue += (factor.summands,)
        elif not isinstance(factor, qml.Identity):
            if isinstance(factor, SProd):
                self.global_phase *= factor.scalar
                factor = factor.base
            wires = factor.wires
            if isinstance(factor, (qml.Identity, qml.PauliX, qml.PauliY, qml.PauliZ)):
                self._add_pauli_factor(factor=factor, wires=wires)
                self._queue_non_pauli_factors(wires=wires)
            else:
                self._add_non_pauli_factor(factor=factor, wires=wires)
                self._queue_pauli_factors(wires=wires)

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
        label = factor.label()
        old_coeff, old_word = self._pauli_factors.get(wire, (1, "I"))
        coeff, new_word = self._pauli_mult[old_word + label]
        self._pauli_factors[wire] = old_coeff * coeff, new_word

    def _add_non_pauli_factor(self, factor: Operator, wires: List[int]):
        """Adds the given non-Pauli factor to the temporary ``self._non_pauli_factors`` dictionary.
        If there alerady exists an identical operator in the dictionary, the two are grouped
        together.

        If there isn't an identical operator in the dictionary, all non Pauli factors that act on
        the same wires are queued.

        Args:
            factor (Operator): Factor to be added.
            wires (List[int]): Factor wires. This argument is added to avoid calling
                ``factor.wires`` several times.
        """
        if isinstance(factor, Pow):
            coeff = factor.z
            factor = factor.base
        else:
            coeff = 1
        op_hash = factor.hash
        old_hash, old_coeff, old_op = self._non_pauli_factors.get(wires, [None, None, None])
        # TODO: Should we create an abstract `Rotation` class and make inherit all rotations from
        # this class? We could group a lot of operator functionalities, and this check will be
        # easier.
        if isinstance(old_op, (qml.RX, qml.RY, qml.RZ)) and factor.name == old_op.name:
            self._non_pauli_factors[wires] = [
                op_hash,
                old_coeff,
                factor.__class__(factor.data[0] + old_op.data[0], wires),
            ]
        elif op_hash == old_hash:
            self._non_pauli_factors[wires][1] += coeff
        else:
            self._queue_non_pauli_factors(wires=wires)
            self._non_pauli_factors[wires] = [op_hash, copy(coeff), factor]

    def _queue_non_pauli_factors(self, wires: List[int]):
        """Remove all factors from the ``self._non_pauli_factors`` dictionary that act on the given
        wires and queue them.

        Args:
            wires (List[int]): Wires of the operators to be queued.
        """
        if not self._non_pauli_factors:
            return
        for wire in wires:
            for key, (_, pow_coeff, op) in list(self._non_pauli_factors.items()):
                if wire in key:
                    self._non_pauli_factors.pop(key)
                    if pow_coeff == 0:
                        continue
                    # TODO: Should we create a qml.pow function that calls op.pow() if possible?
                    op = Pow(base=op, z=pow_coeff).simplify() if pow_coeff != 1 else op
                    if isinstance(op, Prod):
                        self.queue += tuple(
                            (factor,)
                            for factor in op.factors
                            if not isinstance(factor, qml.Identity)
                        )
                    elif not isinstance(op, qml.Identity):
                        self.queue += ((op,),)

    def _queue_pauli_factors(self, wires: List[int]):
        """Remove all Pauli factors from the ``self._pauli_factors`` dictionary that act on the
        given wires and queue them.

        Args:
            wires (List[int]): Wires of the operators to be queued.
        """
        if not self._pauli_factors:
            return
        for wire in wires:
            pauli_coeff, pauli_word = self._pauli_factors.pop(wire, (1, "I"))
            if pauli_word != "I":
                pauli_op = self._paulis[pauli_word](wire)
                self.queue += ((pauli_op,),)
                self.global_phase *= pauli_coeff

    def queue_factors(self, wires: List[int]):
        """Remove all factors from the ``self._pauli_factors`` and ``self._non_pauli_factors``
        dictionaries that act on the given wires and queue then."""
        self._queue_pauli_factors(wires=wires)
        self._queue_non_pauli_factors(wires=wires)
