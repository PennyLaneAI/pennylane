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
from typing import Tuple, Union

import numpy as np

import pennylane as qml
from pennylane import math
from pennylane.operation import Operator, expand_matrix
from pennylane.ops.op_math.sum import Sum


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
            expand_matrix(op.matrix(), op.wires, wire_order=wire_order)
            if not isinstance(op, qml.Hamiltonian)
            else expand_matrix(qml.matrix(op), op.wires, wire_order=wire_order)
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

    @classmethod
    def _simplify_factors(cls, factors: Tuple[Operator]) -> Tuple[Operator]:
        """Reduces the depth of nested factors.

        Returns:
            Tuple[List[~.operation.Operator], List[~.operation.Operator]: reduced sum and non-sum
            factors
        """
        new_factors = ()

        for factor in factors:
            if isinstance(factor, Prod):
                tmp_factors = cls._simplify_factors(factors=factor.factors)
                new_factors += tmp_factors
                continue
            simplified_factor = factor.simplify()
            if isinstance(simplified_factor, Prod):
                new_factors += tuple((factor,) for factor in simplified_factor.factors)
            elif isinstance(simplified_factor, Sum):
                new_factors += (simplified_factor.summands,)
            elif not isinstance(simplified_factor, qml.Identity):
                new_factors += ((simplified_factor,),)

        return new_factors

    def simplify(self) -> Union["Prod", Sum]:
        factors = self._simplify_factors(factors=self.factors)
        factors = list(itertools.product(*factors))
        if len(factors) == 1:
            factor = factors[0]
            return factor[0] if len(factor) == 1 else Prod(*factor)
        factors = [Prod(*factor).simplify() if len(factor) > 1 else factor[0] for factor in factors]

        return Sum(*factors)


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
    if np.intersect1d(wires1, wires2).size != 0:
        return False
    return np.min(wires1) > np.min(wires2)
