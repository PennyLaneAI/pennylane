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
from functools import reduce

import numpy as np

import pennylane as qml
from pennylane import math
from pennylane.operation import Operator


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


def _sum(mats_gen, dtype=None, cast_like=None):
    r"""Private method to compute the sum of matrices.

    Args:
        mats_gen (Generator): a python generator which produces the matrices which
            will be summed together.

    Keyword Args:
        dtype (str): a string representing the data type of the entries in the result.
        cast_like (Tensor): a tensor with the desired data type in its entries.

    Returns:
        res (Tensor): the tensor which is the sum of the matrices obtained from mats_gen.
    """
    # Note this method is currently inefficient (improve addition by looking at wire subgroups)
    res = reduce(math.add, mats_gen)

    if dtype is not None:
        res = math.cast(res, dtype)
    if cast_like is not None:
        res = math.cast_like(res, cast_like)

    return res


class Sum(Operator):
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
        tensor([-0.09347337, -0.18884787, -0.28818254], requires_grad=True)"""

    _eigs = {}  # cache eigen vectors and values like in qml.Hermitian

    def __init__(
        self, *summands: Operator, do_queue=True, id=None
    ):  # pylint: disable=super-init-not-called
        """Initialize a Symbolic Operator class corresponding to the Sum of operations."""
        self._name = "Sum"
        self._id = id
        self.queue_idx = None

        if len(summands) < 2:
            raise ValueError(f"Require at least two operators to sum; got {len(summands)}")

        self.summands = summands
        self.data = [s.parameters for s in self.summands]
        self._wires = qml.wires.Wires.all_wires([s.wires for s in self.summands])

        if do_queue:
            self.queue()

    def __repr__(self):
        """Constructor-call-like representation."""
        return " + ".join([f"{f}" for f in self.summands])

    def __copy__(self):
        cls = self.__class__
        copied_op = cls.__new__(cls)
        copied_op.data = self.data.copy()  # copies the combined parameters
        copied_op.summands = tuple(s.__copy__() for s in self.summands)

        for attr, value in vars(self).items():
            if attr not in {"data", "summands"}:
                setattr(copied_op, attr, value)

        return copied_op

    @property
    def batch_size(self):
        """Batch size of input parameters."""
        raise ValueError("Batch size is not defined for Sum operators.")

    @property
    def ndim_params(self):
        """ndim_params of input parameters."""
        raise ValueError("Dimension of parameters is not currently implemented for Sum operators.")

    @property
    def num_wires(self):
        return len(self.wires)

    @property
    def num_params(self):
        return sum(op.num_params for op in self.summands)

    @property
    def is_hermitian(self):
        """If all of the terms in the sum are hermitian, then the Sum is hermitian."""
        return all(s.is_hermitian for s in self.summands)

    def terms(self):
        r"""Representation of the operator as a linear combination of other operators.

        .. math:: O = \sum_i c_i O_i

        A ``TermsUndefinedError`` is raised if no representation by terms is defined.

        .. seealso:: :meth:`~.Operator.compute_terms`

        Returns:
            tuple[list[tensor_like or float], list[.Operation]]: list of coefficients :math:`c_i`
            and list of operations :math:`O_i`
        """
        return [1.0] * len(self.summands), self.summands

    @property
    def eigendecomposition(self):
        r"""Return the eigendecomposition of the matrix specified by the Hermitian observable.

        This method uses pre-stored eigenvalues for standard observables where
        possible and stores the corresponding eigenvectors from the eigendecomposition.

        It transforms the input operator according to the wires specified.

        Returns:
            dict[str, array]: dictionary containing the eigenvalues and the eigenvectors of the operator
        """
        Hmat = self.matrix()
        Hmat = qml.math.to_numpy(Hmat)
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
        r"""Return the eigenvalues of the specified Hermitian observable.

        This method uses pre-stored eigenvalues for standard observables where
        possible and stores the corresponding eigenvectors from the eigendecomposition.

        Returns:
            array: array containing the eigenvalues of the Hermitian observable
        """
        return self.eigendecomposition["eigval"]

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
            wire_order (Iterable): global wire order, must contain all wire labels from the operator's wires

        Returns:
            tensor_like: matrix representation
        """

        def matrix_gen(summands, wire_order=None):
            """Helper function to construct a generator of matrices"""
            for op in summands:
                yield op.matrix(wire_order=wire_order)

        if wire_order is None:
            wire_order = self.wires

        return _sum(matrix_gen(self.summands, wire_order))

    @property
    def _queue_category(self):  # don't queue Sum instances because it may not be unitary!
        """Used for sorting objects into their respective lists in `QuantumTape` objects.
        This property is a temporary solution that should not exist long-term and should not be
        used outside of ``QuantumTape._process_queue``.

        Returns: None
        """
        return None

    def queue(self, context=qml.QueuingContext):
        """Updates each operator in the summands owner to Sum, this ensures
        that the summands are not applied to the circuit repeatedly."""
        for op in self.summands:
            context.safe_update_info(op, owner=self)
        return self
