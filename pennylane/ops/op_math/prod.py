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
from copy import copy
from functools import reduce

import numpy as np

import pennylane as qml
from pennylane import math
from pennylane.operation import Operator, expand_matrix


def prod(*ops, do_queue=True, id=None):
    """Represent the tensor product (or matrix product) between operators."""
    return Prod(*ops, do_queue=do_queue, id=id)


def _prod(mats_gen):  # TODO: current multiplication always expands mats, this is not always necessary
    """Multiply matrices together"""
    res = reduce(math.dot, mats_gen)   # Inefficient method (should group by wires like in tensor class)
    return res


class Prod(Operator):
    """Arithmetic operator subclass representing the scalar product of an
    operator with the given scalar."""
    _name = "Prod"
    _eigs = {}  # cache eigen vectors and values like in qml.Hermitian

    def __init__(
            self, *factors, do_queue=True, id=None
    ):  # pylint: disable=super-init-not-called
        """Initialize a Prod instance """
        self._id = id
        self.queue_idx = None

        if len(factors) < 2:
            raise ValueError(f"Require at least two operators to multiply; got {len(factors)}")

        self.factors = factors
        self._wires = qml.wires.Wires.all_wires([s.wires for s in self.factors])

        if do_queue:
            self.queue()

    def __repr__(self):
        """Constructor-call-like representation."""
        return " @ ".join([f"{f}" for f in self.factors])

    def __copy__(self):
        cls = self.__class__
        copied_op = cls.__new__(cls)
        copied_op.factors = tuple(f.__copy__() for f in self.factors)
        copied_op.data = self.data.copy()  # copies the combined parameters

        for attr, value in vars(self).items():
            if attr not in {"data", "factors"}:
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
            op.data = copy(new_entry)

    @property
    def batch_size(self):
        """Batch size of input parameters."""
        raise ValueError("Batch size is not defined for Prod operators.")

    @property
    def ndim_params(self):
        """ndim_params of input parameters."""
        raise ValueError("Dimension of parameters is not currently implemented for Prod operators.")

    @property
    def num_params(self):
        return sum(op.num_params for op in self.factors)

    @property
    def num_wires(self):
        return len(self.wires)

    @property
    def is_hermitian(self, run_check=True):   # TODO: cache this value, this check is expensive!
        """check if the product operator is hermitian"""
        if run_check:
            mat = self.matrix()
            adj_mat = qml.math.conjugate(qml.math.transpose(mat))
            if np.allclose(mat, adj_mat):
                return True
            return False
        raise qml.operation.IsHermitianUndefinedErrors

    def decomposition(self):
        """decomposition of the operator into a product of operators. """
        return list(self.factors)

    @property
    def eigendecomposition(self):
        r"""Return the eigendecomposition of the matrix specified by the operator.

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
        r"""Return the eigenvalues of the specified operator.

        This method uses pre-stored eigenvalues for standard observables where
        possible and stores the corresponding eigenvectors from the eigendecomposition.

        Returns:
            array: array containing the eigenvalues of the operator
        """
        return self.eigendecomposition["eigval"]

    def matrix(self, wire_order=None):
        """Representation of the operator as a matrix in the computational basis."""

        def matrix_gen(operators, wire_order=None):
            """Helper function to construct a generator of matrices"""
            for op in operators:
                yield expand_matrix(op.matrix(), op.wires, wire_order=wire_order)

        if wire_order is None:
            wire_order = self.wires

        return _prod(matrix_gen(self.factors, wire_order=wire_order))

    @property
    def _queue_category(self):  # don't queue Prod instances because it may not be unitary!
        """Used for sorting objects into their respective lists in `QuantumTape` objects.
        This property is a temporary solution that should not exist long-term and should not be
        used outside of ``QuantumTape._process_queue``.

        Returns: None
        """
        return None

    def queue(self, context=qml.QueuingContext):
        """Updates each operator's owner to Prod, this ensures
        that the operators are not applied to the circuit repeatedly."""
        for op in self.factors:
            context.safe_update_info(op, owner=self)
        return self
