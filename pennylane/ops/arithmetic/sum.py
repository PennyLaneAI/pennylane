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
import warnings
import numpy as np
import pennylane as qml
from pennylane import math
from pennylane.operation import Operator, expand_matrix


def sum(*summands):
    """Compute the sum of the provided terms"""
    return Sum(*summands)  # a wire order is required when combining operators of varying sizes


class Sum(Operator):
    """Arithmetic operator subclass representing the sum of operators"""

    _eigs = {}  # cache eigen vectors and values like in qml.Hermitian

    def __init__(self, *summands, do_queue=True, id=None):

        if len(summands) < 2:
            raise ValueError(f"Require at least two operators to sum; got {len(summands)}")

        self.summands = summands

        combined_wires = qml.wires.Wires.all_wires([s.wires for s in summands])
        combined_params = [s.parameters for s in summands]
        super().__init__(
            *combined_params, wires=combined_wires, do_queue=do_queue, id=id
        )
        self._name = "Sum"

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
    def num_wires(self):
        return len(self.wires)

    @property
    def is_hermitian(self):
        """If all of the terms in the sum are hermitian, then the Sum is hermitian."""
        return all([s.is_hermitian for s in self.summands])

    def terms(self):
        return [1.0]*len(self.summands), self.summands

    @property
    def eigendecomposition(self):
        """Return the eigendecomposition of the matrix specified by the Hermitian observable.

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
        """Compute diagonalizing_gates (only if op is hermitian)"""

        eigen_vectors = self.eigendecomposition["eigvec"]
        return [qml.QubitUnitary(eigen_vectors.conj().T, wires=self.wires)]

    def eigvals(self):
        """Return the eigenvalues of the specified Hermitian observable.

        This method uses pre-stored eigenvalues for standard observables where
        possible and stores the corresponding eigenvectors from the eigendecomposition.

        Returns:
            array: array containing the eigenvalues of the Hermitian observable
        """
        return self.eigendecomposition["eigval"]

    def matrix(self, wire_order=None):
        """Representation of the operator as a matrix in the computational basis."""

        def matrix_gen(summands, wire_order=None):
            """Helper function to construct a generator of matrices"""
            for op in summands:
                yield expand_matrix(op.matrix(), op.wires, wire_order=wire_order)

        if wire_order is None:
            wire_order = self.wires

        return self._sum(matrix_gen(self.summands, wire_order))

    def _sum(self, mats_gen, dtype=None, cast_like=None):
        """Super inefficient Sum method just as a proof of concept"""
        res = None
        for i, mat in enumerate(mats_gen):
            res = mat if i == 0 else math.add(res, mat)

        if dtype is not None:                     # additional casting logic
            res = math.cast(res, dtype)
        if cast_like is not None:
            res = math.cast_like(res, cast_like)

        return res

    @staticmethod
    def is_commuting(terms, wires):
        """Checks if a sequence of terms all commute with each other"""
        num_terms = len(terms)
        for i in range(num_terms):
            for j in range(i+1, num_terms):
                term_i = terms[i]
                term_j = terms[j]

                if term_i.wires.toset() == term_j.wires.toset():  # same wires no need to expand matrix
                    mat_i = term_i.matrix(wire_order=term_i.wires)
                    mat_j = term_j.matrix(wire_order=term_i.wires)

                else:
                    mat_i = expand_matrix(term_i.matrix(), term_i.wires, wires)
                    mat_j = expand_matrix(term_j.matrix(), term_j.wires, wires)

                if not math.allequal(mat_i @ mat_j, mat_j @ mat_i):
                    return False

        return True

    @property
    def _queue_category(self):  # don't queue Sum instances because it may not be unitary!
        """Used for sorting objects into their respective lists in `QuantumTape` objects.
        This property is a temporary solution that should not exist long-term and should not be
        used outside of ``QuantumTape._process_queue``.

        Returns: None
        """
        return None
