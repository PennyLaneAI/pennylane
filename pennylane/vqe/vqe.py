# Copyright 2019 Xanadu Quantum Technologies Inc.

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
This submodule contains functionality for running Variational Quantum Eigensolver (VQE)
computations using PennyLane.
"""
import numpy as np
from pennylane.ops import Hermitian

class Hamiltonian:
    """
    Lightweight class for representing Hamiltonians for Variational Quantum Eigensolver problems.

    Hamiltonians can be expressed as linear combinations of observables, e.g.,
    :math:`\sum_{k=0}^{N-1}` c_k O_k`.
    This class keeps track of the terms (coefficients and observables) separately.


    Args:
        coeffs (Iterable[float]): coefficients of the Hamiltonian expression
        ops (Iterable[Observable]): observables in the Hamiltonian expression
        tol (float): tolerance used to determine if ops are Hermitian
    """

    def __init__(self, coeffs, ops, tol=1e-5):

        if len(coeffs) != len(ops):
            raise ValueError("Could not create valid Hamiltonian; "
                             "number of coefficients and operators does not match.")
        if any(np.imag(coeffs) != 0):
            raise ValueError("Could not create valid Hamiltonian; "
                             "coefficients are not real-valued.")
        # TODO: to make easier, add a boolean attribute `hermitian` to all pennylane ops
        numeric_ops = [op.parameters[0] for op in ops if isinstance(op, Hermitian)]
        for A in numeric_ops:
            if not np.allclose(A, np.conj(A).T, rtol=tol, atol=0):
                raise ValueError("Could not create valid Hamiltonian; "
                                 "one or more ops are not Hermitian.")

        self._coeffs = coeffs
        self._ops = ops

    @property
    def coeffs(self):
        return self._coeffs

    @property
    def ops(self):
        return self._ops

    @property
    def terms(self):
        """
        The terms of the Hamiltonian expression :math:`\sum_{k=0}^{N-1}` c_k O_k`


        Returns:
            (coeffs, ops), where coeffs/ops is a tuple of floats/operations of length N
        """
        return self.coeffs, self.ops