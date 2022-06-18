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
This module contains the functions needed for resource estimation with double factorization method.
"""
from pennylane import numpy as np
from pennylane.operation import Operation, AnyWires
from .factorization import factorize


class SQ(Operation):
    """Contains the functionality for estimating the number of non-Clifford gates and logical qubits
    for quantum algorithms in second quantization based on the double factorization method.

    Args:
        one_electron (array[array[float]]): one-electron integrals
        two_electron (tensor_like): two-electron integrals
        error (float): target error in the algorithm
        rank_r (int): rank of the first factorization of the two-electron integral tensor
        rank_m (float): average rank of the second factorization of the two-electron integral tensor
        tol_factor (float): threshold error value for discarding the negligible factors
        tol_eigval (float): threshold error value for discarding the negligible factor eigenvalues
        br (int): number of bits for ancilla qubit rotation
        alpha (int): number of bits for the keep register
        beta (int): number of bits for the rotation angles
    """

    def __init__(
        self,
        one_electron,
        two_electron,
        error=0.0016,
        rank_r=None,
        rank_m=None,
        tol_factor=1.0e-5,
        tol_eigval=1.0e-5,
        br=7,
        alpha=10,
        beta=20,
    ):

        self.one_electron = one_electron
        self.two_electron = two_electron
        self.error = error
        self.rank_r = rank_r
        self.rank_m = rank_m
        self.tol_factor = tol_factor
        self.tol_eigval = tol_eigval
        self.br = br
        self.alpha = alpha
        self.beta = beta

        self.n = two_electron.shape[0] * 2

        self.factors, self.eigvals, self.eigvecs = factorize(
            self.two_electron, self.tol_factor, self.tol_eigval
        )

        self.lamb = self.norm(self.one_electron, self.two_electron, self.eigvals)

        if not rank_r:
            self.rank_r = len(self.factors)
        if not rank_m:
            self.rank_m = np.mean([len(v) for v in self.eigvals])

        self.gates = self.gate_cost(
            self.n, self.lamb, self.error, self.rank_r, self.rank_m, self.br, self.alpha, self.beta
        )
        self.qubits = self.qubit_cost(
            self.n, self.lamb, self.error, self.rank_r, self.rank_m, self.br, self.alpha, self.beta
        )

    num_wires = AnyWires
    grad_method = None
