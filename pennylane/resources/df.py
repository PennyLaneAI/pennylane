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


class SQ:
    """This class contains the functionality for estimating the number of non-Clifford gates and
    logical qubits for quantum algorithms in second quantization based on the double factorization
    approach.

    Users must provide the following inputs for their system of interest:
    - one-electron integrals
    - two-electron integrals in the chemist notation
    - target error in quantum phase estimation

    """

    def __init__(self, one_electron, two_electron, error, tol):
        self.one_electron = one_electron
        self.two_electron = two_electron
        self.error = error
        self.tol = tol

        self.factors, self.vals, self.vecs = self.factorize(two_electron, tol)
        self.lamb = self.norm(one_electron, two_electron, self.vals)

        self.rank_r, self.rank_m = self.compute_rank(self.factors, self.vals)

        self.br = 7
        self.aleph = 10
        self.beth = 20

        self.n = two_electron.shape[0] * 2

        self.rank_r = len(self.factors)
        self.rank_m = np.mean([len(v) for v in self.vals])

        self.g_cost = self.gate_cost(
            self.n, self.lamb, self.error, self.rank_r, self.rank_m, self.br, self.aleph, self.beth
        )
        self.q_cost = self.qubit_cost(
            self.n, self.lamb, self.error, self.rank_r, self.rank_m, self.br, self.aleph, self.beth
        )
