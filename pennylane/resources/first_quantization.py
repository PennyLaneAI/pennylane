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
This module contains the functions needed for estimating the number of logical qubits and
non-Clifford gates for quantum algorithms in first quantization using a plane-wave basis.
"""

from pennylane import numpy as np


def success_prob(n, br):
    r"""Return the probability of success for state preparation.

    The expression for computing the probability of success is taken from
    [`arXiv:2105.12767 <https://arxiv.org/abs/2105.12767>`_].

    Args:
        n (int): number of basis states
        br (int): number of bits for ancilla qubit rotation

    Returns:
        float: probability of success for state preparation

    **Example**

    >>> success_prob(10000, 7)
    0.9998814293823286
    """
    c = n / 2 ** np.ceil(np.log2(n))
    d = 2 * np.pi / 2**br

    theta = d * np.round((1 / d) * np.arcsin(np.sqrt(1 / (4 * c))))

    p = c * ((1 + (2 - 4 * c) * np.sin(theta) ** 2) ** 2 + np.sin(2 * theta) ** 2)

    return p


def norm(eta, n, omega, br=7, charge=0):
    r"""Return the 1-norm of a first-quantized Hamiltonian in a plane-wave basis.

    The expression for computing the norm is taken from
    [`arXiv:2105.12767 <https://arxiv.org/abs/2105.12767>`_].

    Args:
        eta (int): number of electrons
        n (int): number of basis states
        omega (float): unit cell volume
        br (int): number of bits for ancilla qubit rotation
        charge (int): total electric charge of the system

    Returns:
        float: the 1-norm of a first-quantized Hamiltonian in a plane-wave basis

    **Example**

    >>> eta = 156
    >>> n = 100000
    >>> omega = 169.69608
    >>> norm(eta, n, omega)
    5128920.595980267
    """
    n_p = np.ceil(np.log2(n ** (1 / 3) + 1))
    l_nu = 4 * np.pi * n ** (1 / 3)  # l_nu = self.lambda_nu(N)
    p_nu = 0.2398  # upper bound from Eq. (29) in arxiv:1807.09802
    p_eq = success_prob(3, 8) * success_prob(3 * eta + 2 * charge, br) * success_prob(eta, br) ** 2

    # lambda_t = (2 ** (2 * n_p - 1) - 1) * (6 * eta * pi ** 2) / (omega ** (2 / 3))
    lambda_t = (2 ** (n_p - 1) - 1) ** 2 * (6 * eta * np.pi**2) / (omega ** (2 / 3))
    lambda_u = l_nu * eta * (eta + charge) / (np.pi * omega ** (1 / 3))
    lambda_v = l_nu * eta * (eta - 1) / (2 * np.pi * omega ** (1 / 3))

    lambda_1 = lambda_t + lambda_u + lambda_v
    lambda_2 = (lambda_u + lambda_v) / (p_nu * (1 - 1 / eta))

    return np.maximum(lambda_1, lambda_2) / p_eq
