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
# pylint: disable= too-many-arguments
from pennylane import numpy as np


def success_prob(n_planewaves, br):
    r"""Return the probability of success for state preparation.

    The expression for computing the probability of success is taken from
    [`PRX Quantum 2, 040332 (2021) <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_],
    Eqs. (59-60).

    Args:
        n_planewaves (int): number of plane waves
        br (int): number of bits for ancilla qubit rotation

    Returns:
        float: probability of success for state preparation

    **Example**

    >>> success_prob(10000, 7)
    0.9998814293823286
    """
    c = n_planewaves / 2 ** np.ceil(np.log2(n_planewaves))
    d = 2 * np.pi / 2**br

    theta = d * np.round((1 / d) * np.arcsin(np.sqrt(1 / (4 * c))))

    p = c * ((1 + (2 - 4 * c) * np.sin(theta) ** 2) ** 2 + np.sin(2 * theta) ** 2)

    return p


def norm(eta, n_planewaves, omega, error, br=7, charge=0):
    r"""Return the 1-norm of a first-quantized Hamiltonian in the plane-wave basis.

    The expression for computing the norm is taken from
    [`PRX Quantum 2, 040332 (2021) <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_].

    Args:
        eta (int): number of electrons
        n_planewaves (int): number of basis states
        omega (float): unit cell volume
        error (float): target error in the algorithm
        br (int): number of bits for ancilla qubit rotation
        charge (int): total electric charge of the system

    Returns:
        float: 1-norm of a first-quantized Hamiltonian in the plane-wave basis

    **Example**

    >>> eta = 156
    >>> n_planewaves = 10000
    >>> omega = 1145.166
    >>> error = 0.001
    >>> norm(eta, n, omega, error)
    1254385.059691027
    """
    l_z = eta + charge

    # target error in the qubitization of U+V which we set to be 0.01 of the algorithm error
    error_uv = 0.01 * error

    # n_p is taken from Eq. (22)
    n_p = int(np.ceil(np.log2(n_planewaves ** (1 / 3) + 1)))

    l_nu = 4 * np.pi * n_planewaves ** (2 / 3)  # computed from Eqs. (25) and (103)
    p_nu = 0.2398  # upper bound from Eq. (29) in arxiv:1807.09802
    n_m = np.log2(  # taken from Eq. (132)
        (2 * eta)
        / (error_uv * np.pi * omega ** (1 / 3))
        * (eta - 1 + 2 * l_z)
        * (7 * 2 ** (n_p + 1) - 9 * n_p - 11 - 3 * 2 ** (-1 * n_p))
    )
    # eps is taken from Eq. (113)
    eps = 4 / 2**n_m * (7 * 2 ** (n_p + 1) + 9 * n_p - 11 - 3 * 2 ** (-1 * n_p))
    l_nu_1 = l_nu + eps

    # lambda_u and lambda_v are taken from Eq. (25)
    lambda_u = eta * l_z * l_nu / (np.pi * omega ** (1 / 3))
    lambda_v = eta * (eta - 1) * l_nu / (2 * np.pi * omega ** (1 / 3))

    # lambda_t_p is taken from Eq. (71)
    lambda_t_p = 6 * eta * np.pi**2 / omega ** (2 / 3) * 2 ** (2 * n_p - 2)

    # lambda_u_1 and lambda_v_1 are taken from Eq. (124)
    lambda_u_1 = lambda_u * l_nu_1 / l_nu
    lambda_v_1 = lambda_v * l_nu_1 / l_nu

    # p_eq is taken from Eq. (63)
    p_eq = success_prob(3, 8) * success_prob(3 * eta + 2 * charge, br) * success_prob(eta, br) ** 2

    # the equations for computing the final lambda value are taken from Eq. (126)
    lambda_a = lambda_t_p + lambda_u_1 + lambda_v_1
    lambda_b = (lambda_u_1 + lambda_v_1 / (1 - 1 / eta)) / p_nu

    return np.maximum(lambda_a, lambda_b) / p_eq
