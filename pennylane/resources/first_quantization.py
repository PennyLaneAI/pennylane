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
from scipy import integrate


def success_prob(n, br):
    r"""Return the probability of success for state preparation.

    The expression for computing the probability of success is taken from
    [`PRX Quantum 2, 040332 (2021) <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_],
    Eqs. (59-60).

    Args:
        n (int): number of plane waves
        br (int): number of bits for ancilla qubit rotation

    Returns:
        float: probability of success for state preparation

    **Example**

    >>> success_prob(10000, 7)
    0.9998814293823286
    """
    if n <= 0 or not isinstance(n, int):
        raise ValueError("The number of planewaves must be a positive integer.")

    if br <= 0 or not isinstance(br, int):
        raise ValueError("br must be a positive integer.")

    c = n / 2 ** np.ceil(np.log2(n))
    d = 2 * np.pi / 2**br

    theta = d * np.round((1 / d) * np.arcsin(np.sqrt(1 / (4 * c))))

    p = c * ((1 + (2 - 4 * c) * np.sin(theta) ** 2) ** 2 + np.sin(2 * theta) ** 2)

    return p


def norm(eta, n, omega, error, br=7, charge=0):
    r"""Return the 1-norm of a first-quantized Hamiltonian in the plane-wave basis.

    The expression for computing the norm is taken from
    [`PRX Quantum 2, 040332 (2021) <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_].

    Args:
        eta (int): number of electrons
        n (int): number of basis states
        omega (float): unit cell volume
        error (float): target error in the algorithm
        br (int): number of bits for ancilla qubit rotation
        charge (int): total electric charge of the system

    Returns:
        float: 1-norm of a first-quantized Hamiltonian in the plane-wave basis

    **Example**

    >>> eta = 156
    >>> n = 10000
    >>> omega = 1145.166
    >>> error = 0.001
    >>> norm(eta, n, omega, error)
    1254385.059691027
    """
    if n <= 0 or not isinstance(n, int):
        raise ValueError("The number of planewaves must be a positive integer.")

    if eta <= 0 or not isinstance(n, int):
        raise ValueError("The number of electrons must be a positive integer.")

    if omega <= 0:
        raise ValueError("The unit cell volume must be a positive number.")

    if error <= 0.0:
        raise ValueError("The target error must be greater than zero.")

    if br <= 0 or not isinstance(br, int):
        raise ValueError("br must be a positive integer.")

    if not isinstance(charge, int):
        raise ValueError("system charge must be an integer.")

    l_z = eta + charge

    # target error in the qubitization of U+V which we set to be 0.01 of the algorithm error
    error_uv = 0.01 * error

    # n_p is taken from Eq. (22)
    n_p = int(np.ceil(np.log2(n ** (1 / 3) + 1)))

    n0 = n ** (1 / 3)
    lambda_nu = (
        4 * np.pi * (np.sqrt(3) * n ** (1 / 3) / 2 - 1)
        + 3
        - 3 / n ** (1 / 3)
        + 3 * integrate.nquad(lambda x, y: 1 / (x**2 + y**2), [[1, n0], [1, n0]])[0]
    )
    n_m = np.log2(  # taken from Eq. (132)
        (2 * eta)
        / (error_uv * np.pi * omega ** (1 / 3))
        * (eta - 1 + 2 * l_z)
        * (7 * 2 ** (n_p + 1) - 9 * n_p - 11 - 3 * 2 ** (-1 * n_p))
    )
    # eps is taken from Eq. (113)
    eps = 4 / 2**n_m * (7 * 2 ** (n_p + 1) + 9 * n_p - 11 - 3 * 2 ** (-1 * n_p))
    lambda_nu_1 = lambda_nu + eps

    p_nu_amp = 0.2398  # upper bound from Eq. (29) in arxiv:1807.09802

    # lambda_u and lambda_v are taken from Eq. (25)
    lambda_u = eta * l_z * lambda_nu / (np.pi * omega ** (1 / 3))
    lambda_v = eta * (eta - 1) * lambda_nu / (2 * np.pi * omega ** (1 / 3))

    # lambda_t_p is taken from Eq. (71)
    lambda_t_p = 6 * eta * np.pi**2 / omega ** (2 / 3) * 2 ** (2 * n_p - 2)

    # lambda_u_1 and lambda_v_1 are taken from Eq. (124)
    lambda_u_1 = lambda_u * lambda_nu_1 / lambda_nu
    lambda_v_1 = lambda_v * lambda_nu_1 / lambda_nu

    # p_eq is taken from Eq. (63)
    p_eq = success_prob(3, 8) * success_prob(3 * eta + 2 * charge, br) * success_prob(eta, br) ** 2

    # the equations for computing the final lambda value are taken from Eq. (126)
    lambda_a = lambda_t_p + lambda_u_1 + lambda_v_1
    lambda_b = (lambda_u_1 + lambda_v_1 / (1 - 1 / eta)) / p_nu_amp

    return np.maximum(lambda_a, lambda_b) / p_eq
