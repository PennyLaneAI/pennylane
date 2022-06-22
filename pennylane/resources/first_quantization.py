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
from scipy import integrate

from pennylane import numpy as np


def success_prob(n, br):
    r"""Return the probability of success for state preparation.

    The expression for computing the probability of success is taken from Eqs. (59, 60) of
    [`PRX Quantum 2, 040332 (2021) <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_].

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


def norm_fq(eta, n, omega, error, br=7, charge=0):
    r"""Return the 1-norm of a first-quantized Hamiltonian in the plane-wave basis.

    The expressions needed for computing the norm are taken from
    [`PRX Quantum 2, 040332 (2021) <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_].
    For numerical convenience, we have used the following modified expressions for computing
    parameters that contain a sum over the elements, :math:`\nu`, of the set of reciprocal lattice
    vectors, :math:`G_0`. For :math:`\lambda_{\nu}` defined in Eq. (25) as

    .. math::

        \lambda_{\nu} = \sum_{\nu \in G_0} \frac{1}{\left \| \nu \right \|^2},

    we have used

    .. math::

        \lambda_{\nu} = 4\pi \left ( \frac{\sqrt{3}}{2} N^{1/3} - 1 \right) + 3 - \frac{3}{N^{1/3}}
        + 3 \int_{x=1}^{N^{1/3}} \int_{y=1}^{N^{1/3}} \frac{1}{x^2 + y^2} dydx.

    We also need to compute :math:`\lambda^{\alpha}_{\nu}` which is defined in Eq. (123) as

    .. math::

        \lambda^{\alpha}_{\nu} = \alpha \sum_{\nu \in G_0} \frac{\left \lceil M(2^{\mu - 2}) / \left
        \| \nu \right \|^2 \right \rceil}{M2^{2\mu - 4}},

    which we compute, following Eq. (113) for :math:`\alpha = 1`, as

    .. math::

        \lambda^{1}_{\nu} = \lambda_{\nu} + \frac{4}{2^{n_m}} (7 \times 2^{n_p + 1} + 9 n_p -
        11 - 3 \times 2^{-n_p}),

    where :math:`M = 2^{n_m}` and :math:`n_m` is defined in Eq. (132). Finally, for :math:`p_\{nu}`
    defined in Eq. (128) as

    .. math::

        p_{\nu} = \sum_{\mu = 2}^{n_p + 1} \sum_{\nu \in B_{\mu}} \frac{\left \lceil M(2^{\mu-2} /
        \left \| \nu \right \|)^2 \right \rceil}{M 2^{2\mu} 2^{n_{\mu} + 1}},

    we use the upper bound from Eq. (29) in [`arXiv:1807.09802v2 <https://arxiv.org/abs/1807.09802v2>`_]
    which gives :math:`p_{\nu} = 0.2398`.

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
    >>> norm_fq(eta, n, omega, error)
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
    # computed using Eq. (113)
    lambda_nu_1 = lambda_nu + 4 / 2**n_m * (
        7 * 2 ** (n_p + 1) + 9 * n_p - 11 - 3 * 2 ** (-1 * n_p)
    )

    p_nu = 0.2398  # upper bound from Eq. (29) in arxiv:1807.09802
    p_nu_amp = np.sin(3 * np.arcsin(np.sqrt(p_nu))) ** 2  # p_nu_amp is taken from Eq. (129)

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
