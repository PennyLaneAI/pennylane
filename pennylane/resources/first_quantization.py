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


def _cost_qrom_fq(lz):
    r"""Return the minimum number of Toffoli gates needed for erasing the output of a QROM.

    Args:
        lz (int): sum of the atomic numbers of nuclei

    Returns:
        int: the minimum cost of erasing the output of a QROM

    **Example**

    >>> _cost_qrom_min(100)
    21
    """
    k_f = np.floor(np.log2(lz) / 2)
    k_c = np.ceil(np.log2(lz) / 2)

    cost_f = int(2**k_f + np.ceil(2 ** (-1 * k_f) * lz))
    cost_c = int(2**k_c + np.ceil(2 ** (-1 * k_c) * lz))

    return min(cost_f, cost_c)


def unitary_cost_fq(n, eta, omega, error, lamb, br=7, charge=0):
    r"""Return the number of Toffoli gates needed to implement the qubitization unitary operator.
    ￼
    The expression for computing the cost is taken from Eq. (125) of
    [`10.1103/PRXQuantum.2.040332 <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_].

    Args:
        n (int): number of plane waves
        eta (int): number of electrons
        omega (float): unit cell volume
        error (float): target error in the algorithm
        lamb (float): 1-norm of the Hamiltonian
        br (int): number of bits for ancilla qubit rotation
        charge (int): total electric charge of the system

    Returns:
        int: the number of Toffoli gates needed to implement the qubitization unitary operator

    **Example**

    >>> n = 100000
    >>> eta = 156
    >>> omega = 169.69608
    >>> error = 0.01
    >>> lamb = 5128920.595980267
    >>> unitary_cost(n, eta, omega, error, lamb)
    12819

    .. details::
        :title: Theory

        The target algorithm error, :math:`\epsilon`, is distributed among four different sources of
        error following Eq. (131) of
        `10.1103/PRXQuantum.2.040332 <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_ such
        that

        .. math::

            \epsilon^2 \geq \epsilon_{qpe}^2 + (\epsilon_{\mathcal{M}} + \epsilon_R + \epsilon_T)^2,

        where :math:`\epsilon_{qpe}` is the quantum phase estimation error and
        :math:`\epsilon_{\mathcal{M}}`, :math:`\epsilon_R`, and :math:`\epsilon_T` are defined in
        Eqs. (132-134) of
        `10.1103/PRXQuantum.2.040332 <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_. Here,
        we assume :math:`\epsilon_{\mathcal{M}} = \epsilon_R = \epsilon_T = \alpha \epsilon` with a
        default value of  :math:`\alpha = 0.01` and obtain

        .. math::

            \epsilon_{qpe} = \sqrt{\epsilon^2 [1 - (3 \alpha)^2]}.

        Note that the user only needs to define the target algorithm error :math:`\epsilon`. The
        error distribution takes place inside the functions.
    """
    alpha = 0.01
    l_z = eta + charge
    l_nu = 2 * np.pi * n ** (2 / 3)

    # n_eta and n_etaz are defined in the third and second paragraphs of page 040332-15
    n_eta = np.ceil(np.log2(eta))
    n_etaz = np.ceil(np.log2(eta + 2 * l_z))

    # n_p is taken from Eq. (22)
    n_p = int(np.ceil(np.log2(n ** (1 / 3) + 1)))

    # errors in Eqs. (132-134) are set to be 0.01 of the algorithm error
    error_t = alpha * error
    error_r = alpha * error
    error_m = alpha * error

    # parameters taken from Eqs. (132-134) of PRX Quantum 2, 040332 (2021)
    n_t = int(np.log2(np.pi * lamb / error_t))  # Eq. (134)
    n_r = int(np.log2((eta * l_z * l_nu) / (error_r * omega ** (1 / 3))))  # Eq. (133)
    n_m = int(
        np.log2(  # Eq. (132)
            (2 * eta)
            / (error_m * np.pi * omega ** (1 / 3))
            * (eta - 1 + 2 * l_z)
            * (7 * 2 ** (n_p + 1) - 9 * n_p - 11 - 3 * 2 ** (-1 * n_p))
        )
    )

    e_r = _cost_qrom_fq(l_z)

    # taken from Eq. (125)
    cost = 2 * (n_t + 4 * n_etaz + 2 * br - 12) + 14 * n_eta + 8 * br - 36
    cost += 3 * n_p**2 + 15 * n_p - 7 + 4 * n_m * (n_p + 1)
    cost += l_z + e_r + 2 * (2 * n_p + 2 * br - 7) + 12 * eta * n_p
    cost += 5 * (n_p - 1) + 2 + 24 * n_p + 6 * n_p * n_r + 18
    cost += n_etaz + 2 * n_eta + 6 * n_p + n_m + 16

    return int(np.ceil(cost))


def estimation_cost_fq(lamb, error):
    r"""Return the number of calls to the unitary needed to achieve the desired error in quantum
    phase estimation.

    The expression for computing the cost is taken from Eq. (125) of
    [`10.1103/PRXQuantum.2.040332 <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_].

    Args:
        lamb (float): 1-norm of a second-quantized Hamiltonian
        error (float): target error in the algorithm

    Returns:
        int: number of calls to unitary

    **Example**

    >>> cost = estimation_cost(72.49779513025341, 0.001)
    >>> print(cost)
    113880

    .. details::
        :title: Theory

        The target algorithm error, :math:`\epsilon`, is distributed among four different sources of
        error following Eq. (131) of
        `10.1103/PRXQuantum.2.040332 <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_ such
        that

        .. math::

            \epsilon^2 \geq \epsilon_{qpe}^2 + (\epsilon_{\mathcal{M}} + \epsilon_R + \epsilon_T)^2,

        where :math:`\epsilon_{qpe}` is the quantum phase estimation error and
        :math:`\epsilon_{\mathcal{M}}`, :math:`\epsilon_R`, and :math:`\epsilon_T` are defined in
        Eqs. (132-134) of
        `10.1103/PRXQuantum.2.040332 <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_. Here,
        we assume :math:`\epsilon_{\mathcal{M}} = \epsilon_R = \epsilon_T = \alpha \epsilon` with a
        default value of  :math:`\alpha = 0.01` and obtain

        .. math::

            \epsilon_{qpe} = \sqrt{\epsilon^2 [1 - (3 \alpha)^2]}.

        Note that the user only needs to define the target algorithm error :math:`\epsilon`. The
        error distribution takes place inside the functions.
    """
    alpha = 0.01
    # qpe_error obtained to satisfy inequality (131)
    error_qpe = np.sqrt(error**2 * (1 - (3 * alpha) ** 2))

    return int(np.ceil(np.pi * lamb / (2 * error_qpe)))


def gate_cost_fq(n, eta, omega, error, lamb, br=7, charge=0):
    r"""Return the total number of Toffoli gates needed to implement the first quantization
    algorithm.

    The expression for computing the cost is taken from Eq. (125) of
    [`10.1103/PRXQuantum.2.040332 <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_].

    Args:
        n (int): number of plane waves
        eta (int): number of electrons
        omega (float): unit cell volume
        error (float): target error in the algorithm
        lamb (float): 1-norm of the Hamiltonian
        br (int): number of bits for ancilla qubit rotation
        charge (int): total electric charge of the system

    Returns:
        int: the number of Toffoli gates needed to implement the first quantization algorithm

    **Example**

    >>> n = 100000
    >>> eta = 156
    >>> omega = 169.69608
    >>> error = 0.01
    >>> lamb = 5128920.595980267
    >>> gate_cost(n, eta, omega, error, lamb)
    10327614069516

    .. details::
        :title: Theory

        The target algorithm error, :math:`\epsilon`, is distributed among four different sources of
        error following Eq. (131) of
        `10.1103/PRXQuantum.2.040332 <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_ such
        that

        .. math::

            \epsilon^2 \geq \epsilon_{qpe}^2 + (\epsilon_{\mathcal{M}} + \epsilon_R + \epsilon_T)^2,

        where :math:`\epsilon_{qpe}` is the quantum phase estimation error and
        :math:`\epsilon_{\mathcal{M}}`, :math:`\epsilon_R`, and :math:`\epsilon_T` are defined in
        Eqs. (132-134) of
        `10.1103/PRXQuantum.2.040332 <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_. Here,
        we assume :math:`\epsilon_{\mathcal{M}} = \epsilon_R = \epsilon_T = \alpha \epsilon` with a
        default value of  :math:`\alpha = 0.01` and obtain

        .. math::

            \epsilon_{qpe} = \sqrt{\epsilon^2 [1 - (3 \alpha)^2]}.

        Note that the user only needs to define the target algorithm error :math:`\epsilon`. The
        error distribution takes place inside the functions.
    """
    e_cost = estimation_cost_fq(lamb, error)
    u_cost = unitary_cost_fq(n, eta, omega, error, lamb, br, charge)

    return e_cost * u_cost


def qubit_cost_fq(n, eta, omega, error, lamb, charge=0):
    r"""Return the number of ancilla qubits needed to implement the first quantization algorithm.
    ￼
    The expression for computing the parameters are taken from
    [`10.1103/PRXQuantum.2.040332 <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_].

    The expression for computing the cost is taken from Eq. (101) of
    [`arXiv:2204.11890v1 <https://arxiv.org/abs/2204.11890v1>`_].

    Args:
        n (int): number of plane waves
        eta (int): number of electrons
        omega (float): unit cell volume
        error (float): target error in the algorithm
        lamb (float): 1-norm of the Hamiltonian
        charge (int): total electric charge of the system

    Returns:
        int: the number of ancilla qubits needed to implement the first quantization algorithm

    **Example**

    >>> n = 100000
    >>> eta = 156
    >>> omega = 169.69608
    >>> error = 0.01
    >>> lamb = 5128920.595980267
    >>> qubit_cost(n, eta, omega, error, lamb)
    4238

    .. details::
        :title: Theory

        The target algorithm error, :math:`\epsilon`, is distributed among four different sources of
        error following Eq. (131) of
        `10.1103/PRXQuantum.2.040332 <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_ such
        that

        .. math::

            \epsilon^2 \geq \epsilon_{qpe}^2 + (\epsilon_{\mathcal{M}} + \epsilon_R + \epsilon_T)^2,

        where :math:`\epsilon_{qpe}` is the quantum phase estimation error and
        :math:`\epsilon_{\mathcal{M}}`, :math:`\epsilon_R`, and :math:`\epsilon_T` are defined in
        Eqs. (132-134) of
        `10.1103/PRXQuantum.2.040332 <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_. Here,
        we assume :math:`\epsilon_{\mathcal{M}} = \epsilon_R = \epsilon_T = \alpha \epsilon` with a
        default value of  :math:`\alpha = 0.01` and obtain

        .. math::

            \epsilon_{qpe} = \sqrt{\epsilon^2 [1 - (3 \alpha)^2]}.

        Note that the user only needs to define the target algorithm error :math:`\epsilon`. The
        error distribution takes place inside the functions.
    """
    alpha = 0.01
    l_z = eta + charge
    l_nu = 2 * np.pi * n ** (2 / 3)

    # n_p is taken from Eq. (22) of 10.1103/PRXQuantum.2.040332
    n_p = np.ceil(np.log2(n ** (1 / 3) + 1))

    # errors in Eqs. (132-134) of 10.1103/PRXQuantum.2.040332, set to be 0.01 of the algorithm error
    error_t = alpha * error
    error_r = alpha * error
    error_m = alpha * error

    # parameters taken from Eqs. (132-134) of PRX Quantum 2, 040332 (2021)
    n_t = int(np.log2(np.pi * lamb / error_t))  # Eq. (134)
    n_r = int(np.log2((eta * l_z * l_nu) / (error_r * omega ** (1 / 3))))  # Eq. (133)
    n_m = int(
        np.log2(  # Eq. (132)
            (2 * eta)
            / (error_m * np.pi * omega ** (1 / 3))
            * (eta - 1 + 2 * l_z)
            * (7 * 2 ** (n_p + 1) - 9 * n_p - 11 - 3 * 2 ** (-1 * n_p))
        )
    )

    # the expression for computing the cost is taken from Eq. (101) of arXiv:2204.11890v1
    qubits = 3 * eta * n_p + 4 * n_m * n_p + 12 * n_p
    qubits += 2 * (np.ceil(np.log2(np.ceil(np.pi * lamb / (2 * error))))) + 5 * n_m
    qubits += 2 * np.ceil(np.log2(eta)) + 3 * n_p**2 + np.ceil(np.log2(eta + 2 * l_z))
    qubits += np.maximum(5 * n_p + 1, 5 * n_r - 4) + np.maximum(n_t, n_r + 1) + 33

    return int(np.ceil(qubits))
