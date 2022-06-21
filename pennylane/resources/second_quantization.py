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
This module contains the functions needed for resource estimation with the double factorization
method.
"""
# pylint: disable= too-many-arguments
from pennylane import numpy as np


def estimation_cost(norm, error):
    r"""Return the number of calls to the unitary needed to achieve the desired error in quantum
    phase estimation.

    The expression for computing the cost is taken from Eq. (45) of
    [`10.1103/PRXQuantum.2.030305 <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.030305>`_].

    Args:
        norm (float): 1-norm of a second-quantized Hamiltonian
        error (float): target error in the algorithm

    Returns:
        int: number of calls to unitary

    **Example**

    >>> cost = estimation_cost(72.49779513025341, 0.001)
    >>> print(cost)
    113880
    """
    if error <= 0.0:
        raise ValueError("The target error must be greater than zero.")

    if norm <= 0.0:
        raise ValueError("The 1-norm must be greater than zero.")

    return int(np.ceil(np.pi * norm / (2 * error)))


def _qrom_cost(constants):
    r"""Return the number of Toffoli gates and the expansion factor needed to implement a QROM for
    the double factorization method.

    The complexity of a QROM computation in the most general form is given by (see Eq. (C39) in
    [`10.1103/PRXQuantum.2.030305 <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.030305>`_])

    .. math::

        \text{cost} = \left \lceil \frac{a + b}{k} \right \rceil + \left \lceil \frac{c}{k} \right
        \rceil + d \left ( k + e \right ),

    where :math:`a, b, c, d, e` are constants that depend on the nature of the QROM implementation
    and the expansion factor :math:`k = 2^n` minimizes the cost. This function computes the optimum
    :math:`k` and the minimum cost for a QROM specification.

    To obtain the optimum values of :math:`k`, we first assume that the cost function is continuous
    and use differentiation to obtain the value of :math:`k` that minimizes the cost. This value of
    :math:`k` is not necessarily an integer power of 2. We then obtain the value of :math:`n` as
    :math:`n = \log_2(k)` and compute the cost for
    :math:`n_{int}= \left \{\left \lceil n \right \rceil, \left \lfloor n \right \rfloor \right \}`.
    The value of :math:`n_{int}` that gives the smaller cost is used to compute the optimim
    :math:`k`.

    Args:
        constants (tuple[float]): constants specifying a QROM

    Returns:
        tuple(int, int): the cost and the expansion factor for the QROM

    **Example**

    >>> constants = (151.0, 7.0, 151.0, 30.0, -1.0)
    >>> cost_qrom(constants)
    168, 4
    """
    a, b, c, d, e = constants
    n = np.log2(((a + b + c) / d) ** 0.5)
    k = np.array([2 ** np.floor(n), 2 ** np.ceil(n)])
    cost = np.ceil((a + b) / k) + np.ceil(c / k) + d * (k + e)

    return int(cost[np.argmin(cost)]), int(k[np.argmin(cost)])


def unitary_cost(n, rank_r, rank_m, rank_max, br=7, alpha=10, beta=20):
    r"""Return the number of Toffoli gates needed to implement the qubitization unitary operator for
    the double factorization algorithm.

    The expression for computing the cost is taken from Eq. (C39) of
    [`10.1103/PRXQuantum.2.030305 <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.030305>`_].

    Args:
        n (int): number of molecular spin-orbitals
        rank_r (int): rank of the first factorization of the two-electron integral tensor
        rank_m (float): average rank of the second factorization of the two-electron integral tensor
        rank_max (float): maximum rank of the second factorization of the two-electron tensor
        br (int): number of bits for ancilla qubit rotation
        alpha (int): number of bits for the keep register
        beta (int): number of bits for the rotation angles

    Returns:
        int: number of Toffoli gates to implement the qubitization unitary

    **Example**

    >>> n = 14
    >>> rank_r = 26
    >>> rank_m = 5.5
    >>> rank_m = 7
    >>> br = 7
    >>> alpha = 10
    >>> beta = 20
    >>> unitary_cost(n, rank_r, rank_m, br, alpha, beta)
    2007
    """
    if n <= 0 or not isinstance(n, int) or n % 2 != 0:
        raise ValueError("The number of spin-orbitals must be a positive even integer.")

    if rank_r <= 0 or not isinstance(rank_r, int):
        raise ValueError("The rank of the first factorization step must be a positive integer.")

    if rank_m <= 0:
        raise ValueError("The rank of the second factorization step must be a positive number.")

    if rank_max <= 0 or not isinstance(rank_max, int):
        raise ValueError(
            "The maximum rank of the second factorization step must be a positive" " integer."
        )

    if br <= 0 or not isinstance(br, int):
        raise ValueError("br must be a positive integer.")

    if alpha <= 0 or not isinstance(alpha, int):
        raise ValueError("alpha must be a positive integer.")

    if beta <= 0 or not isinstance(beta, int):
        raise ValueError("beta must be a positive integer.")

    rank_rm = rank_r * rank_m

    # eta is computed based on step 1.(a) in page 030305-41 of 10.1103/PRXQuantum.2.030305
    eta = np.array([np.log2(n) for n in range(1, rank_r + 1) if rank_r % n == 0])
    eta = int(np.max([n for n in eta if n % 1 == 0]))

    nxi = np.ceil(np.log2(rank_max))  # Eq. (C14) of 10.1103/PRXQuantum.2.030305
    nl = np.ceil(np.log2(rank_r + 1))  # Eq. (C14) of 10.1103/PRXQuantum.2.030305
    nlxi = np.ceil(np.log2(rank_rm + n / 2))  # Eq. (C15) of 10.1103/PRXQuantum.2.030305

    bp1 = nl + alpha  # Eq. (C27) of 10.1103/PRXQuantum.2.030305
    bo = nxi + nlxi + br + 1  # Eq. (C29) of 10.1103/PRXQuantum.2.030305
    bp2 = nxi + alpha + 2  # Eq. (C31) of 10.1103/PRXQuantum.2.030305

    # cost is computed using Eq. (C39) of 10.1103/PRXQuantum.2.030305
    cost = 9 * nl - 6 * eta + 12 * br + 34 * nxi + 8 * nlxi + 9 * alpha + 3 * n * beta - 6 * n - 43
    cost += _qrom_cost((rank_r, 1, 0, bp1, -1))[0]
    cost += _qrom_cost((rank_r, 1, 0, bo, -1))[0]
    cost += _qrom_cost((rank_r, 1, 0, 1, 0))[0] * 2
    cost += _qrom_cost((rank_rm, n / 2, rank_rm, n * beta, 0))[0]
    cost += _qrom_cost((rank_rm, n / 2, rank_rm, 2, 0))[0] * 2
    cost += _qrom_cost((rank_rm, n / 2, rank_rm, 2 * bp2, -1))[0]

    return int(cost)


def gate_cost(n, norm, error, rank_r, rank_m, rank_max, br=7, alpha=10, beta=20):
    r"""Return the total number of Toffoli gates needed to implement the double factorization
    algorithm.

    The expression for computing the cost is taken from Eqs. (45) and (C39) of
    [`10.1103/PRXQuantum.2.030305 <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.030305>`_].

    Args:
        n (int): number of molecular spin-orbitals
        norm (float): 1-norm of a second-quantized Hamiltonian
        error (float): target error in the algorithm
        rank_r (int): rank of the first factorization of the two-electron integral tensor
        rank_m (float): average rank of the second factorization of the two-electron integral tensor
        rank_max (float): maximum rank of the second factorization of the two-electron tensor
        br (int): number of bits for ancilla qubit rotation
        alpha (int): number of bits for the keep register
        beta (int): number of bits for the rotation angles

    Returns:
        int: the number of Toffoli gates for the double factorization method

    **Example**

    >>> n = 14
    >>> norm = 52.98761457453095
    >>> error = 0.001
    >>> rank_r = 26
    >>> rank_m = 5.5
    >>> rank_max = 7
    >>> br = 7
    >>> alpha = 10
    >>> beta = 20
    >>> gate_cost(n, norm, error, rank_r, rank_m, br, alpha, beta)
    167048631
    """
    if n <= 0 or not isinstance(n, int) or n % 2 != 0:
        raise ValueError("The number of spin-orbitals must be a positive even integer.")

    if error <= 0.0:
        raise ValueError("The target error must be greater than zero.")

    if norm <= 0.0:
        raise ValueError("The 1-norm must be greater than zero.")

    if rank_r <= 0 or not isinstance(rank_r, int):
        raise ValueError("The rank of the first factorization step must be a positive integer.")

    if rank_m <= 0:
        raise ValueError("The rank of the second factorization step must be a positive number.")

    if rank_max <= 0 or not isinstance(rank_max, int):
        raise ValueError(
            "The maximum rank of the second factorization step must be a positive" " integer."
        )

    if br <= 0 or not isinstance(br, int):
        raise ValueError("br must be a positive integer.")

    if alpha <= 0 or not isinstance(alpha, int):
        raise ValueError("alpha must be a positive integer.")

    if beta <= 0 or not isinstance(beta, int):
        raise ValueError("beta must be a positive integer.")

    e_cost = estimation_cost(norm, error)
    u_cost = unitary_cost(n, rank_r, rank_m, rank_max, br, alpha, beta)

    return int(e_cost * u_cost)


def qubit_cost(n, lamb, error, rank_r, rank_m, rank_max, br=7, alpha=10, beta=20):
    r"""Return the number of ancilla qubits needed to implement the double factorization method.

    The expression for computing the cost is taken from Eq. (C40) of
    [`10.1103/PRXQuantum.2.030305 <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.030305>`_].


    Args:
        n (int): number of molecular spin-orbitals
        lamb (float): 1-norm of a second-quantized Hamiltonian
        error (float): target error in the algorithm
        rank_r (int): rank of the first factorization of the two-electron integral tensor
        rank_m (float): average rank of the second factorization of the two-electron integral tensor
        rank_max (float): maximum rank of the second factorization of the two-electron tensor
        br (int): number of bits for ancilla qubit rotation
        alpha (int): number of bits for the keep register
        beta (int): number of bits for the rotation angles

    Returns:
        int: number of ancilla qubits for the double factorization method

    **Example**

    >>> n = 14
    >>> lamb = 52.98761457453095
    >>> error = 0.001
    >>> rank_r = 26
    >>> rank_m = 5.5
    >>> rank_max = 7
    >>> br = 7
    >>> alpha = 10
    >>> beta = 20
    >>> qubit_cost(n, lamb, error, rank_r, rank_m, br, alpha, beta)
    292
    """
    if n <= 0 or not isinstance(n, int) or n % 2 != 0:
        raise ValueError("The number of spin-orbitals must be a positive even integer.")

    if error <= 0.0:
        raise ValueError("The target error must be greater than zero.")

    if lamb <= 0.0:
        raise ValueError("The 1-norm must be greater than zero.")

    if rank_r <= 0 or not isinstance(rank_r, int):
        raise ValueError("The rank of the first factorization step must be a positive integer.")

    if rank_m <= 0:
        raise ValueError("The rank of the second factorization step must be a positive number.")

    if rank_max <= 0 or not isinstance(rank_max, int):
        raise ValueError(
            "The maximum rank of the second factorization step must be a positive" " integer."
        )

    if br <= 0 or not isinstance(br, int):
        raise ValueError("br must be a positive integer.")

    if alpha <= 0 or not isinstance(alpha, int):
        raise ValueError("alpha must be a positive integer.")

    if beta <= 0 or not isinstance(beta, int):
        raise ValueError("beta must be a positive integer.")

    rank_rm = rank_r * rank_m

    nxi = np.ceil(np.log2(rank_max))  # Eq. (C14) of 10.1103/PRXQuantum.2.030305
    nl = np.ceil(np.log2(rank_r + 1))  # Eq. (C14) of 10.1103/PRXQuantum.2.030305
    nlxi = np.ceil(np.log2(rank_rm + n / 2))  # Eq. (C15) of 10.1103/PRXQuantum.2.030305

    bo = nxi + nlxi + br + 1  # Eq. (C29) of 10.1103/PRXQuantum.2.030305
    bp2 = nxi + alpha + 2  # Eq. (C31) of 10.1103/PRXQuantum.2.030305
    # kr is taken from Eq. (C39) of 10.1103/PRXQuantum.2.030305
    kr = _qrom_cost((rank_rm, n / 2, rank_rm, n * beta, 0))[1]

    # the cost is computed using Eq. (C40) of 10.1103/PRXQuantum.2.030305
    e_cost = estimation_cost(lamb, error)
    cost = n + 2 * nl + nxi + 3 * alpha + beta + bo + bp2
    cost += kr * n * beta / 2 + 2 * np.ceil(np.log2(e_cost + 1)) + 7

    return int(cost)


def norm(one, two, eigvals):
    r"""Return the 1-norm of a molecular Hamiltonian from the one- and two-electron integrals and
    eigenvalues of the factorized two-electron integral tensor.

    The 1-norm of a double-factorized molecular Hamiltonian is computed as
    [`arXiv:2007.14460 <https://arxiv.org/abs/2007.14460>`_]

    .. math::

        \lambda = ||T|| + \frac{1}{4} \sum_r ||L^{(r)}||^2,

    where the Schatten 1-norm for a given matrix :math:`T` is defined as

    .. math::

        ||T|| = \sum_k |\text{eigvals}[T]_k|.

    The matrices :math:`L^{(r)}` are obtained from factorization of the two-electron integral
    tensor :math:`V` such that

    .. math::

        V_{ijkl} = \sum_r L_{ij}^{(r)} L_{kl}^{(r) T}.

    The matrix :math:`T` is constructed from the one- and two-electron integrals as

    .. math::

        T = h_{ij} - \frac{1}{2} \sum_l V_{illj} + \sum_l V_{llij}.

    Note that the two-electron integral tensor must be arranged in chemist notation.

    Args:
        one (array[array[float]]): one-electron integrals
        two (array[array[float]]): two-electron integrals
        eigvals (array[float]): eigenvalues of the matrices obtained from factorizing the
            two-electron integral tensor

    Returns:
        array[float]: 1-norm of the Hamiltonian

    **Example**

    >>> symbols  = ['H', 'H', 'O']
    >>> geometry = np.array([[0.00000000,  0.00000000,  0.28377432],
    >>>                      [0.00000000,  1.45278171, -1.00662237],
    >>>                      [0.00000000, -1.45278171, -1.00662237]], requires_grad=False)
    >>> mol = qml.qchem.Molecule(symbols, geometry, basis_name='sto-3g')
    >>> core, one, two = qml.qchem.electron_integrals(mol)()
    >>> two = np.swapaxes(two, 1, 3) # convert to the chemists notation
    >>> _, eigvals, _ = factorize(two, 1e-5)
    >>> print(norm(one, two, eigvals))
    52.98762043980203
    """
    lambda_one = 0.25 * np.sum([np.sum(abs(v)) ** 2 for v in eigvals])

    t_matrix = one - 0.5 * np.einsum("illj", two) + np.einsum("llij", two)
    t_eigvals, _ = np.linalg.eigh(t_matrix)
    lambda_two = np.sum(abs(t_eigvals))

    return lambda_one + lambda_two
