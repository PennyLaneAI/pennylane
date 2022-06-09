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

from pennylane import numpy as np


def estimation_cost(norm, error):
    r"""Return the number of calls to the unitary needed to achieve the desired error in phase
    estimation.

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
    return int(np.ceil(np.pi * norm / (2 * error)))


def cost_qrom(constants):
    r"""Return the number of Toffoli gates and the expansion factor needed to implement a QROM.

    The complexity of a QROM computation in the most general form is given by
    [`arXiv:2011.03494 <https://arxiv.org/abs/2011.03494>`_]

    .. math::

        cost = \left \lceil \frac{a + b}{k} \right \rceil + \left \lceil \frac{c}{k} \right \rceil +
        d \left ( k + e \right ),

    where :math:`a, b, c, d, e` are constants that depend on the nature of the QROM implementation
    and the expansion factor :math:`k` is an integer power of two, :math:`k = 2^n`, that minimizes
    the cost. This function computes the optimum :math:`k` and the minimum cost for a QROM
    specification.

    To obtain the optimum values of :math:`k`, we first assume that the cost function is continues
    and use differentiation to obtain the value of :math:`k` that minimizes the cost. This value of
    :math:`k` is not necessarily an integer power of 2. We then obtain the value of :math:`n` as
    :math:`n = \log_2(k)` and compute the cost for
    :math:`n_{int}= \left \{\left \lceil n \right \rceil, \left \lfloor n \right \rfloor \right \}`.
    The value of :math:`n_{int}` that gives the smaller cost is used to compute the optimim
    :math:`k`.

    Args:
        constants (tuple[float]): constants determining a QROM

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


def unitary_cost(n, norm, error, rank_r, rank_m, br=None, aleph=None, beth=None, eta=None):

    if not br:
        br = 7

    if not aleph:
        aleph = np.ceil(2.5 + np.log2(norm / error))

    if not beth:
        beth = np.ceil(5.652 + np.log2(norm * n / error))

    if not eta:
        eta = np.array([np.log2(n) for n in range(1, rank_r + 1) if rank_r % n == 0])
        eta = int(np.max([n for n in eta if n % 1 == 0]))

    nxi = np.ceil(np.log2(rank_m))
    nlxi = np.ceil(np.log2(rank_r * rank_m + n / 2))
    nl = np.ceil(np.log2(rank_r + 1))

    bp1 = nl + aleph
    bp2 = nxi + aleph + 2
    bo = nxi + nlxi + br + 1

    cost = 9 * nl - 6 * eta + 12 * br + 34 * nxi + 8 * nlxi + 9 * aleph + 3 * n * beth - 6 * n - 43

    cost += cost_qrom((rank_r, 1, 0, bp1, -1))
    cost += cost_qrom((rank_r, 1, 0, bo, -1))
    cost += cost_qrom((rank_r, 1, 0, 1, 0)) * 2
    cost += cost_qrom((rank_r * rank_m, n / 2, rank_r * rank_m, n * beth, 0))
    cost += cost_qrom((rank_r * rank_m, n / 2, rank_r * rank_m, 2, 0)) * 2
    cost += cost_qrom((rank_r * rank_m, n / 2, rank_r * rank_m, 2 * bp2, -1))

    return cost


def gate_cost(n, lamb, eps, l, xi, br, aleph=None, beth=None):

    est_cost = estimation_cost(lamb, eps)
    u_cost = unitary_cost(n, lamb, eps, l, xi, br, aleph, beth)

    return est_cost * u_cost


def qubit_cost(n, lamb, eps, l, xi, br, aleph=None, beth=None):

    nxi = np.ceil(np.log2(xi))
    nlxi = np.ceil(np.log2(l * xi + n / 2))
    nl = np.ceil(np.log2(l + 1))

    if not aleph:
        aleph = np.ceil(2.5 + np.log2(lamb / eps))

    if not beth:
        beth = np.ceil(5.652 + np.log2(lamb * n / eps))

    bp1 = nl + aleph
    bp2 = nxi + aleph + 2
    bo = nxi + nlxi + br + 1

    # kp1, ko, kpp1, kpo, kr, kpr, kp2, kpp2 = expansion_factor(n, l, bp1, bo, bp2, xi, beth)

    iteration = estimation_cost(lamb, eps)

    cost = n + 2 * nl + nxi + 2 * aleph + aleph + beth + bo + bp2
    cost += kr * n * beth / 2 + 2 * np.ceil(np.log2(iteration + 1)) + 7

    return cost
