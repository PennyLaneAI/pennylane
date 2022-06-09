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
    """
    return int(np.ceil(np.pi * norm / (2 * error)))


def unitary_cost(n, lamb, eps, l, xi, br, aleph=None, beth=None):

    nxi = np.ceil(np.log2(xi))
    nlxi = np.ceil(np.log2(l * xi + n / 2))
    nl = np.ceil(np.log2(l + 1))

    if not aleph:
        aleph = np.ceil(2.5 + np.log2(lamb / eps))

    if not beth:
        beth = np.ceil(5.652 + np.log2(lamb * n / eps))

    eta = np.array([np.log2(n) for n in range(1, l + 1) if l % n == 0])
    eta = int(np.max([n for n in eta if n % 1 == 0]))

    bp1 = nl + aleph
    bp2 = nxi + aleph + 2
    bo = nxi + nlxi + br + 1

    kp1, ko, kpp1, kpo, kr, kpr, kp2, kpp2 = expansion_factor(n, l, bp1, bo, bp2, xi, beth)

    cost = 9 * nl - 6 * eta + 12 * br
    cost += np.ceil((l + 1) / kp1) + bp1 * (kp1 - 1)
    cost += np.ceil((l + 1) / ko) + bo * (ko - 1)
    cost += np.ceil((l + 1) / kpp1) + kpp1
    cost += np.ceil((l + 1) / kpo) + kpo
    cost += np.ceil((l * xi + n / 2) / kr) + np.ceil((l * xi) / kr) + n * beth * kr
    cost += np.ceil((l * xi + n / 2) / kpr) + np.ceil((l * xi) / kpr) + 2 * kpr
    cost += 34 * nxi + 8 * nlxi
    cost += np.ceil((l * xi + n / 2) / kp2) + np.ceil((l * xi) / kp2) + 2 * bp2 * (kp2 - 1)
    cost += np.ceil((l * xi + n / 2) / kpp2) + np.ceil((l * xi) / kpp2) + 2 * kpp2
    cost += 3 * aleph + 6 * aleph + 3 * n * beth - 6 * n - 43

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

    kp1, ko, kpp1, kpo, kr, kpr, kp2, kpp2 = expansion_factor(n, l, bp1, bo, bp2, xi, beth)

    iteration = estimation_cost(lamb, eps)

    cost = n + 2 * nl + nxi + 2 * aleph + aleph + beth + bo + bp2
    cost += kr * n * beth / 2 + 2 * np.ceil(np.log2(iteration + 1)) + 7

    return cost
