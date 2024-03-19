# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r""" Private functions implementing the error analysis for Trotter-Suzuki product formulas."""

import math
from functools import reduce, lru_cache

import pennylane as qml


# General Helper functions
def _compute_repetitions(order, n):
    """Compute Upsilon"""
    if order == 1:
        return n

    k = order // 2
    return n * (5 ** (k - 1)) * 2


def _spectral_norm(op, fast=True):
    if pr := op.pauli_rep:  # Pauli rep is not none
        if fast or len(pr) <= 1:
            return sum(map(abs, pr.values()))

    elif fast:
        return qml.math.norm(qml.matrix(op), ord="fro")

    return qml.math.max(qml.math.svd(qml.matrix(op), compute_uv=False))


# Compute one-norm error:
def _one_norm_error(h_ops, t, p, n, fast):
    upsilon = _compute_repetitions(p, n)
    h_one_norm = 0

    for op in h_ops:
        h_one_norm += _spectral_norm(op, fast=fast)

    c = (h_one_norm * t) ** (p + 1) / (math.factorial(p + 1) * n**p)
    return c * (upsilon ** (p + 1) + 1)


# Compute alpha_comm
@lru_cache
def _generate_combinations(num_variables, required_sum):
    if num_variables == 0:
        return ()

    if required_sum == 0:
        return ((0,) * num_variables,)

    if num_variables == 1:
        return ((required_sum,),)

    if num_variables == 2:
        return tuple((i, required_sum - i) for i in range(required_sum + 1))

    master_lst = []
    for i in range(required_sum + 1):
        for sub_perms in _generate_combinations(num_variables - 1, required_sum - i):
            master_lst.append((i,) + sub_perms)

    return tuple(master_lst)


@lru_cache
def _recursive_nested_commutator(A, B, alpha):
    """Recursive commutator"""
    if alpha == 0:
        return B

    if alpha == 1:
        return qml.comm(A, B)

    return qml.comm(A, _recursive_nested_commutator(A, B, alpha - 1))


# Compute commutator error:
def _commutator_error(h_ops, t, p, n, fast):
    upsilon = _compute_repetitions(p, n)
    pre_factor = (2 * upsilon * t ** (p + 1)) / (p + 1)

    ops_index_lst, coeffs_lst = _flatten_trotter(len(h_ops), p, n)

    num_factors = len(coeffs_lst)
    alpha_combinations = _generate_combinations(num_factors, p)

    h_comm_norm = 0
    for h_gamma in h_ops:
        for alpha_lst in alpha_combinations:
            c = reduce(lambda x, y: x * y, map(math.factorial, alpha_lst))

            nested_comm = h_gamma
            for index in range(num_factors - 1, -1, -1):  # iterate in reverse order
                alpha_i = alpha_lst[index]
                H_i = qml.s_prod(coeffs_lst[index], h_ops[ops_index_lst[index]])

                nested_comm = _recursive_nested_commutator(H_i, nested_comm, alpha_i)

            h_comm_norm += _spectral_norm(nested_comm, fast=fast) / c

    return pre_factor * h_comm_norm


# Flatten the product formula
def _recursive_flatten(order, num_ops, scalar_t):
    ops_index_lst = list(range(num_ops))

    if order == 1:
        return ops_index_lst, [1 * scalar_t] * num_ops

    if order == 2:
        return ops_index_lst + ops_index_lst[::-1], [0.5 * scalar_t] * (2 * num_ops)

    scalar_1 = qml.templates.subroutines.trotter._scalar(order)  # pylint: disable=protected-access
    scalar_2 = 1 - 4 * scalar_1

    ops_index_lst_1, coeff_lst_1 = _recursive_flatten(order - 2, num_ops, scalar_1 * scalar_t)
    ops_index_lst_2, coeff_lst_2 = _recursive_flatten(order - 2, num_ops, scalar_2 * scalar_t)

    return (
        (2 * ops_index_lst_1) + ops_index_lst_2 + (2 * ops_index_lst_1),
        (2 * coeff_lst_1) + coeff_lst_2 + (2 * coeff_lst_1),
    )


def _simplify(ops_index, coeffs):
    final_ops = []
    final_coeffs = []

    iter_limit = len(coeffs)
    i = 0
    while i < iter_limit:
        final_ops.append(ops_index[i])
        final_coeffs.append(coeffs[i])
        shift = 1

        if (i + 1 < iter_limit) and (ops_index[i] == ops_index[i + 1]):
            final_coeffs[-1] += coeffs[i + 1]
            shift = 2

        i += shift
    return (final_ops, final_coeffs)


def _flatten_trotter(num_ops, order, n):
    ops_index_lst, coeffs_lst = _recursive_flatten(order, num_ops, 1 / n)
    ops_index_lst, coeffs_lst = _simplify(ops_index_lst * n, coeffs_lst * n)
    return ops_index_lst, coeffs_lst
