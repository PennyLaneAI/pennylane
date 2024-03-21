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
def _compute_repetitions(order: int):
    r"""Count the number of times the Hamiltonian is repeated ("stages")
    in the Suzuki-Trotter  product formula for a given order.

    See the definition of upsilon from section 2.3 (equation 15) in
    `Childs et al. (2021) <https://arxiv.org/abs/1912.08854>`_.

    Args:
        order (int): The order of the product formula.
        n (int): The number of trotter steps used in the product formula.

    Returns:
        int: Number of repetitions of the product formula.
    """
    if order == 1:
        return 1

    k = order // 2
    return (5 ** (k - 1)) * 2


def _spectral_norm(op, fast: bool = True):
    r"""Compute the spectral norm of the operator.

    Args:
        op (Operator): The operator we want to compute spectral norm of.
        fast (bool, optional): If true, uses the Frobenius norm as an upper-bound
            to the spectral norm. Defaults to True.

    Returns:
        float: The spectral norm of the operator.
    """
    if pr := op.pauli_rep:  # Pauli rep is not none
        if fast or len(pr) <= 1:
            return sum(map(abs, pr.values()))

    elif fast:
        return qml.math.norm(qml.matrix(op), ord="fro")

    return qml.math.max(qml.math.svd(qml.matrix(op), compute_uv=False))


# Compute one-norm error:
def _one_norm_error(h_ops, t: float, p: int, n: int, fast: bool):
    r"""Compute an upper-bound on the spectral norm error for approximating
    the time evolution of a Hamiltonian using a Suzuki-Trotter product formula.

    This function implements the Trotter error with 1-norm scaling following
    (lemma 6, equation 22 and equation 23) `Childs et al. (2021) <https://arxiv.org/abs/1912.08854>`_.
    (Assuming all hermitian terms).

    Args:
        h_ops (list[Operator]): The terms of the Hamiltonian (specifying the product formula)
        t (float): The time interval for evolution.
        p (int): The order of the product formula.
        n (int): The number of Trotter steps (repetitions).
        fast (bool): If True, a Frobenius bound is used to approximate the spectral norms of each term.

    Returns:
        float: An upper-bound for the spectral norm error of the product formula.
    """
    upsilon = _compute_repetitions(p)
    h_one_norm = 0

    for op in h_ops:
        h_one_norm += _spectral_norm(op, fast=fast)

    c = (h_one_norm * t) ** (p + 1) / (math.factorial(p + 1) * n**p)
    return c * (upsilon ** (p + 1) + 1)


# Compute alpha_comm
@lru_cache
def _generate_combinations(num_variables: int, required_sum: int):
    r"""A helper function which generates a sequence of valid combinations which
    satisfy the required sum constraint.

    **Example:**

        Suppose you have :math:`k` variables :math:`{a_1, a_2, ..., a_k}`. How many unique combinations
        of positive integers are there that satisfy the constraint :math:`\Sum_{i=1}^{k}(a_i) = s`?

        This function produces a list of all sequences which satisfy this constraint:

        >>> _generate_combinations(num_variables=3, required_sum=2)
        ((0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0))

    Args:
        num_variables (int): The total number of variables in a valid combination.
        required_sum (int): The sum of all variables in a valid combination.

    Returns:
        tuple(tuple(int)): The sequence of valid combinations.
    """
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
def _recursive_nested_commutator(A, B, alpha: int):
    r"""Compute the alpha-fold nested commutator of A and B.

    This function is defined mathematically as:

    .. math::

        Ad^{alpha}_{A}(B) := [A, [A, ... [A, B]] ... ]

    Where there are :math:`\alpha`-many nested commutators.

    Args:
        A (Operator): Nested operator in the nested commutator.
        B (Operator): Base operator of the nested commutator.
        alpha (int): Depth of the nested commutation.

    Returns:
        Operator: The resulting operator from evaluating the nested commutation.
    """
    if alpha == 0:
        return B

    if alpha == 1:
        return qml.comm(A, B)

    return qml.comm(A, _recursive_nested_commutator(A, B, alpha - 1))


# Compute commutator error:
def _commutator_error(h_ops, t: float, p: int, n: int, fast: bool) -> float:
    r"""Compute an upper-bound on the spectral norm error for approximating
    the time evolution of a Hamiltonian using a Suzuki-Trotter product formula.

    This function implements the Trotter error with commutator scaling following
    (appendix C, equation 189) `Childs et al. (2021) <https://arxiv.org/abs/1912.08854>`_.
    (Assuming all hermitian terms).

    Args:
        h_ops (list[Operator]): The terms of the hamiltonian (specifying the product formula)
        t (float): The time interval for evolution.
        p (int): The order of the product formula.
        n (int): The number of Trotter steps (repetitions).
        fast (bool): If True, a Frobenius bound is used to approximate the spectral norms of each term.

    Returns:
        float: An upper-bound for the spectral norm error of the product formula.
    """
    upsilon = _compute_repetitions(p)
    pre_factor = (2 * upsilon * t ** (p + 1)) / ((p + 1) * n**p)

    ops_index_lst, coeffs_lst = _flatten_trotter(len(h_ops), p)

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
def _recursive_flatten(order: int, num_ops: int, scalar_t: float):
    r"""Constructs a flattened list representation of the Suzuki-Trotter product formula.

    Suppose we have a Hamiltonian :math:`H = 1.2*X + 0.5*Y`, an associated 2nd order product formula is:

    .. math::

        W_2(t) = exp(-it*\frac{1.2*X}{2}) * exp(-it*\frac{0.5*Y}{2}) * exp(-it*\frac{0.5*Y}{2}) * exp(-it*\frac{1.2*X}{2})

    We represent this product formula using two lists. The first which stores the operators and the second stores the
    coefficients. :code:`ops = [X(0), Y(0), Y(0), X(0)], coeffs = [1.2/2, 0.5/2, 0.5/2, 1.2/2]`. We can
    further compress the memory needed to store the product formula by simply storing the indicies of the operators as
    they appear in the hamiltonian. Since :math:`H = 1.2*X + 0.5*Y`, we have :code:`ops_index = [0, 1, 1, 0]`.

    The Suzuki-Trotter product formula is defined recursively (see :class:`~.TrotterProduct`). This function recursively
    constructs the flattened list representation for the product formula.

    Args:
        order (int): The order of the product formula.
        num_ops (int): The number of terms in the Hamiltonian.
        scalar_t (float): The time interval for evolution.

    Returns:
        ([int], [float]): The flattened product formula
    """
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
    r"""A helper function which collects like terms and combines coefficients to return a simplified product.

    **Example:**

        Suppose we have a Hamiltonian :math:`H = 1.2*X + 0.5*Y`, an associated 2nd order product formula is:

        .. math::

            W_2(t) = exp(-it*\frac{1.2*X}{2}) * exp(-it*\frac{0.5*Y}{2}) * exp(-it*\frac{0.5*Y}{2}) * exp(-it*\frac{1.2*X}{2})

        We represent this product formula using two lists. The first which stores the operators and the second stores the
        coefficients. :code:`ops = [X(0), Y(0), Y(0), X(0)], coeffs = [1.2/2, 0.5/2, 0.5/2, 1.2/2]`. We can
        further compress the memory needed to store the product formula by simply storing the indicies of the operators as
        they appear in the hamiltonian. Since :math:`H = 1.2*X + 0.5*Y`, we have :code:`ops_index = [0, 1, 1, 0]`.

        Note, in the product formula above, the 2nd and 3rd terms in the product have the same base operator, they can
        be combined together to simplify the product formula:

        .. math::

            W_2(t) = exp(-it*\frac{1.2*X}{2}) * exp(-it*0.5*Y) * exp(-it*\frac{1.2*X}{2})

        This function acts on the list representation of a product formula to perform such simplifications and returns
        (in list representation) the simplified product formula (returns :code:`[0, 1, 0], [1.2/2, 0.5, 1.2/2]`).

    Args:
        ops_index (tuple(int)): A tuple storing the indicies of operators to be exponentiated and multiplied.
        coeffs (tuple(float)): A tuple storing the coefficients associated with the operators in the `ops_index`.

    Returns:
       final_ops, final_coeffs (tuple(ints), tuple(floats)): The simplified operator indicies and associated coefficients.
    """
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


def _flatten_trotter(num_ops: int, order: int):
    r"""Compute the simplified flattened list representation of the Suzuki-Trotter product formula.

    This function computes the simplified Suzuki-Trotter product formula for a certain number of
    Trotter steps, in the flattened list representation.

    Args:
        num_ops (int): The number of operators in the hamiltonian.
        order (int): The order of the product formula.

    Returns:
        ([int], [float]): The flattened, simplified product formula.
    """
    ops_index_lst, coeffs_lst = _recursive_flatten(order, num_ops, 1)
    ops_index_lst, coeffs_lst = _simplify(ops_index_lst, coeffs_lst)
    return ops_index_lst, coeffs_lst
