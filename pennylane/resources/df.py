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
This module contains the functions needed for resource estimation with double factorization method.
"""

from pennylane import numpy as np


def norm(one, two, eigvals):
    r"""Return the 1-norm of a molecular Hamiltonian from the one- and two-electron integrals and
    eigenvalues of the factorized two-electron integral tensor.

    The 1-norm of a double-factorized molecular Hamiltonian is computed as
    [`arXiv:2007.14460 <https://arxiv.org/abs/2007.14460>`_]

    .. math::

        \lambda = ||T|| + \frac{1}{4} \sum_r ||L^{(r)}||^2,

    where the Schatten norm for a given matrix :math:`T` is defined as

    .. math::

        ||T|| = \sum_k |\text{eigvals}[T]_k|.

    The matrices :math:`L^{(r)}` are obtained from a rank-r factorizing the two-electron integral
    tensor :math:`V`, in the chemist notation, such that

    .. math::

        V_{ijkl} = \sum_r L_{ij}^{(r)} L_{kl}^{(r) T}.

    The matrix :math:`T` is constructed from the one- and two-electron integrals

    .. math::

        T = h_{ij} - \frac{1}{2} \sum_l V_{illj} + \sum_l V_{llij}.

    Note that the two-electron integral tensor must be arranged in the chemist notation.

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
