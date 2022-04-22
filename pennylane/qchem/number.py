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
This module contains the functions needed for computing the particle number observable.
"""
from pennylane import numpy as np

from .observable_hf import qubit_observable


def particle_number(orbitals):
    r"""Compute the particle number observable :math:`\hat{N}=\sum_\alpha \hat{n}_\alpha`
    in the Pauli basis.

    The particle number operator is given by

    .. math::

        \hat{N} = \sum_\alpha \hat{c}_\alpha^\dagger \hat{c}_\alpha,

    where the index :math:`\alpha` runs over the basis of single-particle states
    :math:`\vert \alpha \rangle`, and the operators :math:`\hat{c}^\dagger` and :math:`\hat{c}` are
    the particle creation and annihilation operators, respectively.

    Args:
        orbitals (int): Number of *spin* orbitals. If an active space is defined, this is
            the number of active spin-orbitals.

    Returns:
        pennylane.Hamiltonian: the particle number observable

    Raises:
        ValueError: If orbitals is less than or equal to 0

    **Example**

    >>> orbitals = 4
    >>> print(particle_number(orbitals))
    (2.0) [I0]
    + (-0.5) [Z0]
    + (-0.5) [Z1]
    + (-0.5) [Z2]
    + (-0.5) [Z3]
    """

    if orbitals <= 0:
        raise ValueError(f"'orbitals' must be greater than 0; got for 'orbitals' {orbitals}")

    r = np.arange(orbitals)
    table = np.vstack([r, r, np.ones([orbitals])]).T

    coeffs = np.array([])
    ops = []

    for i in table:
        coeffs = np.concatenate((coeffs, np.array([i[2]])))
        ops.append([int(i[0]), int(i[1])])

    return qubit_observable((coeffs, ops))
