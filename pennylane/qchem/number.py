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
from pennylane.fermi import FermiSentence, FermiWord

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
    (
        2.0 * I(0)
      + -0.5 * Z(0)
      + -0.5 * Z(1)
      + -0.5 * Z(2)
      + -0.5 * Z(3)
    )
    """

    if orbitals <= 0:
        raise ValueError(f"'orbitals' must be greater than 0; got for 'orbitals' {orbitals}")

    sentence = FermiSentence({FermiWord({(0, i): "+", (1, i): "-"}): 1.0 for i in range(orbitals)})
    sentence.simplify()

    return qubit_observable(sentence)
