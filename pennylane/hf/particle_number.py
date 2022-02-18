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
import autograd.numpy as anp
import pennylane as qml
from pennylane import numpy as np
from pennylane.hf.hamiltonian import _generate_qubit_operator, _return_pauli, simplify


def particle_number(orbitals):
    r"""Computes the particle number observable :math:`\hat{N}=\sum_\alpha \hat{n}_\alpha`
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

    return qubit_operator((coeffs, ops))


def qubit_operator(o_ferm, cutoff=1.0e-12):
    r"""Convert a fermionic observable to a PennyLane qubit observable.

    The fermionic operator is a tuple containing the fermionic coefficients and operators. The
    one-body fermionic operator :math:`a_2^\dagger a_0` is constructed as [2, 0] and the two-body
    operator :math:`a_4^\dagger a_3^\dagger a_2 a_1` is constructed as [4, 3, 2, 1].

    Args:
        o_ferm tuple(array[float], list[int]): fermionic operator
        cutoff (float): cutoff value for discarding the negligible terms

    Returns:
        Hamiltonian: Simplified PennyLane Hamiltonian

    **Example**

    >>> coeffs = np.array([1.0, 1.0])
    >>> ops = [[0, 0], [0, 0]]
    >>> f = (coeffs, ops)
    >>> print(qubit_operator(f))
    ((-1+0j)) [Z0]
    + ((1+0j)) [I0]
    """
    ops = []
    coeffs = anp.array([])

    for n, t in enumerate(o_ferm[1]):

        if len(t) == 0:
            coeffs = anp.array([0.0])
            coeffs = coeffs + np.array([o_ferm[0][n]])
            ops = ops + [qml.Identity(0)]

        else:
            op = _generate_qubit_operator(t)
            if op != 0:
                for i, o in enumerate(op[1]):
                    if len(o) == 0:
                        op[1][i] = qml.Identity(0)
                    if len(o) == 1:
                        op[1][i] = _return_pauli(o[0][1])(o[0][0])
                    if len(o) > 1:
                        k = qml.Identity(0)
                        for o_ in o:
                            k = k @ _return_pauli(o_[1])(o_[0])
                        op[1][i] = k
                coeffs = np.concatenate([coeffs, np.array(op[0]) * o_ferm[0][n]])
                ops = ops + op[1]

    return simplify(qml.Hamiltonian(coeffs, ops), cutoff=cutoff)
