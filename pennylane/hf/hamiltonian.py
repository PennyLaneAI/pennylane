# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
This module contains the functions needed for computing the molecular Hamiltonian.
"""
import autograd.numpy as anp
from pennylane.hf.hartree_fock import generate_scf, nuclear_energy
from pennylane import numpy as np
import pennylane as qml


def generate_electron_integrals(mol, core=None, active=None):
    r"""Return a function that computes the one- and two-electron integrals in the atomic orbital
    basis.

    The one- and two-electron integrals in the molecular orbital basis can be written in terms of
    the integrals in the atomic orbital basis, by recalling that
    :math:`\phi_i = \sum_{\nu}c_{\nu}^i \chi_{\nu}`, as

    .. math::

        h_{pq} = \sum_{\mu \nu} C_{p \mu} h_{\mu \nu} C_{\nu q},

    and

    .. math::

        h_{pqrs} = \sum_{\mu \nu \rho \sigma} C_{p \mu} C_{q \nu} h_{\mu \nu \rho \sigma} C_{\rho r} C_{\sigma s}.


    The :math:`h_{\mu \nu}` and :math:`h_{\mu \nu \rho \sigma}` terms refer to the elements of the
    core matrix and the electron repulsion tensor, respectively.

    Args:
        mol (Molecule): the molecule object
        core (list[int]): indices of the core orbitals
        active (list[int]): indices of the active orbitals

    Returns:
        function: function that computes the core energy, the one- and two-electron integrals

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
    >>> mol = Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> generate_electron_integrals(mol)(*args)
    array([ 0.00000000e+00, -1.39021927e+00,  0.00000000e+00,  0.00000000e+00,
           -2.91653313e-01,  7.14439078e-01, -2.77555756e-17,  5.55111512e-17,
            1.70241443e-01,  5.55111512e-17,  1.70241443e-01,  7.01853154e-01,
            6.66133815e-16, -1.38777878e-16,  7.01853154e-01,  1.70241443e-01,
            2.22044605e-16,  1.70241443e-01, -4.44089210e-16,  6.66133815e-16,
            7.38836690e-01])
    """

    def electron_integrals(*args):
        r"""Compute the one- and two-electron integrals in the atomic orbital basis.

        Args:
            args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            array[float]: 1D array containing the core energy, the one- and two-electron integrals
        """
        v_fock, coeffs, fock_matrix, h_core, repulsion_tensor = generate_scf(mol)(*args)
        one = anp.einsum("qr,rs,st->qt", coeffs.T, h_core, coeffs)
        two = anp.swapaxes(
            anp.einsum(
                "ab,cd,bdeg,ef,gh->acfh", coeffs.T, coeffs.T, repulsion_tensor, coeffs, coeffs
            ),
            1,
            3,
        )
        e_core = nuclear_energy(mol.nuclear_charges, mol.coordinates)(*args)

        if core is None and active is None:
            return e_core, one, two

        else:
            for i in core:
                e_core = e_core + 2 * one[i][i]
                for j in core:
                    e_core = e_core + 2 * two[i][j][j][i] - two[i][j][i][j]

            for p in active:
                for q in active:
                    for i in core:
                        o = anp.zeros(one.shape)
                        o[p, q] = 1.0
                        one = one + (2 * two[i][p][q][i] - two[i][p][i][q]) * o

            one = one[anp.ix_(active, active)]
            two = two[anp.ix_(active, active, active, active)]

            return e_core, one, two

    return electron_integrals


def generate_fermionic_hamiltonian(mol, cutoff=1.0e-12):
    r"""Return a function that computes the fermionic hamiltonian.

    Args:
        mol (Molecule): the molecule object
        cutoff (float): cutoff value for discarding the negligible electronic integrals

    Returns:
        function: function that computes the the fermionic hamiltonian

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
    >>> mol = Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> h = generate_fermionic_hamiltonian(mol)(*args)
    >>> h[0]
    array([ 1.        , -1.39021927,  0.35721954,  0.08512072,  0.35721954,
            0.35721954,  0.08512072,  0.08512072,  0.08512072,  0.35092658,
            0.08512072,  0.08512072,  0.35092658,  0.35092658, -1.39021927,
            0.35721954,  0.08512072,  0.08512072,  0.35092658,  0.35092658,
            0.08512072,  0.35092658,  0.35092658,  0.08512072,  0.08512072,
           -0.29165331,  0.08512072,  0.36941834,  0.08512072,  0.08512072,
            0.36941834,  0.36941834,  0.35092658,  0.08512072, -0.29165331,
            0.08512072,  0.36941834])
    """

    def fermionic_hamiltonian(*args):
        r"""Compute the fermionic hamiltonian.

        Args:
            args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            tuple(array[float], list[list[int]]): the Hamiltonian coefficients and operators
        """
        e_core, one, two = generate_electron_integrals(mol)(*args)

        e_core = anp.array([e_core])

        indices_one = anp.argwhere(abs(one) >= cutoff)
        operators_one = (indices_one * 2).tolist() + (
            indices_one * 2 + 1
        ).tolist()  # up-up + down-down terms
        coeffs_one = anp.tile(one[abs(one) >= cutoff], 2)

        indices_two = anp.argwhere(abs(two) >= cutoff)
        n = len(indices_two)
        operators_two = (
            [(indices_two[i] * 2).tolist() for i in range(n)]  # up-up-up-up term
            + [
                (indices_two[i] * 2 + [0, 1, 1, 0]).tolist() for i in range(n)
            ]  # up-down-down-up term
            + [
                (indices_two[i] * 2 + [1, 0, 0, 1]).tolist() for i in range(n)
            ]  # down-up-up-down term
            + [(indices_two[i] * 2 + 1).tolist() for i in range(n)]  # down-down-down-down term
        )
        coeffs_two = anp.tile(two[abs(two) >= cutoff], 4) / 2

        coeffs = anp.concatenate((e_core, coeffs_one, coeffs_two))
        operators = [[]] + operators_one + operators_two
        indices_sort = [operators.index(i) for i in sorted(operators)]

        return coeffs[indices_sort], sorted(operators)

    return fermionic_hamiltonian


def _generate_qubit_operator(op):
    r"""Convert a fermionic operator to a qubit operator.

    The one-body fermionic operator ::math::`a_0^\dagger a_0` is constructed as [0, 0] and its
    corresponding qubit operator returned by the function is [(0.5+0j), (-0.5+0j)], [[], [(0, 'Z')]]
    which represents ::math::`\frac{1}{2}(I_0 - Z_0)`. The two-body operator
    ::math::`a_0^\dagger a_2^\dagger a_0 a_2` is constructed as [0, 2, 0, 2].

    Args:
        list[int]: the fermionic operator

    Returns
        tuple(list[complex], list[list[int, str]): list of coefficients and the qubit-operator terms

    **Example**

    >>> f  = [0, 0]
    >>> q = _generate_qubit_operator([0, 0])
    >>> q
    ([(0.5+0j), (-0.5+0j)], [[], [(0, 'Z')]])
    """
    if len(op) == 2:
        op = [((op[0], 1), (op[1], 0))]

    if len(op) == 4:
        op = [((op[0], 1), (op[1], 1), (op[2], 0), (op[3], 0))]

        if op[0][0][0] == op[0][1][0] or op[0][2][0] == op[0][3][0]:
            return 0

    for t in op:
        for l in t:
            z = [(index, "Z") for index in range(l[0])]
            x = z + [(l[0], "X"), 0.5]

            if l[1]:
                y = z + [(l[0], "Y"), -0.5j]

            else:
                y = z + [(l[0], "Y"), 0.5j]

            if t.index(l) == 0:
                q = [x, y]
            else:
                m = []
                for t1 in q:
                    for t2 in [x, y]:
                        q1, c1 = _pauli_mult(t1[:-1], t2[:-1], t1[-1], t2[-1])
                        m.append(q1 + [c1])
                q = m

    c = [p[-1] for p in q]
    o = [p[:-1] for p in q]

    for item in o:
        k = [i for i, x in enumerate(o) if x == item]
        if len(k) >= 2:
            for j in k[::-1][:-1]:
                del o[j]
                c[k[0]] = c[k[0]] + c[j]
                del c[j]

    return c, o


def _pauli_mult(p1, p2, c1, c2):
    c = c1 * c2

    t1 = [t[0] for t in p1]
    t2 = [t[0] for t in p2]

    K = []

    for i in p1:
        if i[0] in t1 and i[0] not in t2:
            K.append((i[0], D[i[1]]))
        for j in p2:
            if j[0] in t2 and j[0] not in t1:
                K.append((j[0], D[j[1]]))

            if i[0] == j[0]:
                if i[1] + j[1] in C:
                    K.append((i[0], D[i[1] + j[1]]))
                    c = c * C[i[1] + j[1]]
                else:
                    K.append((i[0], D[i[1] + j[1]]))

    K = [k for k in K if "I" not in k[1]]

    for item in K:
        k_ = [i for i, x in enumerate(K) if x == item]
        if len(k_) >= 2:
            for j in k_[::-1][:-1]:
                del K[j]

    return K, c


D = {
    "X": "X",
    "Y": "Y",
    "Z": "Z",
    "XX": "I",
    "YY": "I",
    "ZZ": "I",
    "ZX": "Y",
    "XZ": "Y",
    "ZY": "X",
    "YZ": "X",
    "XY": "Z",
    "YX": "Z",
    "IX": "X",
    "IY": "Y",
    "IZ": "Z",
    "XI": "X",
    "YI": "Y",
    "ZI": "Z",
    "I": "I",
    "II": "I",
}

C = {
    "ZX": 1.0j,
    "XZ": -1.0j,
    "ZY": -1.0j,
    "YZ": 1.0j,
    "XY": 1.0j,
    "YX": -1.0j,
}

#
#
# def calc_mult_0(term1, term2, c):
#
#     t1 = [t[0] for t in term1]
#     t2 = [t[0] for t in term2]
#
#     K = []
#
#     for i in term1:
#         if i[0] in t1 and i[0] not in t2:
#             K.append((i[0], D[i[1]]))
#         for j in term2:
#             if j[0] in t2 and j[0] not in t1:
#                 K.append((j[0], D[j[1]]))
#
#             if i[0] == j[0]:
#                 if i[1] + j[1] in C:
#                     K.append((i[0], D[i[1] + j[1]]))
#                     c = c * C[i[1] + j[1]]
#                 else:
#                     K.append((i[0], D[i[1] + j[1]]))
#
#     K = [k for k in K if "1" not in k[1]]
#
#     for item in K:
#         k_ = [i for i, x in enumerate(K) if x == item]
#         if len(k_) >= 2:
#             for j in k_[::-1][:-1]:
#                 del K[j]
#
#     return K, c
#
#
# def calc_mult(k1, k2):
#
#     M = []
#     for term1 in k1:
#         for term2 in k2:
#             m, c = calc_mult_0(term1[:-1], term2[:-1], term1[-1] * term2[-1])
#             M.append(m + [c])
#
#     return M
#
#
#
#
# # D_qml_ = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
#
#
# def ham_jw(h):
#
#     H = qml.Hamiltonian([1.0], [qml.Identity(0)])
#
#     for n, t in enumerate(h[1]):
#
#         if len(t) == 0:
#
#             H = H + qml.Hamiltonian([h[0][n]], [qml.Identity(0)])
#
#         elif len(t) == 2:
#             op_q = jordan_wigner_fermion_operator(t)
#
#             if op_q != 0:
#
#                 for i, o in enumerate(op_q[1]):
#                     if len(o) == 0:
#                         op_q[1][i] = qml.Identity(0)
#
#                     if len(o) == 1:
#                         op_q[1][i] = D_qml_[o[0][1]](o[0][0])
#
#                 H = H + qml.Hamiltonian(np.array(op_q[0]) * h[0][n], op_q[1])
#
#         elif len(t) == 4:
#             op_q = jordan_wigner_fermion_operator(t)
#
#             if op_q != 0:
#
#                 for i, o in enumerate(op_q[1]):
#
#                     if len(o) == 0:
#                         op_q[1][i] = qml.Identity(0)
#
#                     if len(o) == 1:
#                         op_q[1][i] = D_qml_[o[0][1]](o[0][0])
#
#                     if len(o) > 1:
#                         k = qml.Identity(0)
#                         for j, o_ in enumerate(o):
#                             k = k @ D_qml_[o_[1]](o_[0])
#
#                         op_q[1][i] = k
#
#                 H = H + qml.Hamiltonian(np.array(op_q[0]) * h[0][n], op_q[1])
#
#     return H
