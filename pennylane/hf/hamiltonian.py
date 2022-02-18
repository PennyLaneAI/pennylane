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
# pylint: disable= too-many-branches, too-many-arguments, too-many-locals, too-many-nested-blocks
import autograd.numpy as anp
import pennylane as qml
from pennylane import numpy as np
from pennylane.hf.hartree_fock import generate_scf, nuclear_energy


def generate_electron_integrals(mol, core=None, active=None):
    r"""Return a function that computes the one- and two-electron integrals in the molecular orbital
    basis.

    The one- and two-electron integrals are required to construct a molecular Hamiltonian in the
    second-quantized form

    .. math::

        H = \sum_{pq} h_{pq} c_p^{\dagger} c_q + \frac{1}{2} \sum_{pqrs} h_{pqrs} c_p^{\dagger} c_q^{\dagger} c_r c_s,

    where :math:`c^{\dagger}` and :math:`c` are the creation and annihilation operators,
    respectively, and :math:`h_{pq}` and :math:`h_{pqrs}` are the one- and two-electron integrals.
    These integrals can be computed by integrating over molecular orbitals :math:`\phi` as

    .. math::

        h_{pq} = \int \phi_p(r)^* \left ( -\frac{\nabla_r^2}{2} - \sum_i \frac{Z_i}{|r-R_i|} \right )  \phi_q(r) dr,

    and

    .. math::

        h_{pqrs} = \int \frac{\phi_p(r_1)^* \phi_q(r_2)^* \phi_r(r_2) \phi_s(r_1)}{|r_1 - r_2|} dr_1 dr_2.

    The molecular orbitals are constructed as a linear combination of atomic orbitals as

    .. math::

        \phi_i = \sum_{\nu}c_{\nu}^i \chi_{\nu}.

    The one- and two-electron integrals can be written in the molecular orbital basis as

    .. math::

        h_{pq} = \sum_{\mu \nu} C_{p \mu} h_{\mu \nu} C_{\nu q},

    and

    .. math::

        h_{pqrs} = \sum_{\mu \nu \rho \sigma} C_{p \mu} C_{q \nu} h_{\mu \nu \rho \sigma} C_{\rho r} C_{\sigma s}.

    The :math:`h_{\mu \nu}` and :math:`h_{\mu \nu \rho \sigma}` terms refer to the elements of the
    core matrix and the electron repulsion tensor, respectively, and :math:`C` is the molecular
    orbital expansion coefficient matrix.

    Args:
        mol (Molecule): the molecule object
        core (list[int]): indices of the core orbitals
        active (list[int]): indices of the active orbitals

    Returns:
        function: function that computes the core constant, and the one- and two-electron integrals

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
    >>> mol = qml.hf.Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> generate_electron_integrals(mol)(*args)
    (1.0,
     array([[-1.3902192695e+00,  0.0000000000e+00],
            [-4.4408920985e-16, -2.9165331336e-01]]),
     array([[[[ 7.1443907755e-01, -2.7755575616e-17],
              [ 5.5511151231e-17,  1.7024144301e-01]],
             [[ 5.5511151231e-17,  1.7024144301e-01],
              [ 7.0185315353e-01,  6.6613381478e-16]]],
            [[[-1.3877787808e-16,  7.0185315353e-01],
              [ 1.7024144301e-01,  2.2204460493e-16]],
             [[ 1.7024144301e-01, -4.4408920985e-16],
              [ 6.6613381478e-16,  7.3883668974e-01]]]]))
    """

    def electron_integrals(*args):
        r"""Compute the one- and two-electron integrals in the molecular orbital basis.

        Args:
            args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            tuple[array[float]]: 1D tuple containing core constant, one- and two-electron integrals
        """
        _, coeffs, _, h_core, repulsion_tensor = generate_scf(mol)(*args)
        one = anp.einsum("qr,rs,st->qt", coeffs.T, h_core, coeffs)
        two = anp.swapaxes(
            anp.einsum(
                "ab,cd,bdeg,ef,gh->acfh", coeffs.T, coeffs.T, repulsion_tensor, coeffs, coeffs
            ),
            1,
            3,
        )
        core_constant = nuclear_energy(mol.nuclear_charges, mol.coordinates)(*args)

        if core is None and active is None:
            return core_constant, one, two

        for i in core:
            core_constant = core_constant + 2 * one[i][i]
            for j in core:
                core_constant = core_constant + 2 * two[i][j][j][i] - two[i][j][i][j]

        for p in active:
            for q in active:
                for i in core:
                    o = anp.zeros(one.shape)
                    o[p, q] = 1.0
                    one = one + (2 * two[i][p][q][i] - two[i][p][i][q]) * o

        one = one[anp.ix_(active, active)]
        two = two[anp.ix_(active, active, active, active)]

        return core_constant, one, two

    return electron_integrals


def generate_fermionic_hamiltonian(mol, cutoff=1.0e-12, core=None, active=None):
    r"""Return a function that computes the fermionic hamiltonian.

    Args:
        mol (Molecule): the molecule object
        cutoff (float): cutoff value for discarding the negligible electronic integrals

    Returns:
        function: function that computes the fermionic hamiltonian

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
    >>> mol = qml.hf.Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> h = generate_fermionic_hamiltonian(mol)(*args)
    """

    def fermionic_hamiltonian(*args):
        r"""Compute the fermionic hamiltonian.

        Args:
            args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            tuple(array[float], list[list[int]]): the Hamiltonian coefficients and operators
        """
        core_constant, one, two = generate_electron_integrals(mol, core, active)(*args)

        core_constant = anp.array([core_constant])

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

        coeffs = anp.concatenate((core_constant, coeffs_one, coeffs_two))
        operators = [[]] + operators_one + operators_two
        indices_sort = [operators.index(i) for i in sorted(operators)]

        return coeffs[indices_sort], sorted(operators)

    return fermionic_hamiltonian


def generate_hamiltonian(mol, cutoff=1.0e-12, core=None, active=None):
    r"""Return a function that computes the qubit hamiltonian.

    Args:
        mol (Molecule): the molecule object
        cutoff (float): cutoff value for discarding the negligible electronic integrals

    Returns:
        function: function that computes the qubit hamiltonian

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
    >>> mol = qml.hf.Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> h = generate_hamiltonian(mol)(*args)
    >>> h.coeffs
    tensor([ 0.29817879+0.j,  0.20813365+0.j,  0.20813365+0.j,
             0.17860977+0.j,  0.04256036+0.j, -0.04256036+0.j,
            -0.04256036+0.j,  0.04256036+0.j, -0.34724873+0.j,
             0.13290293+0.j, -0.34724873+0.j,  0.17546329+0.j,
             0.17546329+0.j,  0.13290293+0.j,  0.18470917+0.j], requires_grad=True)
    """

    def hamiltonian(*args):
        r"""Compute the qubit hamiltonian.

        Args:
            args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            Hamiltonian: the qubit Hamiltonian
        """
        h_ferm = generate_fermionic_hamiltonian(mol, cutoff, core, active)(*args)

        ops = []

        for n, t in enumerate(h_ferm[1]):

            if len(t) == 0:
                coeffs = anp.array([0.0])
                coeffs = coeffs + np.array([h_ferm[0][n]])
                ops = ops + [qml.Identity(0)]

            elif len(t) == 2:
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
                    coeffs = np.concatenate([coeffs, np.array(op[0]) * h_ferm[0][n]])
                    ops = ops + op[1]

            elif len(t) == 4:
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
                    coeffs = np.concatenate([coeffs, np.array(op[0]) * h_ferm[0][n]])
                    ops = ops + op[1]

        h = simplify(qml.Hamiltonian(coeffs, ops), cutoff=cutoff)

        return h

    return hamiltonian


def _generate_qubit_operator(op):
    r"""Convert a fermionic operator to a qubit operator using the Jordan-Wigner mapping.

    The one-body fermionic operator :math:`a_2^\dagger a_0` is constructed as [2, 0] and the
    two-body operator :math:`a_4^\dagger a_3^\dagger a_2 a_1` is constructed as [4, 3, 2, 1].

    Args:
        op (list[int]): the fermionic operator

    Returns
        tuple(list[complex], list[list[int, str]]): list of coefficients and the qubit-operator terms

    **Example**

    >>> f  = [0, 0]
    >>> q = _generate_qubit_operator(f)
    >>> q
    ([(0.5+0j), (-0.5+0j)], [[], [(0, 'Z')]]) # corresponds to :math:`\frac{1}{2}(I_0 - Z_0)`
    """
    if len(op) == 1:
        op = [((op[0], 1),)]

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
                        q1, c1 = _pauli_mult(t1[:-1], t2[:-1])
                        m.append(q1 + [c1 * t1[-1] * t2[-1]])
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


def simplify(h, cutoff=1.0e-12):
    r"""Add together identical terms in the Hamiltonian.

    The Hamiltonian terms with identical Pauli words are added together and eliminated if the
    overall coefficient is smaller than a cutoff value.

    Args:
        h (Hamiltonian): PennyLane Hamiltonian
        cutoff (float): cutoff value for discarding the negligible terms

    Returns:
        Hamiltonian: Simplified PennyLane Hamiltonian

    **Example**

    >>> c = np.array([0.5, 0.5])
    >>> h = qml.Hamiltonian(c, [qml.PauliX(0) @ qml.PauliY(1), qml.PauliX(0) @ qml.PauliY(1)])
    >>> print(simplify(h))
    (1.0) [X0 Y1]
    """
    wiremap = dict(zip(h.wires, range(len(h.wires) + 1)))

    c = []
    o = []
    for i, op in enumerate(h.terms()[1]):
        op = qml.operation.Tensor(op).prune()
        op = qml.grouping.pauli_word_to_string(op, wire_map=wiremap)
        if op not in o:
            c.append(h.terms()[0][i])
            o.append(op)
        else:
            c[o.index(op)] += h.terms()[0][i]

    coeffs = []
    ops = []
    nonzero_ind = np.argwhere(abs(np.array(c)) > cutoff).flatten()
    for i in nonzero_ind:
        coeffs.append(c[i])
        ops.append(qml.grouping.string_to_pauli_word(o[i], wire_map=wiremap))

    try:
        coeffs = qml.math.stack(coeffs)
    except ValueError:
        pass

    return qml.Hamiltonian(coeffs, ops)


def _pauli_mult(p1, p2):
    r"""Return the result of multiplication between two tensor products of Pauli operators.

    The Pauli operator :math:`(P_0)` is denoted by [(0, 'P')], where :math:`P` represents
    :math:`X`, :math:`Y` or :math:`Z`.

    Args:
        p1 (list[tuple[int, str]]): the first tensor product of Pauli operators
        p2 (list[tuple[int, str]]): the second tensor product of Pauli operators

    Returns
        tuple(list[tuple[int, str]], complex): list of the Pauli operators and the coefficient

    **Example**

    >>> p1 = [(0, "X"), (1, "Y")],  # X_0 @ Y_1
    >>> p2 = [(0, "X"), (2, "Y")],  # X_0 @ Y_2
    >>> _pauli_mult(p1, p2)
    ([(2, "Y"), (1, "Y")], 1.0) # p1 @ p2 = X_0 @ Y_1 @ X_0 @ Y_2
    """
    c = 1.0

    t1 = [t[0] for t in p1]
    t2 = [t[0] for t in p2]

    k = []

    for i in p1:
        if i[0] in t1 and i[0] not in t2:
            k.append((i[0], pauli_mult[i[1]]))
        for j in p2:
            if j[0] in t2 and j[0] not in t1:
                k.append((j[0], pauli_mult[j[1]]))

            if i[0] == j[0]:
                if i[1] + j[1] in pauli_coeff:
                    k.append((i[0], pauli_mult[i[1] + j[1]]))
                    c = c * pauli_coeff[i[1] + j[1]]
                else:
                    k.append((i[0], pauli_mult[i[1] + j[1]]))

    k = [i for i in k if "I" not in i[1]]

    for item in k:
        k_ = [i for i, x in enumerate(k) if x == item]
        if len(k_) >= 2:
            for j in k_[::-1][:-1]:
                del k[j]

    return k, c


def _return_pauli(p):
    r"""Return the PennyLane Pauli operator.

    Args:
        args (str): symbol representing the Pauli operator

    Returns:
        pennylane.ops: the PennyLane Pauli operator

    **Example**

    >>> _return_pauli('X')
    qml.PauliX
    """
    if p == "X":
        return qml.PauliX
    if p == "Y":
        return qml.PauliY

    return qml.PauliZ


pauli_mult = {
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
    "X": "X",
    "Y": "Y",
    "Z": "Z",
}

pauli_coeff = {
    "ZX": 1.0j,
    "XZ": -1.0j,
    "ZY": -1.0j,
    "YZ": 1.0j,
    "XY": 1.0j,
    "YX": -1.0j,
}
