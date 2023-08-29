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
This module contains the functions needed for creating fermionic and qubit observables.
"""
import warnings

# pylint: disable= too-many-branches, too-many-return-statements

import pennylane as qml
from pennylane import numpy as np
from pennylane.fermi import FermiSentence, FermiWord
from pennylane.operation import active_new_opmath
from pennylane.pauli.utils import simplify


def fermionic_observable(constant, one=None, two=None, cutoff=1.0e-12, fs=False):
    r"""Create a fermionic observable from molecular orbital integrals.

    Args:
        constant (array[float]): the contribution of the core orbitals and nuclei
        one (array[float]): the one-particle molecular orbital integrals
        two (array[float]): the two-particle molecular orbital integrals
        cutoff (float): cutoff value for discarding the negligible integrals
        fs (bool): if True, a fermi sentence will be returned

    Returns:
        Union[FermiSentence, tuple(array[float], list[list[int]])]: fermionic Hamiltonian

    **Example**

    >>> constant = np.array([1.0])
    >>> integral = np.array([[0.5, -0.8270995], [-0.8270995, 0.5]])
    >>> coeffs, ops = fermionic_observable(constant, integral)
    >>> ops
    [[], [0, 0], [0, 2], [1, 1], [1, 3], [2, 0], [2, 2], [3, 1], [3, 3]]

    If the `fs` kwarg is `True`, a fermionic operator is returned.

    >>> constant = np.array([1.0])
    >>> integral = np.array([[0.5, -0.8270995], [-0.8270995, 0.5]])
    >>> fermionic_observable(constant, integral, fs=True)
    1.0 * I
    + 0.5 * a⁺(0) a(0)
    + -0.8270995 * a⁺(0) a(2)
    + 0.5 * a⁺(1) a(1)
    + -0.8270995 * a⁺(1) a(3)
    + -0.8270995 * a⁺(2) a(0)
    + 0.5 * a⁺(2) a(2)
    + -0.8270995 * a⁺(3) a(1)
    + 0.5 * a⁺(3) a(3)
    """
    coeffs = qml.math.array([])

    if not qml.math.allclose(constant, 0.0):
        coeffs = qml.math.concatenate((coeffs, constant))
        operators = [[]]
    else:
        operators = []

    if one is not None:
        indices_one = qml.math.argwhere(abs(one) >= cutoff)
        # up-up + down-down terms
        operators_one = (indices_one * 2).tolist() + (indices_one * 2 + 1).tolist()
        coeffs_one = qml.math.tile(one[abs(one) >= cutoff], 2)
        coeffs = qml.math.convert_like(coeffs, one)
        coeffs = qml.math.concatenate((coeffs, coeffs_one))
        operators = operators + operators_one

    if two is not None:
        indices_two = np.array(qml.math.argwhere(abs(two) >= cutoff))
        n = len(indices_two)
        operators_two = (
            [(indices_two[i] * 2).tolist() for i in range(n)]  # up-up-up-up
            + [(indices_two[i] * 2 + [0, 1, 1, 0]).tolist() for i in range(n)]  # up-down-down-up
            + [(indices_two[i] * 2 + [1, 0, 0, 1]).tolist() for i in range(n)]  # down-up-up-down
            + [(indices_two[i] * 2 + 1).tolist() for i in range(n)]  # down-down-down-down
        )
        coeffs_two = qml.math.tile(two[abs(two) >= cutoff], 4) / 2

        coeffs = qml.math.concatenate((coeffs, coeffs_two))
        operators = operators + operators_two

    indices_sort = [operators.index(i) for i in sorted(operators)]
    if indices_sort:
        indices_sort = qml.math.array(indices_sort)

    if fs:
        sentence = FermiSentence({FermiWord({}): constant[0]})
        for c, o in zip(coeffs[indices_sort], sorted(operators)):
            if len(o) == 2:
                sentence.update({FermiWord({(0, o[0]): "+", (1, o[1]): "-"}): c})
            if len(o) == 4:
                sentence.update(
                    {FermiWord({(0, o[0]): "+", (1, o[1]): "+", (2, o[2]): "-", (3, o[3]): "-"}): c}
                )
        sentence.simplify()

        return sentence

    warnings.warn(
        "This function will return a fermionic operator by default in the next release. For details, "
        "see the Fermionic Operators tutorial: "
        "https://pennylane.ai/qml/demos/tutorial_fermionic_operators. "
        "Currently, a fermionic operator can be returned by setting the `fs` kwarg to `True`."
    )

    return coeffs[indices_sort], sorted(operators)


def qubit_observable(o_ferm, cutoff=1.0e-12):
    r"""Convert a fermionic observable to a PennyLane qubit observable.

    The fermionic operator is a tuple containing the fermionic coefficients and operators. For
    instance, the one-body fermionic operator :math:`a_2^\dagger a_0` is specified as [2, 0] and the
    two-body operator :math:`a_4^\dagger a_3^\dagger a_2 a_1` is specified as [4, 3, 2, 1].

    Args:
        o_ferm Union[FermiSentence, tuple(array[float], list[list[int]])]: fermionic operator
        cutoff (float): cutoff value for discarding the negligible terms

    Returns:
        (Operator): Simplified PennyLane Hamiltonian

    **Example**

    >>> coeffs = np.array([1.0, 1.0])
    >>> ops = [[0, 0], [0, 0]]
    >>> f = (coeffs, ops)
    >>> print(qubit_observable(f))
    ((-1+0j)) [Z0]
    + ((1+0j)) [I0]

    The input can also be a fermionic operator.

    >>> w1 = qml.fermi.FermiWord({(0, 0) : '+', (1, 1) : '-'})
    >>> w2 = qml.fermi.FermiWord({(0, 1) : '+', (1, 2) : '-'})
    >>> s = qml.fermi.FermiSentence({w1 : 1.2, w2: 3.1})
    >>> print(qubit_observable(s))
      (-0.3j) [Y0 X1]
    + (0.3j) [X0 Y1]
    + (-0.775j) [Y1 X2]
    + (0.775j) [X1 Y2]
    + ((0.3+0j)) [Y0 Y1]
    + ((0.3+0j)) [X0 X1]
    + ((0.775+0j)) [Y1 Y2]
    + ((0.775+0j)) [X1 X2]

    If the new op-math is active, an arithmetic operator is returned.

    >>> qml.operation.enable_new_opmath()
    >>> coeffs = np.array([1.0, 1.0])
    >>> ops = [[0, 0], [0, 0]]
    >>> f = (coeffs, ops)
    >>> print(qubit_observable(f))
    Identity(wires=[0]) + ((-1+0j)*(PauliZ(wires=[0])))
    """
    if isinstance(o_ferm, (FermiWord, FermiSentence)):
        h = qml.jordan_wigner(o_ferm, ps=True)
        h.simplify(tol=cutoff)

        if active_new_opmath():
            if not h.wires:
                return h.operation(wire_order=[0])
            return h.operation()

        if not h.wires:
            h = h.hamiltonian(wire_order=[0])
            return qml.Hamiltonian(
                h.coeffs, [qml.Identity(0) if o.name == "Identity" else o for o in h.ops]
            )

        h = h.hamiltonian()

        return simplify(
            qml.Hamiltonian(
                h.coeffs, [qml.Identity(0) if o.name == "Identity" else o for o in h.ops]
            )
        )

    warnings.warn(
        "Tuple input for the qubit_observable function is deprecated; please use the fermionic "
        "operators format. For details, see the Fermionic Operators tutorial: "
        "https://pennylane.ai/qml/demos/tutorial_fermionic_operators"
    )

    ops = []
    coeffs = qml.math.array([])

    for n, t in enumerate(o_ferm[1]):
        if len(t) == 0:
            ops = ops + [qml.Identity(0)]
            coeffs = qml.math.array([0.0])
            coeffs = coeffs + o_ferm[0][n]
        else:
            op = jordan_wigner(t)
            if op != 0:
                ops = ops + op[1]
                coeffs = qml.math.concatenate([coeffs, qml.math.array(op[0]) * o_ferm[0][n]])

    if active_new_opmath():
        ps = qml.dot(coeffs, ops, pauli=True)
        ps.simplify(tol=cutoff)

        if len(ps) == 0:
            return qml.s_prod(
                0, qml.Identity(ops[0].wires[0])
            )  # use any op and any wire to represent the null op
        if (len(ps) == 1) and ((identity := qml.pauli.PauliWord({})) in ps):
            return qml.s_prod(
                ps[identity], qml.Identity(ops[0].wires[0])
            )  # use any op and any wire to represent the null op
        return ps.operation()

    return simplify(qml.Hamiltonian(coeffs, ops), cutoff=cutoff)


def jordan_wigner(op: list, notation="physicist"):  # pylint:disable=too-many-branches
    r"""Convert a fermionic operator to a qubit operator using the Jordan-Wigner mapping.

    For instance, the one-body fermionic operator :math:`a_2^\dagger a_0` should be constructed as
    [2, 0]. The two-body operator :math:`a_4^\dagger a_3^\dagger a_2 a_1` should be constructed
    as [4, 3, 2, 1] with ``notation='physicist'``. If ``notation`` is set to ``'chemist'``, the
    two-body operator [4, 3, 2, 1] is constructed as :math:`a_4^\dagger a_3 a_2^\dagger a_1`.

    Args:
        op (list[int]): the fermionic operator
        notation (str): notation specifying the order of the two-body fermionic operators

    Returns:
        tuple(list[complex], list[Operation]): list of coefficients and qubit operators

    **Example**

    >>> f  = [0, 0]
    >>> q = jordan_wigner(f)
    >>> q # corresponds to :math:`\frac{1}{2}(I_0 - Z_0)`
    ([(0.5+0j), (-0.5+0j)], [Identity(wires=[0]), PauliZ(wires=[0])])
    """

    warnings.warn(
        "Use of qml.qchem.jordan_wigner is deprecated; please use qml.jordan_wigner instead."
    )

    return qml.fermi.jordan_wigner(op, notation=notation)
