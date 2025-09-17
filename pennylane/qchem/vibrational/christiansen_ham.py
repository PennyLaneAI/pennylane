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
"""The functions related to the construction of the Christiansen form Hamiltonian."""

import numpy as np

from pennylane.bose import BoseSentence, BoseWord, christiansen_mapping

from .christiansen_utils import christiansen_integrals, christiansen_integrals_dipole

# pylint: disable = too-many-branches,too-many-nested-blocks,


def christiansen_bosonic(one, two=None, three=None, ordered=True):
    r"""Generates a Christiansen bosonic vibrational Hamiltonian.

    The Christiansen vibrational Hamiltonian is defined based on Eqs. D4-D7
    of `arXiv:2504.10602 <https://arxiv.org/abs/2504.10602>`_ as:

    .. math::

        H = \sum_{i}^M \sum_{k_i, l_i}^{N_i} C_{k_i, l_i}^{(i)} b_{k_i}^{\dagger} b_{l_i} +
        \sum_{i<j}^{M} \sum_{k_i,l_i}^{N_i} \sum_{k_j,l_j}^{N_j} C_{k_i k_j, l_i l_j}^{(i,j)}
        b_{k_i}^{\dagger} b_{k_j}^{\dagger} b_{l_i} b_{l_j},


    where :math:`b^{\dagger}` and :math:`b` are the bosonic creation and annihilation
    operators, :math:`M` represents the number of normal modes and :math:`N` is the number of
    modals. The coefficients :math:`C` represent the one-mode and two-mode integrals defined as

    .. math::

        C_{k_i, l_i}^{(i)} = \int \phi_i^{k_i}(Q_i) \left( T(Q_i) +
        V_1^{(i)}(Q_i) \right) \phi_i^{h_i}(Q_i),

    and

    .. math::

        C_{k_i, k_j, l_i, l_j}^{(i,j)} = \int \int \phi_i^{k_i}(Q_i) \phi_j^{k_j}(Q_j)
        V_2^{(i,j)}(Q_i, Q_j) \phi_i^{l_i}(Q_i) \phi_j^{l_j}(Q_j) \; \text{d} Q_i \text{d} Q_j,

    where :math:`\phi` represents a modal, :math:`Q` represents a normal coordinate, :math:`T`
    represents the kinetic energy operator and :math:`V` represents the potential energy operator.
    Similarly, the three-mode integrals can be obtained following
    Eq. D7 of `arXiv:2504.10602 <https://arxiv.org/abs/2504.10602>`_.

    Args:
        one (TensorLike[float]): one-body integrals with shape ``(m, n, n)`` where ``m`` and ``n``
            are the number of modes and the maximum number of bosonic states per mode, repectively
        two (TensorLike[float]): two-body integrals with shape ``(m, m, n, n, n, n)`` where ``m``
            and ``n`` are the number of modes and the maximum number of bosonic states per mode,
            repectively. Default is ``None`` which means that the two-body terms will not be
            included in the Hamiltonian.
        three (TensorLike[float]): three-body integrals with shape ``(m, m, m, n, n, n, n, n, n)``
            where ``m`` and ``n`` are the number of modes and the maximum number of bosonic states
            per mode, repectively. Default is ``None`` which means that the two-body terms will not
            be included in the Hamiltonian.
        cutoff (float): tolerance for discarding the negligible coefficients
        ordered (bool): indicates if integral matrix elements are already ordered. Default is ``True``.

    Returns:
        pennylane.bose.BoseSentence: the constructed bosonic operator

    **Example**

    >>> symbols  = ['H', 'F']
    >>> geometry = np.array([[0.0, 0.0, -0.40277116], [0.0, 0.0, 1.40277116]])
    >>> mol = qml.qchem.Molecule(symbols, geometry)
    >>> pes = qml.qchem.vibrational_pes(mol, optimize=False)
    >>> integrals = qml.qchem.vibrational.christiansen_integrals(pes, n_states = 4)
    >>> print(qml.qchem.christiansen_bosonic(integrals[0]))
    0.010354801267111937 * b⁺(0) b(0)
    + 0.0019394049410426685 * b⁺(0) b(1)
    + 0.00046435758469677135 * b⁺(0) b(2)
    + 0.001638099727072391 * b⁺(0) b(3)
    + 0.0019394049410426685 * b⁺(1) b(0)
    + 0.03139978085503162 * b⁺(1) b(1)
    + 0.005580004725710029 * b⁺(1) b(2)
    + 0.0013758584515161654 * b⁺(1) b(3)
    + 0.00046435758469677135 * b⁺(2) b(0)
    + 0.005580004725710029 * b⁺(2) b(1)
    + 0.05314478483410301 * b⁺(2) b(2)
    + 0.010479092552439511 * b⁺(2) b(3)
    + 0.001638099727072391 * b⁺(3) b(0)
    + 0.0013758584515161654 * b⁺(3) b(1)
    + 0.010479092552439511 * b⁺(3) b(2)
    + 0.07565063279464881 * b⁺(3) b(3)
    """

    modes = np.shape(one)[0]

    imax = np.shape(one)[1]
    modals = imax * np.ones(modes, dtype=int)

    idx = {}  # dictionary mapping the tuple (l,n) to an index in the qubit register
    counter = 0
    for l in range(modes):
        for n in range(modals[l]):
            key = (l, n)
            idx[key] = counter
            counter += 1

    obs = {}  # second-quantized Hamiltonian

    # one-body terms
    for l in range(modes):
        for k_l in range(modals[l]):
            for h_l in range(modals[l]):
                (i0, i1) = ((l, k_l), (l, h_l))
                w = BoseWord({(0, idx[i0]): "+", (1, idx[i1]): "-"})
                obs[w] = one[l, k_l, h_l]

    # two-body terms
    if not two is None:
        for l in range(modes):
            if ordered is False:
                m_range = [p for p in range(modes) if p != l]
            else:
                m_range = range(l)
            for m in m_range:
                for k_l in range(modals[l]):
                    for h_l in range(modals[l]):
                        for k_m in range(modals[m]):
                            for h_m in range(modals[m]):
                                (i0, i1, i2, i3) = (
                                    (l, k_l),
                                    (m, k_m),
                                    (l, h_l),
                                    (m, h_m),
                                )
                                w = BoseWord(
                                    {
                                        (0, idx[i0]): "+",
                                        (1, idx[i1]): "+",
                                        (2, idx[i2]): "-",
                                        (3, idx[i3]): "-",
                                    }
                                )
                                obs[w] = two[l, m, k_l, k_m, h_l, h_m]

    # three-body terms
    if not three is None:
        for l in range(modes):
            if not ordered:
                m_range = [p for p in range(modes) if p != l]
            else:
                m_range = range(l)
            for m in m_range:
                if not ordered:
                    n_range = [p for p in range(modes) if p not in (l, m)]
                else:
                    n_range = range(m)
                for n in n_range:
                    for k_l in range(modals[l]):
                        for h_l in range(modals[l]):
                            for k_m in range(modals[m]):
                                for h_m in range(modals[m]):
                                    for k_n in range(modals[n]):
                                        for h_n in range(modals[n]):
                                            (i0, i1, i2, i3, i4, i5) = (
                                                (l, k_l),
                                                (m, k_m),
                                                (n, k_n),
                                                (l, h_l),
                                                (m, h_m),
                                                (n, h_n),
                                            )
                                            w = BoseWord(
                                                {
                                                    (0, idx[i0]): "+",
                                                    (1, idx[i1]): "+",
                                                    (2, idx[i2]): "+",
                                                    (3, idx[i3]): "-",
                                                    (4, idx[i4]): "-",
                                                    (5, idx[i5]): "-",
                                                }
                                            )
                                            obs[w] = three[l, m, n, k_l, k_m, k_n, h_l, h_m, h_n]

    obs_sq = BoseSentence(obs)

    return obs_sq


def christiansen_hamiltonian(pes, n_states=16, cubic=False, wire_map=None, tol=1e-12):
    r"""Generates a Christiansen vibrational Hamiltonian.

    The Christiansen vibrational Hamiltonian is defined based on Eqs. D4-D7
    of `arXiv:2504.10602 <https://arxiv.org/abs/2504.10602>`_ as:

    .. math::

        H = \sum_{i}^M \sum_{k_i, l_i}^{N_i} C_{k_i, l_i}^{(i)} b_{k_i}^{\dagger} b_{l_i} +
        \sum_{i<j}^{M} \sum_{k_i,l_i}^{N_i} \sum_{k_j,l_j}^{N_j} C_{k_i k_j, l_i l_j}^{(i,j)}
        b_{k_i}^{\dagger} b_{k_j}^{\dagger} b_{l_i} b_{l_j},


    where :math:`b^{\dagger}` and :math:`b` are the bosonic creation and annihilation
    operators, :math:`M` represents the number of normal modes and :math:`N` is the number of
    modals. The coefficients :math:`C` represent the one-mode and two-mode integrals defined as

    .. math::

        C_{k_i, l_i}^{(i)} = \int \phi_i^{k_i}(Q_i) \left( T(Q_i) +
        V_1^{(i)}(Q_i) \right) \phi_i^{h_i}(Q_i),

    and

    .. math::

        C_{k_i, k_j, l_i, l_j}^{(i,j)} = \int \int \phi_i^{k_i}(Q_i) \phi_j^{k_j}(Q_j)
        V_2^{(i,j)}(Q_i, Q_j) \phi_i^{l_i}(Q_i) \phi_j^{l_j}(Q_j) \; \text{d} Q_i \text{d} Q_j,

    where :math:`\phi` represents a modal, :math:`Q` represents a normal coordinate, :math:`T`
    represents the kinetic energy operator and :math:`V` represents the potential energy operator.
    Similarly, the three-mode integrals can be obtained following
    Eq. D7 of `arXiv:2504.10602 <https://arxiv.org/abs/2504.10602>`_.

    The bosonic creation and annihilation operators are then mapped to the Pauli operators as

    .. math::

        b^\dagger_0 = \left(\frac{X_0 - iY_0}{2}\right), \:\: \text{...,} \:\:
        b^\dagger_n = \left(\frac{X_n - iY_n}{2}\right),

    and

    .. math::

        b_0 = \left(\frac{X_0 + iY_0}{2}\right), \:\: \text{...,} \:\:
        b_n = \left(\frac{X_n + iY_n}{2}\right),

    where :math:`X` and :math:`Y` are the Pauli operators.

    Args:
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        n_states(int): maximum number of bosonic states per mode
        cubic(bool): Whether to include three-mode couplings. Default is ``False``.
        wire_map (dict): A dictionary defining how to map the states of the Bose operator to qubit
            wires. If ``None``, integers used to label the bosonic states will be used as wire
            labels. Defaults to ``None``.
        tol (float): tolerance for discarding the imaginary part of the coefficients

    Returns:
        Operator: the Christiansen Hamiltonian in the qubit basis

    **Example**

    >>> symbols  = ['H', 'F']
    >>> geometry = np.array([[0.0, 0.0, -0.40277116], [0.0, 0.0, 1.40277116]])
    >>> mol = qml.qchem.Molecule(symbols, geometry)
    >>> pes = qml.qchem.vibrational_pes(mol, optimize=False)
    >>> qml.qchem.vibrational.christiansen_hamiltonian(pes, n_states = 4)
    (
        0.08527499987546708 * I(0)
      + -0.0051774006335491545 * Z(0)
      + 0.0009697024705108074 * (X(0) @ X(1))
      + 0.0009697024705108074 * (Y(0) @ Y(1))
      + 0.0002321787923591865 * (X(0) @ X(2))
      + 0.0002321787923591865 * (Y(0) @ Y(2))
      + 0.0008190498635406456 * (X(0) @ X(3))
      + 0.0008190498635406456 * (Y(0) @ Y(3))
      + -0.015699890427524253 * Z(1)
      + 0.002790002362847834 * (X(1) @ X(2))
      + 0.002790002362847834 * (Y(1) @ Y(2))
      + 0.000687929225764568 * (X(1) @ X(3))
      + 0.000687929225764568 * (Y(1) @ Y(3))
      + -0.026572392417060237 * Z(2)
      + 0.005239546276220405 * (X(2) @ X(3))
      + 0.005239546276220405 * (Y(2) @ Y(3))
      + -0.037825316397333435 * Z(3)
    )
    """

    h_arr = christiansen_integrals(pes, n_states=n_states, cubic=cubic)

    one = h_arr[0]
    two = h_arr[1]
    three = h_arr[2] if len(h_arr) == 3 else None
    cform_bosonic = christiansen_bosonic(one=one, two=two, three=three)
    cform_qubit = christiansen_mapping(bose_operator=cform_bosonic, wire_map=wire_map, tol=tol)

    return cform_qubit


def christiansen_dipole(pes, n_states=16):
    r"""Returns Christiansen dipole operator.

    The Christiansen dipole operator is constructed similar to the vibrational Hamiltonian operator
    defined in Eqs. D4-D7 of `arXiv:2504.10602 <https://arxiv.org/abs/2504.10602>`_. The dipole
    operator is defined as

    .. math::

        \mu = \sum_{i}^M \sum_{k_i, l_i}^{N_i} C_{k_i, l_i}^{(i)} b_{k_i}^{\dagger} b_{l_i} +
        \sum_{i<j}^{M} \sum_{k_i,l_i}^{N_i} \sum_{k_j,l_j}^{N_j} C_{k_i k_j, l_i l_j}^{(i,j)}
        b_{k_i}^{\dagger} b_{k_j}^{\dagger} b_{l_i} b_{l_j},

    where :math:`b^{\dagger}` and :math:`b` are the bosonic creation and annihilation
    operators, :math:`M` represents the number of normal modes and :math:`N` is the number of
    modals. The coefficients :math:`C` represent the one-mode and two-mode integrals defined as

    .. math::

        C_{k_i, l_i}^{(i)} = \int \phi_i^{k_i}(Q_i) \left( D_1^{(i)}(Q_i) \right) \phi_i^{h_i}(Q_i),

    and

    .. math::

        C_{k_i, k_j, l_i, l_j}^{(i,j)} = \int \int \phi_i^{k_i}(Q_i) \phi_j^{k_j}(Q_j)
        D_2^{(i,j)}(Q_i, Q_j) \phi_i^{l_i}(Q_i) \phi_j^{l_j}(Q_j) \; \text{d} Q_i \text{d} Q_j,

    where :math:`\phi` represents a modal, :math:`Q` represents a normal coordinate and :math:`D`
    represents the dipole function obtained from the expansion

    .. math::

        D({Q}) = \sum_i D_1(Q_i) + \sum_{i>j} D_2(Q_i,Q_j) + ....

    Similarly, the three-mode integrals can be obtained following
    Eq. D7 of `arXiv:2504.10602 <https://arxiv.org/abs/2504.10602>`_.

    The bosonic creation and annihilation operators are then mapped to the Pauli operators as

    .. math::

        b^\dagger_0 = \left(\frac{X_0 - iY_0}{2}\right), \:\: \text{...,} \:\:
        b^\dagger_n = \left(\frac{X_n - iY_n}{2}\right),

    and

    .. math::

        b_0 = \left(\frac{X_0 + iY_0}{2}\right), \:\: \text{...,} \:\:
        b_n = \left(\frac{X_n + iY_n}{2}\right),

    where :math:`X` and :math:`Y` are the Pauli operators.

    Args:
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        n_states(int): maximum number of bosonic states per mode

    Returns:
        tuple: a tuple containing:
            - Operator: the Christiansen dipole operator in the qubit basis for x-displacements
            - Operator: the Christiansen dipole operator in the qubit basis for y-displacements
            - Operator: the Christiansen dipole operator in the qubit basis for z-displacements

    **Example**

    >>> symbols  = ['H', 'F']
    >>> geometry = np.array([[0.0, 0.0, -0.40277116], [0.0, 0.0, 1.40277116]])
    >>> mol = qml.qchem.Molecule(symbols, geometry)
    >>> pes = qml.qchem.vibrational_pes(mol, optimize=False, dipole_level=3, cubic=True)
    >>> dipole = qml.qchem.vibrational.christiansen_dipole(pes, n_states = 4)
    >>> dipole[2]
    (
        (-0.005512522132269153+0j) * I(0)
      + (0.00037053485106913064+0j) * Z(0)
      + -0.011436347025770977 * (X(0) @ X(1))
      + (-0.011436347025770977+0j) * (Y(0) @ Y(1))
      + -0.0005031491268437766 * (X(0) @ X(2))
      + (-0.0005031491268437766+0j) * (Y(0) @ Y(2))
      + 4.230790346195971e-05 * (X(0) @ X(3))
      + (4.230790346195971e-05+0j) * (Y(0) @ Y(3))
      + (0.001082095170147779+0j) * Z(1)
      + -0.01610015762949269 * (X(1) @ X(2))
      + (-0.01610015762949269+0j) * (Y(1) @ Y(2))
      + -0.0008228492926524582 * (X(1) @ X(3))
      + (-0.0008228492926524582+0j) * (Y(1) @ Y(3))
      + (0.001734095461712748+0j) * Z(2)
      + -0.01960990751144681 * (X(2) @ X(3))
      + (-0.01960990751144681+0j) * (Y(2) @ Y(3))
      + (0.002325796649339495+0j) * Z(3)
    )
    """

    d_arr = christiansen_integrals_dipole(pes, n_states=n_states)

    one_x = d_arr[0][0, :, :, :]
    two_x = d_arr[1][0, :, :, :, :, :, :] if len(d_arr) > 1 else None
    three_x = d_arr[2][0, :, :, :, :, :, :, :, :, :] if len(d_arr) == 3 else None
    cform_bosonic_x = christiansen_bosonic(one=one_x, two=two_x, three=three_x)
    cform_qubit_x = christiansen_mapping(cform_bosonic_x)

    one_y = d_arr[0][1, :, :, :]
    two_y = d_arr[1][1, :, :, :, :, :, :] if len(d_arr) > 1 else None
    three_y = d_arr[2][1, :, :, :, :, :, :, :, :, :] if len(d_arr) == 3 else None
    cform_bosonic_y = christiansen_bosonic(one=one_y, two=two_y, three=three_y)
    cform_qubit_y = christiansen_mapping(cform_bosonic_y)

    one_z = d_arr[0][2, :, :, :]
    two_z = d_arr[1][2, :, :, :, :, :, :] if len(d_arr) > 1 else None
    three_z = d_arr[2][2, :, :, :, :, :, :, :, :, :] if len(d_arr) == 3 else None
    cform_bosonic_z = christiansen_bosonic(one=one_z, two=two_z, three=three_z)
    cform_qubit_z = christiansen_mapping(cform_bosonic_z)

    return cform_qubit_x, cform_qubit_y, cform_qubit_z
