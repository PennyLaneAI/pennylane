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

# pylint: disable = too-many-branches, too-many-positional-arguments, too-many-arguments, too-many-nested-blocks,


def christiansen_bosonic(one, modes=None, modals=None, two=None, three=None, ordered=True):
    r"""Return Christiansen bosonic vibrational Hamiltonian.

    The construction of the Hamiltonian is based on Eqs. 19-21 of
    `J. Chem. Theory Comput. 2023, 19, 24, 9329–9343 <https://pubs.acs.org/doi/10.1021/acs.jctc.3c00902?ref=PDF>`_.

    Args:
        one (TensorLike[float]): one-body matrix elements
        modes (int): number of vibrational modes. If ``None``, it is obtained from the length of ``one``.
        modals (list(int)): number of allowed vibrational modals for each mode. If ``None``, it is obtained from the shape of ``one``.
        two (TensorLike[float]): two-body matrix elements
        three (TensorLike[float]): three-body matrix elements
        cutoff (float): tolerance for discarding the negligible coefficients
        ordered (bool): indicates if matrix elements are already ordered. Default is ``True``.

    Returns:
        pennylane.bose.BoseSentence: the constructed bosonic operator
    """
    if modes is None:
        modes = np.shape(one)[0]

    if modals is None:
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
    r"""Return Christiansen vibrational Hamiltonian.

    The construction of the Hamiltonian is based on Eqs. 19-21 of
    `J. Chem. Theory Comput. 2023, 19, 24, 9329–9343 <https://pubs.acs.org/doi/10.1021/acs.jctc.3c00902?ref=PDF>`_.

    where the bosonic creation and annihilation operators are mapped to the Pauli operators as

    .. math::

        b^\dagger_0 = \left(\frac{X_0 - iY_0}{2}\right), \:\: \text{...,} \:\:
        b^\dagger_n = \left(\frac{X_n - iY_n}{2}\right),

    and

    .. math::

        b_0 = \left(\frac{X_0 + iY_0}{2}\right), \:\: \text{...,} \:\:
        b_n = \left(\frac{X_n + iY_n}{2}\right),

    where :math:`X`, :math:`Y`, and :math:`Z` are the Pauli operators.

    Args:
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        n_states(int): maximum number of bosonic states per mode
        cubic(bool): Flag to include three-mode couplings. Default is ``False``.
        wire_map (dict): A dictionary defining how to map the states of the Bose operator to qubit
            wires. If ``None``, integers used to label the bosonic states will be used as wire labels.
            Defaults to ``None``.
        tol (float): tolerance for discarding the imaginary part of the coefficients

    Returns:
        Operator: the Christiansen Hamiltonian in the qubit basis
    """

    h_arr = christiansen_integrals(pes, n_states=n_states, cubic=cubic)

    one = h_arr[0]
    two = h_arr[1]
    three = h_arr[2] if len(h_arr) == 3 else None
    cform_bosonic = christiansen_bosonic(one=one, two=two, three=three)
    cform_qubit = christiansen_mapping(bose_operator=cform_bosonic, wire_map=wire_map, tol=tol)

    return cform_qubit


def christiansen_dipole(pes, n_states=16):
    """Return Christiansen dipole operator.

    Args:
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        n_states(int): maximum number of bosonic states per mode

    Returns:
        tuple: a tuple containing:
            - Operator: the Christiansen dipole operator in the qubit basis for x-displacements
            - Operator: the Christiansen dipole operator in the qubit basis for y-displacements
            - Operator: the Christiansen dipole operator in the qubit basis for z-displacements
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
