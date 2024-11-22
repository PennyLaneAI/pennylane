from functools import singledispatch
from typing import Union

import numpy as np

import pennylane as qml
from pennylane.pauli import PauliSentence, PauliWord
from pennylane.bose import BoseSentence, BoseWord
from .christiansen_utils import christiansen_integrals, christiansen_integrals_dipole

def christiansen_bosonic(
    one, modes=None, modals=None, two=None, three=None, cutoff=1e-5, ordered=True
):
    r"""Build a vibrational observable in the Christiansen form (C-form) and map it
    to the Pauli basis

    Args:
        one (array): 3D array with one-body matrix elements
        modes (int): number of vibrational modes, detects from 'one' if none is provided
        modals (array): 1D array with the number of allowed vibrational modals for each mode, detects from 'one' if none is provided
        two (array): 6D array with two-body matrix elements
        three (array): 9D array with three-body matrix elements
        cutoff (float): magnitude beneath which terms are not incorporated in final expression
        ordered (bool): set True if matrix elements are ordered, i.e. two[m,n,::] = 0 for all n >= m and three[m,n,l,::] = 0 for all n >= m and l >= n

    Returns:
        tuple[int, Union[PauliSentence, Operator]]: the number of qubits and a linear combination of qubit operators
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
            if ordered is False:
                m_range = [p for p in range(modes) if p != l]
            else:
                m_range = range(l)
            for m in m_range:
                if ordered is False:
                    n_range = [p for p in range(modes) if p != l and p != m]
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


# def christiansen_hamiltonian(pes, nbos=16, do_cubic=False):

#     h_arr = christiansen_integrals(pes, nbos=nbos, do_cubic=do_cubic)

#     one = h_arr[0]
#     two = h_arr[1]
#     three = h_arr[2] if len(h_arr) == 3 else None
#     cform_bosonic = christiansen_bosonic(one=one, two=two, three=three)
#     cform_qubit = christiansen_mapping(cform_bosonic)

#     return cform_qubit


# def christiansen_dipole(pes, nbos=16, do_cubic=False):

#     d_arr = christiansen_integrals_dipole(pes, nbos=nbos, do_cubic=do_cubic)

#     one_x = d_arr[0][0, :, :, :]
#     two_x = d_arr[1][0, :, :, :, :, :, :] if len(d_arr) > 1 else None
#     three_x = d_arr[2][0, :, :, :, :, :, :, :, :, :] if len(d_arr) == 3 else None
#     cform_bosonic_x = christiansen_bosonic(one=one_x, two=two_x, three=three_x)
#     print(cform_bosonic_x)
#     cform_qubit_x = christiansen_mapping(cform_bosonic_x)

#     one_y = d_arr[0][1, :, :, :]
#     two_y = d_arr[1][1, :, :, :, :, :, :] if len(d_arr) > 1 else None
#     three_y = d_arr[2][1, :, :, :, :, :, :, :, :, :] if len(d_arr) == 3 else None
#     cform_bosonic_y = christiansen_bosonic(one=one_y, two=two_y, three=three_y)
#     cform_qubit_y = christiansen_mapping(cform_bosonic_y)

#     one_z = d_arr[0][2, :, :, :]
#     two_z = d_arr[1][2, :, :, :, :, :, :] if len(d_arr) > 1 else None
#     three_z = d_arr[2][2, :, :, :, :, :, :, :, :, :] if len(d_arr) == 3 else None
#     cform_bosonic_z = christiansen_bosonic(one=one_z, two=two_z, three=three_z)
#     cform_qubit_z = christiansen_mapping(cform_bosonic_z)

#     return cform_qubit_x, cform_qubit_y, cform_qubit_z
