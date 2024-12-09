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
"""This module contains functions to perform VSCF calculation."""

import numpy as np

# pylint: disable=too-many-arguments


def _build_Fock(mode, active_ham_terms, active_terms, modals, h_mat, mode_rot):
    r"""Builds the fock matrix for each vibrational mode.

    Args:
        mode (int): index of the active mode for which the Fock matrix is being calculated
        active_ham_terms (dict): A dictionary containing the active Hamiltonian terms.
        active_terms (list): A list containing the active Hamiltonian terms for the given mode.
        modals (list[int]): A list containing the maximum number of modals to consider for each mode.
        h_mat (Array[Array[float]]): the Hamiltonian matrix
        mode_rot (TensorLike[float]): rotation matrix for the given vibrational mode
    Returns:
        Array[Array[float]]: fock matrix for the given mode
    """

    fock_matrix = np.zeros((modals[mode], modals[mode]))
    for term in active_terms:
        ham_term = active_ham_terms[term]
        h_val = ham_term[0]
        modal_indices = [*ham_term[1]]
        excitation_indices = [*ham_term[2]]

        if mode in modal_indices:
            modal_idx = modal_indices.index(mode)

        im = excitation_indices[modal_idx]
        jm = excitation_indices[modal_idx + len(modal_indices)]
        if im < modals[mode] and jm < modals[mode]:
            h_val = h_val * np.prod(
                [h_mat[term, modal] for modal in modal_indices if modal != mode]
            )
            fock_matrix[im, jm] += h_val

    fock_matrix = mode_rot.T @ fock_matrix @ mode_rot
    return fock_matrix


def _update_h(h_mat, mode, active_ham_terms, mode_rot, ts_act, modals):
    r"""Updates value of h matrix for a mode after other modes have been rotated.

    Args:
        h_mat (Array[Array[float]]): the hamiltonian matrix
        mode (int): index of the active mode for which the h matrix is being updated.
        active_ham_terms (dict): A dictionary containing the active Hamiltonian terms.
        mode_rot (array[array[float]]): rotation matrix for a given mode
        active_terms (list):  A list containing active Hamiltonian terms for the given mode.
        modals (list[int]): A list containing the maximum number of modals to consider for each mode.

    Returns:
        Array[Array[float]]: the updated hamiltonian matrix

    """
    u_coeffs = mode_rot[:, 0]

    for t in ts_act:
        ham_term = active_ham_terms[t]
        modal_indices = ham_term[1]
        excitation_indices = ham_term[2]

        if mode in modal_indices:
            modal_idx = modal_indices.index(mode)

        im = excitation_indices[modal_idx]
        jm = excitation_indices[modal_idx + len(modal_indices)]

        if im < modals[mode] and jm < modals[mode]:
            h_mat[t, mode] = u_coeffs[im] * u_coeffs[jm]

    return h_mat


def _find_active_terms(ham_integrals, modals, cutoff):
    r"""Identifies the active terms in the Hamiltonian

    Args:
        ham_integrals (list[TensorLike[float]]): A list of n-mode expansion of Hamiltonian integrals.
        modals (list[int]): A list containing the maximum number of modals to consider for each mode.
        cutoff (float): threshold value for including matrix elements into operator.

    Returns:
        tuple: A tuple containing the following:
         - dict: A dictionary containing the active Hamiltonian terms.
         - dict: A dictionary containing the active Hamiltonian terms for all vibrational modes.
         - int: total number of active terms in the Hamiltonian

    """
    active_ham_terms = {}
    active_num = 0
    nmodes = np.shape(ham_integrals[0])[0]

    for n, ham_n in enumerate(ham_integrals):
        ham_n = ham_integrals[n]
        for idx, h_val in np.ndenumerate(ham_n):
            if np.abs(h_val) < cutoff:
                continue

            M_arr = idx[: n + 1]

            right_order = True
            for i in range(n):
                if M_arr[i] <= M_arr[i + 1]:
                    right_order = False
                    break

            if right_order:
                exc_arr = idx[n + 1 :]
                exc_in_modals = True
                for m_idx, m in enumerate(M_arr):
                    im = exc_arr[m_idx]
                    jm = exc_arr[m_idx + len(M_arr)]
                    if im >= modals[m] or jm >= modals[m]:
                        exc_in_modals = False
                        break
                if exc_in_modals:
                    active_ham_terms[active_num] = [h_val, M_arr, exc_arr]
                    active_num += 1

    active_mode_terms = {}
    for m in range(nmodes):
        ts_for_m_arr = []
        for t in range(active_num):
            if m in active_ham_terms[t][1]:
                ts_for_m_arr.append(t)
        active_mode_terms[m] = ts_for_m_arr
    return active_ham_terms, active_mode_terms, active_num


def _fock_energy(h_mat, active_ham_terms, active_mode_terms, modals, mode_rots):
    r"""Calculates vibrational energy

    Args:
        h_mat (Array[Array[float]]): the Hamiltonian matrix
        active_ham_terms (dict): A dictionary containing the active Hamiltonian terms
        active_mode_terms (dict): A list containing the active Hamiltonian terms for all vibrational modes.
        modals (list[int]): A list containing the maximum number of modals to consider for each mode.
        mode_rots (List[TensorLike[float]]): A list of rotation matrices for each vibrational mode.

    Returns:
        float: vibrational energy

    """
    nmodes = h_mat.shape[1]
    e0s = np.zeros(nmodes)
    for mode in range(nmodes):
        fock_mat = _build_Fock(
            mode, active_ham_terms, active_mode_terms[mode], modals, h_mat, mode_rots[mode]
        )
        e0s[mode] = fock_mat[0, 0]

    return np.sum(e0s)


def _vscf(ham_integrals, modals, cutoff, tol=1e-8, max_iters=10000):
    r"""Performs the VSCF calculation.

    Args:
        ham_integrals (list(TensorLike[float])): a list containing Hamiltonian integral matrices
        modals (list(int)): A list containing the maximum number of modals to consider for each mode.
        cutoff (float): threshold value for including matrix elements into operator.
        tol (float): convergence tolerance for vscf calculation
        max_iters (int): Maximum number of iterations for vscf to converge.

    Returns:
        tuple : A tuple of the following:
         - float: vscf energy
         - List[TensorLike[float]]: A list of rotation matrices for all vibrational modes.

    """

    nmodes = np.shape(ham_integrals[0])[0]

    active_ham_terms, active_mode_terms, active_num = _find_active_terms(
        ham_integrals, modals, cutoff
    )
    mode_rots = [np.eye(modals[mode]) for mode in range(nmodes)]
    h_mat = np.zeros((active_num, nmodes))
    for mode in range(nmodes):
        h_mat = _update_h(
            h_mat, mode, active_ham_terms, mode_rots[mode], active_mode_terms[mode], modals
        )

    e0 = _fock_energy(h_mat, active_ham_terms, active_mode_terms, modals, mode_rots)

    enew = e0 + 2 * tol
    curr_iter = 0
    while curr_iter <= max_iters and np.abs(enew - e0) > tol:
        if curr_iter != 0:
            e0 = enew

        curr_iter += 1

        for mode in range(nmodes):
            fock = _build_Fock(
                mode, active_ham_terms, active_mode_terms[mode], modals, h_mat, mode_rots[mode]
            )
            _, eigvec = np.linalg.eigh(fock)
            mode_rots[mode] = mode_rots[mode] @ eigvec
            h_mat = _update_h(
                h_mat, mode, active_ham_terms, mode_rots[mode], active_mode_terms[mode], modals
            )
            fock = np.transpose(eigvec) @ fock @ eigvec

        enew = _fock_energy(h_mat, active_ham_terms, active_mode_terms, modals, mode_rots)
    e0 = enew
    return e0, mode_rots


def _rotate_one_body(h1, nmodes, mode_rots, modals):
    r"""Rotates one body integrals.

    Args:
        h1 (TensorLike[float]): one-body integrals
        nmodes (int): number of vibrational modes
        mode_rots (list[TensorLike[float]]): list of rotation matrices for all vibrational modes
        modals (list[int]): A list containing the maximum number of modals to consider for each mode.

    Returns:
        TensorLike[float]: rotated one-body integrals.

    """
    imax = np.max(modals)
    if imax > np.shape(h1)[1]:
        raise ValueError(
            "Number of maximum modals cannot be greater than the modals for unrotated integrals."
        )

    h1_rot = np.zeros((nmodes, imax, imax))
    for m in range(nmodes):
        h1_rot[m, : modals[m], : modals[m]] = np.einsum(
            "ij,ia,jb->ab", h1[m, :, :], mode_rots[m][:, : modals[m]], mode_rots[m][:, : modals[m]]
        )

    return h1_rot


def _rotate_two_body(h2, nmodes, mode_rots, modals):
    r"""Rotates two body integrals.

    Args:
        h2 (TensorLike[float]): two-body integrals
        nmodes (int): number of vibrational modes
        mode_rots (list[TensorLike[float]]): list of rotation matrices for all vibrational modes
        modals (list[int]): A list containing the maximum number of modals to consider for each mode.

    Returns:
        TensorLike[float]: rotated two-body integrals.

    """
    imax = np.max(modals)

    U_mats = [mode_rots[m] for m in range(nmodes)]
    h2_rot = np.zeros((nmodes, nmodes, imax, imax, imax, imax))
    for m1 in range(nmodes):
        for m2 in range(nmodes):
            h2_rot[m1, m2, : modals[m1], : modals[m2], : modals[m1], : modals[m2]] = np.einsum(
                "ijkl,ia,jb,kc,ld->abcd",
                h2[m1, m2, :, :, :, :],
                U_mats[m1][:, : modals[m1]],
                U_mats[m2][:, : modals[m2]],
                U_mats[m1][:, : modals[m1]],
                U_mats[m2][:, : modals[m2]],
            )

    return h2_rot


def _rotate_three_body(h3, nmodes, mode_rots, modals):
    r"""Rotates three body integrals.

    Args:
        h3 (TensorLike[float]): three-body integrals
        nmodes (int): number of vibrational modes
        mode_rots (list[TensorLike[float]]): list of rotation matrices for all vibrational modes
        modals (list[int]): A list containing the maximum number of modals to consider for each mode.

    Returns:
        TensorLike[float]: rotated three-body integrals.

    """
    imax = np.max(modals)

    h3_rot = np.zeros((nmodes, nmodes, nmodes, imax, imax, imax, imax, imax, imax))
    for m1 in range(nmodes):
        for m2 in range(nmodes):
            for m3 in range(nmodes):
                h3_rot[
                    m1,
                    m2,
                    m3,
                    : modals[m1],
                    : modals[m2],
                    : modals[m3],
                    : modals[m1],
                    : modals[m2],
                    : modals[m3],
                ] = np.einsum(
                    "ijklmn,ia,jb,kc,ld,me,nf->abcdef",
                    h3[m1, m2, m3, :, :, :, :, :, :],
                    mode_rots[m1][:, : modals[m1]],
                    mode_rots[m2][:, : modals[m2]],
                    mode_rots[m3][:, : modals[m3]],
                    mode_rots[m1][:, : modals[m1]],
                    mode_rots[m2][:, : modals[m2]],
                    mode_rots[m3][:, : modals[m3]],
                )

    return h3_rot


def vscf_hamiltonian(ham_integrals, modals=None, cutoff=None, cutoff_ratio=1e-6):
    r"""Generates VSCF rotated integrals for vibrational Hamiltonian.

    Args:
        ham_integrals (list[TensorLike[float]]): A list of n-mode expansion of Hamiltonian integrals.
        modals (list[int]): A list containing the maximum number of modals to consider for each mode. Defaults to maximum value if none is provided
        cutoff (float): threshold value for including matrix elements into operator.
        cutoff_ratio (float): default ratio for zeroing elements which are smaller than this ratio with respect to biggest element in all H1,H2,...'s

    Returns:
        List[TensorLike[float]]: List of n-mode expansion of Hamiltonian integrals in VSCF basis.

    """
    n = len(ham_integrals)

    if n > 3:
        raise ValueError(f"Building n-mode Hamiltonian not implemented for {n}=>3!")

    nmodes = np.shape(ham_integrals[0])[0]

    imax = np.shape(ham_integrals[0])[1]
    max_modals = nmodes * [imax]
    if modals is None:
        modals = max_modals
    else:
        imax = np.max(modals)

    if cutoff is None:
        max_val = np.max([np.max(np.abs(H)) for H in ham_integrals])
        cutoff = max_val * cutoff_ratio

    _, mode_rots = _vscf(ham_integrals, modals=max_modals, cutoff=cutoff)
    h1_rot = _rotate_one_body(ham_integrals[0], nmodes, mode_rots, modals)
    if n == 1:
        return [h1_rot]

    h2_rot = _rotate_two_body(ham_integrals[1], nmodes, mode_rots, modals)
    if n == 2:
        return [h1_rot, h2_rot]

    h3_rot = _rotate_three_body(ham_integrals[2], nmodes, mode_rots, modals)
    return [h1_rot, h2_rot, h3_rot]
