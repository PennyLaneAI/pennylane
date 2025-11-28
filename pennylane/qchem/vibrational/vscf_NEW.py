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
from pennylane import math
import logging
logger = logging.getLogger(__name__)

# pylint: disable=too-many-arguments, too-many-positional-arguments

def _build_active_hamiltonian(h_integrals, modals, cutoff, like=None):
    r"""Identifies the active terms in the Hamiltonian, and converts the sparse Hamiltonian into
    dense arrays organized by mode. This function is based on the equations 20-22
    in `J. Chem. Theory Comput. 2010, 6, 235–248 <https://pubs.acs.org/doi/10.1021/ct9004454>`_.
    The equation suggests that if mode m is not contained in a Hamiltonian term, it evaluates to zero.

    Args:
        h_integrals (list[TensorLike[float]]): list of n-mode expansion of Hamiltonian integrals
        modals (list[int]): list containing the maximum number of modals to consider for each mode
        cutoff (float): threshold value for including matrix elements into operator
        like (TensorLike): if provided, the returned arrays will be converted to the same framework() and device) as `like`

    Returns:
        tuple: A tuple containing the following:
            - list[dict]: A list where index 'm' contains the dense tensors for mode 'm'.
            - int: total number of active terms in the Hamiltonian

    """
    
    nmodes = np.shape(h_integrals[0])[0]
    active_num = 0
    
    collectors = [
        {"term_map": [], "i": [], "j": [], "coeffs": [], "mask": []} 
        for _ in range(nmodes)
    ]

    for n, ham_n in enumerate(h_integrals):
        for idx, h_val in np.ndenumerate(ham_n):
            if np.abs(h_val) < cutoff:
                continue

            modal_indices = idx[: n + 1]

            if not all(modal_indices[k] > modal_indices[k + 1] for k in range(n)):
                continue

            excitation_indices = idx[n + 1 :]
            temp_pairs = []
            term_valid = True
            
            for m_idx, m in enumerate(modal_indices):
                # Chemist notation 
                i = excitation_indices[m_idx]
                j = excitation_indices[m_idx + len(modal_indices)]
                if i >= modals[m] or j >= modals[m]:
                    term_valid = False
                    break
                temp_pairs.append((i, j))
            
            if not term_valid:
                continue

            # this term is valid and should be stored
            
            # pre-calculate the interaction mask for this term
            full_term_mask = np.zeros(nmodes, dtype=bool)
            full_term_mask[list(modal_indices)] = True

            for m_idx, m in enumerate(modal_indices):
                i, j = temp_pairs[m_idx]
                
                collectors[m]["term_map"].append(active_num)
                collectors[m]["i"].append(i)
                collectors[m]["j"].append(j)
                collectors[m]["coeffs"].append(h_val)
                
                # create mask for this specific mode (exclude self)
                m_mask = full_term_mask.copy()
                m_mask[m] = False
                collectors[m]["mask"].append(m_mask)

            active_num += 1

    # convert to arrays
    # NOTE  should be moved to use qml.math
    dense_data = []
    for mode in range(nmodes):
        c = collectors[mode]
        if c["term_map"]:
            mode_data = {
                "term_map": np.array(c["term_map"], dtype=int),
                "i": np.array(c["i"], dtype=int),
                "j": np.array(c["j"], dtype=int),
                "coeffs": np.array(c["coeffs"], dtype=float),
                "mask": np.array(c["mask"], dtype=bool),
            }
            
            # Convert to desired framework
            if like is not None:
                for key in mode_data:
                    mode_data[key] = math.convert_like(mode_data[key], like)
            
        dense_data.append(mode_data)

    return dense_data, active_num

def _fock_energy(all_modes_data, modals, h_mat, mode_rots):
    r"""Calculates vibrational energy.

    Args:
        all_modes_data (list[dict]): a list where index 'm' contains the dense tensors for mode 'm'.
        modals (list[int]): list containing the maximum number of modals to consider for each mode
        h_mat (array[array[float]]): the Hamiltonian matrix
        mode_rots (List[TensorLike[float]]): list of rotation matrices for each vibrational mode

    Returns:
        float: vibrational energy

    """
    nmodes = math.shape(h_mat)[1]
    e_calc = 0.0
    for m in range(nmodes):
        # energy calculation is just F[0,0] in the new basis
        f = _build_fock(all_modes_data[m], modals[m], h_mat, mode_rots[m])
        e_calc += f[0,0]
    return float(e_calc) # casting to avoid framework issues


def _accumulate_fock(modals_count, i_idxs, j_idxs, vals, like_tensor):
    """
    Backend-agnostic equivalent of np.add.at for the Fock matrix.
    Uses One-Hot encoding + Einstein Summation to accumulate values.
    """
    # 1. Create a range vector [0, 1, ..., M-1]
    # We use qml.math.cast_like to ensure it's on the correct device/dtype
    m_range = math.convert_like(np.arange(modals_count), like_tensor)
    
    # 2. Create One-Hot Encodings for Rows (i) and Columns (j)
    # Shape becomes (Num_Terms, Modals)
    # Logic: (i[:, None] == range[None, :]) -> Boolean Mask -> Float
    
    # Expand dims for broadcasting
    i_expanded = math.reshape(i_idxs, (-1, 1))
    j_expanded = math.reshape(j_idxs, (-1, 1))
    range_expanded = math.reshape(m_range, (1, -1))
    
    # Create masks and cast to the same type as 'vals' (float)
    # Note: explicit cast is needed because the comparison returns Bool
    row_one_hot = math.cast(math.equal(i_expanded, range_expanded), vals.dtype)
    col_one_hot = math.cast(math.equal(j_expanded, range_expanded), vals.dtype)
    
    # 3. Contract: sum_k (val_k * row_oh_ka * col_oh_kb) -> F_ab
    # 'n' is the term index, 'a' is row index, 'b' is col index
    fock_matrix = math.einsum('n, na, nb -> ab', vals, row_one_hot, col_one_hot)
    
    return fock_matrix


def _build_fock(mode_data, modals_count, h_mat, mode_rot):
    """
    Builds the Fock matrix using purely vectorized operations.
    """
    
    fock_matrix = math.convert_like(np.zeros((modals_count, modals_count)), h_mat)
    if len(mode_data) == 0:
        return fock_matrix

    # let's extract the hamiltonian terms that affect the current mode
    subset_h = h_mat[mode_data["term_map"]]
    
    # we want product(h_mat[t, m]) only for coupled modes.
    # If mask is 1: keeps h value. If mask is 0: becomes 1.0 (identity for product).
    mask = mode_data["mask"] # shape (N_terms, N_modes)
    factors = subset_h * mask + (~mask) # using bitwise NOT for bool array ~mask is 1-mask
    mean_fields = math.prod(factors, axis=1) #collapse to shape (N_terms,)
    
    # let's accumulate into Fock matrix 
    vals = mode_data["coeffs"] * mean_fields    # effective H values
    
    # accumulate into sparse locations (i, j)
    
    if isinstance(h_mat, np.ndarray):
        # Use numpy's built-in function if possible for better performance
        # np.add.at is the vectorized equivalent of `for k: fock[i[k], j[k]] += val[k]`
        np.add.at(fock_matrix, (mode_data["i"], mode_data["j"]), vals)
    else:
        # --- REPLACEMENT FOR np.add.at ---
        fock_matrix = _accumulate_fock(modals_count, mode_data["i"], mode_data["j"], vals, h_mat)

    fock_matrix = mode_rot.T @ fock_matrix @ mode_rot
    
    return fock_matrix

def _update_h(h_mat, mode, mode_data, mode_rot):
    """
    Updates H matrix columns using vectorized assignment.
    """
    if len(mode_data) == 0:
        return h_mat

    u = mode_rot[:, 0]      # ground state coefficients
    new_vals = u[mode_data["i"]] * u[mode_data["j"]]
    h_mat[mode_data["term_map"], mode] = new_vals

    return h_mat


def _vscf(h_integrals, modals, cutoff, like=None, tol=1e-8, max_iters=10000):
    
    nmodes = np.shape(h_integrals[0])[0]
    
    
    
    dense_data, active_num = _build_active_hamiltonian(h_integrals, modals, cutoff, like)
    mode_rots = [np.eye(modals[mode]) for mode in range(nmodes)]
    h_mat = np.zeros((active_num, nmodes))
    
    # eventual switch to GPU 
    if not like is None:
        h_mat = math.convert_like(h_mat, like)
        mode_rots = [math.convert_like(mat, like) for mat in mode_rots]


    # initialization of the hamiltonian matrix
    for mode in range(nmodes):
        h_mat = _update_h(h_mat, mode, dense_data[mode], mode_rots[mode])

    e0 = _fock_energy(dense_data, modals, h_mat, mode_rots)
    
    # SCF Loop
    for _ in range(max_iters):
        
        for mode in range(nmodes):
            fock = _build_fock(dense_data[mode], modals[mode], h_mat, mode_rots[mode])
            _, eigvec = math.linalg.eigh(fock)
            mode_rots[mode] = mode_rots[mode] @ eigvec
            h_mat = _update_h(h_mat, mode, dense_data[mode], mode_rots[mode])

        # check convergence
        enew = _fock_energy(dense_data, modals, h_mat, mode_rots)
        
        if np.abs(enew - e0) <= tol:
            return enew, mode_rots
            
        e0 = enew

    if not like is None:
        mode_rots = [math.convert_like(mat, np.ndarray) for mat in mode_rots]

    return enew, mode_rots





def _rotate_one_body(h1, nmodes, mode_rots, modals):
    r"""Rotates one body integrals.

    Args:
        h1 (TensorLike[float]): one-body integrals
        nmodes (int): number of vibrational modes
        mode_rots (list[TensorLike[float]]): list of rotation matrices for all vibrational modes
        modals (list[int]): list containing the maximum number of modals to consider for each mode

    Returns:
        TensorLike[float]: rotated one-body integrals

    """
    imax = np.max(modals)
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
        modals (list[int]): list containing the maximum number of modals to consider for each mode

    Returns:
        TensorLike[float]: rotated two-body integrals

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
        modals (list[int]): list containing the maximum number of modals to consider for each mode

    Returns:
        TensorLike[float]: rotated three-body integrals

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


def _rotate_dipole(d_integrals, mode_rots, modals):
    r"""Generates VSCF rotated dipole.

    Args:
        d_integrals (list[TensorLike[float]]): list of n-mode expansion of dipole integrals
        mode_rots (list[TensorLike[float]]): list of rotation matrices for all vibrational modes
        modals (list[int]): list containing the maximum number of modals to consider for each mode

    Returns:
        tuple(TensorLike[float]): a tuple of rotated dipole integrals

    """
    n = len(d_integrals)

    nmodes = np.shape(d_integrals[0])[0]
    imax = np.max(modals)
    d1_rot = np.zeros((3, nmodes, imax, imax))

    for alpha in range(3):
        d1_rot[alpha, ::] = _rotate_one_body(
            d_integrals[0][alpha, ::], nmodes, mode_rots, modals=modals
        )
    dip_data = [d1_rot]

    if n > 1:
        d2_rot = np.zeros((3, nmodes, nmodes, imax, imax, imax, imax))
        for alpha in range(3):
            d2_rot[alpha, ::] = _rotate_two_body(
                d_integrals[1][alpha, ::], nmodes, mode_rots, modals=modals
            )
        dip_data = [d1_rot, d2_rot]

    if n > 2:
        d3_rot = np.zeros((3, nmodes, nmodes, nmodes, imax, imax, imax, imax, imax, imax))
        for alpha in range(3):
            d3_rot[alpha, ::] = _rotate_three_body(
                d_integrals[2][alpha, ::], nmodes, mode_rots, modals=modals
            )
        dip_data = [d1_rot, d2_rot, d3_rot]

    return dip_data


def _rotate_hamiltonian(h_integrals, mode_rots, modals):
    r"""Generates VSCF rotated Hamiltonian.

    Args:
        h_integrals (list[TensorLike[float]]): list of n-mode expansion of Hamiltonian integrals
        mode_rots (list[TensorLike[float]]): list of rotation matrices for all vibrational modes
        modals (list[int]): list containing the maximum number of modals to consider for each mode

    Returns:
        tuple(TensorLike[float]): tuple of rotated Hamiltonian integrals

    """

    n = len(h_integrals)
    nmodes = np.shape(h_integrals[0])[0]

    h1_rot = _rotate_one_body(h_integrals[0], nmodes, mode_rots, modals)
    h_data = [h1_rot]

    if n > 1:
        h2_rot = _rotate_two_body(h_integrals[1], nmodes, mode_rots, modals)
        h_data = [h1_rot, h2_rot]

    if n > 2:
        h3_rot = _rotate_three_body(h_integrals[2], nmodes, mode_rots, modals)
        h_data = [h1_rot, h2_rot, h3_rot]

    return h_data


def vscf_integrals(h_integrals, d_integrals=None, modals=None, cutoff=None, cutoff_ratio=1e-6):
    r"""Generates vibrational self-consistent field rotated integrals.

    This function converts the Christiansen vibrational Hamiltonian integrals obtained in the harmonic
    oscillator basis to integrals in the vibrational self-consistent field (VSCF) basis.
    The implementation is based on the method described in
    `J. Chem. Theory Comput. 2010, 6, 235–248 <https://pubs.acs.org/doi/10.1021/ct9004454>`_.

    Args:
        h_integrals (list[TensorLike[float]]): List of Hamiltonian integrals for up to 3 coupled vibrational modes.
            Look at the Usage Details for more information.
        d_integrals (list[TensorLike[float]]): List of dipole integrals for up to 3 coupled vibrational modes.
            Look at the Usage Details for more information.
        modals (list[int]): list containing the maximum number of modals to consider for each vibrational mode.
            Default value is the maximum number of modals.
        cutoff (float): threshold value for including matrix elements into operator
        cutoff_ratio (float): ratio for discarding elements with respect to biggest element in the integrals.
            Default value is ``1e-6``.

    Returns:
        tuple: a tuple containing:
            - list[TensorLike[float]]: List of Hamiltonian integrals in VSCF basis for up to 3 coupled vibrational modes.
            - list[TensorLike[float]]: List of dipole integrals in VSCF basis for up to 3 coupled vibrational modes.

        ``None`` is returned if ``d_integrals`` is ``None``.

    **Example**

    >>> h1 = np.array([[[0.00968289, 0.00233724, 0.0007408,  0.00199125],
    ...                 [0.00233724, 0.02958449, 0.00675431, 0.0021936],
    ...                 [0.0007408,  0.00675431, 0.0506012,  0.01280986],
    ...                 [0.00199125, 0.0021936,  0.01280986, 0.07282307]]])
    >>> qml.qchem.vscf_integrals(h_integrals=[h1], modals=[4,4,4])
    ([array([[[ 9.36124041e-03, -4.20128342e-19,  3.25260652e-19,
            1.08420217e-18],
            [-9.21571847e-19,  2.77803512e-02, -3.46944695e-18,
            5.63785130e-18],
            [-3.25260652e-19, -8.67361738e-19,  4.63297357e-02,
            -1.04083409e-17],
            [ 1.30104261e-18,  5.20417043e-18, -1.38777878e-17,
            7.92203227e-02]]])],
    None)


    .. details::
        :title: Usage Details

        The ``h_integral`` tensor must have one of these dimensions:

        - 1-mode coupled integrals: `(n, m, m)`
        - 2-mode coupled integrals: `(n, n, m, m, m, m)`
        - 3-mode coupled integrals: `(n, n, n, m, m, m, m, m, m)`

        where ``n`` is the number of vibrational modes in the molecule and ``m`` represents the number
        of modals.

        The ``d_integral`` tensor must have one of these dimensions:

        - 1-mode coupled integrals: `(3, n, m)`
        - 2-mode coupled integrals: `(3, n, n, m, m, m, m)`
        - 3-mode coupled integrals: `(3, n, n, n, m, m, m, m, m, m)`

        where ``n`` is the number of vibrational modes in the molecule, ``m`` represents the number
        of modals and the first axis represents the ``x, y, z`` component of the dipole. Default is ``None``.

    """

    if len(h_integrals) > 3:
        raise ValueError(
            f"Building n-mode Hamiltonian is not implemented for n equal to {len(h_integrals)}."
        )

    if d_integrals is not None:
        if len(d_integrals) > 3:
            raise ValueError(
                f"Building n-mode dipole is not implemented for n equal to {len(d_integrals)}."
            )

    nmodes = np.shape(h_integrals[0])[0]

    imax = np.shape(h_integrals[0])[1]
    max_modals = nmodes * [imax]
    if modals is None:
        modals = max_modals
    else:
        if np.max(modals) > imax:
            raise ValueError(
                "Number of maximum modals cannot be greater than the modals for unrotated integrals."
            )
        imax = np.max(modals)

    if cutoff is None:
        max_val = np.max([np.max(np.abs(H)) for H in h_integrals])
        cutoff = max_val * cutoff_ratio

    _, mode_rots = _vscf(h_integrals, modals=max_modals, cutoff=cutoff)

    h_data = _rotate_hamiltonian(h_integrals, mode_rots, modals)

    if d_integrals is not None:
        dip_data = _rotate_dipole(d_integrals, mode_rots, modals)
        return h_data, dip_data

    return h_data, None
