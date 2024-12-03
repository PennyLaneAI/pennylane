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
"""This module contains functions to localize normal modes."""

import warnings

import numpy as np
import scipy

import pennylane as qml

# pylint: disable=dangerous-default-value, too-many-statements


def _mat_transform(u, qmat):
    r"""Returns the rotated displacement vectors matrix for a given rotation unitary u
    and displacement vectors matrix qmat.

    Args:
        u (TensorLike[float]): unitary rotation matrix
        qmat (TensorLike[float]): matrix of displacement vectors

    Returns:
        TensorLike[float]: rotated matrix of displacement vectors

    """
    qloc = np.einsum("qp,iaq->iap", u, qmat)

    return qloc


def _params_to_unitary(params, nmodes):
    r"""Transforms a one-dimensional vector of parameters specifying a unitary rotation
    into its associated matrix u.

    Args:
        params (list[float]): parameters for unitary rotation
        nmodes (int): number of normal modes

    Returns:
        TensorLike[float]: unitary rotation matrix

    """
    ugen = np.zeros((nmodes, nmodes))

    idx = 0
    for m1 in range(nmodes):
        for m2 in range(m1):
            ugen[m1, m2] += params[idx]
            ugen[m2, m1] -= params[idx]
            idx += 1

    return scipy.linalg.expm(ugen)


def _params_cost(params, qmat, nmodes):
    r"""Returns the Pipek-Mezey cost function to be minimized for localized displacements.

    Args:
        params (list[float]): initial parameters
        qmat (TensorLike[float]): matrix of displacement vectors
        nmodes (int): number of normal modes
    Returns:
        float: Pipek-Mezek cost function

    """
    uparams = _params_to_unitary(params, nmodes)
    qrot = _mat_transform(uparams, qmat)
    xi_pm = np.sum(np.sum(qrot**2, axis=1) ** 2)

    return -xi_pm


def _normalize_q(qmat):
    r"""Returns the normalized displacement vectors.

    Args:
        qmat (TensorLike[float]): matrix of displacement vectors

    Returns:
        TensorLike[float]: normalized matrix of displacement vectors

    """

    qnormalized = np.zeros_like(qmat)
    nmodes = qmat.shape[2]

    for m in range(nmodes):
        m_norm = np.sum(np.abs(qmat[:, :, m]) ** 2)
        qnormalized[:, :, m] = qmat[:, :, m] / np.sqrt(m_norm)

    return qnormalized


def _localization_unitary(qmat):
    r"""Calculates the unitary matrix to localize the displacement vectors and
    displacement vectors.

    Args:
        qmat (TensorLike[float]): matrix of displacement vectors associated with normal-modes

    Returns:
        tuple: A tuple containing the following:
         - TensorLike[float] : unitary matrix to localize the displacement vectors
         - TensorLike[float] : localized displacement vectors

    """

    nmodes = qmat.shape[2]
    num_params = nmodes * (nmodes - 1) // 2

    rng = qml.math.random.default_rng(1000)
    params = 2 * np.pi * rng.random(num_params)

    qnormalized = _normalize_q(qmat)

    optimization_res = scipy.optimize.minimize(_params_cost, params, args=(qnormalized, nmodes))

    # Check if the minimization was successful; if it wasn't, proceed with the normal modes.
    if not optimization_res.success:
        warnings.warn(
            "Mode localization finished unsuccessfully, returning normal modes..."
        )  # pragma: no cover
        return _params_to_unitary(np.zeros_like(params), nmodes), qmat  # pragma: no cover

    params_opt = optimization_res.x
    uloc = _params_to_unitary(params_opt, nmodes)

    qloc = _mat_transform(uloc, qmat)

    return uloc, qloc


def _localize_modes(freqs, vecs):
    r"""Performs the mode localization for a given set of frequencies and displacement vectors.

    Args:
        freqs (list[float]): normal mode frequencies
        vecs (list[float]): displacement vectors along the normal modes

    Returns:
        tuple: A tuple containing the following:
         - list[float] : localized frequencies
         - TensorLike[float] : localized displacement vectors
         - TensorLike[float] : localization matrix

    """
    nmodes = len(freqs)
    hess_normal = np.diag(np.square(freqs))

    qmat = np.array([vecs[m] for m in range(nmodes)]).transpose(1, 2, 0)

    uloc, qloc = _localization_unitary(qmat)
    hess_loc = uloc.transpose() @ hess_normal @ uloc
    loc_freqs = np.sqrt(np.array([hess_loc[m, m] for m in range(nmodes)]))

    loc_perm = np.argsort(loc_freqs)
    loc_freqs = loc_freqs[loc_perm]
    qloc = qloc[:, :, loc_perm]
    uloc = uloc[:, loc_perm]

    return loc_freqs, qloc, uloc


def localize_normal_modes(freqs, vecs, bins=[2600]):
    """
    Localizes vibrational normal modes.

    The normal modes are localized by separating frequencies into specified ranges following the
    procedure described in `J. Chem. Phys. 141, 104105 (2014)
    <https://pubs.aip.org/aip/jcp/article-abstract/141/10/104105/74317/
    Efficient-anharmonic-vibrational-spectroscopy-for?redirectedFrom=fulltext>`_.

    Args:
        freqs (list[float]): normal mode frequencies in ``cm^-1``
        vecs (TensorLike[float]): displacement vectors for normal modes
        bins (list[float]): List of upper bound frequencies in ``cm^-1`` for creating separation bins .
            Default is ``[2600]`` which means having one bin for all frequencies between ``0`` and  ``2600 cm^-1``.

    Returns:
        tuple: A tuple containing the following:
         - list[float] : localized frequencies
         - TensorLike[float] : localized displacement vectors
         - TensorLike[float] : localization matrix describing the relationship between
           original and localized modes.

    """
    if not bins:
        raise ValueError("The `bins` list cannot be empty.")

    nmodes = len(freqs)

    num_seps = len(bins)
    natoms = vecs.shape[1]

    modes_arr = [min_modes := np.nonzero(freqs <= bins[0])[0]]
    freqs_arr = [freqs[min_modes]]
    displ_arr = [vecs[min_modes]]

    for sep_idx in range(num_seps - 1):
        mid_modes = np.nonzero((bins[sep_idx] <= freqs) & (bins[sep_idx + 1] >= freqs))[0]
        modes_arr.append(mid_modes)
        freqs_arr.append(freqs[mid_modes])
        displ_arr.append(vecs[mid_modes])

    modes_arr.append(max_modes := np.nonzero(freqs >= bins[-1])[0])
    freqs_arr.append(freqs[max_modes])
    displ_arr.append(vecs[max_modes])

    loc_freqs_arr, qlocs_arr, ulocs_arr = [], [], []
    for idx in range(num_seps + 1):
        num_freqs = len(freqs_arr[idx])
        freqs_block, qloc, uloc_block = [], np.zeros((natoms, 3, 0)), np.zeros((0, 0))
        if num_freqs > 1:
            freqs_block, qloc, uloc_block = _localize_modes(freqs_arr[idx], displ_arr[idx])
        elif num_freqs == 1:
            freqs_block = freqs_arr[idx]
            qloc = np.zeros((natoms, 3, 1))
            qloc[:, :, 0] = displ_arr[idx][0]
            uloc_block = np.eye(1)

        loc_freqs_arr.append(freqs_block)
        qlocs_arr.append(qloc)
        ulocs_arr.append(uloc_block)

    uloc = np.zeros((nmodes, nmodes))
    for idx, indices in enumerate(modes_arr):
        uloc[np.ix_(indices, indices)] = ulocs_arr[idx]

    loc_freqs = np.concatenate(loc_freqs_arr)
    loc_vecs = [
        qlocs_arr[idx][:, :, m]
        for idx in range(num_seps + 1)
        for m in range(len(loc_freqs_arr[idx]))
    ]

    return loc_freqs, loc_vecs, uloc
