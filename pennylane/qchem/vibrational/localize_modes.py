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

# Factor to change units of frequency (Hartree to cm^-1)
AU_TO_CM = 219475

# pylint: disable=dangerous-default-value, too-many-statements


def _pm_cost(q):
    r"""Pipek-Mezey cost function whose minimization yields localized displacements.

    Args:
       q: matrix of displacement vectors

    """

    nnuc, _, nmodes = q.shape

    xi_pm = 0.0

    for p in range(nmodes):
        for i in range(nnuc):
            q2 = 0.0
            for alpha in range(3):
                q2 += q[i, alpha, p] ** 2
            xi_pm += q2**2

    return -xi_pm


def _mat_transform(u, qmat):
    r"""Returns the rotated displacement vectors matrix for a given rotation unitary u and displacement vectors matrix qmat.

    Args:
       u: unitary rotation matrix
       qmat: matrix of displacement vectors

    Returns:
       rotated matrix of displacement vectors

    """
    qloc = np.einsum("qp,iaq->iap", u, qmat)

    return qloc


def _params_to_unitary(params, nmodes):
    r"""Transforms a one-dimensional vector of parameters specifying a unitary rotation into its associated matrix u.

    Args:
       params: parameters for unitary rotation
       nmodes: number of normal modes

    Returns:
       unitary rotation matrix

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
    r"""Returns the cost function to be minimized for localized displacements.

    Args:
       params: initial parameters
       qmat: matrix of displacement vectors
       nmodes: number of normal modes

    """
    uparams = _params_to_unitary(params, nmodes)
    qrot = _mat_transform(uparams, qmat)

    return _pm_cost(qrot)


def _normalize_q(qmat):
    r"""Returns the normalized displacement vectors.

    Args:
       qmat: matrix of displacement vectors

    Returns:
       normalized matrix of displacement vectors

    """

    qnormalized = np.zeros_like(qmat)
    nmodes = qmat.shape[2]

    for m in range(nmodes):
        m_norm = np.sum(np.abs(qmat[:, :, m]) ** 2)
        qnormalized[:, :, m] = qmat[:, :, m] / np.sqrt(m_norm)

    return qnormalized


def _localization_unitary(qmat):
    r"""Calculates the unitary matrix to localize the displacement vectors and displacement vectors.

    Args:
       qmat: matrix of displacement vectors associated with normal-modes

    Returns:
       a tuple of unitary matrix to localize the displacement vectors and localized displacement vectors

    """

    nmodes = qmat.shape[2]
    num_params = int(nmodes * (nmodes - 1) / 2)

    rng = qml.math.random.default_rng(1000)
    params = 2 * np.pi * rng.random(num_params)

    qnormalized = _normalize_q(qmat)

    optimization_res = scipy.optimize.minimize(_params_cost, params, args=(qnormalized, nmodes))

    # Check if the minimization was successful; if it wasn't, proceed with the normal modes.
    if not optimization_res.success:
        warnings.warn(
            "Mode localization finished unsuccessfully, returning normal modes..."
        )  # pragma: no cover
        return _params_to_unitary(0 * params, nmodes), qmat  # pragma: no cover

    params_opt = optimization_res.x
    uloc = _params_to_unitary(params_opt, nmodes)

    qloc = _mat_transform(uloc, qmat)

    return uloc, qloc


def _localize_modes(freqs, displ_vecs, order=True):
    r"""Performs the mode localization for a given set of frequencies and displacement vectors.

    Args:
       freqs: normal mode frequencies
       displ_vecs: displacement vectors along the normal modes

    Returns:
       a tuple of localized frequencies, displacement vectors, and localization matrix

    """
    nmodes = len(freqs)
    hess_normal = np.zeros((nmodes, nmodes))
    for m in range(nmodes):
        hess_normal[m, m] = freqs[m] ** 2

    natoms, _ = np.shape(displ_vecs[0])

    qmat = np.zeros((natoms, 3, nmodes))
    for m in range(nmodes):
        dvec = displ_vecs[m]
        for i in range(natoms):
            for alpha in range(3):
                qmat[i, alpha, m] = dvec[i, alpha]

    uloc, qloc = _localization_unitary(qmat)
    hess_loc = uloc.transpose() @ hess_normal @ uloc
    loc_freqs = np.sqrt(np.array([hess_loc[m, m] for m in range(nmodes)]))

    if order:
        loc_perm = np.argsort(loc_freqs)
        loc_freqs = loc_freqs[loc_perm]
        qloc = qloc[:, :, loc_perm]
        uloc = uloc[:, loc_perm]

    return loc_freqs, qloc, uloc


def localize_normal_modes(freqs, displ_vecs, freq_separation=[2600]):
    """
    Localizes normal modes by separating frequencies into specified ranges and applying mode localization.
    The procedure for localization is described in `J. Chem. Phys. 141, 104105 (2014)
    <https://pubs.aip.org/aip/jcp/article-abstract/141/10/104105/74317/Efficient-anharmonic-vibrational-spectroscopy-for?redirectedFrom=fulltext>`_.

    Args:
        freqs: normal mode frequencies in ``cm^-1``
        displ_vecs: displacement vectors for normal modes
        freq_separation (list): list of frequency separation thresholds in cm^-1. Defaults to [2600].

    Returns:
        A tuple of frequencies, normalized displacement vectors and localization matrix indicating the relationship between
        original and localized modes.:

    """
    if not freq_separation:
        raise ValueError("The `freq_separation` list cannot be empty.")

    nmodes = len(freqs)

    num_seps = len(freq_separation)
    natoms = displ_vecs.shape[1]

    modes_arr = [min_modes := np.nonzero(freqs <= freq_separation[0])[0]]
    freqs_arr = [freqs[min_modes]]
    displ_arr = [displ_vecs[min_modes]]

    for sep_idx in range(num_seps - 1):
        mid_modes = np.nonzero(
            (freq_separation[sep_idx] <= freqs) & (freq_separation[sep_idx + 1] >= freqs)
        )[0]
        modes_arr.append(mid_modes)
        freqs_arr.append(freqs[mid_modes])
        displ_arr.append(displ_vecs[mid_modes])

    modes_arr.append(max_modes := np.nonzero(freqs >= freq_separation[-1])[0])
    freqs_arr.append(freqs[max_modes])
    displ_arr.append(displ_vecs[max_modes])

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
    loc_displ_vecs = [
        qlocs_arr[idx][:, :, m]
        for idx in range(num_seps + 1)
        for m in range(len(loc_freqs_arr[idx]))
    ]

    return loc_freqs, loc_displ_vecs, uloc
