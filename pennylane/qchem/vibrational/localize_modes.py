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

# pylint: disable=dangerous-default-value


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
    r"""Computes spatially localized vibrational normal modes.

    The vibrational normal modes are localized using a localizing unitary following the procedure
    described in `J. Chem. Phys. 141, 104105 (2014)
    <https://pubs.aip.org/aip/jcp/article-abstract/141/10/104105/74317/
    Efficient-anharmonic-vibrational-spectroscopy-for?redirectedFrom=fulltext>`_. The localizing
    unitary :math:`U` is defined in terms of the normal and local coordinates, :math:`q` and
    :math:`\tilde{q}`, respectively as:

    .. math::

        \tilde{q} = \sum_{j=1}^M U_{ij} q_j,

    where :math:`M` is the number of modes. The normal modes
    can be separately localized, to prevent mixing between specific groups of normal modes, by
    defining frequency ranges in ``bins``. For instance, ``bins = [2600]`` allows to separately
    localize modes that have frequencies above and below :math:`2600` reciprocal centimetre (:math:`\text{cm}^{-1}`).
    Similarly, ``bins = [1300, 2600]`` allows to separately localize modes in three groups that have
    frequencies below :math:`1300`, between :math:`1300-2600` and above :math:`2600`.

    Args:
        freqs (TensorLike[float]): normal mode frequencies in reciprocal centimetre (:math:`\text{cm}^{-1}`).
        vecs (TensorLike[float]): displacement vectors of the normal modes
        bins (List[float]): grid of frequencies for grouping normal modes.
            Default is ``[2600]``.

    Returns:
        tuple: A tuple containing the following:
         - TensorLike[float] : localized frequencies in reciprocal centimetre (:math:`\text{cm}^{-1}`).
         - List[TensorLike[float]] : localized displacement vectors
         - TensorLike[float] : localization matrix describing the relationship between the
           original and the localized modes

    **Example**

    >>> freqs = np.array([1326.66001461, 2297.26736859, 2299.65032901])
    >>> vectors = np.array([[[ 5.71518696e-18, -4.55642350e-01,  5.20920552e-01],
    ...                      [ 1.13167924e-17,  4.55642350e-01,  5.20920552e-01],
    ...                      [-1.23163569e-17,  5.09494945e-12, -3.27565762e-02]],
    ...                     [[-4.53008817e-17,  4.90364125e-01,  4.90363894e-01],
    ...                      [-1.98591028e-16,  4.90361513e-01, -4.90361744e-01],
    ...                      [-2.78235498e-18, -3.08350419e-02, -6.75886679e-08]],
    ...                     [[ 5.75393451e-17,  5.37047963e-01,  4.41957355e-01],
    ...                      [ 6.53049347e-17, -5.37050348e-01,  4.41959740e-01],
    ...                      [-5.49709883e-17,  7.49851221e-08, -2.77912798e-02]]])
    >>> freqs_loc, vecs_loc, uloc = qml.qchem.localize_normal_modes(freqs, vectors)
    >>> freqs_loc
    array([1332.62013257, 2296.73453455, 2296.73460655])

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
