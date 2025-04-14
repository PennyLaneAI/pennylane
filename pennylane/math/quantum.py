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
"""Differentiable quantum functions"""
import functools

# pylint: disable=import-outside-toplevel
import itertools
from string import ascii_letters

import scipy as sp
import scipy.sparse.linalg as spla
from autoray import numpy as np
from numpy import float64, sqrt  # pylint:disable=wrong-import-order
from scipy.sparse import csc_matrix, issparse

import pennylane as qml

from . import single_dispatch  # pylint:disable=unused-import
from .interface_utils import get_interface
from .matrix_manipulation import _permute_dense_matrix
from .multi_dispatch import diag, dot, einsum, scatter_element_add
from .utils import allclose, cast, cast_like, convert_like, is_abstract

ascii_letter_arr = np.array(list(ascii_letters))


def cov_matrix(prob, obs, wires=None, diag_approx=False):
    """Calculate the covariance matrix of a list of commuting observables, given
    the joint probability distribution of the system in the shared eigenbasis.

    .. note::
        This method only works for **commuting observables.**
        If the probability distribution is the result of a quantum circuit,
        the quantum state must be rotated into the shared
        eigenbasis of the list of observables before measurement.

    Args:
        prob (tensor_like): probability distribution
        obs (list[.Observable]): a list of observables for which
            to compute the covariance matrix
        diag_approx (bool): if True, return the diagonal approximation
        wires (.Wires): The wire register of the system. If not provided,
            it is assumed that the wires are labelled with consecutive integers.

    Returns:
        tensor_like: the covariance matrix of size ``(len(obs), len(obs))``

    **Example**

    Consider the following ansatz and observable list:

    >>> obs_list = [qml.X(0) @ qml.Z(1), qml.Y(2)]
    >>> ansatz = qml.templates.StronglyEntanglingLayers

    We can construct a QNode to output the probability distribution in the shared eigenbasis of the
    observables:

    .. code-block:: python

        from pennylane import numpy as np

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev, interface="autograd")
        def circuit(weights):
            ansatz(weights, wires=[0, 1, 2])
            # rotate into the basis of the observables
            for o in obs_list:
                o.diagonalizing_gates()
            return qml.probs(wires=[0, 1, 2])

    We can now compute the covariance matrix:

    >>> shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=3)
    >>> weights = np.random.random(shape, requires_grad=True)
    >>> cov = qml.math.cov_matrix(circuit(weights), obs_list)
    >>> cov
    tensor([[0.98125435, 0.4905541 ],
            [0.4905541 , 0.99920878]], requires_grad=True)

    Autodifferentiation is fully supported using all interfaces.
    Here we use autograd:

    >>> cost_fn = lambda weights: qml.math.cov_matrix(circuit(weights), obs_list)[0, 1]
    >>> qml.grad(cost_fn)(weights)
    array([[[ 4.94240914e-17, -2.33786398e-01, -1.54193959e-01],
            [-3.05414996e-17,  8.40072236e-04,  5.57884080e-04],
            [ 3.01859411e-17,  8.60411436e-03,  6.15745204e-04]],
           [[ 6.80309533e-04, -1.23162742e-03,  1.08729813e-03],
            [-1.53863193e-01, -1.38700657e-02, -1.36243323e-01],
            [-1.54665054e-01, -1.89018172e-02, -1.56415558e-01]]])
    """
    variances = []

    # diagonal variances
    for i, o in enumerate(obs):
        eigvals = cast(o.eigvals(), dtype=float64)
        w = o.wires.labels if wires is None else wires.indices(o.wires)
        p = marginal_prob(prob, w)

        res = dot(eigvals**2, p) - (dot(eigvals, p)) ** 2
        variances.append(res)

    cov = diag(variances)

    if diag_approx:
        return cov

    for i, j in itertools.combinations(range(len(obs)), r=2):
        o1 = obs[i]
        o2 = obs[j]

        o1wires = o1.wires.labels if wires is None else wires.indices(o1.wires)
        o2wires = o2.wires.labels if wires is None else wires.indices(o2.wires)
        shared_wires = set(o1wires + o2wires)

        l1 = cast(o1.eigvals(), dtype=float64)
        l2 = cast(o2.eigvals(), dtype=float64)
        l12 = cast(np.kron(l1, l2), dtype=float64)

        p1 = marginal_prob(prob, o1wires)
        p2 = marginal_prob(prob, o2wires)
        p12 = marginal_prob(prob, shared_wires)

        res = dot(l12, p12) - dot(l1, p1) * dot(l2, p2)

        cov = scatter_element_add(cov, [i, j], res)
        cov = scatter_element_add(cov, [j, i], res)

    return cov


def marginal_prob(prob, axis):
    """Compute the marginal probability given a joint probability distribution expressed as a tensor.
    Each random variable corresponds to a dimension.

    If the distribution arises from a quantum circuit measured in computational basis, each dimension
    corresponds to a wire. For example, for a 2-qubit quantum circuit `prob[0, 1]` is the probability of measuring the
    first qubit in state 0 and the second in state 1.

    Args:
        prob (tensor_like): 1D tensor of probabilities. This tensor should of size
            ``(2**N,)`` for some integer value ``N``.
        axis (list[int]): the axis for which to calculate the marginal
            probability distribution

    Returns:
        tensor_like: the marginal probabilities, of
        size ``(2**len(axis),)``

    **Example**

    >>> x = tf.Variable([1, 0, 0, 1.], dtype=tf.float64) / np.sqrt(2)
    >>> marginal_prob(x, axis=[0, 1])
    <tf.Tensor: shape=(4,), dtype=float64, numpy=array([0.70710678, 0.        , 0.        , 0.70710678])>
    >>> marginal_prob(x, axis=[0])
    <tf.Tensor: shape=(2,), dtype=float64, numpy=array([0.70710678, 0.70710678])>
    """
    prob = np.flatten(prob)
    num_wires = int(np.log2(len(prob)))

    if num_wires == len(axis):
        return prob

    inactive_wires = tuple(set(range(num_wires)) - set(axis))
    prob = np.reshape(prob, [2] * num_wires)
    prob = np.sum(prob, axis=inactive_wires)
    return np.flatten(prob)


def reduce_dm(density_matrix, indices, check_state=False, c_dtype="complex128"):
    """Compute the density matrix from a state represented with a density matrix.

    Args:
        density_matrix (tensor_like): 2D or 3D density matrix tensor. This tensor should be of size ``(2**N, 2**N)`` or
            ``(batch_dim, 2**N, 2**N)``, for some integer number of wires``N``.
        indices (list(int)): List of indices in the considered subsystem.
        check_state (bool): If True, the function will check the state validity (shape and norm).
        c_dtype (str): Complex floating point precision type.

    Returns:
        tensor_like: Density matrix of size ``(2**len(indices), 2**len(indices))`` or ``(batch_dim, 2**len(indices), 2**len(indices))``

    .. seealso:: :func:`pennylane.math.reduce_statevector`, and :func:`pennylane.density_matrix`

    **Example**

    >>> x = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    >>> reduce_dm(x, indices=[0])
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]]

    >>> y = [[0.5, 0, 0.5, 0], [0, 0, 0, 0], [0.5, 0, 0.5, 0], [0, 0, 0, 0]]
    >>> reduce_dm(y, indices=[0])
    [[0.5+0.j 0.5+0.j]
     [0.5+0.j 0.5+0.j]]

    >>> reduce_dm(y, indices=[1])
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]]

    >>> z = tf.Variable([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=tf.complex128)
    >>> reduce_dm(z, indices=[1])
    tf.Tensor(
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]], shape=(2, 2), dtype=complex128)

    >>> x = np.array([[[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    ...               [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])
    >>> reduce_dm(x, indices=[1])
    array([[[1.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j]],
           [[0.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j]]])
    """
    density_matrix = cast(density_matrix, dtype=c_dtype)

    if check_state:
        _check_density_matrix(density_matrix)

    if len(np.shape(density_matrix)) == 2:
        batch_dim, dim = None, density_matrix.shape[0]
    else:
        batch_dim, dim = density_matrix.shape[:2]

    num_indices = int(np.log2(dim))
    consecutive_indices = list(range(num_indices))

    # Return the full density matrix if all the wires are given, potentially permuted
    if len(indices) == num_indices:
        return _permute_dense_matrix(density_matrix, consecutive_indices, indices, batch_dim)

    if batch_dim is None:
        density_matrix = qml.math.stack([density_matrix])

    # Compute the partial trace
    traced_wires = [x for x in consecutive_indices if x not in indices]
    density_matrix = partial_trace(density_matrix, traced_wires, c_dtype=c_dtype)

    if batch_dim is None:
        density_matrix = density_matrix[0]

    # Permute the remaining indices of the density matrix
    return _permute_dense_matrix(density_matrix, sorted(indices), indices, batch_dim)


def partial_trace(matrix, indices, c_dtype="complex128"):
    """Compute the reduced density matrix by tracing out the provided indices.

    Args:
        matrix (tensor_like): 2D or 3D density matrix tensor. For a 2D tensor, the size is assumed to be
            ``(2**n, 2**n)``, for some integer number of wires ``n``. For a 3D tensor, the first dimension is assumed to be the batch dimension, ``(batch_dim, 2**N, 2**N)``.

        indices (list(int)): List of indices to be traced.

    Returns:
        tensor_like: (reduced) Density matrix of size ``(2**len(wires), 2**len(wires))``

    .. seealso:: :func:`pennylane.math.reduce_dm`, and :func:`pennylane.math.reduce_statevector`

    **Example**

    We can compute the partial trace of the matrix ``x`` with respect to its 0th index.

    >>> x = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    >>> partial_trace(x, indices=[0])
    array([[1.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j]])

    We can also pass a batch of matrices ``x`` to the function and return the partial trace of each matrix with respect to each matrix's 0th index.

    >>> x = np.array([
    ... [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    ... [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    ... ])
    >>> partial_trace(x, indices=[0])
    array([[[1.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j]],
           [[0.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j]]])

    The partial trace can also be computed with respect to multiple indices within different frameworks such as TensorFlow.

    >>> x = tf.Variable([[[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    ... [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]], dtype=tf.complex128)
    >>> partial_trace(x, indices=[1])
    <tf.Tensor: shape=(2, 2, 2), dtype=complex128, numpy=
    array([[[1.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j]],
           [[0.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j]]])>

    """
    # Autograd does not support same indices sum in backprop, and tensorflow
    # has a limit of 8 dimensions if same indices are used
    matrix = cast(matrix, dtype=c_dtype)
    if qml.math.ndim(matrix) == 2:
        is_batched = False
        batch_dim, dim = 1, matrix.shape[1]
    else:
        is_batched = True
        batch_dim, dim = matrix.shape[:2]

    if get_interface(matrix) in ["autograd", "tensorflow"]:
        return _batched_partial_trace_nonrep_indices(matrix, is_batched, indices, batch_dim, dim)

    # Dimension and reshape
    num_indices = int(np.log2(dim))
    rho_dim = 2 * num_indices

    matrix = np.reshape(matrix, [batch_dim] + [2] * 2 * num_indices)
    indices = np.sort(indices)

    # For loop over wires
    for i, target_index in enumerate(indices):
        target_index = target_index - i
        state_indices = ascii_letters[1 : rho_dim - 2 * i + 1]
        state_indices = list(state_indices)

        target_letter = state_indices[target_index]
        state_indices[target_index + num_indices - i] = target_letter
        state_indices = "".join(state_indices)

        einsum_indices = f"a{state_indices}"
        matrix = einsum(einsum_indices, matrix)

    number_wires_sub = num_indices - len(indices)
    reduced_density_matrix = np.reshape(
        matrix, (batch_dim, 2**number_wires_sub, 2**number_wires_sub)
    )
    return reduced_density_matrix if is_batched else reduced_density_matrix[0]


def _batched_partial_trace_nonrep_indices(matrix, is_batched, indices, batch_dim, dim):
    """Compute the reduced density matrix for autograd interface by tracing out the provided indices with the use
    of projectors as same subscripts indices are not supported in autograd backprop.
    """

    num_indices = int(np.log2(dim))
    rho_dim = 2 * num_indices
    matrix = np.reshape(matrix, [batch_dim] + [2] * 2 * num_indices)

    kraus = cast(np.eye(2), matrix.dtype)

    kraus = np.reshape(kraus, (2, 1, 2))
    kraus_dagger = np.asarray([np.conj(np.transpose(k)) for k in kraus])

    kraus = convert_like(kraus, matrix)
    kraus_dagger = convert_like(kraus_dagger, matrix)
    # For loop over wires
    for target_wire in indices:
        # Tensor indices of density matrix
        state_indices = ascii_letters[1 : rho_dim + 1]
        # row indices of the quantum state affected by this operation
        row_wires_list = [target_wire + 1]
        row_indices = "".join(ascii_letter_arr[row_wires_list].tolist())
        # column indices are shifted by the number of wires
        col_wires_list = [w + num_indices for w in row_wires_list]
        col_indices = "".join(ascii_letter_arr[col_wires_list].tolist())
        # indices in einsum must be replaced with new ones
        num_partial_trace_wires = 1
        new_row_indices = ascii_letters[rho_dim + 1 : rho_dim + num_partial_trace_wires + 1]
        new_col_indices = ascii_letters[
            rho_dim + num_partial_trace_wires + 1 : rho_dim + 2 * num_partial_trace_wires + 1
        ]
        # index for summation over Kraus operators
        kraus_index = ascii_letters[
            rho_dim + 2 * num_partial_trace_wires + 1 : rho_dim + 2 * num_partial_trace_wires + 2
        ]
        # new state indices replace row and column indices with new ones
        new_state_indices = functools.reduce(
            lambda old_string, idx_pair: old_string.replace(idx_pair[0], idx_pair[1]),
            zip(col_indices + row_indices, new_col_indices + new_row_indices),
            state_indices,
        )
        # index mapping for einsum, e.g., 'iga,abcdef,idh->gbchef'
        einsum_indices = (
            f"{kraus_index}{new_row_indices}{row_indices}, a{state_indices},"
            f"{kraus_index}{col_indices}{new_col_indices}->a{new_state_indices}"
        )
        matrix = einsum(einsum_indices, kraus, matrix, kraus_dagger)

    number_wires_sub = num_indices - len(indices)
    reduced_density_matrix = np.reshape(
        matrix, (batch_dim, 2**number_wires_sub, 2**number_wires_sub)
    )
    return reduced_density_matrix if is_batched else reduced_density_matrix[0]


def reduce_statevector(state, indices, check_state=False, c_dtype="complex128"):
    """Compute the density matrix from a state vector.

    Args:
        state (tensor_like): 1D or 2D tensor state vector. This tensor should of size ``(2**N,)``
            or ``(batch_dim, 2**N)``, for some integer value ``N``.
        indices (list(int)): List of indices in the considered subsystem.
        check_state (bool): If True, the function will check the state validity (shape and norm).
        c_dtype (str): Complex floating point precision type.

    Returns:
        tensor_like: Density matrix of size ``(2**len(indices), 2**len(indices))`` or ``(batch_dim, 2**len(indices), 2**len(indices))``

    .. seealso:: :func:`pennylane.math.reduce_dm` and :func:`pennylane.density_matrix`

    **Example**

    >>> x = np.array([1, 0, 0, 0])
    >>> reduce_statevector(x, indices=[0])
    [[1.+0.j 0.+0.j]
    [0.+0.j 0.+0.j]]

    >>> y = [1, 0, 1, 0] / np.sqrt(2)
    >>> reduce_statevector(y, indices=[0])
    [[0.5+0.j 0.5+0.j]
     [0.5+0.j 0.5+0.j]]

    >>> reduce_statevector(y, indices=[1])
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]]

    >>> z = tf.Variable([1, 0, 0, 0], dtype=tf.complex128)
    >>> reduce_statevector(z, indices=[1])
    tf.Tensor(
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]], shape=(2, 2), dtype=complex128)

    >>> x = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    >>> reduce_statevector(x, indices=[1])
    array([[[1.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j]],
           [[0.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j]]])
    """
    state = cast(state, dtype=c_dtype)

    # Check the format and norm of the state vector
    if check_state:
        _check_state_vector(state)

    if len(np.shape(state)) == 1:
        batch_dim, dim = None, np.shape(state)[0]
    else:
        batch_dim, dim = np.shape(state)[:2]

        # batch dim exists but is unknown; cast to int so that reshaping works
        if batch_dim is None:
            batch_dim = -1

    # Get dimension of the quantum system and reshape
    num_wires = int(np.log2(dim))
    consecutive_wires = list(range(num_wires))

    if batch_dim is None:
        state = qml.math.stack([state])

    state = np.reshape(state, [batch_dim if batch_dim is not None else 1] + [2] * num_wires)

    # Get the system to be traced
    # traced_system = [x + 1 for x in consecutive_wires if x not in indices]

    # trace out the subsystem
    indices1 = ascii_letters[1 : num_wires + 1]
    indices2 = "".join(
        [
            ascii_letters[num_wires + i + 1] if i in indices else ascii_letters[i + 1]
            for i in consecutive_wires
        ]
    )
    target = "".join(
        [ascii_letters[i + 1] for i in sorted(indices)]
        + [ascii_letters[num_wires + i + 1] for i in sorted(indices)]
    )
    density_matrix = einsum(
        f"a{indices1},a{indices2}->a{target}",
        state,
        np.conj(state),
        optimize="greedy",
    )

    # Return the reduced density matrix by using numpy tensor product
    # density_matrix = np.tensordot(state, np.conj(state), axes=(traced_system, traced_system))

    if batch_dim is None:
        density_matrix = np.reshape(density_matrix, (2 ** len(indices), 2 ** len(indices)))
    else:
        density_matrix = np.reshape(
            density_matrix, (batch_dim, 2 ** len(indices), 2 ** len(indices))
        )

    return _permute_dense_matrix(density_matrix, sorted(indices), indices, batch_dim)


def dm_from_state_vector(state, check_state=False, c_dtype="complex128"):
    """
    Convenience function to compute a (full) density matrix from
    a state vector.

    Args:
        state (tensor_like): 1D or 2D tensor state vector. This tensor should of size ``(2**N,)``
            or ``(batch_dim, 2**N)``, for some integer value ``N``.
        check_state (bool): If True, the function will check the state validity (shape and norm).
        c_dtype (str): Complex floating point precision type.

    Returns:
        tensor_like: Density matrix of size ``(2**N, 2**N)`` or ``(batch_dim, 2**N, 2**N)``

    **Example**

    >>> x = np.array([1, 0, 1j, 0]) / np.sqrt(2)
    >>> dm_from_state_vector(x)
    array([[0.5+0.j , 0. +0.j , 0. -0.5j, 0. +0.j ],
           [0. +0.j , 0. +0.j , 0. +0.j , 0. +0.j ],
           [0. +0.5j, 0. +0.j , 0.5+0.j , 0. +0.j ],
           [0. +0.j , 0. +0.j , 0. +0.j , 0. +0.j ]])

    """
    num_wires = int(np.log2(np.shape(state)[-1]))
    return reduce_statevector(
        state,
        indices=list(range(num_wires)),
        check_state=check_state,
        c_dtype=c_dtype,
    )


def purity(state, indices, check_state=False, c_dtype="complex128"):
    r"""Computes the purity of a density matrix.

    .. math::
        \gamma = \text{Tr}(\rho^2)

    where :math:`\rho` is the density matrix. The purity of a normalized quantum state satisfies
    :math:`\frac{1}{d} \leq \gamma \leq 1`, where :math:`d` is the dimension of the Hilbert space.
    A pure state has a purity of 1.

    It is possible to compute the purity of a sub-system from a given state. To find the purity of
    the overall state, include all wires in the ``indices`` argument.

    Args:
        state (tensor_like): Density matrix of shape ``(2**N, 2**N)`` or ``(batch_dim, 2**N, 2**N)``
        indices (list(int)): List of indices in the considered subsystem.
        check_state (bool): If ``True``, the function will check the state validity (shape and norm).
        c_dtype (str): Complex floating point precision type.

    Returns:
        float: Purity of the considered subsystem.

    **Example**

    >>> x = [[1/2, 0, 0, 1/2], [0, 0, 0, 0], [0, 0, 0, 0], [1/2, 0, 0, 1/2]]
    >>> purity(x, [0, 1])
    1.0
    >>> purity(x, [0])
    0.5

    >>> x = [[1/2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1/2]]
    >>> purity(x, [0, 1])
    0.5
    """
    # Cast as a c_dtype array
    state = cast(state, dtype=c_dtype)

    density_matrix = reduce_dm(state, indices, check_state)
    return _compute_purity(density_matrix)


def _compute_purity(density_matrix):
    """Compute the purity from a density matrix

    Args:
        density_matrix (tensor_like): ``(2**N, 2**N)`` or ``(batch_dim, 2**N, 2**N)`` tensor for an integer `N`.

    Returns:
        float: Purity of the density matrix.

    **Example**

    >>> x = [[1/2, 0], [0, 1/2]]
    >>> _compute_purity(x)
    0.5

    >>> x = [[1/2, 1/2], [1/2, 1/2]]
    >>> _compute_purity(x)
    1

    """
    batched = len(qml.math.shape(density_matrix)) > 2

    if batched:
        return qml.math.real(qml.math.einsum("abc,acb->a", density_matrix, density_matrix))

    return qml.math.real(qml.math.einsum("ab,ba", density_matrix, density_matrix))


def vn_entropy(state, indices, base=None, check_state=False, c_dtype="complex128"):
    r"""Compute the Von Neumann entropy from a density matrix on a given subsystem. It supports all
    interfaces (NumPy, Autograd, Torch, TensorFlow and Jax).

    .. math::
        S( \rho ) = -\text{Tr}( \rho \log ( \rho ))

    Args:
        state (tensor_like): Density matrix of shape ``(2**N, 2**N)`` or ``(batch_dim, 2**N, 2**N)``.
        indices (list(int)): List of indices in the considered subsystem.
        base (float): Base for the logarithm. If None, the natural logarithm is used.
        check_state (bool): If True, the function will check the state validity (shape and norm).
        c_dtype (str): Complex floating point precision type.

    Returns:
        float: Von Neumann entropy of the considered subsystem.

    **Example**

    The entropy of a subsystem for any state vectors can be obtained. Here is an example for the
    maximally entangled state, where the subsystem entropy is maximal (default base for log is exponential).

    >>> x = [1, 0, 0, 1] / np.sqrt(2)
    >>> x = dm_from_state_vector(x)
    >>> vn_entropy(x, indices=[0])
    0.6931472

    The logarithm base can be switched to 2 for example.

    >>> vn_entropy(x, indices=[0], base=2)
    1.0

    .. seealso:: :func:`pennylane.vn_entropy`

    """
    density_matrix = reduce_dm(state, indices, check_state, c_dtype)
    entropy = _compute_vn_entropy(density_matrix, base)
    return entropy


def _compute_vn_entropy(density_matrix, base=None):
    """Compute the Von Neumann entropy from a density matrix

    Args:
        density_matrix (tensor_like): ``(2**N, 2**N)`` or ``(batch_dim, 2**N, 2**N)`` tensor for an integer `N`.
        base (float, int): Base for the logarithm. If None, the natural logarithm is used.

    Returns:
        float: Von Neumann entropy of the density matrix.

    **Example**

    >>> x = [[1/2, 0], [0, 1/2]]
    >>> _compute_vn_entropy(x)
    0.6931472

    >>> x = [[1/2, 0], [0, 1/2]]
    >>> _compute_vn_entropy(x, base=2)
    1.0

    """
    # Change basis if necessary
    if base:
        div_base = np.log(base)
    else:
        div_base = 1

    evs = qml.math.eigvalsh(density_matrix)
    evs = qml.math.where(evs > 0, evs, 1.0)
    entropy = qml.math.entr(evs) / div_base

    return entropy


# pylint: disable=too-many-arguments
def mutual_info(
    state,
    indices0,
    indices1,
    base=None,
    check_state=False,
    c_dtype="complex128",
):
    r"""Compute the mutual information between two subsystems given a state:

    .. math::

        I(A, B) = S(\rho^A) + S(\rho^B) - S(\rho^{AB})

    where :math:`S` is the von Neumann entropy.

    The mutual information is a measure of correlation between two subsystems.
    More specifically, it quantifies the amount of information obtained about
    one system by measuring the other system. It supports all interfaces
    (NumPy, Autograd, Torch, TensorFlow and Jax).

    Each state must be given as a density matrix. To find the mutual information given
    a pure state, call :func:`~.math.dm_from_state_vector` first.

    Args:
        state (tensor_like): ``(2**N, 2**N)`` or ``(batch_dim, 2**N, 2**N)`` density matrix.
        indices0 (list[int]): List of indices in the first subsystem.
        indices1 (list[int]): List of indices in the second subsystem.
        base (float): Base for the logarithm. If None, the natural logarithm is used.
        check_state (bool): If True, the function will check the state validity (shape and norm).
        c_dtype (str): Complex floating point precision type.

    Returns:
        float: Mutual information between the subsystems

    **Examples**

    The mutual information between subsystems for a state vector can be returned as follows:

    >>> x = np.array([1, 0, 0, 1]) / np.sqrt(2)
    >>> x = qml.math.dm_from_state_vector(x)
    >>> qml.math.mutual_info(x, indices0=[0], indices1=[1])
    1.3862943611198906

    It is also possible to change the log basis.

    >>> qml.math.mutual_info(x, indices0=[0], indices1=[1], base=2)
    2.0

    Similarly the quantum state can be provided as a density matrix:

    >>> y = np.array([[1/2, 1/2, 0, 1/2], [1/2, 0, 0, 0], [0, 0, 0, 0], [1/2, 0, 0, 1/2]])
    >>> qml.math.mutual_info(y, indices0=[0], indices1=[1])
    0.4682351577408206

    .. seealso:: :func:`~.math.vn_entropy` and :func:`pennylane.mutual_info`

    """

    # the subsystems cannot overlap
    if len([index for index in indices0 if index in indices1]) > 0:
        raise ValueError("Subsystems for computing mutual information must not overlap.")

    return _compute_mutual_info(
        state,
        indices0,
        indices1,
        base=base,
        check_state=check_state,
        c_dtype=c_dtype,
    )


# pylint: disable=too-many-arguments
def _compute_mutual_info(
    state,
    indices0,
    indices1,
    base=None,
    check_state=False,
    c_dtype="complex128",
):
    """Compute the mutual information between the subsystems."""
    all_indices = sorted([*indices0, *indices1])
    vn_entropy_1 = vn_entropy(
        state,
        indices=indices0,
        base=base,
        check_state=check_state,
        c_dtype=c_dtype,
    )
    vn_entropy_2 = vn_entropy(
        state,
        indices=indices1,
        base=base,
        check_state=check_state,
        c_dtype=c_dtype,
    )
    vn_entropy_12 = vn_entropy(
        state,
        indices=all_indices,
        base=base,
        check_state=check_state,
        c_dtype=c_dtype,
    )

    return vn_entropy_1 + vn_entropy_2 - vn_entropy_12


def _check_hermitian_operator(operators):
    """Check the shape, and if the matrix is hermitian."""
    dim = operators.shape[-1]

    if (
        len(operators.shape) not in (2, 3)
        or operators.shape[-2] != dim
        or not np.log2(dim).is_integer()
    ):
        raise ValueError(
            "Operator matrix must be of shape (2**wires,2**wires) "
            "or (batch_dim, 2**wires, 2**wires)."
        )

    if len(operators.shape) == 2:
        operators = qml.math.stack([operators])

    if not is_abstract(operators):
        for ops in operators:
            conj_trans = np.transpose(np.conj(ops))
            if not allclose(ops, conj_trans):
                raise ValueError("The matrix is not Hermitian.")


def expectation_value(
    operator_matrix, state_vector, check_state=False, check_operator=False, c_dtype="complex128"
):
    r"""Compute the expectation value of an operator with respect to a pure state.

    The expectation value is the probabilistic expected result of an experiment.
    Given a pure state, i.e., a state which can be represented as a single
    vector :math:`\ket{\psi}` in the Hilbert space, the expectation value of an
    operator :math:`A` can computed as

    .. math::
        \langle A \rangle_\psi = \bra{\psi} A \ket{\psi}


    Args:
        operator_matrix (tensor_like): operator matrix with shape ``(2**N, 2**N)`` or ``(batch_dim, 2**N, 2**N)``.
        state_vector (tensor_like): state vector with shape ``(2**N)`` or ``(batch_dim, 2**N)``.
        check_state (bool): if True, the function will check the validity of the state vector
            via its shape and the norm.
        check_operator (bool): if True, the function will check the validity of the operator
            via its shape and whether it is hermitian.
        c_dtype (str): complex floating point precision type.

    Returns:
        float: Expectation value of the operator for the state vector.

    **Example**

    The expectation value for any operator can obtained by passing their matrix representation as an argument.
    For example, for a 2 qubit state, we can compute the expectation value of the operator :math:`Z \otimes I` as

    >>> import pennylane as qml
    >>> import numpy as np
    >>> state_vector = [1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0]
    >>> operator_matrix = qml.matrix(qml.PauliZ(0), wire_order=[0, 1])
    >>> qml.math.expectation_value(operator_matrix, state_vector)
    tensor(-2.23711432e-17+0.j, requires_grad=True)

    .. seealso:: :func:`pennylane.math.fidelity`

    """
    state_vector = cast(state_vector, dtype=c_dtype)
    operator_matrix = cast(operator_matrix, dtype=c_dtype)

    if check_state:
        _check_state_vector(state_vector)

    if check_operator:
        _check_hermitian_operator(operator_matrix)

    if qml.math.shape(operator_matrix)[-1] != qml.math.shape(state_vector)[-1]:
        raise qml.QuantumFunctionError(
            "The operator and the state vector must have the same number of wires."
        )

    # The overlap <psi|A|psi>
    expval = qml.math.einsum(
        "...i,...i->...",
        qml.math.conj(state_vector),
        qml.math.einsum("...ji,...i->...j", operator_matrix, state_vector, optimize="greedy"),
        optimize="greedy",
    )
    return expval


# pylint: disable=too-many-arguments
def vn_entanglement_entropy(
    state, indices0, indices1, base=None, check_state=False, c_dtype="complex128"
):
    r"""Compute the Von Neumann entanglement entropy between two subsystems in a given state.

    .. math::

        S(\rho_A) = -\text{Tr}[\rho_A \log \rho_A] = -\text{Tr}[\rho_B \log \rho_B] = S(\rho_B)

    where :math:`S` is the von Neumann entropy, and :math:`\rho_A = \text{Tr}_B [\rho_{AB}]` and
    :math:`\rho_B = \text{Tr}_A [\rho_{AB}]` are the reduced density matrices for each partition.

    The Von Neumann entanglement entropy is a measure of the degree of quantum entanglement between
    two subsystems constituting a pure bipartite quantum state. The entropy of entanglement is the
    Von Neumann entropy of the reduced density matrix for any of the subsystems. If it is non-zero,
    it indicates the two subsystems are entangled.

    Each state must be given as a density matrix. To find the mutual information given
    a pure state, call :func:`~.math.dm_from_state_vector` first.

    Args:
        state (tensor_like): ``(2**N, 2**N)`` or ``(batch_dim, 2**N, 2**N)`` density matrix.
        indices0 (list[int]): Indices of the qubits in the first subsystem.
        indices1 (list[int]): Indices of the qubits in the second subsystem.
        base (float): Base for the logarithm. If ``None``, the natural logarithm is used.
        check_state (bool): If True, the function will check the state validity (shape and norm).
        c_dtype (str): Complex floating point precision type.

    Returns:
        float: The von Neumann entanglement entropy of the bipartite state.

    **Examples**

    The entanglement entropy between subsystems for a state vector can be returned as follows:

    >>> x = np.array([0, -1, 1, 0]) / np.sqrt(2)
    >>> x = qml.math.dm_from_state_vector(x)
    >>> qml.math.vn_entanglement_entropy(x, indices0=[0], indices1=[1])
    0.6931471805599453

    It is also possible to change the logarithm base:

    >>> qml.math.vn_entanglement_entropy(x, indices0=[0], indices1=[1], base=2)
    1

    Similarly, the quantum state can be provided as a density matrix:

    >>> y = np.array([[1, 1, -1, -1], [1, 1, -1, -1], [-1, -1, 1, 1], [-1, -1, 1, 1]]) * 0.25
    >>> qml.math.vn_entanglement_entropy(y, indices0=[0], indices1=[1])
    0

    """

    # The subsystems cannot overlap
    if len([index for index in indices0 if index in indices1]) > 0:
        raise ValueError("Subsystems for computing the entanglement entropy must not overlap.")

    return _compute_vn_entanglement_entropy(
        state, indices0, indices1, base=base, check_state=check_state, c_dtype=c_dtype
    )


def _compute_vn_entanglement_entropy(
    state, indices0, _, base=None, check_state=False, c_dtype="complex128"
):
    """Computes the Von Neumann entanglement entropy between the subsystems."""

    vn_entropy_1 = vn_entropy(
        state, indices=indices0, base=base, check_state=check_state, c_dtype=c_dtype
    )

    # The Von Neumann entropy of the two subsystems should be the same if the overall state is a
    # pure state. Here we trust that the user only uses this function for pure states, and do not
    # perform any checks so that the code is compatible with jax.jit
    return vn_entropy_1


def sqrt_matrix(density_matrix):
    r"""Compute the square root matrix of a density matrix where :math:`\rho = \sqrt{\rho} \times \sqrt{\rho}`

    Args:
        density_matrix (tensor_like): 2D or 3D (with batching) density matrix of the quantum system.

    Returns:
        (tensor_like): Square root of the density matrix.
    """
    evs, vecs = qml.math.linalg.eigh(density_matrix)
    evs = qml.math.real(evs)
    evs = qml.math.where(evs > 0.0, evs, 0.0)
    if not is_abstract(evs):
        evs = qml.math.cast_like(evs, vecs)

    shape = qml.math.shape(density_matrix)
    if len(shape) > 2:
        # broadcasting case
        i = qml.math.cast_like(qml.math.convert_like(qml.math.eye(shape[-1]), evs), evs)
        sqrt_evs = qml.math.expand_dims(qml.math.sqrt(evs), 1) * i
        return vecs @ sqrt_evs @ qml.math.conj(qml.math.transpose(vecs, (0, 2, 1)))

    return vecs @ qml.math.diag(qml.math.sqrt(evs)) @ qml.math.conj(qml.math.transpose(vecs))


def sqrt_matrix_sparse(sparse_matrix):
    r"""Compute the square root matrix of a positive-definite Hermitian matrix where :math:`\rho = \sqrt{\rho} \times \sqrt{\rho}`

    Args:
        sparse_matrix (sparse): 2D sparse matrix of the quantum system.

    Returns:
       (sparse): Square root of the sparse matrix. Even for data types like `csr_matrix` or `csc_matrix`, the output matrix is not guaranteed to be sparse as well.

    """
    if not issparse(sparse_matrix):
        raise TypeError(
            f"sqrt_matrix_sparse currently only supports scipy.sparse matrices, but received {type(sparse_matrix)}. "
        )
    if sparse_matrix.nnz == 0:
        return sparse_matrix
    # NOTE: the following steps should be re-visited in the future to establish
    # better understanding and control over the heuristics we chose
    # 1. choice of max iteration and tolerance for denman beavers, sc-85713
    # 2. different methods for sparse matrix square root, sc-85710
    return _denman_beavers_iterations(sparse_matrix, max_iter=100, tol=1e-10)


def _inv_newton(M, guess):
    """
    Compute the inverse of a matrix using Newton's method.

    Args:
        M (array-like): The matrix to be inverted.
        guess (array-like): An initial guess for the inverse of the matrix.

    Returns:
        array-like: An improved estimate of the inverse of the matrix.
    """
    return 2 * guess - guess @ M @ guess


def _denman_beavers_iterations(mat, max_iter=100, tol=1e-13):
    """Compute matrix square root using the Denman-Beavers iteration.

    The Denman–Beavers iteration was introduced by E. D. Denman and A. N. Beavers in 1976
    and stems from Newton-type methods originally used to compute the matrix sign function.
    In this adaptation for matrix square roots, two matrices (Y and Z) are refined in each
    step until convergence. This technique is often effective for sparse or structured
    matrices, particularly those that are positive semidefinite or invertible.

    Args:
        mat (sparse): Sparse input matrix
        max_iter (int): Maximum number of iterations
        tol (float): Convergence tolerance (absolute tolerance). Measured using the Frobenius norm of the difference between input mat and the square of the output.

    Returns:
        scipy.sparse.spmatrix: Square root of the input matrix

    Raises:
        LinAlgError: If matrix inversion fails
        ValueError: If NaN values or overflow are encountered during computation
    """
    if mat.shape == (1, 1):
        return sqrt(mat)
    try:
        mat = csc_matrix(mat)
        Y = mat
        Z = sp.sparse.eye(mat.shape[0], format="csc")

        # Keep track of previous iteration for convergence check
        Y_prev = None
        norm_diff = None

        for iter_num in range(max_iter):
            # Compute next iteration
            if iter_num < 2:
                Zinv = spla.inv(Z) if iter_num > 0 else Z
                Yinv = spla.inv(Y)
            else:
                # Take Newton step
                Zinv = _inv_newton(Z, Zinv)
                Yinv = _inv_newton(Y, Yinv)

            Y = 0.5 * (Y + Zinv)
            Z = 0.5 * (Z + Yinv)

            # Check for NaN or infinite values
            if not (np.all(np.isfinite(Y.data)) and np.all(np.isfinite(Z.data))):
                raise ValueError(
                    "Invalid values encountered during computation: nan or inf"
                    f"Input matrix: {mat.toarray()}"
                )

            # Check convergence every 10 iterations
            if iter_num % 10 == 0 and iter_num > 0:
                if Y_prev is not None:
                    # Compute Frobenius norm of difference
                    diff = Y - Y_prev
                    norm_diff = spla.norm(diff)
                    if norm_diff < tol:
                        break
                Y_prev = Y.copy()

        numerical_error = spla.norm((Y @ Y - mat))
        if (norm_diff and norm_diff > tol) or numerical_error > tol:
            raise ValueError(
                f"Convergence threshold not reached after {max_iter} iterations, "
                f"with final norm error {norm_diff} and numerical error {numerical_error}"
            )
        return Y
    except RuntimeError as e:
        raise ValueError(
            "Invalid values encountered during matrix multiplication: "
            f"Input matrix: {mat.toarray()}"
            f"system error: {e}"
        ) from e


def _compute_relative_entropy(rho, sigma, base=None):
    r"""
    Compute the quantum relative entropy of density matrix rho with respect to sigma.

    .. math::
        S(\rho\,\|\,\sigma)=-\text{Tr}(\rho\log\sigma)-S(\rho)=\text{Tr}(\rho\log\rho)-\text{Tr}(\rho\log\sigma)
        =\text{Tr}(\rho(\log\rho-\log\sigma))

    where :math:`S` is the von Neumann entropy.
    """
    if base:
        div_base = np.log(base)
    else:
        div_base = 1

    evs_rho, u_rho = qml.math.linalg.eigh(rho)
    evs_sig, u_sig = qml.math.linalg.eigh(sigma)

    # cast all eigenvalues to real
    evs_rho, evs_sig = np.real(evs_rho), np.real(evs_sig)

    # zero eigenvalues need to be treated very carefully here
    # we use the convention that 0 * log(0) = 0
    evs_sig = qml.math.where(evs_sig == 0, 0.0, evs_sig)
    rho_nonzero_mask = qml.math.where(evs_rho == 0.0, False, True)

    ent = qml.math.entr(qml.math.where(rho_nonzero_mask, evs_rho, 1.0))

    # whether the inputs are batched
    rho_batched = len(qml.math.shape(rho)) > 2
    sig_batched = len(qml.math.shape(sigma)) > 2

    indices_rho = "abc" if rho_batched else "bc"
    indices_sig = "abd" if sig_batched else "bd"
    target = "acd" if rho_batched or sig_batched else "cd"

    # the matrix of inner products between eigenvectors of rho and eigenvectors
    # of sigma; this is a doubly stochastic matrix
    rel = qml.math.einsum(
        f"{indices_rho},{indices_sig}->{target}",
        np.conj(u_rho),
        u_sig,
        optimize="greedy",
    )
    rel = np.abs(rel) ** 2

    if sig_batched:
        evs_sig = qml.math.expand_dims(evs_sig, 1)

    rel = qml.math.sum(qml.math.where(rel == 0.0, 0.0, np.log(evs_sig) * rel), -1)
    rel = -qml.math.sum(qml.math.where(rho_nonzero_mask, evs_rho * rel, 0.0), -1)

    return (rel - ent) / div_base


def relative_entropy(state0, state1, base=None, check_state=False, c_dtype="complex128"):
    r"""
    Compute the quantum relative entropy of one state with respect to another.

    .. math::
        S(\rho\,\|\,\sigma)=-\text{Tr}(\rho\log\sigma)-S(\rho)=\text{Tr}(\rho\log\rho)-\text{Tr}(\rho\log\sigma)
        =\text{Tr}(\rho(\log\rho-\log\sigma))

    Roughly speaking, quantum relative entropy is a measure of distinguishability between two
    quantum states. It is the quantum mechanical analog of relative entropy.

    Each state must be given as a density matrix. To find the relative entropy given
    a pure state, call :func:`~.math.dm_from_state_vector` first.

    Args:
        state0 (tensor_like): ``(2**N, 2**N)`` or ``(batch_dim, 2**N, 2**N)`` density matrix.
        state1 (tensor_like): ``(2**N, 2**N)`` or ``(batch_dim, 2**N, 2**N)`` density matrix.
        base (float): Base for the logarithm. If None, the natural logarithm is used.
        check_state (bool): If True, the function will check the state validity (shape and norm).
        c_dtype (str): Complex floating point precision type.

    Returns:
        float: Quantum relative entropy of state0 with respect to state1

    **Examples**

    The relative entropy between two equal states is always zero:

    >>> x = np.array([1, 0])
    >>> x = qml.math.dm_from_state_vector(x)
    >>> qml.math.relative_entropy(x, x)
    0.0

    and the relative entropy between two non-equal pure states is always infinity:

    >>> y = np.array([1, 1]) / np.sqrt(2)
    >>> y = qml.math.dm_from_state_vector(y)
    >>> qml.math.relative_entropy(x, y)
    inf

    The quantum states can be provided as density matrices, allowing for computation
    of relative entropy between mixed states:

    >>> rho = np.array([[0.3, 0], [0, 0.7]])
    >>> sigma = np.array([[0.5, 0], [0, 0.5]])
    >>> qml.math.relative_entropy(rho, sigma)
    0.08228288

    It is also possible to change the log base:

    >>> qml.math.relative_entropy(rho, sigma, base=2)
    0.1187091
    """
    # Cast as a c_dtype array
    state0 = cast(state0, dtype=c_dtype)

    # Cannot be cast_like if jit
    if not is_abstract(state0):
        state1 = cast_like(state1, state0)

    if check_state:
        # pylint: disable=expression-not-assigned
        _check_density_matrix(state0)
        _check_density_matrix(state1)

    # Compare the number of wires on both subsystems
    if qml.math.shape(state0)[-1] != qml.math.shape(state1)[-1]:
        raise qml.QuantumFunctionError("The two states must have the same number of wires.")

    return _compute_relative_entropy(state0, state1, base=base)


def _check_density_matrix(density_matrix):
    """Check the shape, the trace and the positive semi-definitiveness of a matrix."""
    dim = density_matrix.shape[-1]
    if (
        len(density_matrix.shape) not in (2, 3)
        or density_matrix.shape[-2] != dim
        or not np.log2(dim).is_integer()
    ):
        raise ValueError("Density matrix must be of shape (2**N, 2**N) or (batch_dim, 2**N, 2**N).")

    if len(density_matrix.shape) == 2:
        density_matrix = qml.math.stack([density_matrix])

    if not is_abstract(density_matrix):
        for dm in density_matrix:
            # Check trace
            trace = np.trace(dm)
            if not allclose(trace, 1.0, atol=1e-10):
                raise ValueError("The trace of the density matrix should be one.")

            # Check if the matrix is Hermitian
            conj_trans = np.transpose(np.conj(dm))
            if not allclose(dm, conj_trans):
                raise ValueError("The matrix is not Hermitian.")

            # Check if positive semi-definite
            evs, _ = qml.math.linalg.eigh(dm)
            evs = np.real(evs)
            evs_non_negative = [ev for ev in evs if ev >= -1e-7]
            if len(evs) != len(evs_non_negative):
                raise ValueError("The matrix is not positive semi-definite.")


def _check_state_vector(state_vector):
    """Check the shape and the norm of a state vector."""
    dim = state_vector.shape[-1]
    if len(np.shape(state_vector)) not in (1, 2) or not np.log2(dim).is_integer():
        raise ValueError("State vector must be of shape (2**wires,) or (batch_dim, 2**wires)")

    if len(state_vector.shape) == 1:
        state_vector = qml.math.stack([state_vector])

    # Check norm
    if not is_abstract(state_vector):
        for sv in state_vector:
            norm = np.linalg.norm(sv, ord=2)
            if not allclose(norm, 1.0, atol=1e-10):
                raise ValueError("Sum of amplitudes-squared does not equal one.")


def max_entropy(state, indices, base=None, check_state=False, c_dtype="complex128"):
    r"""Compute the maximum entropy of a density matrix on a given subsystem. It supports all
    interfaces (NumPy, Autograd, Torch, TensorFlow and Jax).

    .. math::
        S_{\text{max}}( \rho ) = \log( \text{rank} ( \rho ))

    Args:
        state (tensor_like): Density matrix of shape ``(2**N, 2**N)`` or ``(batch_dim, 2**N, 2**N)``.
        indices (list(int)): List of indices in the considered subsystem.
        base (float): Base for the logarithm. If None, the natural logarithm is used.
        check_state (bool): If True, the function will check the state validity (shape and norm).
        c_dtype (str): Complex floating point precision type.

    Returns:
        float: The maximum entropy of the considered subsystem.

    **Example**

    The maximum entropy of a subsystem for any state vector can be obtained by first calling
    :func:`~.math.dm_from_state_vector` on the input. Here is an example for the
    maximally entangled state, where the subsystem entropy is maximal (default base for log is exponential).

    >>> x = [1, 0, 0, 1] / np.sqrt(2)
    >>> x = dm_from_state_vector(x)
    >>> max_entropy(x, indices=[0])
    0.6931472

    The logarithm base can be changed. For example:

    >>> max_entropy(x, indices=[0], base=2)
    1.0

    The maximum entropy can be obtained by providing a quantum state as a density matrix. For example:

    >>> y = [[1/2, 0, 0, 1/2], [0, 0, 0, 0], [0, 0, 0, 0], [1/2, 0, 0, 1/2]]
    >>> max_entropy(y, indices=[0])
    0.6931472

    The maximum entropy is always greater or equal to the Von Neumann entropy. In this maximally
    entangled example, they are equal:

    >>> vn_entropy(x, indices=[0])
    0.6931472

    However, in general, the Von Neumann entropy is lower:

    >>> x = [np.cos(np.pi/8), 0, 0, -1j*np.sin(np.pi/8)]
    >>> x = dm_from_state_vector(x)
    >>> vn_entropy(x, indices=[1])
    0.4164955
    >>> max_entropy(x, indices=[1])
    0.6931472

    """
    density_matrix = reduce_dm(state, indices, check_state, c_dtype)
    maximum_entropy = _compute_max_entropy(density_matrix, base)
    return maximum_entropy


def _compute_max_entropy(density_matrix, base):
    """Compute the maximum entropy from a density matrix

    Args:
        density_matrix (tensor_like): ``(2**N, 2**N)`` or ``(batch_dim, 2**N, 2**N)`` tensor for an integer `N`.
        base (float, int): Base for the logarithm. If None, the natural logarithm is used.

    Returns:
        float: Maximum entropy of the density matrix.

    **Example**

    >>> x = [[1/2, 0], [0, 1/2]]
    >>> _compute_max_entropy(x)
    0.6931472

    >>> x = [[1/2, 0], [0, 1/2]]
    >>> _compute_max_entropy(x, base=2)
    1.0

    """
    # Change basis if necessary
    if base:
        div_base = np.log(base)
    else:
        div_base = 1

    evs = qml.math.eigvalsh(density_matrix)
    evs = qml.math.real(evs)
    rank = qml.math.sum(evs / qml.math.where(evs > 1e-8, evs, 1.0), -1)
    maximum_entropy = qml.math.log(rank) / div_base

    return maximum_entropy


def min_entropy(state, indices, base=None, check_state=False, c_dtype="complex128"):
    r"""Compute the minimum entropy from a density matrix.

    .. math::
        S_{\text{min}}( \rho ) = -\log( \max_{i} ( p_{i} ))

    Args:
        state (tensor_like): Density matrix of shape ``(2**N, 2**N)`` or ``(batch_dim, 2**N, 2**N)``.
        indices (list(int)): List of indices in the considered subsystem.
        base (float): Base for the logarithm. If None, the natural logarithm is used.
        check_state (bool): If True, the function will check the state validity (shape and norm).
        c_dtype (str): Complex floating point precision type.

    Returns:
        float: The minimum entropy of the considered subsystem.

    **Example**

    The minimum entropy of a subsystem for any state vector can be obtained by first calling
    :func:`~.math.dm_from_state_vector` on the input. Here is an example for the
    maximally entangled state, where the subsystem entropy is maximal (default base for log is exponential).

    >>> x = [1, 0, 0, 1] / np.sqrt(2)
    >>> x = dm_from_state_vector(x)
    >>> min_entropy(x, indices=[0])
    0.6931472

    The logarithm base can be changed. For example:

    >>> min_entropy(x, indices=[0], base=2)
    1.0

    The minimum entropy can be obtained by providing a quantum state as a density matrix. For example:

    >>> y = [[1/2, 0, 0, 1/2], [0, 0, 0, 0], [0, 0, 0, 0], [1/2, 0, 0, 1/2]]
    >>> min_entropy(y, indices=[0])
    0.6931472

    The Von Neumann entropy is always greater than the minimum entropy.

    >>> x = [np.cos(np.pi/8), 0, 0, -1j*np.sin(np.pi/8)]
    >>> x = dm_from_state_vector(x)
    >>> vn_entropy(x, indices=[1])
    0.4164955
    >>> min_entropy(x, indices=[1])
    0.1583472

    """
    density_matrix = reduce_dm(state, indices, check_state, c_dtype)
    minimum_entropy = _compute_min_entropy(density_matrix, base)

    return minimum_entropy


def _compute_min_entropy(density_matrix, base):
    r"""Compute the minimum entropy from a density matrix

    Args:
        density_matrix (tensor_like): ``(2**N, 2**N)`` tensor density matrix for an integer `N`.
        base (float, int): Base for the logarithm. If None, the natural logarithm is used.

    Returns:
        float: Minimum entropy of the density matrix.

    **Example**

    >>> x = [[1/2, 0], [0, 1/2]]
    >>> _compute_min_entropy(x)
    0.6931472

    >>> x = [[1/2, 0], [0, 1/2]]
    >>> _compute_min_entropy(x, base=2)
    1.0
    """
    # Change basis if necessary
    div_base = np.log(base) if base else 1

    evs, _ = qml.math.linalg.eigh(density_matrix)
    evs = qml.math.real(evs)
    minimum_entropy = -qml.math.log(qml.math.max(evs)) / div_base

    return minimum_entropy


def trace_distance(state0, state1, check_state=False, c_dtype="complex128"):
    r"""
    Compute the trace distance between two quantum states.

    .. math::
        T(\rho, \sigma)=\frac12\|\rho-\sigma\|_1
        =\frac12\text{Tr}\left(\sqrt{(\rho-\sigma)^{\dagger}(\rho-\sigma)}\right)

    where :math:`\|\cdot\|_1` is the Schatten :math:`1`-norm.

    The trace distance measures how close two quantum states are. In particular, it upper-bounds
    the probability of distinguishing two quantum states.

    Args:
        state0 (tensor_like): ``(2**N, 2**N)`` or ``(batch_dim, 2**N, 2**N)`` density matrix.
        state1 (tensor_like): ``(2**N, 2**N)`` or ``(batch_dim, 2**N, 2**N)`` density matrix.
        check_state (bool): If True, the function will check the states' validity (shape and norm).
        c_dtype (str): Complex floating point precision type.

    Returns:
        float: Trace distance between state0 and state1

    **Examples**

    The trace distance between two equal states is always zero:

    >>> x = np.array([[1, 0], [0, 0]])
    >>> qml.math.trace_distance(x, x)
    0.0

    It is possible to use state vectors by first transforming them into density matrices via the
    :func:`~reduce_statevector` function:

    >>> y = qml.math.reduce_statevector(np.array([0.2, np.sqrt(0.96)]), [0])
    >>> qml.math.trace_distance(x, y)
    0.9797958971132713

    The quantum states can also be provided as batches of density matrices:

    >>> batch0 = np.array([np.eye(2) / 2, np.ones((2, 2)) / 2, np.array([[1, 0],[0, 0]])])
    >>> batch1 = np.array([np.ones((2, 2)) / 2, np.ones((2, 2)) / 2, np.array([[1, 0],[0, 0]])])
    >>> qml.math.trace_distance(batch0, batch1)
    array([0.5, 0. , 0. ])

    If only one of the two states represent a single element, then the trace distances are taken
    with respect to that element:

    >>> rho = np.ones((2, 2)) / 2
    >>> qml.math.trace_distance(rho, batch0)
    array([0.5       , 0.        , 0.70710678])
    """
    # Cast as a c_dtype array
    state0 = cast(state0, dtype=c_dtype)

    # Cannot be cast_like if jit
    if not is_abstract(state0):
        state1 = cast_like(state1, state0)

    if check_state:
        _check_density_matrix(state0)
        _check_density_matrix(state1)

    if state0.shape[-1] != state1.shape[-1]:
        raise qml.QuantumFunctionError("The two states must have the same number of wires.")

    if len(state0.shape) == len(state1.shape) == 3 and state0.shape[0] != state1.shape[0]:
        raise ValueError(
            "The two states must be batches of the same size, or one of them must contain a single "
            "element."
        )

    eigvals = qml.math.abs(qml.math.eigvalsh(state0 - state1))

    return qml.math.sum(eigvals, axis=-1) / 2
