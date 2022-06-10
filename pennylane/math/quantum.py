# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
import itertools
import functools

from string import ascii_letters as ABC
from autoray import numpy as np
from numpy import float64

from . import single_dispatch  # pylint:disable=unused-import
from .multi_dispatch import diag, dot, scatter_element_add, einsum, get_interface
from .utils import is_abstract, allclose, cast, convert_like

ABC_ARRAY = np.array(list(ABC))


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

    >>> obs_list = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliY(2)]
    >>> ansatz = qml.templates.StronglyEntanglingLayers

    We can construct a QNode to output the probability distribution in the shared eigenbasis of the
    observables:

    .. code-block:: python

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
    array([[0.98707611, 0.03665537],
         [0.03665537, 0.99998377]])

    Autodifferentiation is fully supported using all interfaces.
    Here we use autograd:

    >>> cost_fn = lambda weights: qml.math.cov_matrix(circuit(weights), obs_list)[0, 1]
    >>> qml.grad(cost_fn)(weights)[0]
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


def _density_matrix_from_matrix(density_matrix, indices, check_state=False):
    """Compute the density matrix from a state represented with a density matrix.

    Args:
        density_matrix (tensor_like): 2D density matrix tensor. This tensor should be of size ``(2**N, 2**N)`` for some
            integer number of wires``N``.
        indices (list(int)): List of indices in the considered subsystem.
        check_state (bool): If True, the function will check the state validity (shape and norm).

    Returns:
        tensor_like: Density matrix of size ``(2**len(wires), 2**len(wires))``

    **Example**

    >>> x = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    >>> _density_matrix_from_matrix(x, indices=[0])
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]]

    >>> y = [[0.5, 0, 0.5, 0], [0, 0, 0, 0], [0.5, 0, 0.5, 0], [0, 0, 0, 0]]
    >>> _density_matrix_from_matrix(y, indices=[0])
    [[0.5+0.j 0.5+0.j]
     [0.5+0.j 0.5+0.j]]

    >>> _density_matrix_from_matrix(y, indices=[1])
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]]

    >>> z = tf.Variable([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=tf.complex128)
    >>> _density_matrix_from_matrix(x, indices=[1])
    tf.Tensor(
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]], shape=(2, 2), dtype=complex128)


    """
    shape = density_matrix.shape[0]
    num_indices = int(np.log2(shape))

    if check_state:
        # Check format
        if (
            len(density_matrix.shape) != 2
            or density_matrix.shape[0] != density_matrix.shape[1]
            or not np.log2(shape).is_integer()
        ):
            raise ValueError("Density matrix must be of shape (2**N, 2**N).")
        # Check trace
        trace = np.trace(density_matrix)
        if not is_abstract(trace):
            if not allclose(trace, 1.0, atol=1e-10):
                raise ValueError("The trace of the density matrix should be one.")
            # Check if the matrix is hermitian
            conj_trans = np.transpose(np.conj(density_matrix))
            if not allclose(density_matrix, conj_trans):
                raise ValueError("The matrix is not hermitian.")
            # Check if positive semi definite
            evs = np.linalg.eigvalsh(density_matrix)
            evs_non_negative = [ev for ev in evs if ev >= 0.0]
            if len(evs) != len(evs_non_negative):
                raise ValueError("The matrix is not positive semi-definite.")

    consecutive_indices = list(range(0, num_indices))

    # Return the full density matrix if all the wires are given
    if tuple(indices) == tuple(consecutive_indices):
        return density_matrix

    traced_wires = [x for x in consecutive_indices if x not in indices]
    density_matrix = _partial_trace(density_matrix, traced_wires)
    return density_matrix


def _partial_trace(density_matrix, indices):
    """Compute the reduced density matrix by tracing out the provided indices.

    Args:
        density_matrix (tensor_like): 2D density matrix tensor. This tensor should be of size ``(2**N, 2**N)`` for some
            integer number of wires ``N``.
        indices (list(int)): List of indices to be traced.

    Returns:
        tensor_like: (reduced) Density matrix of size ``(2**len(wires), 2**len(wires))``

    **Example**

    >>> x = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    >>> _partial_trace(x, indices=[0])
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]]


    >>> x = tf.Variable([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=tf.complex128)
    >>> _partial_trace(x, indices=[1])
    tf.Tensor(
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]], shape=(2, 2), dtype=complex128)
    """
    # Does not support same indices sum in backprop
    if get_interface(density_matrix) == "autograd":
        density_matrix = _partial_trace_autograd(density_matrix, indices)
        return density_matrix

    # Dimension and reshape
    shape = density_matrix.shape[0]
    num_indices = int(np.log2(shape))
    rho_dim = 2 * num_indices

    density_matrix = np.reshape(density_matrix, [2] * 2 * num_indices)
    indices = np.sort(indices)

    # For loop over wires
    for i, target_index in enumerate(indices):
        target_index = target_index - i
        state_indices = ABC[: rho_dim - 2 * i]
        state_indices = list(state_indices)

        target_letter = state_indices[target_index]
        state_indices[target_index + num_indices - i] = target_letter
        state_indices = "".join(state_indices)

        einsum_indices = f"{state_indices}"
        density_matrix = einsum(einsum_indices, density_matrix)

    number_wires_sub = num_indices - len(indices)
    reduced_density_matrix = np.reshape(
        density_matrix, (2**number_wires_sub, 2**number_wires_sub)
    )
    return reduced_density_matrix


def _partial_trace_autograd(density_matrix, indices):
    """Compute the reduced density matrix for autograd interface by tracing out the provided indices with the use
    of projectors as same subscripts indices are not supported in autograd backprop.
    """
    # Dimension and reshape
    shape = density_matrix.shape[0]
    num_indices = int(np.log2(shape))
    rho_dim = 2 * num_indices
    density_matrix = np.reshape(density_matrix, [2] * 2 * num_indices)

    kraus = cast(np.eye(2), density_matrix.dtype)

    kraus = np.reshape(kraus, (2, 1, 2))
    kraus_dagger = np.asarray([np.conj(np.transpose(k)) for k in kraus])

    kraus = convert_like(kraus, density_matrix)
    kraus_dagger = convert_like(kraus_dagger, density_matrix)
    # For loop over wires
    for target_wire in indices:
        # Tensor indices of density matrix
        state_indices = ABC[:rho_dim]
        # row indices of the quantum state affected by this operation
        row_wires_list = [target_wire]
        row_indices = "".join(ABC_ARRAY[row_wires_list].tolist())
        # column indices are shifted by the number of wires
        col_wires_list = [w + num_indices for w in row_wires_list]
        col_indices = "".join(ABC_ARRAY[col_wires_list].tolist())
        # indices in einsum must be replaced with new ones
        num_partial_trace_wires = 1
        new_row_indices = ABC[rho_dim : rho_dim + num_partial_trace_wires]
        new_col_indices = ABC[
            rho_dim + num_partial_trace_wires : rho_dim + 2 * num_partial_trace_wires
        ]
        # index for summation over Kraus operators
        kraus_index = ABC[
            rho_dim + 2 * num_partial_trace_wires : rho_dim + 2 * num_partial_trace_wires + 1
        ]
        # new state indices replace row and column indices with new ones
        new_state_indices = functools.reduce(
            lambda old_string, idx_pair: old_string.replace(idx_pair[0], idx_pair[1]),
            zip(col_indices + row_indices, new_col_indices + new_row_indices),
            state_indices,
        )
        # index mapping for einsum, e.g., 'iga,abcdef,idh->gbchef'
        einsum_indices = (
            f"{kraus_index}{new_row_indices}{row_indices}, {state_indices},"
            f"{kraus_index}{col_indices}{new_col_indices}->{new_state_indices}"
        )
        density_matrix = einsum(einsum_indices, kraus, density_matrix, kraus_dagger)
    number_wires_sub = num_indices - len(indices)
    reduced_density_matrix = np.reshape(
        density_matrix, (2**number_wires_sub, 2**number_wires_sub)
    )
    return reduced_density_matrix


def _density_matrix_from_state_vector(state, indices, check_state=False):
    """Compute the density matrix from a state vector.

    Args:
        state (tensor_like): 1D tensor state vector. This tensor should of size ``(2**N,)`` for some integer value ``N``.
        indices (list(int)): List of indices in the considered subsystem.
        check_state (bool): If True, the function will check the state validity (shape and norm).

    Returns:
        tensor_like: Density matrix of size ``(2**len(indices), 2**len(indices))``

    **Example**

    >>> x = np.array([1, 0, 0, 0])
    >>> _density_matrix_from_state_vector(x, indices=[0])
    [[1.+0.j 0.+0.j]
    [0.+0.j 0.+0.j]]

    >>> y = [1, 0, 1, 0] / np.sqrt(2)
    >>> _density_matrix_from_state_vector(y, indices=[0])
    [[0.5+0.j 0.5+0.j]
     [0.5+0.j 0.5+0.j]]

    >>> _density_matrix_from_state_vector(y, indices=[1])
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]]

    >>> z = tf.Variable([1, 0, 0, 0], dtype=tf.complex128)
    >>> _density_matrix_from_state_vector(z, indices=[1])
    tf.Tensor(
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]], shape=(2, 2), dtype=complex128)

    """
    len_state = np.shape(state)[0]

    # Check the format and norm of the state vector
    if check_state:
        # Check format
        if len(np.shape(state)) != 1 or not np.log2(len_state).is_integer():
            raise ValueError("State vector must be of length 2**wires.")
        # Check norm
        norm = np.linalg.norm(state, ord=2)
        if not is_abstract(norm):
            if not allclose(norm, 1.0, atol=1e-10):
                raise ValueError("Sum of amplitudes-squared does not equal one.")

    # Get dimension of the quantum system and reshape
    num_indices = int(np.log2(len_state))
    consecutive_wires = list(range(num_indices))
    state = np.reshape(state, [2] * num_indices)

    # Get the system to be traced
    traced_system = [x for x in consecutive_wires if x not in indices]

    # Return the reduced density matrix by using numpy tensor product
    density_matrix = np.tensordot(state, np.conj(state), axes=(traced_system, traced_system))
    density_matrix = np.reshape(density_matrix, (2 ** len(indices), 2 ** len(indices)))

    return density_matrix


def reduced_dm(state, indices, check_state=False, c_dtype="complex128"):
    """Compute the reduced density matrix from a state vector or a density matrix.

    Args:
        state (tensor_like): ``(2**N)`` state vector or ``(2**N, 2**N)`` density matrix.
        indices (Sequence(int)): List of indices in the considered subsystem.
        check_state (bool): If True, the function will check the state validity (shape and norm).
        c_dtype (str): Complex floating point precision type.

    Returns:
        tensor_like: Reduced density matrix of size ``(2**len(indices), 2**len(indices))``


    **Example**

    >>> x = [1, 0, 1, 0] / np.sqrt(2)
    >>> reduced_dm(x, indices=[0])
    [[0.5+0.j 0.5+0.j]
     [0.5+0.j 0.5+0.j]]

    >>> reduced_dm(x, indices=[1])
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]]

    >>> y = tf.Variable([1, 0, 0, 0], dtype=tf.complex128)
    >>> reduced_dm(z, indices=[1])
    tf.Tensor(
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]], shape=(2, 2), dtype=complex128)

    >>> z = [[0.5, 0, 0.0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]]
    >>> reduced_dm(z, indices=[0])
    [[0.5+0.j 0.0+0.j]
     [0.0+0.j 0.5+0.j]]

    >>> reduced_dm(z, indices=[1])
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]]

    >>> y_mat_tf = tf.Variable([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=tf.complex128)
    >>> reduced_dm(y_mat_tf, indices=[1])
    tf.Tensor(
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]], shape=(2, 2), dtype=complex128)
    """
    # Cast as a c_dtype array
    state = cast(state, dtype=c_dtype)
    len_state = state.shape[0]
    # State vector
    if state.shape == (len_state,):
        density_matrix = _density_matrix_from_state_vector(state, indices, check_state)
        return density_matrix

    density_matrix = _density_matrix_from_matrix(state, indices, check_state)

    return density_matrix
