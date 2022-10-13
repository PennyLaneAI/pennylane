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
# pylint: disable=import-outside-toplevel
import itertools
import functools

from string import ascii_letters as ABC
from autoray import numpy as np
from numpy import float64

import pennylane as qml

from . import single_dispatch  # pylint:disable=unused-import
from .multi_dispatch import diag, dot, scatter_element_add, einsum, get_interface
from .utils import is_abstract, allclose, cast, convert_like, cast_like

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
        _check_density_matrix(density_matrix)

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
    # Autograd does not support same indices sum in backprop
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
        _check_state_vector(state)

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
    """Compute the reduced density matrix from a state vector or a density matrix. It supports all interfaces (Numpy,
    Autograd, Torch, Tensorflow and Jax).

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
    >>> reduced_dm(y, indices=[1])
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

    .. seealso:: :func:`pennylane.qinfo.transforms.reduced_dm` and :func:`pennylane.density_matrix`
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


def vn_entropy(state, indices, base=None, check_state=False, c_dtype="complex128"):
    r"""Compute the Von Neumann entropy from a state vector or density matrix on a given subsystem. It supports all
    interfaces (Numpy, Autograd, Torch, Tensorflow and Jax).

    .. math::
        S( \rho ) = -\text{Tr}( \rho \log ( \rho ))

    Args:
        state (tensor_like): ``(2**N)`` state vector or ``(2**N, 2**N)`` density matrix.
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
    >>> vn_entropy(x, indices=[0])
    0.6931472

    The logarithm base can be switched to 2 for example.

    >>> vn_entropy(x, indices=[0], base=2)
    1.0

    The entropy can be obtained by providing a quantum state as a density matrix, for example:

    >>> y = [[1/2, 0, 0, 1/2], [0, 0, 0, 0], [0, 0, 0, 0], [1/2, 0, 0, 1/2]]
    >>> vn_entropy(x, indices=[0])
    0.6931472

    .. seealso:: :func:`pennylane.qinfo.transforms.vn_entropy` and :func:`pennylane.vn_entropy`
    """
    density_matrix = reduced_dm(state, indices, check_state, c_dtype)
    entropy = _compute_vn_entropy(density_matrix, base)

    return entropy


def _compute_vn_entropy(density_matrix, base=None):
    """Compute the Von Neumann entropy from a density matrix

    Args:
        density_matrix (tensor_like): ``(2**N, 2**N)`` tensor density matrix for an integer `N`.
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
def mutual_info(state, indices0, indices1, base=None, check_state=False, c_dtype="complex128"):
    r"""Compute the mutual information between two subsystems given a state:

    .. math::

        I(A, B) = S(\rho^A) + S(\rho^B) - S(\rho^{AB})

    where :math:`S` is the von Neumann entropy.

    The mutual information is a measure of correlation between two subsystems.
    More specifically, it quantifies the amount of information obtained about
    one system by measuring the other system. It supports all interfaces
    (Numpy, Autograd, Torch, Tensorflow and Jax).

    Each state can be given as a state vector in the computational basis, or
    as a density matrix.

    Args:
        state (tensor_like): ``(2**N)`` state vector or ``(2**N, 2**N)`` density matrix.
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
    >>> qml.math.mutual_info(x, indices0=[0], indices1=[1])
    1.3862943611198906

    It is also possible to change the log basis.

    >>> qml.math.mutual_info(x, indices0=[0], indices1=[1], base=2)
    2.0

    Similarly the quantum state can be provided as a density matrix:

    >>> y = np.array([[1/2, 1/2, 0, 1/2], [1/2, 0, 0, 0], [0, 0, 0, 0], [1/2, 0, 0, 1/2]])
    >>> qml.math.mutual_info(y, indices0=[0], indices1=[1])
    0.4682351577408206

    .. seealso:: :func:`~.math.vn_entropy`, :func:`pennylane.qinfo.transforms.mutual_info` and :func:`pennylane.mutual_info`
    """

    # the subsystems cannot overlap
    if len([index for index in indices0 if index in indices1]) > 0:
        raise ValueError("Subsystems for computing mutual information must not overlap.")

    # Cast to a complex array
    state = cast(state, dtype=c_dtype)

    state_shape = state.shape
    if len(state_shape) > 0:
        len_state = state_shape[0]
        if state_shape in [(len_state,), (len_state, len_state)]:
            return _compute_mutual_info(
                state, indices0, indices1, base=base, check_state=check_state, c_dtype=c_dtype
            )

    raise ValueError("The state is not a state vector or a density matrix.")


# pylint: disable=too-many-arguments
def _compute_mutual_info(
    state, indices0, indices1, base=None, check_state=False, c_dtype="complex128"
):
    """Compute the mutual information between the subsystems."""
    all_indices = sorted([*indices0, *indices1])
    vn_entropy_1 = vn_entropy(
        state, indices=indices0, base=base, check_state=check_state, c_dtype=c_dtype
    )
    vn_entropy_2 = vn_entropy(
        state, indices=indices1, base=base, check_state=check_state, c_dtype=c_dtype
    )
    vn_entropy_12 = vn_entropy(
        state, indices=all_indices, base=base, check_state=check_state, c_dtype=c_dtype
    )

    return vn_entropy_1 + vn_entropy_2 - vn_entropy_12


def fidelity(state0, state1, check_state=False, c_dtype="complex128"):
    r"""Compute the fidelity for two states (a state can be a state vector or a density matrix) acting on quantum
    systems with the same size.

    The fidelity for two mixed states given by density matrices :math:`\rho` and :math:`\sigma`
    is defined as

    .. math::
        F( \rho , \sigma ) = \text{Tr}( \sqrt{\sqrt{\rho} \sigma \sqrt{\rho}})^2

    If one of the states is pure, say :math:`\rho=\ket{\psi}\bra{\psi}`, then the expression
    for fidelity simplifies to

    .. math::
        F( \ket{\psi} , \sigma ) = \bra{\psi} \sigma \ket{\psi}

    Finally, if both states are pure, :math:`\sigma=\ket{\phi}\bra{\phi}`, then the
    fidelity is simply

    .. math::
        F( \ket{\psi} , \ket{\phi}) = \left|\braket{\psi, \phi}\right|^2

    .. note::
        It supports all interfaces (Numpy, Autograd, Torch, Tensorflow and Jax). The second state is coerced
        to the type and dtype of the first state. The fidelity is returned in the type of the interface of the
        first state.

    Args:
        state0 (tensor_like): 1D state vector or 2D density matrix
        state1 (tensor_like): 1D state vector or 2D density matrix
        check_state (bool): If True, the function will check the validity of both states; it checks (shape, norm) for
            state vectors or (shape, trace, positive-definitiveness) for density matrices.
        c_dtype (str): Complex floating point precision type.

    Returns:
        float: Fidelity between the two quantum states.

    **Example**

    Two state vectors can be used as arguments and the fidelity (overlap) is returned, e.g.:

    >>> state0 = [0.98753537-0.14925137j, 0.00746879-0.04941796j]
    >>> state1 = [0.99500417+0.j, 0.09983342+0.j]
    >>> qml.math.fidelity(state0, state1)
    0.9905158135644924

    Alternatively one can give a state vector and a density matrix as arguments, e.g.:

    >>> state0 = [0, 1]
    >>> state1 = [[0, 0], [0, 1]]
    >>> qml.math.fidelity(state0, state1)
    1.0

    It also works with two density matrices, e.g.:

    >>> state0 = [[1, 0], [0, 0]]
    >>> state1 = [[0, 0], [0, 1]]
    >>> qml.math.fidelity(state0, state1)
    0.0

    .. seealso:: :func:`pennylane.qinfo.transforms.fidelity`

    """
    # Cast as a c_dtype array
    state0 = cast(state0, dtype=c_dtype)
    len_state0 = state0.shape[0]

    # Cannot be cast_like if jit
    if not is_abstract(state0):
        state1 = cast_like(state1, state0)

    len_state1 = state1.shape[0]

    if check_state:
        if state0.shape == (len_state0,):
            _check_state_vector(state0)
        else:
            _check_density_matrix(state0)

        if state1.shape == (len_state1,):
            _check_state_vector(state1)
        else:
            _check_density_matrix(state1)

    # Get dimension of the quantum system and reshape
    num_indices0 = int(np.log2(len_state0))
    num_indices1 = int(np.log2(len_state1))

    if num_indices0 != num_indices1:
        raise qml.QuantumFunctionError("The two states must have the same number of wires.")

    # Two pure states, squared overlap
    if state1.shape == (len_state1,) and state0.shape == (len_state0,):
        overlap = np.tensordot(state0, np.transpose(np.conj(state1)), axes=1)
        overlap = np.abs(overlap) ** 2
        return overlap
    # First state mixed, second state pure
    if state1.shape == (len_state1,) and state0.shape != (len_state0,):
        overlap = np.tensordot(state0, np.transpose(np.conj(state1)), axes=1)
        overlap = np.tensordot(state1, overlap, axes=1)
        overlap = np.real(overlap)
        return overlap
    # First state pure, second state mixed
    if state0.shape == (len_state0,) and state1.shape != (len_state1,):
        overlap = np.tensordot(state1, np.transpose(np.conj(state0)), axes=1)
        overlap = np.tensordot(state0, overlap, axes=1)
        overlap = np.real(overlap)
        return overlap
    # Two mixed states
    fid = _compute_fidelity(state0, state1)
    return fid


def sqrt_matrix(density_matrix):
    r"""Compute the square root matrix of a density matrix where :math:`\rho = \sqrt{\rho} \times \sqrt{\rho}`
    Args:
        density_matrix (tensor_like): 2D density matrix of the quantum system.
    Returns:
        (tensor_like): Square root of the density matrix.
    """
    evs, vecs = qml.math.linalg.eigh(density_matrix)
    evs = np.real(evs)
    evs = qml.math.where(evs > 0.0, evs, 0.0)
    if not is_abstract(evs):
        evs = qml.math.cast_like(evs, vecs)
    return vecs @ qml.math.diag(np.sqrt(evs)) @ np.conj(np.transpose(vecs))


def _compute_fidelity(density_matrix0, density_matrix1):
    r"""Compute the fidelity for two density matrices with the same number of wires.

    .. math::
            F( \rho , \sigma ) = -\text{Tr}( \sqrt{\sqrt{\rho} \sigma \sqrt{\rho}})^2
    """
    # Implementation in single dispatches (sqrt(rho))
    sqrt_mat = qml.math.sqrt_matrix(density_matrix0)

    # sqrt(rho) * sigma * sqrt(rho)
    sqrt_mat_sqrt = sqrt_mat @ density_matrix1 @ sqrt_mat

    # extract eigenvalues
    evs = qml.math.eigvalsh(sqrt_mat_sqrt)
    evs = np.real(evs)
    evs = qml.math.where(evs > 0.0, evs, 0.0)

    trace = (qml.math.sum(qml.math.sqrt(evs))) ** 2

    return trace


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

    # the matrix of inner products between eigenvectors of rho and eigenvectors
    # of sigma; this is a doubly stochastic matrix
    rel = np.abs(qml.math.dot(np.transpose(np.conj(u_rho)), u_sig)) ** 2

    rel = qml.math.sum(qml.math.where(rel == 0.0, 0.0, np.log(evs_sig) * rel), axis=1)
    rel = -qml.math.sum(qml.math.where(rho_nonzero_mask, evs_rho * rel, 0.0))

    return (rel - ent) / div_base


def relative_entropy(state0, state1, base=None, check_state=False, c_dtype="complex128"):
    r"""
    Compute the quantum relative entropy of one state with respect to another.

    .. math::
        S(\rho\,\|\,\sigma)=-\text{Tr}(\rho\log\sigma)-S(\rho)=\text{Tr}(\rho\log\rho)-\text{Tr}(\rho\log\sigma)
        =\text{Tr}(\rho(\log\rho-\log\sigma))

    Roughly speaking, quantum relative entropy is a measure of distinguishability between two
    quantum states. It is the quantum mechanical analog of relative entropy.

    Each state can be given as a state vector in the computational basis or
    as a density matrix.

    Args:
        state0 (tensor_like): ``(2**N)`` state vector or ``(2**N, 2**N)`` density matrix.
        state1 (tensor_like): ``(2**N)`` state vector or ``(2**N, 2**N)`` density matrix.
        base (float): Base for the logarithm. If None, the natural logarithm is used.
        check_state (bool): If True, the function will check the state validity (shape and norm).
        c_dtype (str): Complex floating point precision type.

    Returns:
        float: Quantum relative entropy of state0 with respect to state1

    **Examples**

    The relative entropy between two equal states is always zero:

    >>> x = np.array([1, 0])
    >>> qml.math.relative_entropy(x, x)
    0.0

    and the relative entropy between two non-equal pure states is always infinity:

    >>> y = np.array([1, 1]) / np.sqrt(2)
    >>> qml.math.relative_entropy(x, y)
    inf

    The quantum states can be provided as density matrices, allowing for computation
    of relative entropy between mixed states:

    >>> rho = np.array([[0.3, 0], [0, 0.7]])
    >>> sigma = np.array([[0.5, 0], [0, 0.5]])
    >>> qml.math.relative_entropy(rho, sigma)
    tensor(0.08228288, requires_grad=True)

    It is also possible to change the log base:

    >>> qml.math.relative_entropy(rho, sigma, base=2)
    tensor(0.1187091, requires_grad=True)

    .. seealso:: :func:`pennylane.qinfo.transforms.relative_entropy`
    """
    # Cast as a c_dtype array
    state0 = cast(state0, dtype=c_dtype)
    len_state0 = state0.shape[0]

    # Cannot be cast_like if jit
    if not is_abstract(state0):
        state1 = cast_like(state1, state0)

    len_state1 = state1.shape[0]

    if check_state:
        if state0.shape == (len_state0,):
            _check_state_vector(state0)
        else:
            _check_density_matrix(state0)

        if state1.shape == (len_state1,):
            _check_state_vector(state1)
        else:
            _check_density_matrix(state1)

    # Get dimension of the quantum system and reshape
    num_indices0 = int(np.log2(len_state0))
    num_indices1 = int(np.log2(len_state1))

    if num_indices0 != num_indices1:
        raise qml.QuantumFunctionError("The two states must have the same number of wires.")

    if state0.shape == (len_state0,):
        state0 = qml.math.outer(state0, np.conj(state0))

    if state1.shape == (len_state1,):
        state1 = qml.math.outer(state1, np.conj(state1))

    return _compute_relative_entropy(state0, state1, base=base)


def _check_density_matrix(density_matrix):
    """Check the shape, the trace and the positive semi-definitiveness of a matrix."""
    shape = density_matrix.shape[0]
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
        # Check if the matrix is Hermitian
        conj_trans = np.transpose(np.conj(density_matrix))
        if not allclose(density_matrix, conj_trans):
            raise ValueError("The matrix is not Hermitian.")
        # Check if positive semi-definite
        evs = np.linalg.eigvalsh(density_matrix)
        evs = np.real(evs)
        evs_non_negative = [ev for ev in evs if ev >= 0.0]
        if len(evs) != len(evs_non_negative):
            raise ValueError("The matrix is not positive semi-definite.")


def _check_state_vector(state_vector):
    """Check the shape and the norm of a state vector."""
    len_state = state_vector.shape[0]
    # Check format
    if len(np.shape(state_vector)) != 1 or not np.log2(len_state).is_integer():
        raise ValueError("State vector must be of length 2**wires.")
    # Check norm
    norm = np.linalg.norm(state_vector, ord=2)
    if not is_abstract(norm):
        if not allclose(norm, 1.0, atol=1e-10):
            raise ValueError("Sum of amplitudes-squared does not equal one.")


def min_entropy(state, base=None, check_state=False, c_dtype="complex128"):
    r"""
    Compute the min-entropy of a state.

    .. math::
        `\Min Entropy` = \RE{\infty}{X} = - \log_2 \max_{x \in \mathcal{X}} p(x)

    The min entropy of a state is the negative log of the largest eigenvalue of the corresponding density matrix.

    Args:
        state (tensor_like): ``(2**N)`` state vector or ``(2**N, 2**N)`` density matrix.
        base (float): Base for the logarithm. If None, the natural logarithm is used.
        check_state (bool): If True, the function will check the state validity (shape and norm).
        c_dtype (str): Complex floating point precision type.

    Returns:
        float: Min-Entropy of the input state.

    **Examples**

    Each state can be given as a state vector in the computational basis or as a density matrix.

    >>> x = np.array([1, 0])
    >>> qml.math.min_entropy(x)
    0.0

    >>> y = np.array([1/np.sqrt(3), np.sqrt(2/3)])
    >>> qml.math.min_entropy(y)
    4.262249472335989e-06

    The quantum states can be provided as density matrices as show in the example below:

    >>> rho = np.array([[0.3, 0], [0, 0.7]])
    >>> qml.math.min_entropy(rho)
    0.35667494393873245

    It is also possible to change the log base:

    >>> qml.math.relative_entropy(rho, base=2)
    0.5145731728297583
    
    """

    # Cast as a c_dtype array
    state = cast(state, dtype=c_dtype)
    len_state = state.shape[0]

    if check_state:
        if state.shape == (len_state,):
            _check_state_vector(state)
        else:
            _check_density_matrix(state)

    if state.shape == (len_state,):
        state = qml.math.outer(state, np.conj(state))

    if base:
        div_base = np.log(base)
    else:
        div_base = 1

    # eigenvalues of the input state
    evs_state = qml.math.linalg.eigh(state)[0]

    # cast all eigenvalues to real
    evs_state = np.real(evs_state)

    min_ent = -np.log(evs_state.max()) / div_base

    return min_ent
