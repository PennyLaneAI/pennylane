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

from autoray import numpy as np
from numpy import float64

from . import single_dispatch  # pylint:disable=unused-import
from .multi_dispatch import cast, diag, dot, scatter_element_add


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
        l = cast(o.get_eigvals(), dtype=float64)
        w = o.wires.labels if wires is None else wires.indices(o.wires)
        p = marginal_prob(prob, w)

        res = dot(l**2, p) - (dot(l, p)) ** 2
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

        l1 = cast(o1.get_eigvals(), dtype=float64)
        l2 = cast(o2.get_eigvals(), dtype=float64)
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
