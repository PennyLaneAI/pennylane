# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Contains the QNode transform function cov_matrix, which outputs a QNode
which computes the covariance matrix with respect to the input QNode's observables.
"""
# pylint: disable=protected-access,import-outside-toplevel

import itertools

import numpy as np
import pennylane as qml
from pennylane.tape import QNode

from . import functions as fn


def get_interface_library(obj):
    """Returns the correct library wrapper for classical processing,
    given an object datatype."""
    interface = obj.__class__.__module__.split(".")[0]
    return fn[interface]


def _marginal_prob(prob, wires):
    """Compute the marginal probability given joint probabilities of the full system.

    Args:
        prob (array[float] or tensor[float]): probability vector of size ``(2**num_wires,)``
        wires (list[int]): list of integers for which to calculate the marginal
            probability distribution

    Returns:
        array[float] or tensor[float]: the marginal probabilities, of
        size ``(2**len(wires),)``
    """
    fn = get_interface_library(prob)
    prob = fn.flatten(prob)
    num_wires = int(np.log2(len(prob)))

    if num_wires == len(wires):
        return prob

    inactive_wires = list(set(range(num_wires)) - set(wires.labels))
    prob = fn.reshape(prob, [2] * num_wires)
    prob = fn.reduce_sum(prob, inactive_wires)
    return fn.flatten(prob)


def _cov_matrix(prob, obs, diag_approx=False):
    """Calculate the covariance matrix of a list of commuting observables, given
    the joint probability distribution of the system in the shared eigenbasis.

    .. note::

        This method only works for **commuting observables.**

        The probability distribution must be rotated into the shared
        eigenbasis of the list of observables.

    Args:
        prob (array or tensor): probability distribution
        obs (list[Observable]): a list of observables for which
            to compute the covariance matrix for

    Returns:
        array or tensor: the covariance matrix of size ``(len(obs), len(obs))``
    """
    fn = get_interface_library(prob)

    diag = []

    # diagonal variances
    for i, o in enumerate(obs):
        l = fn.cast(fn.asarray(o.eigvals), dtype=fn.float64)
        p = _marginal_prob(prob, o.wires)

        res = fn.dot(l ** 2, p) - (fn.dot(l, p)) ** 2
        diag.append(res)

    diag = fn.diag(diag)

    if diag_approx:
        return diag

    for i, j in itertools.combinations(range(len(obs)), r=2):
        o1 = obs[i]
        o2 = obs[j]

        l1 = fn.cast(fn.asarray(o1.eigvals), dtype=fn.float64)
        l2 = fn.cast(fn.asarray(o2.eigvals), dtype=fn.float64)
        l12 = fn.cast(fn.asarray(np.kron(l1, l2)), dtype=fn.float64)

        p1 = _marginal_prob(prob, o1.wires)
        p2 = _marginal_prob(prob, o2.wires)
        p12 = _marginal_prob(prob, o1.wires + o2.wires)

        res = fn.dot(l12, p12) - fn.dot(l1, p1) * fn.dot(l2, p2)

        diag = fn.scatter_element_add(diag, [i, j], res)
        diag = fn.scatter_element_add(diag, [j, i], res)

    return diag


def cov_matrix(qnode):
    r"""QNode transform for computing the covariance matrix of a QNode
    with respect to its measured observables.

    Since a QNode only permits each wire to be measured once, by design
    all observables will be commuting.

    .. todo::

        This transform could be made more general, by perhaps accepting a
        QNode/quantum tape and a list of observables. However, this would at the
        same time require more logic, to take into account non-commuting
        observables.

    Args:
        qnode (.tape.QNode): the QNode to compute the covariance matrix of

    Returns:
        .tape.QNode: A QNode that, when evaluated, outputs the covariance matrix
        of the input QNode. The rows and columns of the covariance matrix correspond
        to the observables measured, in order, of the input QNode.

    **Example**

    Consider a QNode with various measured observables:

    .. code-block:: python

        qml.enable_tape()
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="autograd")
        def circuit(weights):
            ansatz(weights)
            return qml.expval(qml.PauliX(0) @ qml.PauliZ(1)), qml.expval(qml.PauliY(2))

    We can use this transform to return a new QNode, that computes the covariance
    matrix of the measured observables :math:`X_0 \otimes Z_1` and :math:`Y_2` with
    respect to the circuit ansatz:

    >>> cov_circuit = qml.tape.transforms.cov_matrix.cov_matrix(circuit)
    >>> cov_circuit
    <pennylane.tape.qnode.QNode at 0x7ff39dcb2c50>

    This QNode can be evaluated, returning the covariance matrix:

    >>> res = cov_circuit(weights)
    >>> res
    tensor([[ 0.96476749, -0.63958103],
            [-0.63958103,  0.99031256]], requires_grad=True)

    The output covariance matrix is end-to-end differentiable, in all interfaces.
    For example, using the autograd interface, let's differentiate the `(0, 1)`
    element of the covariance matrix (corresponding to :math:`\text{cov}(X_0 \otimes Z_1, Y_2)`):

    >>> grad_fn = qml.grad(lambda weights: cov_qnode(weights)[0, 1])
    >>> print(grad_fn(weights))
    (array([[ 2.91145666e-02, -1.10066756e-01, -4.54650047e-17],
            [ 3.86166688e-01,  2.47478347e-01,  5.27788755e-01],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]),)
    """

    def construct(self, args, kwargs):
        QNode.construct(self, args, kwargs)

        # extract the observables from the QNode
        self.observables = [
            o for o in self.qtape.observables if isinstance(o, qml.operation.Observable)
        ]

        # Expand the quantum tape, so that the final quantum state is in the eigenbasis
        # of all measured observables.
        self.qtape = self.qtape.expand(expand_measurements=True)

        # replace the computational basis measurements with a probability measurement
        self.qtape._measurements = [
            qml.tape.MeasurementProcess(qml.operation.Probability, wires=self.qtape.wires)
        ]
        self.qtape._update()

    def process(self, output):
        # process the output probabilities of the QNode to return the covariance matrix.
        return _cov_matrix(output, self.observables)

    import copy

    new_qnode = copy.deepcopy(qnode)
    new_qnode.construct = construct.__get__(new_qnode, qml.tape.QNode)
    new_qnode.process = process.__get__(new_qnode, qml.tape.QNode)
    return new_qnode
