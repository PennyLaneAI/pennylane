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
Contains an implementation of the metric tensor using QNode collections.
"""
# pylint: disable=protected-access,import-outside-toplevel

import itertools

import numpy as np
from scipy.linalg import block_diag
import pennylane as qml

from . import functions as fn


def _marginal_prob(prob, wires, interface):
    """Compute the marginal probability.

    Args:
        prob (array[float] or tensor[float]): probability vector of size ``(2**num_wires,)``
        wires (list[int]): list of integers for which to calculate the marginal
            probability distribution
        interface (str): the classical interface to target

    Returns:
        array[float] or tensor[float]: the marginal probabilities, of
        size ``(2**len(wires),)``
    """
    num_wires = int(np.log2(len(prob)))
    inactive_wires = list(set(range(num_wires)) - set(wires))
    prob = fn[interface].reshape(prob, [2] * num_wires)
    prob = fn[interface].reduce_sum(prob, inactive_wires)
    return fn[interface].flatten(prob)


class MetricTensor:
    """Construct a QNode collection that computes the metric tensor of a given
    QNode.

    Args:
        qnode (QNode): a qubit QNode for which the metric tensor should be computed
        device (Device, Sequence[Device]): Corresponding device(s) where the resulting
            metric tensor should be executed. This can either be a single device, or a list
            of devices of length corresponding to the number of parametric layers in ``qnode``.


    **Example**

    Consider the following QNode:

    .. code-block:: python

        @qml.qnode(dev, interface="tf")
        def qnode(params):
            qml.templates.StronglyEntanglingLayers(params, wires=[0, 1, 2])
            return qml.expval(qml.PauliY(0) @ qml.PauliZ(1))

    By using ``qml.MetricTensor()``, we create a metric tensor object that can be evaluated
    to compute the metric tensor of the QNode:

    >>> g = qml.MetricTensor(qnode, dev)

    We can evaluate this function at various points in the QNode parameter space to
    compute the metric tensor:

    >>> weights = qml.init.strong_ent_layers_normal(n_wires=3, n_layers=2)
    >>> t1 = g(weights)
    >>> t1.shape
    (18, 18)

    The ``MetricTensor`` object computes the metric tensor by generating a :class:`~.QNodeCollection`;
    each QNode in the collection computes a block diagonal element of the the metric tensor on hardware.
    These QNode are accessible via the ``qnodes`` attribute:

    >>> print(g.qnodes)
    <pennylane.collections.qnode_collection.QNodeCollection object at 0x7f62dcb39310>

    In this case, 14 QNodes are used to compute the metric tensor.

    >>> len(g.qnodes)
    14

    The output of ``g`` is also end-to-end differentiable using your
    :doc:`chosen ML library </introduction/interfaces>`.
    When the gradient of the metric tensor is requested, the parameter-shift rule is used to
    to compute quantum gradients automatically, on quantum hardware if required.

    In this example, we are using the TensorFlow interface, so we can use TensorFlow to
    construct a cost function using the metric tensor and to differentiate it.

    >>> with tf.GradientTape() as tape:
    ...     loss = tf.reduce_sum(g(weights))
    >>> grad = tape.gradient(loss, weights)
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, qnode, device):
        self.qnode = qnode
        self.dev = device
        self.interface = qnode.interface
        self._qnode_type = self.qnode.__class__

        self.qnodes = None
        self.obs = None
        self.coeffs = None
        self.params = None

        self.fn = fn[self.interface]

    def __call__(self, *args, **kwargs):
        parallel = kwargs.pop("parallel", False)
        diag_approx = kwargs.pop("diag_approx", False)

        if self.qnodes is None:
            self.qnodes, self.obs, self.coeffs, self.params = self._make_qnodes(args, kwargs)

        if len(args) == 1:
            args = args[0]

        probs = self.qnodes(args, parallel=parallel)
        self.fn.reshape(probs, [len(self.qnodes), -1])

        gs = []

        for prob, obs, coeffs in zip(probs, self.obs, self.coeffs):
            # calculate the covariance matrix of this layer
            scale = self.fn.asarray(np.outer(coeffs, coeffs))
            g = scale * self.cov_matrix(prob, obs, diag_approx=diag_approx)
            gs.append(g)

        perm = np.array([item for sublist in self.params for item in sublist])

        # create the block diagonal metric tensor
        tensor = self.fn.block_diag(*gs)

        # permute rows
        tensor = self.fn.gather(tensor, perm)

        # permute columns
        tensor = self.fn.gather(self.fn.transpose(tensor), perm)
        return tensor

    def cov_matrix(self, prob, obs, diag_approx=False):
        """For a given list of observables, calculate the covariance matrix
        of a probability distribution.

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
        diag = []

        # diagonal variances
        for i, o in enumerate(obs):
            l = self.fn.cast(self.fn.asarray(o.eigvals), dtype=self.fn.float64)
            p = _marginal_prob(prob, o.wires, self.interface)
            res = self.fn.dot(l ** 2, p) - (self.fn.dot(l, p)) ** 2
            diag.append(res)

        diag = self.fn.diag(diag)

        if diag_approx:
            return diag

        for i, j in itertools.combinations(range(len(obs)), r=2):
            o1 = obs[i]
            o2 = obs[j]

            l1 = self.fn.cast(self.fn.asarray(o1.eigvals), dtype=self.fn.float64)
            l2 = self.fn.cast(self.fn.asarray(o2.eigvals), dtype=self.fn.float64)
            l12 = self.fn.cast(self.fn.asarray((o1 @ o2).eigvals), dtype=self.fn.float64)

            p1 = _marginal_prob(prob, o1.wires, self.interface)
            p2 = _marginal_prob(prob, o2.wires, self.interface)
            p12 = _marginal_prob(prob, o1.wires + o2.wires, self.interface)

            res = self.fn.dot(l12, p12) - self.fn.dot(l1, p1) * self.fn.dot(l2, p2)

            diag = self.fn.scatter_element_add(diag, [i, j], res)
            diag = self.fn.scatter_element_add(diag, [j, i], res)

        return diag

    def _make_qnodes(self, args, kwargs):
        """Helper method to construct the QNodes that generate the metric tensor."""
        if hasattr(self.qnode, "circuit"):
            if self.qnode.circuit is None or self.qnode.mutable:
                # construct the circuit
                self.qnode._construct(args, kwargs)
        else:
            self.qnode(*args, **kwargs)

        circuit = getattr(self.qnode, "circuit", None) or getattr(
            self.qnode._qnode, "circuit", None
        )

        qnodes = qml.QNodeCollection()

        obs_list = []
        coeffs = []
        params = []

        for queue, curr_ops, param_idx, _ in circuit.iterate_parametrized_layers():
            params.append(param_idx)
            coeffs.append([])
            obs_list.append([])

            # for each operation in the layer, get the generator
            for op in curr_ops:
                gen, s = op.generator
                w = op.wires

                if gen is None:
                    raise qml.qnodes.QuantumFunctionError(
                        "Can't generate metric tensor, operation {}"
                        "has no defined generator".format(op)
                    )

                coeffs[-1].append(s)

                # get the observable corresponding to the generator of the current operation
                if isinstance(gen, np.ndarray):
                    # generator is a Hermitian matrix
                    obs_list[-1].append(qml.Hermitian(gen, w))

                elif issubclass(gen, qml.operation.Observable):
                    # generator is an existing PennyLane operation
                    obs_list[-1].append(gen(w))

                else:
                    raise qml.qnodes.QuantumFunctionError(
                        "Can't generate metric tensor, generator {}"
                        "has no corresponding observable".format(gen)
                    )

            @qml.qnode(self.dev, interface=self.interface, mutable=False)
            def qn(
                weights, _queue=queue, _obs_list=obs_list[-1], _dev=self.dev
            ):  # pylint: disable=dangerous-default-value
                for op in _queue:
                    op.queue()

                for o in _obs_list:
                    o.diagonalizing_gates()

                return qml.probs(wires=range(_dev.num_wires))

            qnodes.append(qn)

        return qnodes, obs_list, coeffs, params
