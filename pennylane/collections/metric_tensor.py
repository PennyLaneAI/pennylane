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

from .dot import _get_dot_func
from .sum import _get_sum_func


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

    if interface == "tf":
        import tensorflow as tf

        prob = tf.reshape(prob, [2] * num_wires)
    else:
        # NumPy and Torch both have reshape methods
        prob = prob.reshape([2] * num_wires)

    for a in reversed(inactive_wires):
        prob = _get_sum_func(interface)(prob, axis=a)

    if interface == "tf":
        return tf.reshape(prob, [-1])

    return prob.flatten()


class MetricTensor:
    """Construct a QNode collection that computes the metric tensor of a given
    QNode.

    Args:
        qnode (QubitQNode): a Qubit QNode for which the metric tensor should be computed
        device (Device, Sequence[Device]): Corresponding device(s) where the resulting
            metric tensor should be executed. This can either be a single device, or a list
            of devices of length corresponding to the number of parametric layers in ``qnode``.
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

        if self.interface == "tf":
            import tensorflow as tf

            self.np = tf
        elif self.interface == "torch":
            import torch

            self.np = torch
        elif self.interface == "autograd":
            from autograd import numpy as np  # pylint: disable=redefined-outer-name

            self.np = np
        else:
            self.np = np

        self.dot = _get_dot_func(self.interface)[0]

    def __call__(self, *args, **kwargs):
        parallel = kwargs.pop("parallel", False)
        diag_approx = kwargs.pop("diag_approx", False)

        if self.qnodes is None:
            self.qnodes, self.obs, self.coeffs, self.params = self._make_qnodes(args, kwargs)

        if len(args) == 1:
            args = args[0]

        probs = self.qnodes(args, parallel=parallel)
        self.np.reshape(probs, [len(self.qnodes), -1])

        gs = []

        for prob, obs, coeffs in zip(probs, self.obs, self.coeffs):
            # calculate the covariance matrix of this layer
            scale = np.outer(coeffs, coeffs)

            if self.interface == "torch":
                scale = self.np.tensor(scale)  # pylint: disable=not-callable

            g = scale * self.cov_matrix(prob, obs, diag_approx=diag_approx)
            gs.append(g)

        perm = np.array([item for sublist in self.params for item in sublist])

        if self.interface == "torch":
            # torch allows tensor assignment
            tensor = self.np.zeros([len(perm), len(perm)], dtype=self.np.float64)

            for g, p in zip(gs, self.params):
                row = np.array(p).reshape(-1, 1)
                col = np.array(p).reshape(1, -1)
                tensor[row, col] = g

            return tensor

        if self.interface == "tf":
            # tf doesn't allow tensor assignment, but does provide
            # support for constructing block diagonal operators

            tfl = self.np.linalg
            linop_blocks = [tfl.LinearOperatorFullMatrix(block) for block in gs]
            linop_block_diagonal = tfl.LinearOperatorBlockDiag(linop_blocks)

            # permute rows
            tensor = self.np.gather(linop_block_diagonal.to_dense(), perm)

            # permute columns
            tensor = self.np.gather(self.np.transpose(tensor), perm)
            return tensor

        # autograd
        tensor = block_diag(*gs)[:, perm][perm]
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
            dot_fn, l = _get_dot_func(self.interface, o.eigvals)
            p = _marginal_prob(prob, o.wires, self.interface)
            res = dot_fn(l ** 2, p) - (dot_fn(l, p)) ** 2
            diag.append(res)

        if self.interface == "tf":
            diag = self.np.linalg.diag(diag)
        elif self.interface == "torch":
            diag = self.np.diag(self.np.stack(diag))
        else:
            diag = self.np.diag(diag)

        if diag_approx:
            return diag

        # off-diagonal covariances
        off_diag = diag

        if self.interface == "tf":
            off_diag = np.zeros([len(obs), len(obs)]).tolist()

        for i, j in itertools.combinations(range(len(obs)), r=2):

            o1 = obs[i]
            o2 = obs[j]

            _, l1 = _get_dot_func(self.interface, o1.eigvals)
            _, l2 = _get_dot_func(self.interface, o2.eigvals)
            _, l12 = _get_dot_func(self.interface, (o1 @ o2).eigvals)

            p1 = _marginal_prob(prob, o1.wires, self.interface)
            p2 = _marginal_prob(prob, o2.wires, self.interface)
            p12 = _marginal_prob(prob, o1.wires + o2.wires, self.interface)
            off_diag[i][j] = off_diag[j][i] = dot_fn(l12, p12) - dot_fn(l1, p1) * dot_fn(l2, p2)

        if self.interface == "tf":
            return diag + self.np.cast(self.np.stack(off_diag), dtype=self.np.float64)

        return off_diag

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
                weights, _queue=queue, _obs_list=obs_list[-1], _dev=self.dev, _params=params
            ):  # pylint: disable=dangerous-default-value
                counter = 0
                p_idx = np.argsort([item for sublist in _params for item in sublist])

                for op in _queue:
                    p = []

                    for x in op.params:
                        if not isinstance(x, qml.variable.Variable):
                            p.append(x)
                        else:
                            p.append(weights[p_idx[counter]])
                            counter += 1

                    op.__class__(*p, wires=op.wires)

                for o in _obs_list:
                    o.diagonalizing_gates()

                return qml.probs(wires=range(_dev.num_wires))

            qnodes.append(qn)

        return qnodes, obs_list, coeffs, params
