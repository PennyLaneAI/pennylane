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
Contains an implementation of the covariance matrix using QNode collections.
"""
import itertools
import numpy as np
import pennylane as qml
from pennylane.utils import expand_matrix, _flatten
from pennylane.operation import Tensor, Operation, CVOperation
from pennylane import PauliX, PauliY, PauliZ, Hadamard, Identity, Hermitian


def _get_matrix(obs):
    if isinstance(obs, Operation):
        return obs.matrix

    return obs._matrix(*obs.params)


_PAULIS = {"PauliX", "PauliY", "PauliZ"}

# All qubit observables: X, Y, Z, H, Hermitian, Identity, (Tensor)
def symmetric_product(obs1, obs2):
    if isinstance(obs1, CVObservable) or isinstance(obs2, CVObservable):
        raise ValueError(
            "The symmetric product is currently only supported for Qubit observables."
            " The following observables were provided: {}, {}.".format(obs1.name, obs2.name)
        )

    wires1 = obs1.wires if not isinstance(obs1, Tensor) else list(_flatten(obs1.wires))
    wires2 = obs2.wires if not isinstance(obs2, Tensor) else list(_flatten(obs2.wires))

    if set(wires1).isdisjoint(set(wires2)):
        return obs1 @ obs2

    # By now, the observables are guaranteed to have the same wires
    try:
        if obs1.name in _PAULIS and obs2.name in _PAULIS:
            if obs1.name == obs2.name:
                return Identity(wires=wires1)

            # TODO: Add a Zero observable / Multiple of Identity observable
            return Hermitian(np.zeros((2, 2)), wires=wires1)
    except TypeError:
        # This occurs when we use in with a tensor's name
        pass

    # TODO: add further cases with a lookup table and/or logic
    # the following code will then be needed only for the case
    # if isinstance(obs1, Hermitian) or isinstance(obs2, Hermitian):

    expanded_wires = list(set(wires1 + wires2))
    mat1 = expand_matrix(_get_matrix(obs1), wires1, expanded_wires)
    mat2 = expand_matrix(_get_matrix(obs2), wires2, expanded_wires)
    sym_mat = 0.5 * (mat1 @ mat2 + mat2 @ mat1)

    return Hermitian(sym_mat, wires=expanded_wires)


class CovarianceMatrix:
    def __init__(self, ansatz, observables, device, interface="autograd", diff_method="best"):
        self.num_observables = len(observables)

        product_observables = []
        self.product_observable_indices = {}
        for i, j in itertools.combinations(range(self.num_observables), r=2):
            product_observables.append(symmetric_product(observables[i], observables[j]))
            self.product_observable_indices[(i, j)] = len(product_observables) - 1

        squared_observables = []
        for i in range(self.num_observables):
            squared_observables.append(symmetric_product(observables[i], observables[i]))

        self.product_qnodes = qml.map(
            ansatz, product_observables, device, interface=interface, diff_method=diff_method
        )

        self.square_qnodes = qml.map(
            ansatz, squared_observables, device, interface=interface, diff_method=diff_method
        )

        self.expval_qnodes = qml.map(
            ansatz, observables, device, interface=interface, diff_method=diff_method
        )

    def __call__(self, *args, **kwargs):
        squares = self.square_qnodes(*args, **kwargs)
        expvals = self.expval_qnodes(*args, **kwargs)
        products = self.product_qnodes(*args, **kwargs)

        cov_mat = np.diag(squares - expvals ** 2)

        for i, j in itertools.combinations(range(self.num_observables), r=2):
            cov_mat[i, j] = (
                products[self.product_observable_indices[(i, j)]] - expvals[i] * expvals[j]
            )
            cov_mat[j, i] = cov_mat[i, j]

        return cov_mat


class DiagonalObservablesCovarianceMatrix:
    def __init__(self, ansatz, diagonals, device, interface="autograd", diff_method="best"):
        self.diagonals = diagonals
        self.num_observables = len(diagonals)

        def func(params, wires, **kwargs):
            ansatz(params, wires, **kwargs)

            return qml.probs(wires)

        self.qnode = qml.QNode(func, device, interface=interface, diff_method=diff_method)

    def __call__(self, *args, **kwargs):
        probs = self.qnode(*args, **kwargs)

        squares = [np.dot(probs, diagonal**2) for diagonal in self.diagonals]
        expvals = [np.dot(probs, diagonal) for diagonal in self.diagonals]

        cov_mat = np.diag(squares - expvals ** 2)

        for i, j in itertools.combinations(range(self.num_observables), r=2):
            cov_mat[i, j] = (
                np.dot(probs, self.diagonals[i] * self.diagonals[j]) - expvals[i] * expvals[j]
            )
            cov_mat[j, i] = cov_mat[i, j]

        return cov_mat
