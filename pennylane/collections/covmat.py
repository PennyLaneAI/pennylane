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


def expand(matrix, original_wires, expanded_wires):
    N = len(original_wires)
    M = len(expanded_wires)
    D = M - N

    dims = [2] * (2 * N)
    tensor = matrix.reshape(dims)

    if D > 0:
        extra_dims = [2] * (2 * D)
        identity = np.eye(len(extra_dims)).reshape(extra_dims)
        expanded_tensor = np.tensordot(tensor, identity, axes=0)
        # Fix order of tensor factors
        expanded_tensor = np.moveaxis(expanded_tensor, range(2 * N, 2 * N + D), range(N, N + D))
    else:
        expanded_tensor = tensor

    wire_indices = []
    for wire in original_wires:
        wire_indices.append(expanded_wires.index(wire))

    wire_indices = np.array(wire_indices)

    # Order tensor factors according to wires
    original_indices = np.array(range(N))
    expanded_tensor = np.moveaxis(expanded_tensor, original_indices, wire_indices)
    expanded_tensor = np.moveaxis(expanded_tensor, original_indices + M, wire_indices + M)

    return expanded_tensor.reshape((2 ** M, 2 ** M))


def symmetric_product(obs1, obs2):
    if isinstance(obs1, qml.Hermitian) and isinstance(obs2, qml.Hermitian):
        expanded_wires = list(set(obs1.wires + obs2.wires))
        mat1 = expand(np.array(obs1.params[0]), obs1.wires, expanded_wires)
        mat2 = expand(np.array(obs2.params[0]), obs2.wires, expanded_wires)
        sym_mat = 0.5 * (mat1 @ mat2 + mat2 @ mat1)

        return qml.Hermitian(sym_mat, wires=expanded_wires)
    else:
        raise NotImplementedException("Only Hermitian supported so far.")


class CovarianceMatrix:
    def __init__(self, ansatz, observables, device, interface="autograd", diff_method="best"):
        self.N = len(observables)
        product_observables = []
        self.product_observable_indices = {}
        for i, j in itertools.combinations(range(self.N), r=2):
            product_observables.append(symmetric_product(observables[i], observables[j]))
            self.product_observable_indices[(i, j)] = len(product_observables) - 1
            self.product_observable_indices[(j, i)] = len(product_observables) - 1

        self.product_qnodes = qml.map(
            ansatz, product_observables, device, interface=interface, diff_method=diff_method
        )
        self.var_qnodes = qml.map(
            ansatz, observables, device, measure="var", interface=interface, diff_method=diff_method
        )
        self.expval_qnodes = qml.map(
            ansatz, observables, device, interface=interface, diff_method=diff_method
        )

    def __call__(self, *args, **kwargs):
        variances = self.var_qnodes(*args, **kwargs)
        expvals = self.expval_qnodes(*args, **kwargs)
        products = self.product_qnodes(*args, **kwargs)

        cov_mat = np.diag(variances)

        for i, j in itertools.combinations(range(self.N), r=2):
            cov_mat[i, j] = (
                products[self.product_observable_indices[(i, j)]] - expvals[i] * expvals[j]
            )
            cov_mat[j, i] = cov_mat[i, j]

        return cov_mat
