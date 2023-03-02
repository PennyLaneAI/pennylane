# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Contains core operations used in the Quantum Singular Value Transform framework.
"""
from pennylane import numpy as np
from scipy.linalg import sqrtm, norm

import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.wires import Wires


class BlockEncode(Operation):
    """General Block Encoding"""

    num_params = 1
    num_wires = AnyWires

    def __init__(self, a, wires, do_queue=True, id=None):
        a = np.atleast_2d(a)
        wires = Wires(wires)
        if np.sum(a.shape) <= 2:
            normalization = a if a > 1 else 1
            subspace = (1, 1, 2 ** len(wires))
        else:
            normalization = np.max(
                [norm(a @ np.conj(a).T, ord=np.inf), norm(np.conj(a).T @ a, ord=np.inf)]
            )
            subspace = (*a.shape, 2 ** len(wires))

        a = a / normalization if normalization > 1 else a

        if subspace[2] < (subspace[0] + subspace[1]):
            raise qml.QuantumFunctionError(
                f"Block encoding a {subspace[0]} x {subspace[1]} matrix"
                f"requires a hilbert space of size at least "
                f"{subspace[0] + subspace[1]} x {subspace[0] + subspace[1]}."
                f" Cannot be embedded in a {len(wires)} qubit system."
            )

        super().__init__(a, wires=wires, do_queue=do_queue, id=id)
        self.hyperparameters["norm"] = normalization
        self.hyperparameters["subspace"] = subspace

    @staticmethod
    def compute_matrix(*params, **hyperparams):
        """Get the matrix representation of block encoded unitary."""
        a = params[0]
        n, m, k = hyperparams["subspace"]

        if isinstance(a, int) or isinstance(a, float):
            u = np.block(
                [[a, np.sqrt(1 - a * np.conj(a))], [np.sqrt(1 - a * np.conj(a)), -np.conj(a)]]
            )
        else:
            d1, d2 = a.shape

            col1 = qml.math.vstack([a, qml.math.sqrt_matrix(np.eye(d2) - np.conj(a).T @ a)])
            col2 = qml.math.vstack([qml.math.sqrt_matrix(np.eye(d1) - a @ np.conj(a).T), -np.conj(a).T])

            u = qml.math.hstack([col1, col2])

        if n + m < k:
            r = k - (n + m)
            u = np.block([[u, np.zeros((n + m, r))], [np.zeros((r, n + m)), np.eye(r)]])

        return u
