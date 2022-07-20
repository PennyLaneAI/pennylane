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
"""Classical Shadows baseclass with processing functions"""

from collections.abc import Iterable
import pennylane.numpy as np
import pennylane as qml
import warnings


class ClassicalShadow:
    """TODO: docstring"""

    def __init__(self, bitstrings, recipes):
        self.bitstrings = bitstrings
        self.recipes = recipes

        assert len(bitstrings) == len(recipes)
        self.snapshots = len(bitstrings)

        self.unitaries = [
            qml.matrix(qml.Hadamard(0)),
            qml.matrix(qml.Hadamard(0)) @ qml.matrix(qml.PhaseShift(np.pi / 2, wires=0)),
            qml.matrix(qml.Identity(0)),
        ]

    def expval(self, H, snapshots=None):
        if snapshots is None:
            snapshots = self.snapshots
        return None

    def local_snapshots(self, wires=None, snapshots=None):
        r"""Compute the T x n x 2 x 2 local snapshots
        i.e. compute :math:`3 U_i^\dagger |b_i \rangle \langle b_i| U_i - 1` for each qubit and each snapshot
        """
        bitstrings = self.bitstrings
        recipes = self.recipes

        if isinstance(snapshots, int):
            pick_snapshots = np.random.rand(self.snapshots, size=(snapshots))
            bitstrings = bitstrings[pick_snapshots]
            recipes = recipes[pick_snapshots]

        if isinstance(wires, Iterable):
            bitstrings = bitstrings[:, wires]
            recipes = recipes[:, wires]

        # unitaries as a class attirbute to allow for custom sets
        unitaries = self.unitaries

        # Computational basis states as density matrices
        zero = np.zeros((2, 2), dtype="complex")
        zero[0, 0] = 1.0
        one = np.zeros((2, 2), dtype="complex")
        one[1, 1] = 1.0

        T, n = bitstrings.shape

        # This vectorized approach is relying on clever broadcasting and might need some rework to make it compatible with all interfaces

        U = np.empty((T, n, 2, 2), dtype="complex")
        for i in range(len(unitaries)):
            U[np.where(recipes == i)] = unitaries[i]

        state = np.empty((T, n, 2, 2), dtype="complex")
        state[np.where(bitstrings == 0)] = zero
        state[np.where(bitstrings == 1)] = one

        return (
            3 * qml.math.transpose(qml.math.conj(U), axes=(0, 1, 3, 2)) @ state @ U
            - np.eye(2)[np.newaxis, np.newaxis]
        )

    def global_snapshots(self, wires=None, snapshots=None):
        """Compute the T x 2**n x 2**n global snapshots"""

        local_snapshot = self.local_snapshots(wires, snapshots)
    
        if local_snapshot.shape[1] > 16:
            warnings.warn(
                "Querying density matrices for n_wires > 16 is not recommended, operation will take a long time",
                UserWarning,
            )

        global_snapshots = []
        for T_snapshot in local_snapshot:
            tensor_product = T_snapshot[0]
            for n_snapshot in T_snapshot[1:]:
                tensor_product = np.kron(tensor_product, n_snapshot)
            global_snapshots.append(tensor_product)

        return np.array(global_snapshots)

