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
import numpy as np
from scipy.linalg import sqrtm, norm

import pennylane as qml
from pennylane.operation import Operation, AnyWires


class BlockEncoding(Operation):
    """General Block Encoding"""
    num_params = 1
    num_wires = AnyWires

    def __init__(self, a, wires, do_queue=True, id=None):
        super().__init__(a, wires=wires, do_queue=do_queue, id=id)
        normalization =

    @staticmethod
    def compute_matrix(a):
        """Get the matrix representation of block encoded unitary."""
        d1, d2 = a.shape
        u = np.block([[a, sqrtm(np.eye(d1) - a @ np.conj(a).T)],
                      [sqrtm(np.eye(d2) - np.conj(a).T @ a), -np.conj(a).T]])
        return u


class PiControlledPhase(Operation):
    """A Pi-Controlled Phase gate"""
    num_wires = AnyWires

    def __init__(self, phi, size, wires, do_queue=True, id=None):
        super().__init__(phi, wires=wires, do_queue=do_queue, id=id)
        self.hyperparams["size"] = size

    @staticmethod
    def compute_matrix(a):
        """Get the matrix representation of block encoded unitary."""
        d1, d2 = a.shape
        u = np.block([[a, sqrtm(np.eye(d1) - a @ np.conj(a).T)],
                      [sqrtm(np.eye(d2) - np.conj(a).T @ a), -np.conj(a).T]])
        return u