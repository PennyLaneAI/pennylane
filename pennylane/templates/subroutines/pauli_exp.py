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
"""
This submodule contains the template for PauliExp.
"""
# pylint:disable=abstract-method,arguments-differ,protected-access

import functools
import numpy as np

import pennylane as qml
from pennylane.operation import AnyWires, Operation


class PauliExp(Operation):
    r"""

    """
    num_wires = AnyWires
    grad_method = None

    def __init__(self, coeff, pauli_word,  id=None):

        super().__init__(coeff, pauli_word, wires= pauli_word.wires, id=id)

    @property
    def num_params(self):
        return 2

    @staticmethod
    def compute_decomposition(coeff, pauli_word, wires = None):  # pylint: disable=arguments-differ,unused-argument
        r"""

        """
        decomp_ops = []
        wires = pauli_word.wires
        for i in range(len(wires)):
            if pauli_word.name[i] == "PauliX":
                decomp_ops.append(qml.Hadamard(wires=wires[i]))
            if pauli_word.name[i] == "PauliY":
                decomp_ops.append(qml.adjoint(qml.S)(wires=wires[i]))
                decomp_ops.append(qml.Hadamard(wires=wires[i]))
                decomp_ops.append(qml.S(wires=wires[i]))

        for i in range(len(wires) - 1):
            decomp_ops.append(qml.CNOT(wires=[wires[i], wires[-1]]))

        decomp_ops.append(qml.RZ(2 * coeff, wires=wires[-1]))

        for i in range(len(wires) - 2, -1, -1):
            decomp_ops.append(qml.CNOT(wires=[wires[i], wires[-1]]))

        for i in range(len(wires)):
            if pauli_word.name[i] == "PauliX":
                decomp_ops.append(qml.Hadamard(wires=wires[i]))
            if pauli_word.name[i] == "PauliY":
                decomp_ops.append(qml.adjoint(qml.S)(wires=wires[i]))
                decomp_ops.append(qml.Hadamard(wires=wires[i]))
                decomp_ops.append(qml.S(wires=wires[i]))

        return decomp_ops

