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
This submodule contains the template for AQFT.
"""

import warnings

import numpy as np

import pennylane as qml
from pennylane.operation import Operation




import itertools

def index_to_bitstring(integer, n):
    """ Converts an integer to a bitstring of length n. """
    return bin(integer)[2:].zfill(n)

def sum_by_prefix(data, prefix):
    """ Adds the elements of data whose index binary representation begin with `prefix`. """
    n = len(data).bit_length() - 1
    sum_result = 0
    for i, value in enumerate(data):
        bitstring = index_to_bitstring(i, n)
        if bitstring.startswith(prefix):
            sum_result += value
    return sum_result

def generate_bitstrings(n, add_zero = False):
    # itertools.product creates a cartesian product, which in this case are all combinations of 0s and 1s
    # repeat=n specifies the length of the sequences
    # if add_zero == True, we add one '0' to the end of each bitstring

    if add_zero:
      return [''.join(map(str, bits)) + '0' for bits in itertools.product([0, 1], repeat=n)]
    else:
      return [''.join(map(str, bits)) for bits in itertools.product([0, 1], repeat=n)]

def func_to_b(n_precision, func, x):
  """This function is used to obtain the binary representation of the outputs of a function."""
  return bin(int(2**(n_precision) + 2**(n_precision-3)*func(x)))[-n_precision:]


class StatePrepQROM(Operation):

    def __init__(self, vector, embedding_wires, precision_wires, aux_wires = None, id = None):

        self.vector = vector
        self.hyperparameters["embedding_wires"] = qml.wires.Wires(embedding_wires)
        self.hyperparameters["precision_wires"] = qml.wires.Wires(precision_wires)
        self.hyperparameters["aux_wires"] = qml.wires.Wires(aux_wires) if aux_wires else None

        if aux_wires:

            all_wires = self.hyperparameters["embedding_wires"] + self.hyperparameters["precision_wires"] + self.hyperparameters["aux_wires"]
        else:
            all_wires = self.hyperparameters["embedding_wires"] + self.hyperparameters["precision_wires"]


        super().__init__(vector, wires= all_wires, id=id)

    @property
    def num_params(self):
        return 1

    def decomposition(self):  # pylint: disable=arguments-differ

        return self.compute_decomposition(
            self.vector,
            embedding_wires=self.hyperparameters["embedding_wires"],
            precision_wires=self.hyperparameters["precision_wires"],
            aux_wires=self.hyperparameters["aux_wires"],
        )

    @staticmethod
    def compute_decomposition(vector, embedding_wires, precision_wires, aux_wires):  # pylint: disable=arguments-differ

        vector = np.array(vector) ** 2
        decomp_ops = []

        func = lambda x: 2 * np.arccos(np.sqrt(x))
        n_precision = len(precision_wires)

        set_prefix = generate_bitstrings(1)
        p0 = np.array([sum_by_prefix(vector, prefix=p) for p in set_prefix])
        decomp_ops.append(qml.RY(func(p0[0]), wires=embedding_wires[0]))

        # qml.Barrier()

        wires_reg = qml.registers(
            {"others": int(np.log2(len(vector))), "theta_wires": n_precision, "work_wires": 1 * n_precision})
        #print(wires_reg)

        theta_wires = precision_wires
        work_wires = qml.wires.Wires(aux_wires)

        for iter in range(int(np.log2(len(vector)) - 1)):

            set_prefix = generate_bitstrings(iter + 1)
            p0 = np.array([sum_by_prefix(vector, prefix=p) for p in set_prefix])

            set_prefix = generate_bitstrings(iter + 1, add_zero=True)
            new_p0 = np.array([sum_by_prefix(vector, prefix=p) for p in set_prefix])

            eps = 0.00000001  # to avoid division by 0
            b = [func_to_b(n_precision, func, new_p0[i] / (p0[i] + eps)) for i in range(len(new_p0))]

            if aux_wires:
                aux_work_wires = work_wires[: min(2 ** ((iter + 1) - 1) * n_precision, len(work_wires))]
            else:
                aux_work_wires = None

            decomp_ops.append(qml.QROM(bitstrings=b,
                     target_wires=theta_wires,
                     control_wires=embedding_wires[:iter + 1], #range(iter + 1),
                     work_wires= aux_work_wires,
                     clean=False))

            for ind, wire in enumerate(theta_wires):
                decomp_ops.append(qml.CRY(2 ** (2 - ind), wires=[wire, embedding_wires[iter + 1]]))

            decomp_ops.append(qml.adjoint(qml.QROM)(bitstrings=b,
                                  target_wires=theta_wires,
                                  control_wires=embedding_wires[:iter + 1],
                                  work_wires=aux_work_wires,
                                  clean=False))


        return decomp_ops