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

# pylint: disable=too-few-public-methods,function-redefined

import pennylane as qml


class Sum(qml.operation.Operator):
    """Arithmetic operator subclass representing the sum of two operators"""

    def __init__(self, *summands, do_queue=True, id=None):

        self.hyperparameters["summands"] = summands

        combined_wires = qml.wires.Wires.all_wires([s.wires for s in summands])
        super().__init__(
            *[s.parameters for s in summands], wires=combined_wires, do_queue=do_queue, id=id
        )
        self._name = "Sum(" + ", ".join([f"{f}" for f in summands]) + ")"

    def __repr__(self):
        """Constructor-call-like representation."""
        return " + ".join([f"{f}" for f in self.hyperparameters['summands']])

    @property
    def num_wires(self):
        return len(self.wires)

    @staticmethod
    def compute_terms(*params, summands=None):
        return [1.0]*len(summands), summands

    @staticmethod
    def compute_matrix(*params, summands=None):
        # ugly to compute this here again! Should we have passed wires to the matrix methods in the first place?
        combined_wires = qml.wires.Wires.all_wires([s.wires for s in summands])
        m = summands[0].get_matrix(wire_order=combined_wires)
        for f in summands[1:]:
            m + f.get_matrix(wire_order=combined_wires)
        return m


def sum(*summands):

    if all(isinstance(s, qml.operation.Observable) for s in summands):
        # if everything is an observable, we can create a Hamiltonian
        coeffs, ops = qml.ops.math.utils.flatten_terms([1.]*len(summands), summands)
        return qml.Hamiltonian(coeffs, ops)

    return Sum(*summands)
