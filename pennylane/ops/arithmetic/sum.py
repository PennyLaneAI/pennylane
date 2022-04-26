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
"""
This file contains the implementation of the Sum class which contains logic for
computing the sum of operations.
"""
import warnings
import numpy as np
import pennylane as qml
from pennylane import numpy as qnp


def sum(*summands):
    """Top level function computing the sum of the
    provided summands"""
    return Sum(*summands)


class Sum(qml.operation.Operator):
    """Arithmetic operator subclass representing the sum of operators"""

    def __init__(self, *summands, do_queue=True, id=None):

        if len(summands) < 2:
            raise ValueError(f"Require at least two operators to sum; got {len(summands)}")

        self.summands = summands

        combined_wires = qml.wires.Wires.all_wires([s.wires for s in summands])
        combined_params = [s.parameters for s in summands]
        super().__init__(
            *combined_params, wires=combined_wires, do_queue=do_queue, id=id
        )
        self._name = "Sum(" + ", ".join([f"{f}" for f in summands]) + ")"

    def __repr__(self):
        """Constructor-call-like representation."""
        return " + ".join([f"{f}" for f in self.summands])

    @property
    def num_wires(self):
        return len(self.wires)

    def terms(self):
        return [1.0]*len(self.summands), self.summands

    def get_matrix(self, wire_order=None):
        return self._sum(m.get_matrix(wire_order=wire_order) for m in self.summands)

    def _sum(self, *mats):
        """Super inefficient Sum method just as a proof of concept"""
        res = qnp.zeros(len(self.wires))

        for mat in mats:
            res += mat

        return res
