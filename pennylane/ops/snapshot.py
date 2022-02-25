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
This module contains the Snapshot (pseudo) operation that is common to both
cv and qubit computing paradigms in PennyLane.
"""
from pennylane.operation import AnyWires, AllWires, Operation


# pylint: disable=unused-argument
class Snapshot(Operation):
    r"""pennylane.Snapshot()
    The Snapshot operation preserves the internal simulator state at specific
    points in a circuit. As such it is a pseudo operation as it doesn't affect
    the quantum state.
    """
    num_wires = AnyWires
    num_params = 0
    grad_method = None

    def __init__(self, tag=None):
        self.tag = tag
        super().__init__(wires=AllWires, do_queue=True)

    def label(self, decimals=None, base_label=None):
        return "Snapshot" + ("(" + self.tag + ")" if self.tag else "")

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        return []
