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
import numpy as np

from pennylane.operation import AnyWires, Operation


# pylint: disable=unused-argument
class Snapshot(Operation):
    r"""
    The Snapshot operation saves the internal simulator state at specific
    execution steps of a quantum function. As such, it is a pseudo operation
    with no effect on the quantum state.

    **Details:**

    * Number of wires: AllWires
    * Number of parameters: 0

    Args:
        tag (str or None): An optional custom tag for the snapshot, used to index it
                           in the snapshots dictionary.
    """
    num_wires = AnyWires
    num_params = 0
    grad_method = None

    def __init__(self, tag=None):
        self.tag = tag
        super().__init__(wires=[], do_queue=True)

    def label(self, decimals=None, base_label=None, cache=None):
        return "|S|"

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        return []

    # TODO: remove once pennylane-lightning#242 is resolved
    @staticmethod
    def compute_matrix(*params, **hyperparams):
        return np.eye(2)
