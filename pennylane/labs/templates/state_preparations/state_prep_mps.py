# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Contains the MPSPrep template.
"""

from pennylane.operation import Operation
from pennylane.wires import Wires


class MPSPrep(Operation):
    r"""Prepares an initial state using a MPS representation

    .. note::

        This operator is only supported in ``qml.device(“lightning.tensor”)``


    Args:
        mps (list(arrays)): The list representing the MPS input. Given that an MPS is a product of MPS site matrices,
        the input is a list of arrays of rank-3, and rank-2 tensors on the ends.

        wires (Sequence[int]): The wires where the initial state is prepared.

    **Example**

    .. code-block::

        from pennylane import labs

        mps = # The MPS array

        dev = qml.device("lightning.tensor", wires = 3)
        @qml.qnode(dev)
        def circuit():
            labs.MPSPrep(mps, wires = [0,1,2])
            return qml.sample(wires=x_wires)
    """

    def __init__(self, mps, wires, id=None):
        self.hyperparameters["mps"] = mps
        super().__init__(wires=wires, id=id)

    @property
    def mps(self):
        """list representing the MPS input"""
        return self.hyperparameters["mps"]

    def _flatten(self):
        hyperparameters = (("mps", tuple(self.hyperparameters["mps"])), ("wires", self.wires))
        return self.data, hyperparameters

    @classmethod
    def _unflatten(cls, data, metadata):
        hyperparams_dict = {key: list(value) if key == "mps" else value for key, value in metadata}
        return cls(**hyperparams_dict)

    def map_wires(self, wire_map):
        new_wires = Wires([wire_map.get(wire, wire) for wire in self.wires])
        return MPSPrep(self.mps, new_wires)
