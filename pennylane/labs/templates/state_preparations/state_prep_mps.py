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
Contains the StatePrepMPS template.
"""

from pennylane.operation import Operation
from pennylane.wires import Wires


class MPSPrep(Operation):
    r"""Prepares an initial state using MPS

    .. note::

        This operator is only supported in qml.device(“lightning.tensor”)


    Args:
        mps (list(arrays)): The list representing the MPS input. The dimensions of these arrays are carefully described in usage details.
        wires (Sequence[int]): The wires where the initial state is prepared.

    **Example**

    .. code-block::

        mps = # The MPS array

        dev = qml.device("lightning.tensor", wires = 3)
        @qml.qnode(dev)
        def circuit():
            qml.labs.MPSPrep(mps, wires = [0,1,2])
            return qml.sample(wires=x_wires)

    .. details::
    :title: Usage Details

    Given that an MPS is a product of MPS site matrices, the input would be just a list of numpy arrays of rank-3, and rank-2 tensors on the ends.

    """

    def __init__(self, mps, wires, id=None):
        class MPSsite_wrapper_data:
            def __init__(self, mps):
                self.data = mps
                
        self.mps = MPSsite_wrapper_data(mps)
        super().__init__(self.mps, wires=wires, id=id)

    def _flatten(self):
        return tuple(self.data), self.wires

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(data[0], metadata)

    def map_wires(self, wire_map):
        new_wires = Wires([wire_map.get(wire, wire) for wire in self.wires])
        return MPSPrep(self.mps, new_wires)
