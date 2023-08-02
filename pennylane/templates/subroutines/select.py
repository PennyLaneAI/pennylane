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
Contains the QSVT template and qsvt wrapper function.
"""
# pylint: disable=too-many-arguments

import pennylane as qml
from pennylane.operation import Operation
from pennylane import math
import itertools

class SELECT(Operation):
    r"""
    Applies specific input unitaries depending on the state of a set of control qubits.

    .. math:: SELECT|X\rangle \otimes |\psi\rangle = \X\rangle \otimes U_x |\psi\rangle

    .. note:: The first wire provided corresponds to the **control qubit**.

    Args:
        ops list[Operator]: operations to apply
        control_wires (Sequence[int]): the wires/qubits that control which operation is applied
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional).
            This argument is deprecated, instead of setting it to ``False``
            use :meth:`~.queuing.QueuingManager.stop_recording`.
        id (str or None): String representing the operation (optional)
    """
    def __init__(self, ops, control_wires, do_queue=None, id=None):
        super().__init__(ops, control_wires, wires=None, do_queue=do_queue, id=id)

    @staticmethod
    def compute_decomposition(ops, wires):
        # control_wires = wires

        states = list(itertools.product([0,1],repeat=len(control_wires)))
        decomp_ops= [
            qml.ctrl(ops[index],control_wires, control_values)
            for index, control_values
            in enumerate(states)
            ]
        return decomp_ops