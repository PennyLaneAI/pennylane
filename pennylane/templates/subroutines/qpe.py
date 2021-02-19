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
Contains the ``QuantumPhaseEstimation`` template.
"""
from numpy.linalg import matrix_power

import pennylane as qml
from pennylane.templates.decorator import template
from pennylane.wires import Wires


@template
def QuantumPhaseEstimation(unitary, target_wires, estimation_wires):

    target_wires = Wires(target_wires)
    estimation_wires = Wires(estimation_wires)

    if len(Wires.shared_wires([target_wires, estimation_wires])) != 0:
        raise qml.QuantumFunctionError("The target wires and estimation wires must be different")

    for i, wire in enumerate(estimation_wires):
        qml.Hadamard(wire)
        u = matrix_power(unitary, 2 ** i)
        qml.ControlledQubitUnitary(u, control_wires=wire, target_wires=target_wires)

    qml.QFT(estimation_wires).inv()
