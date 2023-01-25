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
This submodule defines functions to decompose controlled operations
"""

import pennylane as qml
from pennylane.operation import Operator
from pennylane.wires import Wires

# ctrl_ops_dict = {
#     "CNOT": qml.PauliX,
#     "CZ": qml.PauliZ,
#     "CY": qml.PauliY,
#     "CH": qml.Hadamard,
#     "CRX": qml.RX,
#     "CRZ": qml.RZ,
#     "CRY": qml.RY,
#     "CRot": qml.Rot,
#     "ControlledPhaseShift": qml.PhaseShift,
#     # Only focus on controlled operations with one control qubit for now
#     # "CCZ": qml.PauliZ,
#     # "Toffoli": qml.PauliX,
#     # "MultiControlledX": qml.PauliX,
# }


def ctrl_decomp_zyz(target_operation: Operator, control_wires: Wires):
    """Decompose a controlled single-qubit operation given the target operation

    This function decomposes the controlled version of a single-qubit operation
    given a target operation using the decomposition defined in Lemma 4.3 and 5.1
    of `Barenco et al. (1995) <https://arxiv.org/abs/quant-ph/9503016>`_.

    Args:
        target_operation (~.operation.Operator): the target operation to decompose
        control_wires (~.wires.Wires): the control wires of the operation

    Returns:
        list[Operation]: the decomposed operations
    """
    target_wires = target_operation.wires
    try:
        phi, theta, omega = target_operation.single_qubit_rot_angles()
    except NotImplementedError:
        zyz_decomp = qml.transforms.zyz_decomposition(
            qml.matrix(target_operation), target_operation.wires
        )[0]
        phi, theta, omega = zyz_decomp.single_qubit_rot_angles()

    return [
        qml.RZ(phi, wires=target_wires),
        qml.RY(theta / 2, wires=target_wires),
        qml.MultiControlledX(wires=control_wires + target_wires),
        qml.RY(-theta / 2, wires=target_wires),
        qml.RZ(-(phi + omega) / 2, wires=target_wires),
        qml.MultiControlledX(wires=control_wires + target_wires),
        qml.RZ((omega - phi) / 2, wires=target_wires),
    ]
