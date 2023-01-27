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
This submodule defines functions to decompose controlled operations
"""

import pennylane as qml
from pennylane.operation import Operator
from pennylane.wires import Wires


def ctrl_decomp_zyz(target_operation: Operator, control_wires: Wires):
    """Decompose a controlled single-qubit operation given the target operation

    This function decomposes the controlled version of a single-qubit operation
    given a target operation using the decomposition defined in section 5 of
    `Barenco et al. (1995) <https://arxiv.org/abs/quant-ph/9503016>`_.

    .. note:: This method will add a global phase for target operations that do not
    belong to the SU(2) group.

    Args:
        target_operation (~.operation.Operator): the target operation to decompose
        control_wires (~.wires.Wires): the control wires of the operation

    Returns:
        list[Operation]: the decomposed operations

    Raises:
        ValueError: if `target_operation` is not a single-qubit operation

    """
    # TODO: Add support for general unitaries, not just SU(2) matrices
    if len(target_operation.wires) != 1:
        raise ValueError(
            "The target operation must be a single-qubit operation, instead "
            f"got {target_operation.__class__.__name__}."
        )

    target_wire = target_operation.wires

    if len(control_wires) > 2:
        control_gate = qml.MultiControlledX(wires=control_wires + target_wire)
    elif len(control_wires) == 2:
        control_gate = qml.Toffoli(wires=control_wires + target_wire)
    else:
        control_gate = qml.CNOT(wires=control_wires + target_wire)

    try:
        phi, theta, omega = target_operation.single_qubit_rot_angles()
    except NotImplementedError:
        zyz_decomp = qml.transforms.zyz_decomposition(qml.matrix(target_operation), target_wire)[0]
        phi, theta, omega = zyz_decomp.single_qubit_rot_angles()

    return [
        qml.RZ(phi, wires=target_wire),
        qml.RY(theta / 2, wires=target_wire),
        control_gate,
        qml.RY(-theta / 2, wires=target_wire),
        qml.RZ(-(phi + omega) / 2, wires=target_wire),
        control_gate,
        qml.RZ((omega - phi) / 2, wires=target_wire),
    ]
