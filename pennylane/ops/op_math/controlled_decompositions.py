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
    """Decompose the controlled version of a target single-qubit operation

    This function decomposes a controlled single-qubit target operation using the
    decomposition defined in section 5 of
    `Barenco et al. (1995) <https://arxiv.org/abs/quant-ph/9503016>`_.

    .. warning:: This method will add a global phase for target operations that do not
        belong to the SU(2) group.

    Args:
        target_operation (~.operation.Operator): the target operation to decompose
        control_wires (~.wires.Wires): the control wires of the operation

    Returns:
        list[Operation]: the decomposed operations

    Raises:
        ValueError: if ``target_operation`` is not a single-qubit operation

    **Example**

    We can create a controlled operation using `qml.ctrl`, or by creating the
    decomposed controlled version of using `qml.ctrl_decomp_zyz`.

    .. code-block:: python

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def expected_circuit(op):
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            qml.ctrl(op, [0,1])
            return qml.probs()

        @qml.qnode(dev)
        def decomp_circuit(op):
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            qml.ops.ctrl_decomp_zyz(op, [0,1])
            return qml.probs()

    Measurements on both circuits will give us the same results:

    >>> op = qml.RX(0.123, wires=2)
    >>> expected_circuit(op)
    tensor([0.25      , 0.        , 0.25      , 0.        , 0.25      ,
        0.        , 0.24905563, 0.00094437], requires_grad=True)
    >>> decomp_circuit(op)
    tensor([0.25      , 0.        , 0.25      , 0.        , 0.25      ,
        0.        , 0.24905563, 0.00094437], requires_grad=True)

    """
    if len(target_operation.wires) != 1:
        raise ValueError(
            "The target operation must be a single-qubit operation, instead "
            f"got {target_operation.__class__.__name__}."
        )

    target_wire = target_operation.wires

    try:
        phi, theta, omega = target_operation.single_qubit_rot_angles()
    except NotImplementedError:
        with qml.QueuingManager.stop_recording():
            zyz_decomp = qml.transforms.zyz_decomposition(
                qml.matrix(target_operation), target_wire
            )[0]
        phi, theta, omega = zyz_decomp.single_qubit_rot_angles()

    decomp = []

    if not qml.math.isclose(phi, 0.0, atol=1e-8, rtol=0):
        decomp.append(qml.RZ(phi, wires=target_wire))
    if not qml.math.isclose(theta / 2, 0.0, atol=1e-8, rtol=0):
        decomp.extend(
            [
                qml.RY(theta / 2, wires=target_wire),
                qml.MultiControlledX(wires=control_wires + target_wire),
                qml.RY(-theta / 2, wires=target_wire),
            ]
        )
    else:
        decomp.append(qml.MultiControlledX(wires=control_wires + target_wire))
    if not qml.math.isclose(-(phi + omega) / 2, 0.0, atol=1e-6, rtol=0):
        decomp.append(qml.RZ(-(phi + omega) / 2, wires=target_wire))
    decomp.append(qml.MultiControlledX(wires=control_wires + target_wire))
    if not qml.math.isclose((omega - phi) / 2, 0.0, atol=1e-8, rtol=0):
        decomp.append(qml.RZ((omega - phi) / 2, wires=target_wire))

    return decomp
