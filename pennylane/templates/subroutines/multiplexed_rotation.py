# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Contains the MultiplexedRotation template.
"""

import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.templates.state_preparations.mottonen import  _apply_uniform_rotation_dagger


class MultiplexedRotation(Operation):
    r"""Applies the MultiplexedRotation operator.

    This operator applies a sequence of controlled rotations to a target qubit.
    The rotations are selected based on the values encoded in the control qubits.
    Its definition is given by:

    .. math::

       \sum_i | i \rangle \langle i | \otimes R(\alpha_i)

    Here, :math:`| i \rangle` refers to the computational basis states of the control register,
    and :math:`R(\alpha_i)` denotes a unitary rotation applied to the target qubit,
    parametrized by :math:`\alpha_i`.

    For more details, see `Möttönen and Vartiainen (2005), Fig 7a<https://arxiv.org/abs/quant-ph/0504100>`_.

    .. seealso:: :class:`~.Select`.

    Args:
        angles (tensor_like): The rotation angles to be applied.
        control_wires (Sequence[int]): The control qubits used to select the rotation.
        target_wire (Sequence[int]): The wire where the rotations are applied.
        rot_axis (str): The axis about which the rotation is performed.
        It can take the value `X`, `Y` or `Z`. Default is `Z`
    """

    num_wires = AnyWires
    grad_method = None
    ndim_params = (1,)

    def __init__(self, angles, control_wires, target_wire, rot_axis = "Z", id=None):

        self.hyperparameters["control_wires"] = qml.wires.Wires(control_wires)
        self.hyperparameters["target_wire"] = qml.wires.Wires(target_wire)
        self.hyperparameters["rot_axis"] = rot_axis

        all_wires = self.hyperparameters["control_wires"] + self.hyperparameters["target_wire"]
        super().__init__(angles, wires=all_wires, id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(angles, **kwargs):  # pylint: disable=arguments-differ
        r"""

        """

        rot_axis = kwargs["rot_axis"]
        control_wires = kwargs["control_wires"]
        target_wire = kwargs["target_wire"]

        op_list = []

        if rot_axis == "X":
            op_list.append(qml.Hadamard(target_wire))
        elif rot_axis == "Y":
            op_list.append(qml.adjoint(qml.S(target_wire)))
            op_list.append(qml.Hadamard(target_wire))

        op_list.extend(_apply_uniform_rotation_dagger(qml.RZ, angles, control_wires[::-1], target_wire))

        if rot_axis == "X":
            op_list.append(qml.Hadamard(target_wire))
        elif rot_axis == "Y":
            op_list.append(qml.Hadamard(target_wire))
            op_list.append(qml.S(target_wire))

        return op_list
