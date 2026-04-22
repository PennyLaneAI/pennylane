# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Resource operators for PennyLane subroutine templates."""

import pennylane.labs.estimator_beta as qre
from pennylane.estimator import GateCount, resource_rep

# pylint: disable=unused-argument


def selectpaulirot_controlled_resource_decomp(
    num_ctrl_wires: int, num_zero_ctrl: int, target_resource_params: dict
) -> list[GateCount]:
    r"""Returns a list representing the resources of the controlled version of the :class:`~pennylane.estimator.templates.SelectPauliRot` operator.
    Each object in the list
    represents a gate and the number of times it occurs in the circuit.

    Args:
        num_ctrl_wires (int): the number of qubits the operation is controlled on
        num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
        target_resource_params (dict): A dictionary containing the resource parameters
            of the target operator.

    Resources:
        The resources are obtained from the construction scheme given in `Möttönen and Vartiainen
        (2005), Fig 7a <https://arxiv.org/abs/quant-ph/0504100>`_. Specifically, the resources
        for an :math:`n` qubit unitary are given as :math:`2^{n}` instances of the :code:`CNOT`
        gate and :math:`2^{n}` instances of the controlled single qubit rotation gate (:code:`RX`,
        :code:`RY` or :code:`RZ`) depending on the :code:`rot_axis`.

    Returns:
        list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
        represents a specific quantum gate and the number of times it appears
        in the decomposition.
    """

    num_ctrl_wires_base = target_resource_params["num_ctrl_wires"]
    rot_axis = target_resource_params["rot_axis"]
    precision = target_resource_params["precision"]

    rotation_gate_map = {
        "X": qre.RX,
        "Y": qre.RY,
        "Z": qre.RZ,
    }
    gate_lst = []

    gate = resource_rep(
        qre.Controlled,
        {
            "base_cmpr_op": resource_rep(rotation_gate_map[rot_axis], {"precision": precision}),
            "num_ctrl_wires": num_ctrl_wires,
            "num_zero_ctrl": num_zero_ctrl,
        },
    )
    cnot = resource_rep(qre.CNOT)

    gate_lst.append(GateCount(gate, 2**num_ctrl_wires_base))
    gate_lst.append(GateCount(cnot, 2**num_ctrl_wires_base))

    return gate_lst
