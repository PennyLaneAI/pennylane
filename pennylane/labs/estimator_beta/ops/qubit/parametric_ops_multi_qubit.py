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
r"""Resource operators for parametric multi qubit operations."""

import pennylane.labs.estimator_beta as qre
from pennylane.estimator.resource_operator import GateCount, resource_rep

# pylint: disable = unused-argument

PAULI_ROT_SPECIAL_CASES = {
    "X": lambda eps: [GateCount(qre.resource_rep(qre.RX, {"precision": eps}))],
    "Y": lambda eps: [GateCount(qre.resource_rep(qre.RY, {"precision": eps}))],
    "Z": lambda eps: [GateCount(qre.resource_rep(qre.RZ, {"precision": eps}))],
    "XX": lambda eps: [
        GateCount(qre.resource_rep(qre.RX, {"precision": eps})),
        GateCount(qre.resource_rep(qre.CNOT), count=2),
    ],
    "YY": lambda eps: [
        GateCount(qre.resource_rep(qre.RY, {"precision": eps})),
        GateCount(qre.resource_rep(qre.CY), count=2),
    ],
}


def pauliRot_controlled_resource_decomp(
    num_ctrl_wires: int,
    num_zero_ctrl: int,
    target_resource_params: dict,
) -> list[GateCount]:
    r"""Returns a list representing the resources for a controlled version of the :class:`~pennylane.estimator.ops.qubit.PauliRot` operator.

    Args:
        num_ctrl_wires (int): the number of qubits the operation is controlled on
        num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
        target_resource_params (dict): A dictionary containing the resource parameters of the target operator

    Resources:

        The resources are computed based on Section VIII (Figures 3 and 4) of
        `The Bravyi-Kitaev transformation for quantum computation of electronic structure
        <https://arxiv.org/abs/1208.5986>`_, in combination with the following identities:

            When the :code:`pauli_string` is a single Pauli operator (:code:`X, Y, Z, Identity`)
            the cost is the associated controlled single qubit rotation gate: (:code:`CRX`,
            :code:`CRY`, :code:`CRZ`, controlled-\ :code:`GlobalPhase`).

            The resources are derived from the following identity. If an operation :math:`\hat{A}`
            can be expressed as :math:`\hat{A} \ = \ \hat{U} \cdot \hat{B} \cdot \hat{U}^{\dagger}`
            then the controlled operation :math:`C\hat{A}` can be expressed as:

            .. math:: C\hat{A} \ = \ \hat{U} \cdot C\hat{B} \cdot \hat{U}^{\dagger}

            Specifically, the resources are one multi-controlled RZ-gate and a cascade of
            :math:`2 \times (n - 1)` :code:`CNOT` gates where :math:`n` is the number of qubits
            the gate acts on. Additionally, for each :code:`X` gate in the Pauli word we conjugate by
            a pair of :code:`Hadamard` gates, and for each :code:`Y` gate in the Pauli word
            we conjugate by a pair of :code:`Hadamard` and a pair of :code:`S` gates.

            if the :code:`pauli_string` is :code:`XX`, :code:`YY` or :code:`ZZ` the cost is a multi-controlled version of the associated rotation gate
            (:code:`CRX`, :code:`CRY`, :code:`CRZ` respectively) and 2 :code:`CNOT` gates.

    Returns:
        list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
        where each object represents a specific quantum gate and the number of times it appears
        in the decomposition.

    """
    pauli_string = target_resource_params["pauli_string"]
    precision = target_resource_params["precision"]

    if (set(pauli_string) == {"I"}) or (len(pauli_string) == 0):
        ctrl_gp = qre.Controlled.resource_rep(
            qre.resource_rep(qre.GlobalPhase),
            num_ctrl_wires,
            num_zero_ctrl,
        )
        return [GateCount(ctrl_gp)]

    # Special cases:
    if pauli_string in PAULI_ROT_SPECIAL_CASES:
        gate_list = []
        base_resources = PAULI_ROT_SPECIAL_CASES[pauli_string](eps=precision)
        for gate_count in base_resources:
            if gate_count.gate.name in ["RX", "RY", "RZ"]:
                gate_list.append(
                    GateCount(
                        resource_rep(
                            qre.Controlled,
                            {
                                "base_cmpr_op": gate_count.gate,
                                "num_ctrl_wires": num_ctrl_wires,
                                "num_zero_ctrl": num_zero_ctrl,
                            },
                        ),
                        count=gate_count.count,
                    )
                )
            else:
                gate_list.append(gate_count)
        return gate_list

    active_wires = len(pauli_string.replace("I", ""))

    h = qre.Hadamard.resource_rep()
    s = qre.S.resource_rep()
    crz = qre.Controlled.resource_rep(
        qre.resource_rep(qre.RZ, {"precision": precision}),
        num_ctrl_wires,
        num_zero_ctrl,
    )
    s_dagg = qre.resource_rep(
        qre.Adjoint,
        {"base_cmpr_op": qre.resource_rep(qre.S)},
    )
    cnot = qre.CNOT.resource_rep()

    h_count = 0
    s_count = 0

    for gate in pauli_string:
        if gate == "X":
            h_count += 1
        if gate == "Y":
            h_count += 1
            s_count += 1

    gate_types = []
    if h_count:
        gate_types.append(GateCount(h, 2 * h_count))

    if s_count:
        gate_types.append(GateCount(s, s_count))
        gate_types.append(GateCount(s_dagg, s_count))

    gate_types.append(GateCount(crz))
    gate_types.append(GateCount(cnot, 2 * (active_wires - 1)))

    return gate_types
