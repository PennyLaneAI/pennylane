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
Defines `is_commuting`, an function for determining if two functions commute.
"""

import pennylane as qml
from pennylane import numpy as np
from pennylane.grouping.utils import is_pauli_word, pauli_to_binary, _wire_map_from_pauli_pair


def _pword_is_commuting(pauli_word_1, pauli_word_2, wire_map=None):
    r"""Checks if two Pauli words commute.

    To determine if two Pauli words commute, we can check the value of the
    symplectic inner product of their binary vector representations.
    For two binary vectors representing Pauli words, :math:`p_1 = [x_1, z_1]`
    and :math:`p_2 = [x_2, z_2],` the symplectic inner product is defined as
    :math:`\langle p_1, p_2 \rangle_{symp} = z_1 x_2^T + z_2 x_1^T`. If the symplectic
    product is :math:`0` they commute, while if it is :math:`1`, they don't commute.

    Args:
        pauli_word_1 (Observable): first Pauli word in commutator
        pauli_word_2 (Observable): second Pauli word in commutator
        wire_map (dict[Union[str, int], int]): dictionary containing all wire labels used in
            the Pauli word as keys, and unique integer labels as their values

    Returns:
        bool: returns True if the input Pauli commute, False otherwise

    **Example**

    >>> wire_map = {'a' : 0, 'b' : 1, 'c' : 2}
    >>> pauli_word_1 = qml.PauliX('a') @ qml.PauliY('b')
    >>> pauli_word_2 = qml.PauliZ('a') @ qml.PauliZ('c')
    >>> is_commuting(pauli_word_1, pauli_word_2, wire_map=wire_map)
    False
    """

    if wire_map is None:
        wire_map = _wire_map_from_pauli_pair(pauli_word_1, pauli_word_2)

    n_qubits = len(wire_map)

    pauli_vec_1 = pauli_to_binary(pauli_word_1, n_qubits=n_qubits, wire_map=wire_map)
    pauli_vec_2 = pauli_to_binary(pauli_word_2, n_qubits=n_qubits, wire_map=wire_map)

    x1, z1 = pauli_vec_1[:n_qubits], pauli_vec_1[n_qubits:]
    x2, z2 = pauli_vec_2[:n_qubits], pauli_vec_2[n_qubits:]

    return (np.dot(z1, x2) + np.dot(z2, x1)) % 2 == 0


def _get_target_name(op):
    """Get the name for the target operation. If the operation is not controlled, this is
    simplify the operation's name.
    """
    _control_base_map = {
        "CNOT": "PauliX",
        "CZ": "PauliZ",
        "CY": "PauliY",
        "CSWAP": "SWAP",
        "Toffoli": "PauliX",
        "ControlledPhaseShift": "PhaseShift",
        "CRX": "RX",
        "CRY": "RY",
        "CRZ": "RZ",
        "CRot": "Rot",
        "MultiControlledX": "PauliX",
    }
    if op.name in _control_base_map:
        return _control_base_map[op.name]
    if isinstance(op, qml.ops.op_math.Controlled):  # pylint: disable=no-member
        return op.base.name
    return op.name


def _check_mat_commutation(op1, op2):
    """Uses matrices and matrix multiplication to determine whether op1 and op2 commute.

    ``op1`` and ``op2`` must be on the same wires.
    """
    op1_mat = op1.matrix()
    op2_mat = op2.matrix()

    mat_12 = np.matmul(op1_mat, op2_mat)
    mat_21 = np.matmul(op2_mat, op1_mat)

    return qml.math.allclose(mat_12, mat_21)


def _create_commute_function():
    """This function constructs the ``_commutes`` helper utility function while using closure
    to hide the ``commutation_map`` data away from the global scope of the file.
    This function only needs to be called a single time.
    Returns:
        function
    """
    pauliz_group = {
        "PauliZ",
        "ctrl",
        "S",
        "Adjoint(S)",
        "T",
        "Adjoint(T)",
        "RZ",
        "PhaseShift",
        "MultiRZ",
        "Identity",
        "U1",
        "IsingZZ",
        "S.inv",
        "T.inv",
    }
    swap_group = {"SWAP", "ISWAP", "SISWAP", "Identity", "Adjoint(ISWAP)", "Adjoint(SISWAP)"}
    paulix_group = {"PauliX", "SX", "RX", "Identity", "IsingXX", "SX.inv", "Adjoint(SX)"}
    pauliy_group = {"PauliY", "RY", "Identity", "IsingYY"}

    commutation_map = {}
    for group in [paulix_group, pauliy_group, pauliz_group, swap_group]:
        for op in group:
            commutation_map[op] = group

    identity_only = {"Hadamard", "U2", "U3", "Rot"}
    for op in identity_only:
        commutation_map[op] = {"Identity", op}

    commutation_map["Identity"] = pauliz_group.union(
        swap_group, paulix_group, pauliy_group, identity_only
    )

    def commutes_inner(op_name1, op_name2):
        """Determine whether or not two operations commute.

        Relies on ``commutation_map`` from the enclosing namespace of ``_create_commute_function``.

        Args:
            op_name1 (str): name of one operation
            op_name2 (str): name of the second operation

        Returns:
            Bool

        """
        return op_name1 in commutation_map[op_name2]

    return commutes_inner


_commutes = _create_commute_function()


def intersection(wires1, wires2):
    r"""Check if two sets of wires intersect.

    Args:
        wires1 (pennylane.wires.Wires): First set of wires.
        wires2 (pennylane.wires.Wires: Second set of wires.

    Returns:
         bool: True if the two sets of wires are not disjoint and False if disjoint.
    """
    return len(qml.wires.Wires.shared_wires([wires1, wires2])) != 0


def check_commutation_two_non_simplified_crot(operation1, operation2):
    r"""Check commutation for two CRot that were not simplified.

    Args:
        operation1 (pennylane.Operation): First operation.
        operation2 (pennylane.Operation): Second operation.

    Returns:
         Bool: True if commutation, False otherwise.
    """
    # Two non simplified CRot
    target_wires_1 = qml.wires.Wires(
        [w for w in operation1.wires if w not in operation1.control_wires]
    )
    target_wires_2 = qml.wires.Wires(
        [w for w in operation2.wires if w not in operation2.control_wires]
    )

    control_control = intersection(operation1.control_wires, operation2.control_wires)
    target_target = intersection(target_wires_1, target_wires_2)

    if control_control:
        if target_target:
            return _check_mat_commutation(operation1, operation2)
        # control_control and not target_target
        return True

    if target_target:
        return _check_mat_commutation(
            qml.Rot(*operation1.data, wires=operation1.wires[1]),
            qml.Rot(*operation2.data, wires=operation2.wires[1]),
        )
    return False


def check_commutation_two_non_simplified_rotations(operation1, operation2):
    r"""Check that the operations are two non simplified operations. If it is the case, then it checks commutation
    for two rotations that were not simplified.

    Only allowed ops are `U2`, `U3`, `Rot`, `CRot`.

    Args:
        operation1 (pennylane.Operation): First operation.
        operation2 (pennylane.Operation): Second operation.

    Returns:
         Bool: True if commutation, False otherwise, None if not two rotations.
    """

    target_wires_1 = qml.wires.Wires(
        [w for w in operation1.wires if w not in operation1.control_wires]
    )
    target_wires_2 = qml.wires.Wires(
        [w for w in operation2.wires if w not in operation2.control_wires]
    )

    if operation1.name == "CRot":
        if intersection(target_wires_1, operation2.wires):
            op1_rot = qml.Rot(*operation1.data, wires=target_wires_1)
            return _check_mat_commutation(op1_rot, operation2)
        return _commutes(operation2.name, "ctrl")

    if operation2.name == "CRot":
        if intersection(target_wires_2, operation1.wires):
            op2_rot = qml.Rot(*operation2.data, wires=target_wires_2)
            return _check_mat_commutation(op2_rot, operation1)
        return _commutes(operation1.name, "ctrl")

    return _check_mat_commutation(operation1, operation2)


unsupported_operations = [
    "PauliRot",
    "QubitDensityMatrix",
    "CVNeuralNetLayers",
    "ApproxTimeEvolution",
    "ArbitraryUnitary",
    "CommutingEvolution",
    "DisplacementEmbedding",
    "SqueezingEmbedding",
    "Prod",
    "Sum",
]
non_commuting_operations = [
    # State Prep
    "QubitStateVector",
    "BasisState",
    # Templates
    "ArbitraryStatePreparation",
    "BasisStatePreparation",
    "MottonenStatePreparation",
    "QubitCarry",
    "QubitSum",
    "SingleExcitation",
    "SingleExcitationMinus",
    "SingleExcitationPlus",
    "DoubleExcitation",
    "DoubleExcitationPlus",
    "DoubleExcitationMinus",
    "BasicEntanglerLayers",
    "GateFabric",
    "ParticleConservingU1",
    "ParticleConservingU2",
    "RandomLayers",
    "SimplifiedTwoDesign",
    "StronglyEntanglingLayers",
    "AllSinglesDoubles",
    "FermionicDoubleExcitation",
    "FermionicSingleExcitation",
    "Grover",
    "kUpCCGSD",
    "Permute",
    "QFT",
    "QuantumMonteCarlo",
    "QuantumPhaseEstimation",
    "UCCSD",
    "MPS",
    "TTN",
    "AmplitudeEmbedding",
    "AngleEmbedding",
    "BasisEmbedding",
    "IQPEmbedding",
    "QAOAEmbedding",
    # utility ops
    "Barrier",
    "WireCut",
    "Snapshot",
]


def is_commuting(operation1, operation2, wire_map=None):
    r"""Check if two operations are commuting using a lookup table.

    A lookup table is used to check the commutation between the
    controlled, targeted part of operation 1 with the controlled, targeted part of operation 2.

    .. note::

        Most qubit-based PennyLane operations are supported --- CV operations
        are not supported at this time.

        Unsupported qubit-based operations include:

        :class:`~.PauliRot`, :class:`~.QubitDensityMatrix`, :class:`~.CVNeuralNetLayers`,
        :class:`~.ApproxTimeEvolution`, :class:`~.ArbitraryUnitary`, :class:`~.CommutingEvolution`,
        :class:`~.DisplacementEmbedding` and :class:`~.SqueezingEmbedding`.

    Args:
        operation1 (.Operation): A first quantum operation.
        operation2 (.Operation): A second quantum operation.
        wire_map (dict[Union[str, int], int]): dictionary for Pauli word commutation containing all
            wire labels used in the Pauli word as keys, and unique integer labels as their values

    Returns:
         bool: True if the operations commute, False otherwise.

    **Example**

    >>> qml.is_commuting(qml.PauliX(wires=0), qml.PauliZ(wires=0))
    False
    """
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-return-statements

    if operation1.name in unsupported_operations or isinstance(
        operation1, (qml.operation.CVOperation, qml.operation.Channel)
    ):
        raise qml.QuantumFunctionError(f"Operation {operation1.name} not supported.")

    if operation2.name in unsupported_operations or isinstance(
        operation2, (qml.operation.CVOperation, qml.operation.Channel)
    ):
        raise qml.QuantumFunctionError(f"Operation {operation2.name} not supported.")

    if is_pauli_word(operation1) and is_pauli_word(operation2):
        return _pword_is_commuting(operation1, operation2, wire_map)

    # aside from Pauli words, Tensor commutation evaluation is not supported
    if isinstance(operation1, qml.operation.Tensor) or isinstance(operation2, qml.operation.Tensor):
        # pylint: disable=raise-missing-from
        raise qml.QuantumFunctionError("Tensor operations are not supported.")

    # operations are disjoints
    if not intersection(operation1.wires, operation2.wires):
        return True

    # Simplify the rotations if possible
    with qml.queuing.stop_recording():
        operation1 = qml.simplify(operation1)
        operation2 = qml.simplify(operation2)

    # Operation is in the non commuting list
    if operation1.name in non_commuting_operations or operation2.name in non_commuting_operations:
        return False

    # Two CRot that cannot be simplified
    if operation1.name == "CRot" and operation2.name == "CRot":
        return check_commutation_two_non_simplified_crot(operation1, operation2)

    if "Identity" in (operation1.name, operation2.name):
        return True

    # Check if operations are non simplified rotations and return commutation if it is the case.
    op_set = {"U2", "U3", "Rot", "CRot"}
    if operation1.name in op_set and operation2.name in op_set:
        return check_commutation_two_non_simplified_rotations(operation1, operation2)

    ctrl_base_1 = _get_target_name(operation1)
    ctrl_base_2 = _get_target_name(operation2)

    op1_control_wires = getattr(operation1, "control_wires", {})
    op2_control_wires = getattr(operation2, "control_wires", {})

    target_wires_1 = qml.wires.Wires([w for w in operation1.wires if w not in op1_control_wires])
    target_wires_2 = qml.wires.Wires([w for w in operation2.wires if w not in op2_control_wires])

    if intersection(target_wires_1, target_wires_2) and not _commutes(ctrl_base_1, ctrl_base_2):
        return False

    if intersection(target_wires_1, op2_control_wires) and not _commutes("ctrl", ctrl_base_1):
        return False

    if intersection(target_wires_2, op1_control_wires) and not _commutes("ctrl", ctrl_base_2):
        return False

    return True
