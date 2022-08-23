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
    if op.name == "ControlledOperation":
        return op.control_base
    return op.name


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

    no_commutation = {"Barrier", "WireCut", "QubitStateVector", "BasisState"}
    for op in no_commutation:
        commutation_map[op] = {}

    commutation_map["Identity"] = {
        "Hadamard",
        "PauliX",
        "PauliY",
        "PauliZ",
        "SWAP",
        "ctrl",
        "S",
        "T",
        "SX",
        "ISWAP",
        "SISWAP",
        "RX",
        "RY",
        "RZ",
        "PhaseShift",
        "Rot",
        "MultiRZ",
        "Identity",
        "U1",
        "U2",
        "U3",
        "IsingXX",
        "IsingYY",
        "IsingZZ",
        "ECR",
    }

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

    if control_control and target_target:
        return np.all(
            np.allclose(
                np.matmul(operation1.matrix(), operation2.matrix()),
                np.matmul(operation2.matrix(), operation1.matrix()),
            )
        )

    if control_control and not target_target:
        return True

    if not control_control and target_target:
        return np.all(
            np.allclose(
                np.matmul(
                    qml.Rot(*operation1.data, wires=operation1.wires[1]).matrix(),
                    qml.Rot(*operation2.data, wires=operation2.wires[1]).matrix(),
                ),
                np.matmul(
                    qml.Rot(*operation2.data, wires=operation2.wires[1]).matrix(),
                    qml.Rot(*operation1.data, wires=operation1.wires[1]).matrix(),
                ),
            )
        )
    return False


def check_simplify_identity_commutation(operation1, operation2):
    r"""First check that a parametric operation can be simplified to the identity operator, if it is the case then
     return the commutation relation with the second operation. If simplification is not possible, it returns None.

    Args:
        operation1 (pennylane.Operation): First operation.
        operation2 (pennylane.Operation): Second operation.

    Returns:
         Bool: True if commutation, False non-commmutation and None if not possible to simplify.
    """
    if operation1.data and operation1.name != "U2":
        all_zeros = np.allclose(np.mod(operation1.data, 2 * np.pi), 0)
        if all_zeros:
            if operation2.name not in ["Barrier", "WireCut"]:
                return True
            return False
    return None


def check_commutation_two_non_simplified_rotations(operation1, operation2):
    r"""Check that the operations are two non simplified operations. If it is the case, then it checks commutation
    for two rotations that were not simplified.

    Args:
        operation1 (pennylane.Operation): First operation.
        operation2 (pennylane.Operation): Second operation.

    Returns:
         Bool: True if commutation, False otherwise, None if not two rotations.
    """
    # Two non simplified rotations
    if (operation1.name in ["U2", "U3", "Rot", "CRot"]) and (
        operation2.name in ["U2", "U3", "Rot", "CRot"]
    ):
        target_wires_1 = qml.wires.Wires(
            [w for w in operation1.wires if w not in operation1.control_wires]
        )
        target_wires_2 = qml.wires.Wires(
            [w for w in operation2.wires if w not in operation2.control_wires]
        )

        if operation1.name == "CRot":
            if not intersection(target_wires_1, operation2.wires):
                return _commutes(operation2.name, "ctrl")
            return np.all(
                np.allclose(
                    np.matmul(
                        qml.Rot(*operation1.data, wires=target_wires_1).matrix(),
                        operation2.matrix(),
                    ),
                    np.matmul(
                        operation2.matrix(),
                        qml.Rot(*operation1.data, wires=target_wires_1).matrix(),
                    ),
                )
            )

        if operation2.name == "CRot":
            if not intersection(target_wires_2, operation1.wires):
                return _commutes(operation1.name, "ctrl")
            return np.all(
                np.allclose(
                    np.matmul(
                        qml.Rot(*operation2.data, wires=target_wires_2).matrix(),
                        operation1.matrix(),
                    ),
                    np.matmul(
                        operation1.matrix(),
                        qml.Rot(*operation2.data, wires=target_wires_2).matrix(),
                    ),
                )
            )

        return np.all(
            np.allclose(
                np.matmul(
                    operation1.matrix(),
                    operation2.matrix(),
                ),
                np.matmul(
                    operation2.matrix(),
                    operation1.matrix(),
                ),
            )
        )
    return None


unsupported_operations = [
    "PauliRot",
    "QubitDensityMatrix",
    "CVNeuralNetLayers",
    "ApproxTimeEvolution",
    "ArbitraryUnitary",
    "CommutingEvolution",
    "DisplacementEmbedding",
    "SqueezingEmbedding",
]
non_commuting_operations = [
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
]


def is_commuting(operation1, operation2):
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

    if operation1.name == "ControlledOperation" and operation1.control_base == "MultipleTargets":
        raise qml.QuantumFunctionError(f"{operation1.control_base} controlled is not supported.")

    if operation2.name == "ControlledOperation" and operation2.control_base == "MultipleTargets":
        raise qml.QuantumFunctionError(f"{operation2.control_base} controlled is not supported.")

    # Case 1 operations are disjoints
    if not intersection(operation1.wires, operation2.wires):
        return True

    # Simplify the rotations if possible
    operation1 = qml.simplify(operation1)
    operation2 = qml.simplify(operation2)

    # Two CRot that cannot be simplified
    if operation1.name == "CRot" and operation2.name == "CRot":
        return check_commutation_two_non_simplified_crot(operation1, operation2)

    # Parametric operation might implement the identity operator
    commutation_identity_simplification_1 = check_simplify_identity_commutation(
        operation1, operation2
    )
    if commutation_identity_simplification_1 is not None:
        return commutation_identity_simplification_1

    # pylint:disable=arguments-out-of-order
    commutation_identity_simplification_2 = check_simplify_identity_commutation(
        operation2, operation1
    )
    if commutation_identity_simplification_2 is not None:
        return commutation_identity_simplification_2

    # Operation is in the non commuting list
    if operation1.name in non_commuting_operations or operation2.name in non_commuting_operations:
        return False

    # Check if operations are non simplified rotations and return commutation if it is the case.
    two_non_simplified_rot = check_commutation_two_non_simplified_rotations(operation1, operation2)
    if two_non_simplified_rot is not None:
        return two_non_simplified_rot

    control_base_1 = _get_target_name(operation1)
    control_base_2 = _get_target_name(operation2)

    op1_control_wires = getattr(operation1, "control_wires", {})
    op2_control_wires = getattr(operation2, "control_wires", {})

    target_wires_1 = qml.wires.Wires([w for w in operation1.wires if w not in op1_control_wires])
    target_wires_2 = qml.wires.Wires([w for w in operation2.wires if w not in op2_control_wires])

    ops_commute = True
    if intersection(target_wires_1, target_wires_2):
        ops_commute = ops_commute and _commutes(control_base_1, control_base_2)

    if intersection(target_wires_1, op2_control_wires):
        ops_commute = ops_commute and _commutes("ctrl", control_base_1)

    if intersection(target_wires_2, op1_control_wires):
        ops_commute = ops_commute and _commutes("ctrl", control_base_2)

    return ops_commute
