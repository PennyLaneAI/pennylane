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
A transform to obtain the commutation DAG of a quantum circuit.
"""
import heapq
from functools import wraps
from collections import OrderedDict
from networkx.drawing.nx_pydot import to_pydot

import networkx as nx
import pennylane as qml
import pennylane.numpy as np
from pennylane.wires import Wires


def commutation_dag(circuit):
    r"""Construct the pairwise-commutation DAG (directed acyclic graph) representation of a quantum circuit.

    In the DAG, each node represents a quantum operation, and edges represent
    non-commutation between two operations.

    This transform takes into account that not all
    operations can be moved next to each other by pairwise commutation.

    Args:
        circuit (pennylane.QNode, .QuantumTape, or Callable): A quantum node, tape,
            or function that applies quantum operations.

    Returns:
         function: Function which accepts the same arguments as the :class:`qml.QNode`, :class:`qml.tape.QuantumTape`
         or quantum function. When called, this function will return the commutation DAG representation of the circuit.

    **Example**

    .. code-block:: python

        def circuit(x, y, z):
            qml.RX(x, wires=0)
            qml.RX(y, wires=0)
            qml.CNOT(wires=[1, 2])
            qml.RY(y, wires=1)
            qml.Hadamard(wires=2)
            qml.CRZ(z, wires=[2, 0])
            qml.RY(-y, wires=1)
            return qml.expval(qml.PauliZ(0))

    The commutation dag can be returned by using the following code:

    >>> dag_fn = commutation_dag(circuit)
    >>> dag = dag_fn(np.pi / 4, np.pi / 3, np.pi / 2)

    Nodes in the commutation DAG can be accessed via the :meth:`~.get_nodes` method, returning a list of
    the  form ``(ID, CommutationDAGNode)``:

    >>> nodes = dag.get_nodes()
    [(0, <pennylane.transforms.commutation_dag.CommutationDAGNode object at 0x132b03b20>), ...]

    You can also access specific nodes (of type :class:`~.CommutationDAGNode`) by using the :meth:`~.get_node`
    method. See :class:`~.CommutationDAGNode` for a list of available
    node attributes.

    >>> second_node = dag.get_node(2)
    >>> second_node
    <pennylane.transforms.commutation_dag.CommutationDAGNode object at 0x136f8c4c0>
    >>> second_node.op
    CNOT(wires=[1, 2])
    >>> second_node.successors
    [3, 4, 5, 6]
    >>> second_node.predecessors
    []

    For more details, see:

    * Iten, R., Moyard, R., Metger, T., Sutter, D., Woerner, S.
      "Exact and practical pattern matching for quantum circuit optimization" `doi.org/10.1145/3498325
      <https://dl.acm.org/doi/abs/10.1145/3498325>`_

    """

    # pylint: disable=protected-access

    @wraps(circuit)
    def wrapper(*args, **kwargs):

        if isinstance(circuit, qml.QNode):
            # user passed a QNode, get the tape
            circuit.construct(args, kwargs)
            tape = circuit.qtape

        elif isinstance(circuit, qml.tape.QuantumTape):
            # user passed a tape
            tape = circuit

        elif callable(circuit):
            # user passed something that is callable but not a tape or qnode.
            tape = qml.transforms.make_tape(circuit)(*args, **kwargs)
            # raise exception if it is not a quantum function
            if len(tape.operations) == 0:
                raise ValueError("Function contains no quantum operation")

        else:
            raise ValueError("Input is not a tape, QNode, or quantum function")

        # Initialize DAG
        dag = CommutationDAG(tape)

        return dag

    return wrapper


# fmt: off

commutation_map = OrderedDict(
    {
        "Hadamard": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "PauliX": [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        "PauliY": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        "PauliZ": [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        "SWAP": [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "ctrl": [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        "S": [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        "T": [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        "SX": [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        "ISWAP": [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "SISWAP": [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "Barrier": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "WireCut": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "RX": [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        "RY": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        "RZ": [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        "PhaseShift": [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        "Rot": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "MultiRZ": [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        "Identity": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        "U1": [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        "U2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "U3": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "IsingXX": [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        "IsingYY": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        "IsingZZ": [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        "QubitStateVector": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "BasisState": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }
)
"""OrderedDict[str, list[int]]: Represents the commutation relations between each gate. Positions in the array are
the one defined by the position dictionary. 1 represents commutation and 0 non-commutation."""

# fmt: on


position = dict(zip(commutation_map, range(len(commutation_map))))
"""OrderedDict[str, int]: represents the index of each gates in the list of the commutation_map dictionary."""


def intersection(wires1, wires2):
    r"""Check if two sets of wires intersect.

    Args:
        wires1 (pennylane.wires.Wires): First set of wires.
        wires2 (pennylane.wires.Wires: Second set of wires.

    Returns:
         bool: True if the two sets of wires are not disjoint and False if disjoint.
    """
    return len(qml.wires.Wires.shared_wires([wires1, wires2])) != 0


def simplify_rotation(rot):
    r"""Simplify a general one qubit rotation into RX, RY and RZ rotation.

    Args:
        rot (pennylane.Rot): One qubit rotation.

    Returns:
         qml.operation: Simplified rotation if possible.
    """
    wires = rot.wires
    params = rot.parameters

    p0, p1, p2 = np.mod(params, 2 * np.pi)

    if np.allclose(p0, np.pi / 2) and np.allclose(np.mod(rot.data[2], -2 * np.pi), -np.pi / 2):
        return qml.RX(rot.data[1], wires=wires)
    if np.allclose(p0, 0) and np.allclose(p2, 0):
        return qml.RY(rot.data[1], wires=wires)
    if np.allclose(p1, 0):
        return qml.RZ(rot.data[0] + rot.data[2], wires=wires)
    if np.allclose(p0, np.pi) and np.allclose(p1, np.pi / 2) and np.allclose(p2, 0):
        return qml.Hadamard(wires=rot.wires)

    return rot


def simplify_controlled_rotation(crot):
    r"""Simplify a general one qubit controlled rotation into CRX, CRY, CRZ and CH.

    Args:
        crot (pennylane.CRot): One qubit controlled rotation.

    Returns:
         qml.operation: Simplified controlled rotation if possible.
    """
    target_wires = [w for w in crot.wires if w not in crot.control_wires]
    wires = crot.wires
    params = crot.parameters

    p0, p1, p2 = np.mod(params, 2 * np.pi)

    if np.allclose(p0, np.pi / 2) and np.allclose(np.mod(crot.data[2], -2 * np.pi), -np.pi / 2):
        return qml.CRX(crot.data[1], wires=wires)
    if np.allclose(p0, 0) and np.allclose(p2, 0):
        return qml.CRY(crot.data[1], wires=wires)
    if np.allclose(p1, 0):
        return qml.CRZ(crot.data[0] + crot.data[2], wires=wires)
    if np.allclose(p0, np.pi) and np.allclose(p1, np.pi / 2) and np.allclose(p2, 0):
        hadamard = qml.Hadamard
        return qml.ctrl(hadamard, control=crot.control_wires)(wires=target_wires)

    return crot


def simplify_u2(u2):
    r"""Simplify a u2 one qubit rotation into RX and RY rotations.

    Args:
        u2 (pennylane.U2): U2 rotation.

    Returns:
         qml.operation: Simplified rotation if possible.
    """
    wires = u2.wires

    if np.allclose(np.mod(u2.data[1], 2 * np.pi), 0) and np.allclose(
        np.mod(u2.data[0] + u2.data[1], 2 * np.pi), 0
    ):
        return qml.RY(np.pi / 2, wires=wires)
    if np.allclose(np.mod(u2.data[1], np.pi / 2), 0) and np.allclose(
        np.mod(u2.data[0] + u2.data[1], 2 * np.pi), 0
    ):
        return qml.RX(u2.data[1], wires=wires)

    return u2


def simplify_u3(u3):
    r"""Simplify a general U3 one qubit rotation into RX, RY and RZ rotation.

    Args:
        u3 (pennylane.U3): One qubit U3 rotation.

    Returns:
         qml.operation: Simplified rotation if possible.
    """
    wires = u3.wires
    params = u3.parameters

    p0, p1, p2 = np.mod(params, 2 * np.pi)

    if np.allclose(p0, 0) and not np.allclose(p1, 0) and np.allclose(p2, 0):
        return qml.PhaseShift(u3.data[1], wires=wires)
    if (
        np.allclose(p2, np.pi / 2)
        and np.allclose(np.mod(u3.data[1] + u3.data[2], 2 * np.pi), 0)
        and not np.allclose(p0, 0)
    ):
        return qml.RX(u3.data[0], wires=wires)
    if not np.allclose(p0, 0) and np.allclose(p1, 0) and np.allclose(p2, 0):
        return qml.RY(u3.data[0], wires=wires)

    return u3


def simplify(operation):
    r"""Simplify the (controlled) rotation operations :class:`~.Rot`,
    :class:`~.U2`, :class:`~.U3`, and :class:`~.CRot` into one of
    :class:`~.RX`, :class:`~.CRX`, :class:`~.RY`, :class:`~.CRY`, :class:`~.`RZ`,
    :class:`~.CZ`, :class:`~.H` and :class:`~.CH` where possible.

    Args:
        operation (.Operation): Rotation or controlled rotation.

    Returns:
         .Operation: An operation representing the simplified rotation, if possible.
         Otherwise, the original operation is returned.

    **Example**

    You can simplify rotation with certain parameters, for example:

    >>> qml.simplify(qml.Rot(np.pi / 2, 0.1, -np.pi / 2, wires=0))
    qml.RX(0.1, wires=0)

    However, not every rotation can be simplified. The original operation
    is returned if no simplification is possible.

    >>> qml.simplify(qml.Rot(0.1, 0.2, 0.3, wires=0))
    qml.Rot(0.1, 0.2, 0.3, wires=0)
    """
    if operation.name not in ["Rot", "U2", "U3", "CRot"]:
        raise qml.QuantumFunctionError(f"{operation.name} is not a Rot, U2, U3 or CRot.")

    if operation.name == "Rot":
        return simplify_rotation(operation)

    if operation.name == "U2":
        return simplify_u2(operation)

    if operation.name == "U3":
        return simplify_u3(operation)

    return simplify_controlled_rotation(operation)


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
                np.matmul(operation1.get_matrix(), operation2.get_matrix()),
                np.matmul(operation2.get_matrix(), operation1.get_matrix()),
            )
        )

    if control_control and not target_target:
        return True

    if not control_control and target_target:
        return np.all(
            np.allclose(
                np.matmul(
                    qml.Rot(*operation1.data, wires=operation1.wires[1]).get_matrix(),
                    qml.Rot(*operation2.data, wires=operation2.wires[1]).get_matrix(),
                ),
                np.matmul(
                    qml.Rot(*operation2.data, wires=operation2.wires[1]).get_matrix(),
                    qml.Rot(*operation1.data, wires=operation1.wires[1]).get_matrix(),
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
                return bool(commutation_map["ctrl"][position[operation2.name]])
            return np.all(
                np.allclose(
                    np.matmul(
                        qml.Rot(*operation1.data, wires=target_wires_1).get_matrix(),
                        operation2.get_matrix(),
                    ),
                    np.matmul(
                        operation2.get_matrix(),
                        qml.Rot(*operation1.data, wires=target_wires_1).get_matrix(),
                    ),
                )
            )

        if operation2.name == "CRot":
            if not intersection(target_wires_2, operation1.wires):
                return bool(commutation_map[operation1.name][position["ctrl"]])
            return np.all(
                np.allclose(
                    np.matmul(
                        qml.Rot(*operation2.data, wires=target_wires_2).get_matrix(),
                        operation1.get_matrix(),
                    ),
                    np.matmul(
                        operation1.get_matrix(),
                        qml.Rot(*operation2.data, wires=target_wires_2).get_matrix(),
                    ),
                )
            )

        return np.all(
            np.allclose(
                np.matmul(
                    operation1.get_matrix(),
                    operation2.get_matrix(),
                ),
                np.matmul(
                    operation2.get_matrix(),
                    operation1.get_matrix(),
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

    control_base = {
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
        "ControlledOperation": "ControlledOperation",
    }

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

    # Simplify the rotations if possible
    if operation1.name in ["U2", "U3", "Rot", "CRot"]:
        operation1 = simplify(operation1)

    if operation2.name in ["U2", "U3", "Rot", "CRot"]:
        operation2 = simplify(operation2)

    # Case 1 operations are disjoints
    if not intersection(operation1.wires, operation2.wires):
        return True

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

    # Case 2 both operations are controlled
    if control_base.get(operation1.name) and control_base.get(operation2.name):
        return _both_controlled(control_base, operation1, operation2)

    # Case 3: only operation 1 is controlled
    if control_base.get(operation1.name):
        if control_base.get(operation1.name) != "ControlledOperation":
            control_base_1 = control_base.get(operation1.name)
        else:
            control_base_1 = operation1.control_base

        target_wires_1 = qml.wires.Wires(
            [w for w in operation1.wires if w not in operation1.control_wires]
        )

        control_target = intersection(operation1.control_wires, operation2.wires)
        target_target = intersection(target_wires_1, operation2.wires)

        # Case 3.1: control and target 1 overlap with target 2
        if control_target and target_target:
            return bool(commutation_map[control_base_1][position[operation2.name]]) and bool(
                commutation_map["ctrl"][position[operation2.name]]
            )

        # Case 3.2: control operation 1 overlap with target 2
        if control_target and not target_target:
            return bool(commutation_map["ctrl"][position[operation2.name]])

        # Case 3.3: target 1 overlaps with target 2
        if not control_target and target_target:
            return bool(commutation_map[control_base_1][position[operation2.name]])

    # Case 4: only operation 2 is controlled
    if control_base.get(operation2.name):
        if control_base.get(operation2.name) != "ControlledOperation":
            control_base_2 = control_base.get(operation2.name)
        else:
            control_base_2 = operation2.control_base

        target_wires_2 = qml.wires.Wires(
            [w for w in operation2.wires if w not in operation2.control_wires]
        )

        target_control = intersection(operation1.wires, operation2.control_wires)
        target_target = intersection(operation1.wires, target_wires_2)

        # Case 4.1: control and target 2 overlap with target 1
        if target_control and target_target:
            return bool(commutation_map[operation1.name][position[control_base_2]]) and bool(
                commutation_map[operation1.name][position[control_base_2]]
            )

        # Case 4.2: control operation 2 overlap with target 1
        if target_control and not target_target:
            return bool(commutation_map[operation1.name][position["ctrl"]])

        # Case 4.3: target 1 overlaps with target 2
        if not target_control and target_target:
            return bool(commutation_map[operation1.name][position[control_base_2]])

    # Case 5: no controlled operations
    # Case 5.1: no controlled operations we simply check the commutation table
    return bool(commutation_map[operation1.name][position[operation2.name]])


def _both_controlled(control_base, operation1, operation2):
    """Auxiliary function to the is_commuting function for the case when both
    operations are controlled."""
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-return-statements

    if control_base.get(operation1.name) != "ControlledOperation":
        control_base_1 = control_base.get(operation1.name)
    else:
        control_base_1 = operation1.control_base

    if control_base.get(operation2.name) != "ControlledOperation":
        control_base_2 = control_base.get(operation2.name)
    else:
        control_base_2 = operation2.control_base

    target_wires_1 = qml.wires.Wires(
        [w for w in operation1.wires if w not in operation1.control_wires]
    )
    target_wires_2 = qml.wires.Wires(
        [w for w in operation2.wires if w not in operation2.control_wires]
    )

    control_control = intersection(operation1.control_wires, operation2.control_wires)
    target_target = intersection(target_wires_1, target_wires_2)
    control_target = intersection(operation1.control_wires, target_wires_2)
    target_control = intersection(target_wires_1, operation2.control_wires)

    # Case 2.1: disjoint targets
    if control_control and not target_target and not control_target and not target_control:
        return True

    # Case 2.2: disjoint controls
    if not control_control and target_target and not control_target and not target_control:
        return bool(commutation_map[control_base_1][position[control_base_2]])

    # Case 2.3: targets overlap and controls overlap
    if target_target and control_control and not control_target and not target_control:
        return bool(commutation_map[control_base_1][position[control_base_2]])

    # Case 2.4: targets and controls overlap
    if control_target and target_control and not target_target:
        return bool(commutation_map["ctrl"][position[control_base_2]]) and bool(
            commutation_map[control_base_1][position["ctrl"]]
        )

    # Case 2.5: targets overlap with and controls and targets
    if control_target and not target_control and target_target:
        return bool(commutation_map["ctrl"][position[control_base_2]]) and bool(
            commutation_map[control_base_1][position[control_base_2]]
        )

    # Case 2.6: targets overlap with and controls and targets
    if target_control and not control_target and target_target:
        return bool(commutation_map[control_base_1][position["ctrl"]]) and bool(
            commutation_map[control_base_1][position[control_base_2]]
        )

    # Case 2.7: targets overlap with control
    if target_control and not control_target and not target_target:
        return bool(commutation_map[control_base_1][position["ctrl"]])

    # Case 2.8: targets overlap with control
    if not target_control and control_target and not target_target:
        return bool(commutation_map["ctrl"][position[control_base_2]])

    # Case 2.9: targets and controls overlap with targets and controls
    # equivalent to target_control and control_target and target_target:
    return (
        bool(commutation_map[control_base_1][position["ctrl"]])
        and bool(commutation_map["ctrl"][position[control_base_2]])
        and bool(commutation_map[control_base_1][position[control_base_2]])
    )


def _merge_no_duplicates(*iterables):
    """Merge K list without duplicate using python heapq ordered merging.

    Args:
        *iterables: A list of k sorted lists

    Yields:
        Iterator: List from the merging of the k ones (without duplicates)
    """
    last = object()
    for val in heapq.merge(*iterables):
        if val != last:
            last = val
            yield val


class CommutationDAGNode:
    r"""Class to store information about a quantum operation in a node of the
    commutation DAG.

    Args:
        op (.Operation): PennyLane operation.
        wires (.Wires): Wires on which the operation acts on.
        node_id (int): ID of the node in the DAG.
        successors (array[int]): List of the node's successors in the DAG.
        predecessors (array[int]): List of the node's predecessors in the DAG.
        reachable (bool): Attribute used to check reachability by pairwise commutation.
    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    # pylint: disable=too-few-public-methods

    __slots__ = [
        "op",
        "wires",
        "target_wires",
        "control_wires",
        "node_id",
        "successors",
        "predecessors",
        "reachable",
    ]

    def __init__(
        self,
        op=None,
        wires=None,
        target_wires=None,
        control_wires=None,
        successors=None,
        predecessors=None,
        reachable=None,
        node_id=-1,
    ):
        self.op = op
        """Operation: The operation represented by the nodes."""
        self.wires = wires
        """Wires: The wires that the operation acts on."""
        self.target_wires = target_wires
        """Wires: The target wires of the operation."""
        self.control_wires = control_wires if control_wires is not None else []
        """Wires: The control wires of the operation."""
        self.node_id = node_id
        """int: The ID of the operation in the DAG."""
        self.successors = successors if successors is not None else []
        """list(int): List of the node's successors."""
        self.predecessors = predecessors if predecessors is not None else []
        """list(int): List of the node's predecessors."""
        self.reachable = reachable
        """bool: Useful attribute to create the commutation DAG."""


class CommutationDAG:
    r"""Class to represent a quantum circuit as a directed acyclic graph (DAG). This class is useful to build the
    commutation DAG and set up all nodes attributes. The construction of the DAG should be used through the
    transform :class:`qml.transforms.commutation_dag`.

    Args:
        tape (.QuantumTape): PennyLane quantum tape representing a quantum circuit.

    **Reference:**

    [1] Iten, R., Moyard, R., Metger, T., Sutter, D. and Woerner, S., 2020.
    Exact and practical pattern matching for quantum circuit optimization.
    `doi.org/10.1145/3498325 <https://dl.acm.org/doi/abs/10.1145/3498325>`_

    """

    def __init__(self, tape):

        self.num_wires = len(tape.wires)
        self.node_id = -1
        self._multi_graph = nx.MultiDiGraph()

        consecutive_wires = Wires(range(len(tape.wires)))
        wires_map = OrderedDict(zip(tape.wires, consecutive_wires))

        for operation in tape.operations:
            operation._wires = Wires([wires_map[wire] for wire in operation.wires.tolist()])
            self.add_node(operation)

        self._add_successors()

        for obs in tape.observables:
            obs._wires = Wires([wires_map[wire] for wire in obs.wires.tolist()])

        self.observables = tape.observables if tape.observables is not None else []

    def _add_node(self, node):
        self.node_id += 1
        node.node_id = self.node_id
        self._multi_graph.add_node(node.node_id, node=node)

    def add_node(self, operation):
        """Add the operation as a node in the DAG and updates the edges.

        Args:
            operation (qml.operation): PennyLane quantum operation to add to the DAG.
        """
        target_wires = [w for w in operation.wires if w not in operation.control_wires]

        new_node = CommutationDAGNode(
            op=operation,
            wires=operation.wires.tolist(),
            target_wires=target_wires,
            control_wires=operation.control_wires.tolist(),
            successors=[],
            predecessors=[],
        )

        self._add_node(new_node)
        self._update_edges()

    def get_node(self, node_id):
        """Add the operation as a node in the DAG and updates the edges.

        Args:
            node_id (int): PennyLane quantum operation to add to the DAG.

        Returns:
            CommutationDAGNOde: The node with the given id.
        """
        return self._multi_graph.nodes(data="node")[node_id]

    def get_nodes(self):
        """Return iterable to loop through all the nodes in the DAG

        Returns:
            networkx.classes.reportviews.NodeDataView: Iterable nodes.
        """
        return self._multi_graph.nodes(data="node")

    def add_edge(self, node_in, node_out):
        """Add an edge (non commutation) between node_in and node_out.

        Args:
            node_in (int): Id of the ingoing node.
            node_out (int): Id of the outgoing node.

        Returns:
            int: Id of the created edge.
        """
        return self._multi_graph.add_edge(node_in, node_out, commute=False)

    def get_edge(self, node_in, node_out):
        """Get the edge between two nodes if it exists.

        Args:
            node_in (int): Id of the ingoing node.
            node_out (int): Id of the outgoing node.

        Returns:
            dict or None: Default weight is 0, it returns None when there is no edge.
        """
        return self._multi_graph.get_edge_data(node_in, node_out)

    def get_edges(self):
        """Get all edges as an iterable.

        Returns:
            networkx.classes.reportviews.OutMultiEdgeDataView: Iterable over all edges.
        """
        return self._multi_graph.edges.data()

    def direct_predecessors(self, node_id):
        """Return the direct predecessors of the given node.

        Args:
            node_id (int): Id of the node in the DAG.

        Returns:
            list[int]: List of the direct predecessors of the given node.
        """
        dir_pred = list(self._multi_graph.pred[node_id].keys())
        dir_pred.sort()
        return dir_pred

    def predecessors(self, node_id):
        """Return the predecessors of the given node.

        Args:
            node_id (int): Id of the node in the DAG.

        Returns:
            list[int]: List of the predecessors of the given node.
        """
        pred = list(nx.ancestors(self._multi_graph, node_id))
        pred.sort()
        return pred

    def direct_successors(self, node_id):
        """Return the direct successors of the given node.

        Args:
            node_id (int): Id of the node in the DAG.

        Returns:
            list[int]: List of the direct successors of the given node.
        """
        dir_succ = list(self._multi_graph.succ[node_id].keys())
        dir_succ.sort()
        return dir_succ

    def successors(self, node_id):
        """Return the successors of the given node.

        Args:
            node_id (int): Id of the node in the DAG.

        Returns:
            list[int]: List of the successors of the given node.
        """
        succ = list(nx.descendants(self._multi_graph, node_id))
        succ.sort()
        return succ

    @property
    def graph(self):
        """Return the DAG object.

        Returns:
            networkx.MultiDiGraph(): Networkx representation of the DAG.
        """
        return self._multi_graph

    def draw(self, filename="dag.png"):  # pragma: no cover
        """Draw the DAG object.

        Args:
            filename (str): The file name which is in PNG format. Default = 'dag.png'
        """
        draw_graph = nx.MultiDiGraph()

        for node in self.get_nodes():
            wires = ",".join([" " + str(elem) for elem in node[1].op.wires.tolist()])
            label = (
                "ID: "
                + str(node[0])
                + "\n"
                + "Op: "
                + node[1].op.name
                + "\n"
                + "Wires: ["
                + wires[1::]
                + "]"
            )
            draw_graph.add_node(
                node[0], label=label, color="blue", style="filled", fillcolor="lightblue"
            )

        for edge in self.get_edges():
            draw_graph.add_edge(edge[0], edge[1])

        dot = to_pydot(draw_graph)
        dot.write_png(filename)

    def _pred_update(self, node_id):
        self.get_node(node_id).predecessors = []

        for d_pred in self.direct_predecessors(node_id):
            self.get_node(node_id).predecessors.append([d_pred])
            self.get_node(node_id).predecessors.append(self.get_node(d_pred).predecessors)

        self.get_node(node_id).predecessors = list(
            _merge_no_duplicates(*self.get_node(node_id).predecessors)
        )

    def _add_successors(self):

        for node_id in range(len(self._multi_graph) - 1, -1, -1):
            direct_successors = self.direct_successors(node_id)

            for d_succ in direct_successors:
                self.get_node(node_id).successors.append([d_succ])
                self.get_node(node_id).successors.append(self.get_node(d_succ).successors)

            self.get_node(node_id).successors = list(
                _merge_no_duplicates(*self.get_node(node_id).successors)
            )

    def _update_edges(self):

        max_node_id = len(self._multi_graph) - 1
        max_node = self.get_node(max_node_id).op

        for current_node_id in range(0, max_node_id):
            self.get_node(current_node_id).reachable = True

        for prev_node_id in range(max_node_id - 1, -1, -1):
            if self.get_node(prev_node_id).reachable and not is_commuting(
                self.get_node(prev_node_id).op, max_node
            ):
                self.add_edge(prev_node_id, max_node_id)
                self._pred_update(max_node_id)
                list_predecessors = self.get_node(max_node_id).predecessors
                for pred_id in list_predecessors:
                    self.get_node(pred_id).reachable = False
