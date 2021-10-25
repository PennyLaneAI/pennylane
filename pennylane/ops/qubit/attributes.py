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
This file contains a number of attributes that may be held by operators,
and lists all operators satisfying those criteria.
"""
import pennylane as qml


class Attribute(set):
    r"""Class to represent a set of operators with a certain attribute.

    **Example**

    Suppose we would like to store a list of which qubit operations are
    Pauli operators. We can create a new Attribute, ``pauli_ops``, like so,
    listing which operations satisfy this property.

    >>> pauli_ops = Attribute(["PauliX", "PauliZ"])

    We can check either a string or an Operation for inclusion in this set:

    >>> qml.PauliX(0) in pauli_ops
    True
    >>> "Hadamard" in pauli_ops
    False

    We can also dynamically add operators to the sets at runtime, by passing
    either a string, an operation class, or an operation itself. This is useful
    for adding custom operations to the attributes such as
    ``composable_rotations`` and ``self_inverses`` that are used in compilation
    transforms.

    >>> pauli_ops.add("PauliY")
    >>> pauli_ops
    ["PauliX", "PauliY", "PauliZ"]
    """

    def add(self, obj):
        try:
            if isinstance(obj, str):
                return super().add(obj)

            if isinstance(obj, qml.operation.Operator):
                return super().add(obj.name)

            if issubclass(obj, qml.operation.Operator):
                return super().add(obj.__name__)

            raise TypeError

        except TypeError:
            raise TypeError(
                "Only an Operator or string representing an Operator can be added to an attribute."
            )

    def __contains__(self, obj):
        """Check if the attribute contains a given operator."""
        if isinstance(obj, str):
            return super().__contains__(obj)

        try:
            if isinstance(obj, qml.operation.Operator):
                return super().__contains__(obj.name)

            if issubclass(obj, qml.operation.Operator):
                return super().__contains__(obj.__name__)

        except TypeError:
            raise TypeError(
                "Only an Operator or string representing an Operator can be checked for attribute inclusion."
            )

        return False


composable_rotations = Attribute(
    [
        "RX",
        "RY",
        "RZ",
        "PhaseShift",
        "CRX",
        "CRY",
        "CRZ",
        "ControlledPhaseShift",
        "Rot",
    ]
)
"""Operations for which composing multiple copies of the operation results in an
addition (or alternative accumulation) of parameters.

For example, ``qml.RZ`` is a composable rotation. Applying ``qml.RZ(0.1,
wires=0)`` followed by ``qml.RZ(0.2, wires=0)`` is equivalent to performing
a single rotation ``qml.RZ(0.3, wires=0)``.
"""


self_inverses = Attribute(
    ["Hadamard", "PauliX", "PauliY", "PauliZ", "CNOT", "CZ", "CY", "SWAP", "Toffoli"]
)
"""Operations that are their own inverses."""

symmetric_over_all_wires = Attribute(
    [
        "CZ",
        "SWAP",
    ]
)
"""Operations that are the same if you exchange the order of wires.

For example, ``qml.CZ(wires=[0, 1])`` has the same effect as ``qml.CZ(wires=[1,
0])`` due to symmetry of the operation.
"""

symmetric_over_control_wires = Attribute(["Toffoli"])
"""Controlled operations that are the same if you exchange the order of all but
the last (target) wire.

For example, ``qml.Toffoli(wires=[0, 1, 2])`` has the same effect as
``qml.Toffoli(wires=[1, 0, 2])``, but neither are the same as
``qml.Toffoli(wires=[0, 2, 1])``.
"""
