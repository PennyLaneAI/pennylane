# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Base classes for resource estimation."""
from collections import defaultdict
from dataclasses import dataclass, field


class CompressedResourceOp:
    r"""Instantiate the light weight class corressponding to the operator type and parameters.

    Args:
        op_type (Type): the PennyLane type of the operation
        params_tuple (tuple): [tuple(tuple(parameter_name: str, parameter_value))] contains the minimal
            pairs of parameter names and values required to compute the resources for the given operator!

    .. details::

        This representation is the minimal amount of information required to estimate resources for the operator.

        **Example**

        >>> op_tp = CompressedResourceOp(qml.Hadamard, (("num_wires", 1),))
        >>> print(op_tp)
        Hadamard(num_wires=1)

        >>> op_tp = CompressedResourceOp(
                qml.QSVT,
                (
                    ("num_wires", 5),
                    ("num_angles", 100),
                ),
            )
        >>> print(op_tp)
        QSVT(num_wires=5, num_angles=100)
    """

    def __init__(self, op_type: type, params: dict) -> None:
        r"""Instantiate the light weight class corressponding to the operator type and parameters.

        Args:
            op_type (Type): the PennyLane type of the operation
            params_tuple (tuple): [tuple(tuple(parameter_name: str, parameter_value))] contains the minimal
                pairs of parameter names and values required to compute the resources for the given operator!

        .. details::

            This representation is the minimal amount of information required to estimate resources for the operator.

            **Example**

            >>> op_tp = CompressedResourceOp(qml.Hadamard, (("num_wires", 1),))
            >>> print(op_tp)
            Hadamard(num_wires=1)

            >>> op_tp = CompressedResourceOp(
                    qml.QSVT,
                    (
                        ("num_wires", 5),
                        ("num_angles", 100),
                    ),
                )
            >>> print(op_tp)
            QSVT(num_wires=5, num_angles=100)
        """
        self._name = op_type.__name__
        self.op_type = op_type
        self.params = params

        sorted_keys = sorted(list(params))
        self._hashable_params = tuple(
            (key, params[key]) for key in sorted_keys
        )  # tuple of sorted params

    def __hash__(self) -> int:
        return hash((self._name, self._hashable_params))

    def __eq__(self, other: object) -> bool:
        return (self.op_type == other.op_type) and (self.params == other.params)

    def __repr__(self) -> str:
        op_type_str = self._name + "("
        params_str = ", ".join([f"{key}={self.params[key]}" for key in self.params]) + ")"

        return op_type_str + params_str


@dataclass
class Resources:
    r"""Contains attributes which store key resources such as number of gates, number of wires, and gate types.

    Args:
        num_wires (int): number of qubits
        num_gates (int): number of gates
        gate_types (dict): dictionary storing operation names (str) as keys
            and the number of times they are used in the circuit (int) as values

    .. details::

        The resources being tracked can be accessed as class attributes.
        Additionally, the :code:`Resources` instance can be nicely displayed in the console.

        **Example**

        >>> r = Resources(
        ...             num_wires=2,
        ...             num_gates=2,
        ...             gate_types={"Hadamard": 1, "CNOT": 1}
        ...     )
        >>> print(r)
        wires: 2
        gates: 2
        gate_types:
        {'Hadamard': 1, 'CNOT': 1}
    """

    num_wires: int = 0
    num_gates: int = 0
    gate_types: defaultdict = field(default_factory=lambda: defaultdict(int))

    def __str__(self):
        """String representation of the Resources object."""
        keys = ["wires", "gates"]
        vals = [self.num_wires, self.num_gates]
        items = "\n".join([str(i) for i in zip(keys, vals)])
        items = items.replace("('", "")
        items = items.replace("',", ":")
        items = items.replace(")", "")

        gate_type_str = ", ".join(
            [f"'{gate_name}': {count}" for gate_name, count in self.gate_types.items()]
        )
        items += "\ngate_types:\n{" + gate_type_str + "}"
        return items

    def _ipython_display_(self):
        """Displays __str__ in ipython instead of __repr__"""
        print(str(self))
