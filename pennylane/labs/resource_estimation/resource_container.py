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
import copy
from collections import defaultdict
from dataclasses import dataclass, field

import pennylane.labs.resource_estimation.resource_constructor as rc


class CompressedResourceOp:
    r"""Instantiate the light weight class corressponding to the operator type and parameters.

    Args:
        op_type (Type): the PennyLane type of the operation
        params (dict): a dictionary containing the minimal pairs of parameter names and values
                    required to compute the resources for the given operator!

    .. details::

        This representation is the minimal amount of information required to estimate resources for the operator.

        **Example**

        >>> op_tp = CompressedResourceOp(qml.Hadamard, {"num_wires":1})
        >>> print(op_tp)
        Hadamard(num_wires=1)

        >>> op_tp = CompressedResourceOp(
                qml.QSVT,
                {
                    "num_wires": 5,
                    "num_angles": 100,
                },
            )
        >>> print(op_tp)
        QSVT(num_wires=5, num_angles=100)
    """

    def __init__(self, op_type: type, params: dict) -> None:
        r"""Instantiate the light weight class corressponding to the operator type and parameters.

        Args:
            op_type (Type): the PennyLane type of the operation
            params (dict): a dictionary containing the minimal pairs of parameter names and values
                        required to compute the resources for the given operator!

        .. details::

            This representation is the minimal amount of information required to estimate resources for the operator.

            **Example**

            >>> op_tp = CompressedResourceOp(qml.Hadamard, {"num_wires":1})
            >>> print(op_tp)
            Hadamard(num_wires=1)

            >>> op_tp = CompressedResourceOp(
                    qml.QSVT,
                    {
                        "num_wires": 5,
                        "num_angles": 100,
                    },
                )
            >>> print(op_tp)
            QSVT(num_wires=5, num_angles=100)
        """
        if not issubclass(op_type, rc.ResourceConstructor):
            raise TypeError(
                f"op_type must be a subclass of ResourceConstructor. Got type {type(op_type)}."
            )


        self._name = (op_type.__name__).replace("Resource", "")
        self.op_type = op_type
        self.params = params
        self._hashable_params = tuple(params.items())

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

    def __add__(self, other: "Resources") -> "Resources":
        """Add two resources objects in series"""
        return add_in_series(self, other)

    def __mul__(self, scaler: int) -> "Resources":
        """Scale a resources object in series"""
        return mul_in_series(self, scaler)

    __rmul__ = __mul__  # same implementation

    def __iadd__(self, other: "Resources") -> "Resources":
        """Add two resources objects in series"""
        return add_in_series(self, other, in_place=True)

    def __imull__(self, scaler: int) -> "Resources":
        """Scale a resources object in series"""
        return mul_in_series(self, scaler, in_place=True)

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


def add_in_series(first: Resources, other: Resources, in_place=False) -> Resources:
    r"""Add two resources assuming the circuits are executed in series.

    Args:
        first (Resources): first resource object to combine
        other (Resources): other resource object to combine with
        in_place (bool): determines if the first Resources are modified in place (default False)

    Returns:
        Resources: combined resources
    """
    new_wires = max(first.num_wires, other.num_wires)
    new_gates = first.num_gates + other.num_gates
    new_gate_types = _combine_dict(first.gate_types, other.gate_types, in_place=in_place)

    if in_place:
        first.num_wires = new_wires
        first.num_gates = new_gates
        return first

    return Resources(new_wires, new_gates, new_gate_types)


def add_in_parallel(first: Resources, other: Resources, in_place=False) -> Resources:
    r"""Add two resources assuming the circuits are executed in parallel.

    Args:
        first (Resources): first resource object to combine
        other (Resources): other resource object to combine with
        in_place (bool): determines if the first Resources are modified in place (default False)

    Returns:
        Resources: combined resources
    """
    new_wires = first.num_wires + other.num_wires
    new_gates = first.num_gates + other.num_gates
    new_gate_types = _combine_dict(first.gate_types, other.gate_types, in_place=in_place)

    if in_place:
        first.num_wires = new_wires
        first.num_gates = new_gates
        return first

    return Resources(new_wires, new_gates, new_gate_types)


def mul_in_series(first: Resources, scaler: int, in_place=False) -> Resources:
    r"""Multiply two resources assuming the circuits are executed in series.

    Args:
        first (Resources): first resource object to combine
        scaler (int): integer value to scale the resources by
        in_place (bool): determines if the first Resources are modified in place (default False)

    Returns:
        Resources: combined resources
    """
    new_gates = scaler * first.num_gates
    new_gate_types = _scale_dict(first.gate_types, scaler, in_place=in_place)

    if in_place:
        first.num_gates = new_gates
        return first

    return Resources(first.num_wires, new_gates, new_gate_types)


def mul_in_parallel(first: Resources, scaler: int, in_place=False) -> Resources:
    r"""Multiply two resources assuming the circuits are executed in parallel.

    Args:
        first (Resources): first resource object to combine
        scaler (int): integer value to scale the resources by
        in_place (bool): determines if the first Resources are modified in place (default False)

    Returns:
        Resources: combined resources
    """
    new_wires = scaler * first.num_wires
    new_gates = scaler * first.num_gates
    new_gate_types = _scale_dict(first.gate_types, scaler, in_place=in_place)

    if in_place:
        first.num_wires = new_wires
        first.num_gates = new_gates
        return first

    return Resources(new_wires, new_gates, new_gate_types)


def _combine_dict(dict1: defaultdict, dict2: defaultdict, in_place=False):
    r"""Private function which combines two dictionaries together."""
    combined_dict = dict1 if in_place else copy.copy(dict1)

    for k, v in dict2.items():
        combined_dict[k] += v

    return combined_dict


def _scale_dict(dict1: defaultdict, scaler: int, in_place=False):
    r"""Private function which scales the values in a dictionary."""

    combined_dict = dict1 if in_place else copy.copy(dict1)

    for k in combined_dict:
        combined_dict[k] *= scaler

    return combined_dict
