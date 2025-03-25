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

from pennylane.decomposition.resources import CompressedResourceOp as _CompressedResourceOp
from pennylane.labs.resource_estimation import ResourceOperator


class CompressedResourceOp(_CompressedResourceOp):  # pylint: disable=too-few-public-methods
    r"""Instantiate the light weight class corresponding to the operator type and parameters.

    Args:
        op_type (Type): the class object of an operation which inherits from '~.ResourceOperator'
        params (dict): a dictionary containing the minimal pairs of parameter names and values
                    required to compute the resources for the given operator

    .. details::

        This representation is the minimal amount of information required to estimate resources for the operator.

        **Example**

        >>> op_tp = CompressedResourceOp(ResourceHadamard, {"num_wires":1})
        >>> print(op_tp)
        Hadamard(num_wires=1)
    """

    def __init__(self, op_type, params: dict, name=None) -> None:
        r"""Instantiate the light weight class corresponding to the operator type and parameters.

        Args:
            op_type (Type): the class object for an operation which inherits from '~.ResourceOperator'
            params (dict): a dictionary containing the minimal pairs of parameter names and values
                        required to compute the resources for the given operator

        .. details::

            This representation is the minimal amount of information required to estimate resources for the operator.

            **Example**

            >>> op_tp = CompressedResourceOp(ResourceHadamard, {"num_wires":1})
            >>> print(op_tp)
            Hadamard(num_wires=1)
        """
        if not issubclass(op_type, ResourceOperator):
            raise TypeError(f"op_type must be a subclass of ResourceOperator. Got {op_type}.")

        super().__init__(op_type, params)
        self._name = name or op_type.tracking_name(**params)

    def __repr__(self) -> str:
        return self._name


# @dataclass
class Resources:
    r"""Contains attributes which store key resources such as number of gates, number of wires, and gate types.

    Args:
        num_wires (int): number of qubits
        num_gates (int): number of gates
        gate_types (dict): dictionary storing operation names (str) as keys
            and the number of times they are used in the circuit (int) as values

    .. details::

        The resources being tracked can be accessed as class attributes.
        Additionally, the :class:`~.Resources` instance can be nicely displayed in the console.

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

    def __init__(self, num_wires: int = 0, num_gates: int = 0, gate_types: dict = None):
        gate_types = gate_types or {}

        self.num_wires = num_wires
        self.num_gates = num_gates
        self.gate_types = (
            gate_types
            if (isinstance(gate_types, defaultdict) and isinstance(gate_types.default_factory, int))
            else defaultdict(int, gate_types)
        )

    def __add__(self, other: "Resources") -> "Resources":
        """Add two resources objects in series"""
        return add_in_series(self, other)

    def __eq__(self, other: "Resources") -> bool:
        """Test if two resource objects are equal"""
        if self.num_wires != other.num_wires:
            return False
        if self.num_gates != other.num_gates:
            return False

        return self.gate_types == other.gate_types

    def __mul__(self, scalar: int) -> "Resources":
        """Scale a resources object in series"""
        return mul_in_series(self, scalar)

    __rmul__ = __mul__  # same implementation

    def __iadd__(self, other: "Resources") -> "Resources":
        """Add two resources objects in series"""
        return add_in_series(self, other, in_place=True)

    def __imull__(self, scalar: int) -> "Resources":
        """Scale a resources object in series"""
        return mul_in_series(self, scalar, in_place=True)

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

    def __repr__(self):
        """Compact string representation of the Resources object"""
        return {
            "gate_types": self.gate_types,
            "num_gates": self.num_gates,
            "num_wires": self.num_wires,
        }.__repr__()

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


def mul_in_series(first: Resources, scalar: int, in_place=False) -> Resources:
    r"""Multiply the resources by a scalar assuming the circuits are executed in series.

    Args:
        first (Resources): first resource object to combine
        scalar (int): integer value to scale the resources by
        in_place (bool): determines if the first Resources are modified in place (default False)

    Returns:
        Resources: combined resources
    """
    new_gates = scalar * first.num_gates
    new_gate_types = _scale_dict(first.gate_types, scalar, in_place=in_place)

    if in_place:
        first.num_gates = new_gates
        return first

    return Resources(first.num_wires, new_gates, new_gate_types)


def mul_in_parallel(first: Resources, scalar: int, in_place=False) -> Resources:
    r"""Multiply the resources by a scalar assuming the circuits are executed in parallel.

    Args:
        first (Resources): first resource object to combine
        scalar (int): integer value to scale the resources by
        in_place (bool): determines if the first Resources are modified in place (default False)

    Returns:
        Resources: combined resources
    """
    new_wires = scalar * first.num_wires
    new_gates = scalar * first.num_gates
    new_gate_types = _scale_dict(first.gate_types, scalar, in_place=in_place)

    if in_place:
        first.num_wires = new_wires
        first.num_gates = new_gates
        return first

    return Resources(new_wires, new_gates, new_gate_types)


def substitute(
    initial_resources: Resources, gate_name: str, replacement_resources: Resources, in_place=False
) -> Resources:
    """Replaces a specified gate in a :class:`~.resource.Resources` object with the contents of
    another :class:`~.resource.Resources` object.

    Args:
        initial_resources (Resources): the resources to be modified
        gate_name (str): the name of the operation to be replaced
        replacement (Resources): the resources to be substituted instead of the gate
        in_place (bool): determines if the initial resources are modified in place or if a new copy is
            created

    Returns:
        Resources: the updated :class:`~.Resources` after substitution

    .. details::

        **Example**

        In this example we replace the resources for the :code:`RX` gate:

        .. code-block:: python3

            from pennylane.labs.resource_estimation import Resources

            replace_gate_name = "RX"

            initial_resources = Resources(
                num_wires = 2,
                num_gates = 3,
                gate_types = {"RX": 2, "CNOT": 1},
            )

            replacement_rx_resources = Resources(
                num_wires = 1,
                num_gates = 7,
                gate_types = {"Hadamard": 3, "S": 4},
            )

        Executing the substitution produces:

        >>> from pennylane.labs.resource_estimation import substitute
        >>> res = substitute(
        ...     initial_resources, replace_gate_name, replacement_rx_resources,
        ... )
        >>> print(res)
        wires: 2
        gates: 15
        gate_types:
        {'CNOT': 1, 'Hadamard': 6, 'S': 8}
    """

    count = initial_resources.gate_types.get(gate_name, 0)

    if count > 0:
        new_gates = initial_resources.num_gates - count + (count * replacement_resources.num_gates)

        replacement_gate_types = _scale_dict(
            replacement_resources.gate_types, count, in_place=in_place
        )
        new_gate_types = _combine_dict(
            initial_resources.gate_types, replacement_gate_types, in_place=in_place
        )
        new_gate_types.pop(gate_name)

        if in_place:
            initial_resources.num_gates = new_gates
            return initial_resources

        return Resources(initial_resources.num_wires, new_gates, new_gate_types)

    return initial_resources


def _combine_dict(dict1: defaultdict, dict2: defaultdict, in_place=False):
    r"""Private function which combines two dictionaries together."""
    combined_dict = dict1 if in_place else copy.copy(dict1)

    for k, v in dict2.items():
        combined_dict[k] += v

    return combined_dict


def _scale_dict(dict1: defaultdict, scalar: int, in_place=False):
    r"""Private function which scales the values in a dictionary."""

    combined_dict = dict1 if in_place else copy.copy(dict1)

    for k in combined_dict:
        combined_dict[k] *= scalar

    return combined_dict
