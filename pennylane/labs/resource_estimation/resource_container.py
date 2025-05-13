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
from __future__ import annotations

import copy
from decimal import Decimal
from collections import defaultdict
from typing import Hashable, Optional, Type

from pennylane.labs.resource_estimation.resource_operator import ResourceOperator
from pennylane.labs.resource_estimation.qubit_manager import QubitManager


class CompressedResourceOp:  # pylint: disable=too-few-public-methods
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

    def __init__(
        self, op_type: Type[ResourceOperator], params: Optional[dict] = None, name: str = None
    ):

        if not issubclass(op_type, ResourceOperator):
            raise TypeError(f"op_type must be a subclass of ResourceOperator. Got {op_type}.")
        self.op_type = op_type
        self.params = params or {}
        self._hashable_params = _make_hashable(params) if params else ()
        self._name = name or op_type.tracking_name(**self.params)

    def __hash__(self) -> int:
        return hash((self.op_type, self._hashable_params))

    def __eq__(self, other: CompressedResourceOp) -> bool:
        return (
            isinstance(other, CompressedResourceOp)
            and self.op_type == other.op_type
            and self.params == other.params
        )

    def __repr__(self) -> str:
        return self._name


def _make_hashable(d) -> tuple:
    if isinstance(d, Hashable):
        return d
    sorted_keys = sorted(d)
    return tuple((k, _make_hashable(d[k])) for k in sorted_keys)


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

    def __init__(self, qubit_manager, gate_types: dict = None):
        gate_types = gate_types or {}

        self.qubit_manager = qubit_manager
        self.gate_types = (
            gate_types
            if (isinstance(gate_types, defaultdict) and isinstance(gate_types.default_factory, int))
            else defaultdict(int, gate_types)
        )

    def __add__(self, other: "Resources") -> "Resources":
        """Add two resources objects in series"""
        assert isinstance(other, (self.__class__, ResourceOperator))
        return add_in_series(self, other)

    def __and__(self, other: "Resources") -> "Resources":
        """Add two resources objects in parallel"""
        assert isinstance(other, (self.__class__, ResourceOperator))
        return add_in_parallel(self, other)

    def __eq__(self, other: "Resources") -> bool:
        """Test if two resource objects are equal"""
        return (self.gate_types == other.gate_types) and (self.qubit_manager == other.qubit_manager)

    def __mul__(self, scalar: int) -> "Resources":
        """Scale a resources object in series"""
        assert isinstance(scalar, int)
        return mul_in_series(self, scalar)
    
    def __matmul__(self, scalar: int) -> "Resources":
        """Scale a resources object in parallel"""
        assert isinstance(scalar, int)
        return mul_in_parallel(self, scalar)

    __rmul__ = __mul__
    __radd__ = __add__
    __rand__ = __and__
    __rmatmul__ = __matmul__

    @property
    def clean_gate_counts(self):
        clean_gate_counts = defaultdict(int)

        for cmp_res_op, counts in self.gate_types.items():
            clean_gate_counts[cmp_res_op._name] += counts

        return clean_gate_counts

    def __str__(self):
        """String representation of the Resources object."""
        # keys = ["wires", "gates"]
        # vals = [self.num_wires, self.num_gates]
        # items = "\n".join([str(i) for i in zip(keys, vals)])
        # items = items.replace("('", "")
        # items = items.replace("',", ":")
        # items = items.replace(")", "")

        gate_type_str = ", ".join(
            [f"'{gate_name}': {Decimal(count):.3E}" if count>999 else f"'{gate_name}': {count}" for gate_name, count in self.clean_gate_counts.items()]
        )

        items = "--- Resources: ---\n"
        items += f" qubit manager: {self.qubit_manager}\n"

        total_gates = sum(self.clean_gate_counts.values())
        if total_gates>999:
            items += f" total # gates: {total_gates}\n"
        else:
            items += f" total # gates: {Decimal(total_gates):.3E}\n"

        items += " gate_types:\n  {" + gate_type_str + "}"

        return items

    def __repr__(self):
        """Compact string representation of the Resources object"""
        return {
            "qubit manager": self.qubit_manager,
            "gate_types": self.gate_types,
        }.__repr__()

    def _ipython_display_(self):
        """Displays __str__ in ipython instead of __repr__"""
        print(str(self))


def add_in_series(first: Resources, other, in_place=False) -> Resources:  # + 
    r"""Add two resources assuming the circuits are executed in series.

    Args:
        first (Resources): first resource object to combine
        other (Resources): other resource object to combine with
        in_place (bool): determines if the first Resources are modified in place (default False)

    Returns:
        Resources: combined resources
    """
    if isinstance(other, Resources):
        qm1, qm2 = (first.qubit_manager, other.qubit_manager)
        
        new_clean = max(qm1.clean_qubits, qm2.clean_qubits)
        new_dirty = qm1.dirty_qubits + qm2.dirty_qubits
        new_budget = qm1.tight_budget or qm2.tight_budget
        new_logic = max(qm1.algo_qubits, qm2.algo_qubits)

        new_qubit_manager = QubitManager(
            work_wires={"clean": new_clean, "dirty": new_dirty}, 
            tight_budget=new_budget
        )

        new_qubit_manager._logic_qubit_counts = new_logic
        new_gate_types = _combine_dict(first.gate_types, other.gate_types, in_place=False)

    else: 
        qm = first.qubit_manager
        new_logic = max(qm.algo_qubits, other.num_wires)

        new_qubit_manager = QubitManager(
            work_wires={"clean": qm.clean_qubits, "dirty": qm.dirty_qubits}, 
            tight_budget=qm.tight_budget
        )

        new_qubit_manager._logic_qubit_counts = new_logic
        new_gate_types = copy.copy(first.gate_types)
        new_gate_types[other.resource_rep_from_op()] += 1
    
    return Resources(new_qubit_manager, new_gate_types)


def add_in_parallel(first: Resources, other, in_place=False) -> Resources:  # & 
    r"""Add two resources assuming the circuits are executed in parallel.

    Args:
        first (Resources): first resource object to combine
        other (Resources): other resource object to combine with
        in_place (bool): determines if the first Resources are modified in place (default False)

    Returns:
        Resources: combined resources
    """
    if isinstance(other, Resources):
        qm1, qm2 = (first.qubit_manager, other.qubit_manager)
        
        new_clean = max(qm1.clean_qubits, qm2.clean_qubits)
        new_dirty = qm1.dirty_qubits + qm2.dirty_qubits
        new_budget = qm1.tight_budget or qm2.tight_budget
        new_logic = qm1.algo_qubits + qm2.algo_qubits

        new_qubit_manager = QubitManager(
            work_wires={"clean": new_clean, "dirty": new_dirty}, 
            tight_budget=new_budget,
        )

        new_qubit_manager._logic_qubit_counts = new_logic
        new_gate_types = _combine_dict(first.gate_types, other.gate_types, in_place=False)
    
    else: 
        qm = first.qubit_manager
        new_logic = qm.algo_qubits + other.num_wires

        new_qubit_manager = QubitManager(
            work_wires={"clean": qm.clean_qubits, "dirty": qm.dirty_qubits}, 
            tight_budget=qm.tight_budget
        )

        new_qubit_manager._logic_qubit_counts = new_logic
        new_gate_types = copy.copy(first.gate_types)
        new_gate_types[other.resource_rep_from_op()] += 1
    
    return Resources(new_qubit_manager, new_gate_types)


def mul_in_series(first: Resources, scalar: int, in_place=False) -> Resources:  # * 
    r"""Multiply the resources by a scalar assuming the circuits are executed in series.

    Args:
        first (Resources): first resource object to combine
        scalar (int): integer value to scale the resources by
        in_place (bool): determines if the first Resources are modified in place (default False)

    Returns:
        Resources: combined resources
    """
    qm = first.qubit_manager
    
    new_clean = qm.clean_qubits
    new_dirty = scalar * qm.dirty_qubits
    new_budget = qm.tight_budget
    new_logic = qm.algo_qubits

    new_qubit_manager = QubitManager(
        work_wires={"clean": new_clean, "dirty": new_dirty}, 
        tight_budget=new_budget,
    )

    new_qubit_manager._logic_qubit_counts = new_logic
    new_gate_types = _scale_dict(first.gate_types, scalar, in_place=False)

    return Resources(new_qubit_manager, new_gate_types)


def mul_in_parallel(first: Resources, scalar: int, in_place=False) -> Resources:  # @ 
    r"""Multiply the resources by a scalar assuming the circuits are executed in parallel.

    Args:
        first (Resources): first resource object to combine
        scalar (int): integer value to scale the resources by
        in_place (bool): determines if the first Resources are modified in place (default False)

    Returns:
        Resources: combined resources
    """
    qm = first.qubit_manager
    
    new_clean = qm.clean_qubits
    new_dirty = scalar * qm.dirty_qubits
    new_budget = qm.tight_budget
    new_logic = scalar * qm.algo_qubits

    new_qubit_manager = QubitManager(
        work_wires={"clean": new_clean, "dirty": new_dirty}, 
        tight_budget=new_budget,
    )

    new_qubit_manager._logic_qubit_counts = new_logic
    new_gate_types = _scale_dict(first.gate_types, scalar, in_place=False)

    return Resources(new_qubit_manager, new_gate_types)


# def substitute(
#     initial_resources: Resources, gate_name: str, replacement_resources: Resources, in_place=False
# ) -> Resources:
#     """Replaces a specified gate in a :class:`~.resource.Resources` object with the contents of
#     another :class:`~.resource.Resources` object.

#     Args:
#         initial_resources (Resources): the resources to be modified
#         gate_name (str): the name of the operation to be replaced
#         replacement (Resources): the resources to be substituted instead of the gate
#         in_place (bool): determines if the initial resources are modified in place or if a new copy is
#             created

#     Returns:
#         Resources: the updated :class:`~.Resources` after substitution

#     .. details::

#         **Example**

#         In this example we replace the resources for the :code:`RX` gate:

#         .. code-block:: python3

#             from pennylane.labs.resource_estimation import Resources

#             replace_gate_name = "RX"

#             initial_resources = Resources(
#                 num_wires = 2,
#                 num_gates = 3,
#                 gate_types = {"RX": 2, "CNOT": 1},
#             )

#             replacement_rx_resources = Resources(
#                 num_wires = 1,
#                 num_gates = 7,
#                 gate_types = {"Hadamard": 3, "S": 4},
#             )

#         Executing the substitution produces:

#         >>> from pennylane.labs.resource_estimation import substitute
#         >>> res = substitute(
#         ...     initial_resources, replace_gate_name, replacement_rx_resources,
#         ... )
#         >>> print(res)
#         wires: 2
#         gates: 15
#         gate_types:
#         {'CNOT': 1, 'Hadamard': 6, 'S': 8}
#     """

#     count = initial_resources.gate_types.get(gate_name, 0)

#     if count > 0:
#         new_gates = initial_resources.num_gates - count + (count * replacement_resources.num_gates)

#         replacement_gate_types = _scale_dict(
#             replacement_resources.gate_types, count, in_place=in_place
#         )
#         new_gate_types = _combine_dict(
#             initial_resources.gate_types, replacement_gate_types, in_place=in_place
#         )
#         new_gate_types.pop(gate_name)

#         if in_place:
#             initial_resources.num_gates = new_gates
#             return initial_resources

#         return Resources(initial_resources.num_wires, new_gates, new_gate_types)

#     return initial_resources


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
