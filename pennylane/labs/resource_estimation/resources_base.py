# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Base class for storing resources."""
from __future__ import annotations

import copy
from collections import defaultdict
from decimal import Decimal

from pennylane.labs.resource_estimation.qubit_manager import QubitManager


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
        assert isinstance(other, self.__class__)
        return add_in_series(self, other)

    def __and__(self, other: "Resources") -> "Resources":
        """Add two resources objects in parallel"""
        assert isinstance(other, self.__class__)
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
        gate_type_str = ", ".join(
            [
                f"'{gate_name}': {Decimal(count):.3E}" if count > 999 else f"'{gate_name}': {count}"
                for gate_name, count in self.clean_gate_counts.items()
            ]
        )

        items = "--- Resources: ---\n"
        items += f" qubit manager: {self.qubit_manager}\n"

        if (total_gates := sum(self.clean_gate_counts.values())) > 999:
            items += f" total gates: {Decimal(total_gates):.3E}\n"
        else:
            items += f" total gates: {total_gates}\n"

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


def add_in_series(first: Resources, other) -> Resources:  # +
    r"""Add two resources assuming the circuits are executed in series.

    Args:
        first (Resources): first resource object to combine
        other (Resources): other resource object to combine with

    Returns:
        Resources: combined resources
    """
    qm1, qm2 = (first.qubit_manager, other.qubit_manager)

    new_clean = max(qm1.clean_qubits, qm2.clean_qubits)
    new_dirty = qm1.dirty_qubits + qm2.dirty_qubits
    new_budget = qm1.tight_budget or qm2.tight_budget
    new_logic = max(qm1.algo_qubits, qm2.algo_qubits)

    new_qubit_manager = QubitManager(
        work_wires={"clean": new_clean, "dirty": new_dirty}, tight_budget=new_budget
    )

    new_qubit_manager._logic_qubit_counts = new_logic
    new_gate_types = _combine_dict(first.gate_types, other.gate_types)
    return Resources(new_qubit_manager, new_gate_types)


def add_in_parallel(first: Resources, other) -> Resources:  # &
    r"""Add two resources assuming the circuits are executed in parallel.

    Args:
        first (Resources): first resource object to combine
        other (Resources): other resource object to combine with

    Returns:
        Resources: combined resources
    """
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
    new_gate_types = _combine_dict(first.gate_types, other.gate_types)
    return Resources(new_qubit_manager, new_gate_types)


def mul_in_series(first: Resources, scalar: int) -> Resources:  # *
    r"""Multiply the resources by a scalar assuming the circuits are executed in series.

    Args:
        first (Resources): first resource object to combine
        scalar (int): integer value to scale the resources by

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
    new_gate_types = _scale_dict(first.gate_types, scalar)

    return Resources(new_qubit_manager, new_gate_types)


def mul_in_parallel(first: Resources, scalar: int) -> Resources:  # @
    r"""Multiply the resources by a scalar assuming the circuits are executed in parallel.

    Args:
        first (Resources): first resource object to combine
        scalar (int): integer value to scale the resources by

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
    new_gate_types = _scale_dict(first.gate_types, scalar)

    return Resources(new_qubit_manager, new_gate_types)


def _combine_dict(dict1: defaultdict, dict2: defaultdict):
    r"""Private function which combines two dictionaries together."""
    combined_dict = copy.copy(dict1)

    for k, v in dict2.items():
        combined_dict[k] += v

    return combined_dict


def _scale_dict(dict1: defaultdict, scalar: int):
    r"""Private function which scales the values in a dictionary."""
    combined_dict = copy.copy(dict1)

    for k in combined_dict:
        combined_dict[k] *= scalar

    return combined_dict
