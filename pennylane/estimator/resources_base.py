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
from collections import defaultdict, Counter
from decimal import Decimal

from .wires_manager import WireResourceManager


class Resources:
    r"""A container to track and update the resources used throughout a quantum circuit.
    The resources tracked include number of gates, number of wires, and gate types.

    Args:
        wire_manager (WireResourceManager): A wire tracker class which contains the number of available
            work wires, categorized as clean, dirty or algorithmic wires.
        gate_types (dict): A dictionary storing operations (ResourceOperator) as keys and the number
            of times they are used in the circuit (int) as values.

    **Example**

    The resources can be accessed as class attributes. Additionally, the :class:`~.Resources`
    instance can be nicely displayed in the console.

    >>> H = plre.resource_rep(plre.Hadamard)
    >>> X = plre.resource_rep(plre.X)
    >>> Y = plre.resource_rep(plre.Y)
    >>> wm = WireResourceManager(work_wires=3)
    >>> gt = defaultdict(int, {H: 10, X:7, Y:3})
    >>>
    >>> res = plre.Resources(wire_manager=wm, gate_types=gt)
    >>> print(res)
    --- Resources: ---
     Total wires: 3
        algorithmic wires: 0
        allocated wires: 3
             clean wires: 3
             dirty wires: 0
     Total gates : 20
      'Hadamard': 10,
      'X': 7,
      'Y': 3

    .. details::
        :title: Usage Details

        The :class:`Resources` object supports arithmetic operations which allow for quick addition
        and multiplication of resources. When combining resources, we can make a simplifying
        assumption about how they are applied in a quantum circuit (in series or in parallel).

        When assuming the circuits were executed in series, the number of algorithmic wires add
        together. When assuming the circuits were executed in parallel, the maximum of each set of
        algorithmic wires is used. The clean auxiliary wires can be reused between the circuits,
        and thus we always use the maximum of each set when combining the resources. Finally, the
        dirty wires cannot be reused between circuits, thus we always add them together.

        .. code-block::

            from collections import defaultdict

            # Resource reps for each operator:
            H = plre.resource_rep(plre.Hadamard)
            X = plre.resource_rep(plre.X)
            Z = plre.resource_rep(plre.Z)
            CNOT = plre.resource_rep(plre.CNOT)

            # state of wires:
            wm1 = WireResourceManager(work_wires={"clean":2, "dirty":1}, algo_wires=3)
            wm2 = WireResourceManager(work_wires={"clean":1, "dirty":2}, algo_wires=4)

            # state of gates:
            gt1 = defaultdict(int, {H: 10, X:5, CNOT:2})
            gt2 = defaultdict(int, {H: 15, Z:5, CNOT:4})

            # resources:
            res1 = plre.Resources(wire_manager=wm1, gate_types=gt1)
            res2 = plre.Resources(wire_manager=wm2, gate_types=gt2)


        .. code-block:: pycon

            >>> print(res1)
            --- Resources: ---
             Total wires: 6
                algorithmic wires: 3
                allocated qubits: 3
                     clean qubits: 2
                     dirty qubits: 1
             Total gates : 17
              'Hadamard': 10,
              'X': 5,
              'CNOT': 2

            >>> print(res2)
            --- Resources: ---
             Total qubits: 7
                algorithmic qubits: 4
                allocated qubits: 3
                     clean qubits: 1
                     dirty qubits: 2
             Total gates : 24
              'Hadamard': 15,
              'Z': 5,
              'CNOT': 4


        Specifically, users can add together two instances of resources using the :code:`+` and
        :code:`&` operators. These represent combining the resources assuming the circuits were
        executed in series or parallel respectively.

        .. code-block:: pycon

            >>> res_in_series = res1 + res2
            >>> print(res_in_series)
            --- Resources: ---
             Total qubits: 9
                algorithmic qubits: 4
                allocated qubits: 5
                     clean qubits: 2
                     dirty qubits: 3
             Total gates : 41
              'Hadamard': 25,
              'X': 5,
              'CNOT': 6,
              'Z': 5

            >>> res_in_parallel = res1 & res2
            >>> print(res_in_parallel)
            --- Resources: ---
             Total qubits: 12
                algorithmic qubits: 7
                allocated qubits: 5
                     clean qubits: 2
                     dirty qubits: 3
             Total gates : 41
              'Hadamard': 25,
              'X': 5,
              'CNOT': 6,
              'Z': 5

        Similarly, users can scale up the resources for an operator by some integer factor using
        the :code:`*` and :code:`@` operators. These represent scaling the resources assuming the
        circuits were executed in series or parallel respectively.

        .. code-block:: pycon

            >>> res_in_series = 5 * res1
            >>> print(res_in_series)
            --- Resources: ---
             Total qubits: 10
                algorithmic qubits: 3
                allocated qubits: 7
                     clean qubits: 2
                     dirty qubits: 5
             Total gates : 85
              'Hadamard': 50,
              'X': 25,
              'CNOT': 10

            >>> res_in_parallel = 5 @ res1
            >>> print(res_in_parallel)
            --- Resources: ---
             Total qubits: 22
                algorithmic qubits: 15
                allocated qubits: 7
                     clean qubits: 2
                     dirty qubits: 5
             Total gates : 85
              'Hadamard': 50,
              'X': 25,
              'CNOT': 10

    """

    def __init__(self, wire_manager: WireResourceManager, gate_types: None | dict = None):
        """Initialize the Resources class."""
        gate_types = gate_types or {}

        self.wire_manager = wire_manager
        self.gate_types = (
            gate_types
            if (isinstance(gate_types, defaultdict) and isinstance(gate_types.default_factory, int))
            else defaultdict(int, gate_types)
        )

    def __add__(self, other: Resources) -> Resources:
        """Add two resources objects in series"""
        assert isinstance(other, self.__class__)
        return add_in_series(self, other)

    def __and__(self, other: Resources) -> Resources:
        """Add two resources objects in parallel"""
        assert isinstance(other, self.__class__)
        return add_in_parallel(self, other)

    def __eq__(self, other: Resources) -> bool:
        """Determine if two resources objects are equal"""
        return (self.gate_types == other.gate_types) and (self.wire_manager == other.wire_manager)

    def __mul__(self, scalar: int) -> Resources:
        """Scale a resources object in series"""
        assert isinstance(scalar, int)
        return mul_in_series(self, scalar)

    def __matmul__(self, scalar: int) -> Resources:
        """Scale a resources object in parallel"""
        assert isinstance(scalar, int)
        return mul_in_parallel(self, scalar)

    __rmul__ = __mul__
    __radd__ = __add__
    __rand__ = __and__
    __rmatmul__ = __matmul__

    @property
    def clean_gate_counts(self):
        r"""Produce a dictionary which stores the gate counts
        using the operator names as keys.

        Returns:
            dict: A dictionary with operator names (str) as keys
                and the number of occurances in the circuit (int) as values.
        """
        clean_gate_counts = defaultdict(int)

        for cmp_res_op, counts in self.gate_types.items():
            clean_gate_counts[cmp_res_op.name] += counts

        return clean_gate_counts

    def __str__(self):
        """Generates a string representation of the Resources object."""
        wm = self.wire_manager
        total_wires = wm.total_wires
        total_gates = sum(self.clean_gate_counts.values())

        total_gates_str = str(total_gates) if total_gates <= 999 else f"{Decimal(total_gates):.3E}"
        total_wires_str = (
            str(total_wires) if total_wires <= 9999 else f"{Decimal(total_wires):.3E}"
        )

        items = "--- Resources: ---\n"
        items += f" Total wires: {total_wires_str}\n"

        qubit_breakdown_str = f"    algorithmic wires: {wm.algo_wires}\n    allocated wires: {wm.clean_wires+wm.dirty_wires}\n\t clean wires: {wm.clean_wires}\n\t dirty wires: {wm.dirty_wires}\n"
        items += qubit_breakdown_str

        items += f" Total gates : {total_gates_str}\n  "

        gate_type_str = ",\n  ".join(
            [
                f"'{gate_name}': {Decimal(count):.3E}" if count > 999 else f"'{gate_name}': {count}"
                for gate_name, count in self.clean_gate_counts.items()
            ]
        )
        items += gate_type_str

        return items

    def __repr__(self):
        """Compact string representation of the Resources object"""
        return {
            "wire_manager": self.wire_manager,
            "gate_types": self.gate_types,
        }.__repr__()

    def _ipython_display_(self):  # pragma: no cover
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
    wm1, wm2 = (first.wire_manager, other.wire_manager)

    new_clean = max(wm1.clean_wires, wm2.clean_wires)
    new_dirty = wm1.dirty_wires + wm2.dirty_wires
    new_budget = wm1.tight_budget or wm2.tight_budget
    new_logic = max(wm1.algo_wires, wm2.algo_wires)

    new_wire_manager = WireResourceManager(
        work_wires={"clean": new_clean, "dirty": new_dirty}, tight_budget=new_budget
    )

    new_wire_manager.algo_wires = new_logic
    new_gate_types = defaultdict(int, Counter(first.gate_types) + Counter(other.gate_types))
    return Resources(new_wire_manager, new_gate_types)


def add_in_parallel(first: Resources, other) -> Resources:  # &
    r"""Add two resources assuming the circuits are executed in parallel.

    Args:
        first (Resources): first resource object to combine
        other (Resources): other resource object to combine with

    Returns:
        Resources: combined resources
    """
    qm1, qm2 = (first.wire_manager, other.wire_manager)

    new_clean = max(qm1.clean_wires, qm2.clean_wires)
    new_dirty = qm1.dirty_wires + qm2.dirty_wires
    new_budget = qm1.tight_budget or qm2.tight_budget
    new_logic = qm1.algo_wires + qm2.algo_wires

    new_wire_manager = WireResourceManager(
        work_wires={"clean": new_clean, "dirty": new_dirty},
        tight_budget=new_budget,
    )

    new_wire_manager.algo_wires = new_logic
    new_gate_types = defaultdict(int, Counter(first.gate_types) + Counter(other.gate_types))
    return Resources(new_wire_manager, new_gate_types)


def mul_in_series(first: Resources, scalar: int) -> Resources:  # *
    r"""Multiply the resources by a scalar assuming the circuits are executed in series.

    Args:
        first (Resources): first resource object to scale
        scalar (int): integer value to scale the resources by

    Returns:
        Resources: scaled resources
    """
    qm = first.wire_manager

    new_clean = qm.clean_wires
    new_dirty = scalar * qm.dirty_wires
    new_budget = qm.tight_budget
    new_logic = qm.algo_wires

    new_wire_manager = WireResourceManager(
        work_wires={"clean": new_clean, "dirty": new_dirty},
        tight_budget=new_budget,
    )

    new_wire_manager.algo_wires = new_logic
    new_gate_types = defaultdict(int, {k: v * scalar for k, v in first.gate_types.items()})

    return Resources(new_wire_manager, new_gate_types)


def mul_in_parallel(first: Resources, scalar: int) -> Resources:  # @
    r"""Multiply the resources by a scalar assuming the circuits are executed in parallel.

    Args:
        first (Resources): first resource object to scale
        scalar (int): integer value to scale the resources by

    Returns:
        Resources: scaled resources
    """
    qm = first.wire_manager

    new_clean = qm.clean_wires
    new_dirty = scalar * qm.dirty_wires
    new_budget = qm.tight_budget
    new_logic = scalar * qm.algo_wires

    new_wire_manager = WireResourceManager(
        work_wires={"clean": new_clean, "dirty": new_dirty},
        tight_budget=new_budget,
    )

    new_wire_manager.algo_wires = new_logic
    new_gate_types = defaultdict(int, {k: v * scalar for k, v in first.gate_types.items()})

    return Resources(new_wire_manager, new_gate_types)
