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

from collections import Counter, defaultdict
from decimal import Decimal

from .wires_manager import WireResourceManager

DefaultGateSet = {
    "Toffoli",
    "T",
    "CNOT",
    "X",
    "Y",
    "Z",
    "S",
    "Hadamard",
}


class Resources:
    r"""A container to track and update the resources used throughout a quantum circuit.

    The resources tracked include number of gates, number of wires, and gate types.

    Args:
        wire_manager (:class:`~.pennylane.estimator.WireResourceManager`): A wire tracker class which contains the number of available
            work wires, categorized as zero state and any state wires, and algorithmic wires.
        gate_types (dict): A dictionary storing operations (:class:`~.pennylane.estimator.ResourceOperator`) as keys and the number
            of times they are used in the circuit (``int``) as values.

    **Example**

    >>> from pennylane import estimator as qre
    >>> H = qre.resource_rep(qre.Hadamard)
    >>> X = qre.resource_rep(qre.X)
    >>> Y = qre.resource_rep(qre.Y)
    >>> wm = qre.WireResourceManager(work_wires=3)
    >>> gt = defaultdict(int, {H: 10, X:7, Y:3})
    >>>
    >>> res = qre.Resources(wire_manager=wm, gate_types=gt)
    >>> print(res)
    --- Resources: ---
     Total wires: 3
        algorithmic wires: 0
        allocated wires: 3
             zero state: 3
             any state: 0
     Total gates : 20
      'X': 7,
      'Y': 3,
      'Hadamard': 10

    .. details::
        :title: Usage Details

        The :class:`~.pennylane.estimator.Resources` object supports arithmetic operations which allow for quick addition
        and multiplication of resources. When combining resources, we can make a simplifying
        assumption about how they are applied in a quantum circuit: in series or in parallel.

        When assuming the circuits are executed in parallel, the number of algorithmic wires add
        together. When assuming the circuits are executed in series, the maximum of each set of
        algorithmic wires is used. The ``zeroed`` auxiliary wires can be reused between the circuits,
        and thus we always use the maximum of each set when combining the resources. Finally, the
        ``any_state`` wires cannot be reused between circuits, thus we always add them together.

        .. code-block::

            from collections import defaultdict

            # Resource reps for each operator:
            H = plre.resource_rep(plre.Hadamard)
            X = plre.resource_rep(plre.X)
            Z = plre.resource_rep(plre.Z)
            CNOT = plre.resource_rep(plre.CNOT)

            # state of wires:
            wm1 = plre.WireResourceManager(zeroed=2, any_state=1, algo_wires=3)
            wm2 = plre.WireResourceManager(zeroed=1, any_state=2, algo_wires=4)

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
                allocated wires: 3
                     zero state: 2
                     any state: 1
             Total gates : 17
              'CNOT': 2,
              'X': 5,
              'Hadamard': 10

            >>> print(res2)
            --- Resources: ---
             Total wires: 7
                algorithmic wires: 4
                allocated wires: 3
                     zero state: 1
                     any state: 2
             Total gates : 24
              'CNOT': 4,
              'Z': 5,
              'Hadamard': 15

    """

    def __init__(self, wire_manager: WireResourceManager, gate_types: dict | None = None):
        """Initialize the Resources class."""
        gate_types = gate_types or {}

        self.wire_manager = wire_manager
        self.gate_types = (
            gate_types
            if (isinstance(gate_types, defaultdict) and isinstance(gate_types.default_factory, int))
            else defaultdict(int, gate_types)
        )

    def add_series(self, other: "Resources") -> "Resources":
        """Add two resources objects in series.

        When combining resources for serial execution, the following rules apply:
        * Zeroed wires: The total ``zeroed`` auxiliary wires are the maximum of the ``zeroed``
          wires in each circuit, as they can be reused.
        * Any state wires: The ``any_state`` wires are added together, as they cannot be reused.
        * Algorithmic wires: The total ``algo_wires`` are the maximum of the ``algo_wires``
          from each circuit.
        * Gates: The gates from each circuit are added together.

        Args:
            other (:class:`~.pennylane.estimator.Resources`): other resource object to combine with

        Returns:
            :class:`~.pennylane.estimator.Resources`: combined resources

        **Example**

        >>> from pennylane import estimator as qre
        >>> gate_set = {"X", "Y", "Z", "CNOT", "T", "S", "Hadamard"}
        >>> res1 = qre.estimate(qre.Toffoli(), gate_set)
        >>> res2 = qre.estimate(qre.QFT(num_wires=4), gate_set)
        >>> res_in_series = res1.add_series(res2)
        >>> print(res_in_series)
        --- Resources: ---
         Total wires: 6
            algorithmic wires: 4
            allocated wires: 2
                 clean wires: 2
                 dirty wires: 0
         Total gates : 838
          'T': 796,
          'CNOT': 28,
          'Z': 2,
          'S': 3,
          'Hadamard': 9

        """
        if not isinstance(other, self.__class__):
            raise TypeError(f"Cannot add {self.__class__.__name__} object to {type(other)}.")

        wm1 = self.wire_manager
        wm2 = other.wire_manager

        new_zeroed = max(wm1.zeroed, wm2.zeroed)
        new_any = wm1.any_state + wm2.any_state
        new_budget = wm1.tight_budget or wm2.tight_budget
        new_logic = max(wm1.algo_wires, wm2.algo_wires)

        new_wire_manager = WireResourceManager(
            zeroed=new_zeroed, any_state=new_any, algo=new_logic, tight_budget=new_budget
        )

        new_gate_types = defaultdict(int, Counter(self.gate_types) + Counter(other.gate_types))
        return Resources(new_wire_manager, new_gate_types)

    def add_parallel(self, other: "Resources") -> "Resources":
        """Add two resources objects in parallel.

        When combining resources for parallel execution, the following rules apply:
        * Zeroed wires: The maximum of the ``zeroed`` auxiliary wires is used, as they can
          be reused across parallel circuits.
        * Any state wires: The ``any_state`` wires are added together, as they cannot be
          reused between circuits.
        * Algorithmic wires: The ``algo_wires`` are added together, as each circuit is a
          separate unit running simultaneously.
        * Gates: The gates from each circuit are added together.

        Args:
            other (:class:`~.pennylane.estimator.Resources`): other resource object to combine with

        Returns:
            :class:`~.pennylane.estimator.Resources`: combined resources

        **Example**

        >>> from pennylane import estimator as qre
        >>> gate_set = {"X", "Y", "Z", "CNOT", "T", "S", "Hadamard"}
        >>> res1 = qre.estimate(qre.Toffoli(), gate_set)
        >>> res2 = qre.estimate(qre.QFT(num_wires=4), gate_set)
        >>> res_in_parallel = res1.add_parallel(res2)
        >>> print(res_in_parallel)
        --- Resources: ---
         Total wires: 9
            algorithmic wires: 7
            allocated wires: 2
                 clean wires: 2
                 dirty wires: 0
         Total gates : 838
          'T': 796,
          'CNOT': 28,
          'Z': 2,
          'S': 3,
          'Hadamard': 9
        """
        if not isinstance(other, self.__class__):
            raise TypeError(f"Cannot add {self.__class__.__name__} object to {type(other)}.")

        qm1 = self.wire_manager
        qm2 = other.wire_manager

        new_zeroed = max(qm1.zeroed, qm2.zeroed)
        new_any = qm1.any_state + qm2.any_state
        new_budget = qm1.tight_budget or qm2.tight_budget
        new_logic = qm1.algo_wires + qm2.algo_wires

        new_wire_manager = WireResourceManager(
            zeroed=new_zeroed,
            any_state=new_any,
            algo=new_logic,
            tight_budget=new_budget,
        )

        new_gate_types = defaultdict(int, Counter(self.gate_types) + Counter(other.gate_types))
        return Resources(new_wire_manager, new_gate_types)

    def __eq__(self, other: Resources) -> bool:
        """Determine if two resources objects are equal"""
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Cannot compare {self.__class__.__name__} with object of type {type(other)}."
            )
        return (self.gate_types == other.gate_types) and (self.wire_manager == other.wire_manager)

    def multiply_series(self, scalar: int) -> Resources:
        """Scale a resources object in series

        Args:
            scalar (int): integer value by which to scale the resources

        Returns:
            :class:`~.pennylane.estimator.Resources`: scaled resources

        **Example**

        >>> from pennylane import estimator as qre
        >>> gate_set = {"X", "Y", "Z", "CNOT", "T", "S", "Hadamard"}
        >>> res1 = qre.estimate(qre.Toffoli(), gate_set)
        >>> res_in_series = res1.multiply_series(3)
        >>> print(res_in_series)
        --- Resources: ---
         Total wires: 5
            algorithmic wires: 3
            allocated wires: 2
                 clean wires: 2
                 dirty wires: 0
         Total gates : 72
          'T': 12,
          'CNOT': 30,
          'Z': 6,
          'S': 9,
          'Hadamard': 15

        """
        if not isinstance(scalar, int):
            raise TypeError(
                f"Cannot multiply {self.__class__.__name__} object with {type(scalar)}."
            )

        new_wire_manager = WireResourceManager(
            zeroed=self.wire_manager.zeroed,
            any_state=scalar * self.wire_manager.any_state,
            algo=self.wire_manager.algo_wires,
            tight_budget=self.wire_manager.tight_budget,
        )

        new_gate_types = defaultdict(int, {k: v * scalar for k, v in self.gate_types.items()})

        return Resources(new_wire_manager, new_gate_types)

    def multiply_parallel(self, scalar: int) -> Resources:
        """Scale a resources object in parallel

        Args:
            scalar (int): integer value by which to scale the resources

        Returns:
            :class:`~.pennylane.estimator.Resources`: scaled resources

        **Example**

        >>> from pennylane import estimator as qre
        >>> gate_set = {"X", "Y", "Z", "CNOT", "T", "S", "Hadamard"}
        >>> res1 = qre.estimate(qre.Toffoli(), gate_set)
        >>> res_in_parallel = res1.multiply_parallel(3)
        >>> print(res_in_parallel)
        --- Resources: ---
         Total wires: 11
            algorithmic wires: 9
            allocated wires: 2
                 clean wires: 2
                 dirty wires: 0
         Total gates : 72
          'T': 12,
          'CNOT': 30,
          'Z': 6,
          'S': 9,
          'Hadamard': 15
        """

        if not isinstance(scalar, int):
            raise TypeError(
                f"Cannot multiply {self.__class__.__name__} object with {type(scalar)}."
            )

        new_wire_manager = WireResourceManager(
            zeroed=self.wire_manager.zeroed,
            any_state=scalar * self.wire_manager.any_state,
            algo=scalar * self.wire_manager.algo_wires,
            tight_budget=self.wire_manager.tight_budget,
        )

        new_gate_types = defaultdict(int, {k: v * scalar for k, v in self.gate_types.items()})

        return Resources(new_wire_manager, new_gate_types)

    @property
    def gate_counts(self) -> dict:
        r"""Produce a dictionary which stores the gate counts
        using the operator names as keys.

        Returns:
            dict: A dictionary with operator names (str) as keys
                and the number of occurances in the circuit (int) as values.
        """
        gate_counts = defaultdict(int)

        for cmp_res_op, counts in self.gate_types.items():
            gate_counts[cmp_res_op.name] += counts

        return gate_counts

    def __str__(self):
        """Generates a string representation of the Resources object."""

        wm = self.wire_manager
        total_wires = wm.total_wires
        total_gates = sum(self.gate_counts.values())

        total_gates_str = str(total_gates) if total_gates <= 999 else f"{Decimal(total_gates):.3E}"
        total_wires_str = str(total_wires) if total_wires <= 9999 else f"{Decimal(total_wires):.3E}"

        items = "--- Resources: ---\n"
        items += f" Total wires: {total_wires_str}\n"

        qubit_breakdown_str = f"    algorithmic wires: {wm.algo_wires}\n    allocated wires: {wm.zeroed+wm.any_state}\n\t zero state: {wm.zeroed}\n\t any state: {wm.any_state}\n"
        items += qubit_breakdown_str

        items += f" Total gates : {total_gates_str}\n  "

        gate_counts = self.gate_counts
        custom_gates = []
        default_gates = []
        for gate_name, count in gate_counts.items():
            if gate_name not in DefaultGateSet:
                custom_gates.append((gate_name, count))
            else:
                default_gates.append((gate_name, count))

        res_order = ["Toffoli", "T", "CNOT", "X", "Y", "Z", "S", "Hadamard"]

        gate_order_map = {name: i for i, name in enumerate(res_order)}
        default_gates.sort(key=lambda x: gate_order_map.get(x[0], len(res_order)))

        ordered_gates = custom_gates + default_gates
        gate_type_str = ",\n  ".join(
            [
                f"'{gate_name}': {Decimal(count):.3E}" if count > 999 else f"'{gate_name}': {count}"
                for gate_name, count in ordered_gates
            ]
        )
        items += gate_type_str

        return items

    def __repr__(self):
        """Compact string representation of the Resources object"""
        return f"Resources(wire_manager={self.wire_manager}, gate_types={self.gate_types})"
