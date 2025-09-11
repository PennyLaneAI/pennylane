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

DefaultGateSet = frozenset(
    {
        "Toffoli",
        "T",
        "CNOT",
        "X",
        "Y",
        "Z",
        "S",
        "Hadamard",
    }
)


class Resources:
    r"""A container for the resources used throughout a quantum circuit.

    The resources tracked include number of wires, number of gates, and their gate types.

    Args:
        zeroed (int): Number of zeroed state work wires.
        any_state (int): Number of work wires in an unknown state, default is ``0``.
        algo_wires (int): Number of algorithmic wires, default value is ``0``.
        gate_types (dict): A dictionary storing operations (:class:`~.pennylane.estimator.ResourceOperator`) as keys and the number
            of times they are used in the circuit (``int``) as values.

    **Example**

    >>> from pennylane import estimator as qre
    >>> H = qre.resource_rep(qre.Hadamard)
    >>> X = qre.resource_rep(qre.X)
    >>> RX = qre.resource_rep(qre.RX, {"precision":1e-8})
    >>> RX_2 = qre.resource_rep(qre.RX, {"precision":1e-6})
    >>> gt = defaultdict(int, {H: 10, X:7, RX:2, RX_2:2})
    >>>
    >>> res = qre.Resources(zeroed=3, gate_types=gt)
    >>> print(res)
    --- Resources: ---
     Total wires: 2
        algorithmic wires: 0
        allocated wires: 2
             zero state: 2
             any state: 0
     Total gates : 21
      'RX': 4,
      'X': 7,
      'Hadamard': 10
    >>>
    >>> print(res.gate_breakdown())
    RX total: 4
        RX {'eps': 1e-08}: 2
        RX {'eps': 1e-06}: 2
    X total: 7
    Hadamard total: 10

    """

    def __init__(
        self, zeroed: int, any_state: int = 0, algo: int = 0, gate_types: dict | None = None
    ):
        """Initialize the Resources class."""
        gate_types = gate_types or {}

        self.zeroed = zeroed
        self.any_state = any_state
        self.algo_wires = algo
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
            other (:class:`~.pennylane.estimator.Resources`): the other resource object to add in series with

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
                 zero state: 2
                 any state: 0
         Total gates : 838
          'T': 796,
          'CNOT': 28,
          'Z': 2,
          'S': 3,
          'Hadamard': 9

        """
        if not isinstance(other, self.__class__):
            raise TypeError(f"Cannot add {self.__class__.__name__} object to {type(other)}.")

        new_zeroed = max(self.zeroed, other.zeroed)
        new_any = self.any_state + other.any_state
        new_logic = max(self.algo_wires, other.algo_wires)

        new_gate_types = defaultdict(int, Counter(self.gate_types) + Counter(other.gate_types))
        return Resources(
            zeroed=new_zeroed, any_state=new_any, algo=new_logic, gate_types=new_gate_types
        )

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
                 zero state: 2
                 any state: 0
         Total gates : 838
          'T': 796,
          'CNOT': 28,
          'Z': 2,
          'S': 3,
          'Hadamard': 9

        """
        if not isinstance(other, self.__class__):
            raise TypeError(f"Cannot add {self.__class__.__name__} object to {type(other)}.")

        new_zeroed = max(self.zeroed, other.zeroed)
        new_any = self.any_state + other.any_state
        new_logic = self.algo_wires + other.algo_wires

        new_gate_types = defaultdict(int, Counter(self.gate_types) + Counter(other.gate_types))
        return Resources(
            zeroed=new_zeroed, any_state=new_any, algo=new_logic, gate_types=new_gate_types
        )

    def __eq__(self, other: Resources) -> bool:
        """Determine if two resources objects are equal"""
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Cannot compare {self.__class__.__name__} with object of type {type(other)}."
            )
        return (
            (self.gate_types == other.gate_types)
            and (self.zeroed == other.zeroed)
            and (self.any_state == other.any_state)
            and (self.algo_wires == other.algo_wires)
        )

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
                 zero state: 2
                 any state: 0
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

        new_gate_types = defaultdict(int, {k: v * scalar for k, v in self.gate_types.items()})

        return Resources(
            zeroed=self.zeroed,
            any_state=self.any_state * scalar,
            algo=self.algo_wires,
            gate_types=new_gate_types,
        )

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
                 zero state: 2
                 any state: 0
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

        new_gate_types = defaultdict(int, {k: v * scalar for k, v in self.gate_types.items()})

        return Resources(
            zeroed=self.zeroed,
            any_state=self.any_state * scalar,
            algo=self.algo_wires * scalar,
            gate_types=new_gate_types,
        )

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

        total_wires = self.algo_wires + self.zeroed + self.any_state
        total_gates = sum(self.gate_counts.values())

        total_gates_str = str(total_gates) if total_gates <= 999 else f"{Decimal(total_gates):.3E}"
        total_wires_str = str(total_wires) if total_wires <= 9999 else f"{Decimal(total_wires):.3E}"

        items = "--- Resources: ---\n"
        items += f" Total wires: {total_wires_str}\n"

        qubit_breakdown_str = f"    algorithmic wires: {self.algo_wires}\n    allocated wires: {self.zeroed+self.any_state}\n\t zero state: {self.zeroed}\n\t any state: {self.any_state}\n"
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

    def gate_breakdown(self, gate_set=None):
        """Generates a string breakdown of gate counts by type and parameters,
        optionally for a specific set of gates.

        Args:
            gate_set (list): A list of gate names to break down.
                If ``None``, details will be provided for all gate types.

        **Example**

        >>> from pennylane import estimator as qre
        >>> res1 = qre.estimate(qre.SemiAdder(10))
        >>> print(res1.gate_breakdown())
        Toffoli total: 9
            Toffoli {'elbow': 'left'}: 9
        CNOT total: 60
        Hadamard total: 27

        """
        output_lines = []

        default_gate_list = ["Toffoli", "T", "CNOT", "X", "Y", "Z", "S", "Hadamard"]

        def add_gate_to_output(gate_name, counts_dict):
            """Formats and adds a single gate breakdown to the output list."""
            total_count = sum(counts_dict.values())
            if total_count == 0:
                return

            total_count_str = (
                str(total_count) if total_count <= 999 else f"{Decimal(total_count):.3E}"
            )
            output_lines.append(f"{gate_name} total: {total_count_str}")

            if any(params != () for params in counts_dict):
                for params, count in counts_dict.items():
                    if count > 0:
                        count_str = str(count) if count <= 999 else f"{Decimal(count):.3E}"
                        output_lines.append(f"    {gate_name} {dict(params)}: {count_str}")

        if gate_set is None:
            all_gate_counts = defaultdict(lambda: defaultdict(int))
            for op, count in self.gate_types.items():
                params_tuple = tuple(sorted(op.params.items())) if op.params else ()
                all_gate_counts[op.name][params_tuple] += count

            other_gates = sorted(
                [name for name in all_gate_counts if name not in set(default_gate_list)]
            )

            for gate_name in other_gates:
                add_gate_to_output(gate_name, all_gate_counts[gate_name])

            for gate_name in default_gate_list:
                if gate_name in all_gate_counts:
                    add_gate_to_output(gate_name, all_gate_counts[gate_name])

        else:
            for gate_name in gate_set:
                parameter_counts = defaultdict(int)
                for op, count in self.gate_types.items():
                    if op.name == gate_name:
                        params_tuple = tuple(sorted(op.params.items()))
                        parameter_counts[params_tuple] += count

                add_gate_to_output(gate_name, parameter_counts)

        return "\n".join(output_lines)

    def __repr__(self):
        """Compact string representation of the Resources object"""
        return f"Resources(zeroed={self.zeroed}, any_state={self.any_state}, algo={self.algo_wires}, gate_types={self.gate_types})"
