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
    r"""Stores the estimated resource requirements of a quantum circuit.

    The :func:`~pennylane.estimator.estimate` function returns an object of this class. It contains
    estimates of all resource types tracked by the resource estimation pipeline, including the
    number of gates and the number of wires.

    Args:
        zeroed_wires (int): Number of allocated wires returned in the zeroed state.
        any_state_wires (int): Number of allocated wires returned in an unknowned state.
        algo_wires (int): Number of algorithmic wires, default value is ``0``.
        gate_types (dict): A dictionary mapping operations (:class:`~.pennylane.estimator.ResourceOperator`) to
            their number of occurences in the decomposed circuit.

    **Example**

    .. code-block:: python

        import pennylane.estimator as qre

        def circuit():
            qre.Hadamard()
            qre.CNOT()
            qre.RX(precision=1e-8)
            qre.RX(precision=1e-6)
            qre.AliasSampling(num_coeffs=3)

    >>> res = qre.estimate(circuit, gate_set={"RX", "Toffoli", "T", "CNOT", "Hadamard"})()
    >>> print(res)
    --- Resources: ---
     Total wires: 123
       algorithmic wires: 2
       allocated wires: 121
         zero state: 58
         any state: 63
     Total gates : 2.248E+3
       'RX': 2,
       'Toffoli': 65,
       'T': 868,
       'CNOT': 639,
       'Hadamard': 674

    You can also access a more detailed breakdown of resources using the
    :meth:`~.estimator.resources_base.Resources.gate_breakdown` method

    >>> print(res.gate_breakdown())
    RX total: 2
        RX {'precision': 1e-08}: 1
        RX {'precision': 1e-06}: 1
    Toffoli total: 65
        Toffoli {'elbow': None}: 4
        Toffoli {'elbow': 'left'}: 61
    T total: 868
    CNOT total: 639
    Hadamard total: 674

    """

    def __init__(
        self,
        zeroed_wires: int,
        any_state_wires: int = 0,
        algo_wires: int = 0,
        gate_types: dict | None = None,
    ):
        """Initialize the Resources class."""
        gate_types = gate_types or {}
        self.zeroed_wires = zeroed_wires
        self.any_state_wires = any_state_wires
        self.algo_wires = algo_wires
        self.gate_types = defaultdict(int, gate_types)

    def add_series(self, other: "Resources") -> "Resources":
        """Add two Resources objects in series.

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

        >>> import pennylane.estimator as qre
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

        new_zeroed = max(self.zeroed_wires, other.zeroed_wires)
        new_any = self.any_state_wires + other.any_state_wires
        new_logic = max(self.algo_wires, other.algo_wires)

        new_gate_types = defaultdict(int, Counter(self.gate_types) + Counter(other.gate_types))
        return Resources(
            zeroed_wires=new_zeroed,
            any_state_wires=new_any,
            algo_wires=new_logic,
            gate_types=new_gate_types,
        )

    def add_parallel(self, other: "Resources") -> "Resources":
        """Add two Resources objects in parallel.

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

        >>> import pennylane.estimator as qre
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

        new_zeroed = max(self.zeroed_wires, other.zeroed_wires)
        new_any = self.any_state_wires + other.any_state_wires
        new_logic = self.algo_wires + other.algo_wires

        new_gate_types = defaultdict(int, Counter(self.gate_types) + Counter(other.gate_types))
        return Resources(
            zeroed_wires=new_zeroed,
            any_state_wires=new_any,
            algo_wires=new_logic,
            gate_types=new_gate_types,
        )

    def __eq__(self, other: Resources) -> bool:
        """Determine if two resources objects are equal"""
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Cannot compare {self.__class__.__name__} with object of type {type(other)}."
            )
        return (
            (self.gate_types == other.gate_types)
            and (self.zeroed_wires == other.zeroed_wires)
            and (self.any_state_wires == other.any_state_wires)
            and (self.algo_wires == other.algo_wires)
        )

    def multiply_series(self, scalar: int) -> Resources:
        """Scale a Resources object in series

        Args:
            scalar (int): integer value by which to scale the resources

        Returns:
            :class:`~.pennylane.estimator.Resources`: scaled resources

        **Example**

        >>> import pennylane.estimator as qre
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
            zeroed_wires=self.zeroed_wires,
            any_state_wires=self.any_state_wires * scalar,
            algo_wires=self.algo_wires,
            gate_types=new_gate_types,
        )

    def multiply_parallel(self, scalar: int) -> Resources:
        """Scale a Resources object in parallel

        Args:
            scalar (int): integer value by which to scale the resources

        Returns:
            :class:`~.pennylane.estimator.Resources`: scaled resources

        **Example**

        >>> import pennylane.estimator as qre
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
            zeroed_wires=self.zeroed_wires,
            any_state_wires=self.any_state_wires * scalar,
            algo_wires=self.algo_wires * scalar,
            gate_types=new_gate_types,
        )

    @property
    def gate_counts(self) -> dict:
        r"""Produce a dictionary which stores the gate counts
        using the operator names as keys.

        Returns:
            dict: A dictionary with operator names (str) as keys
                and the number of occurrences in the circuit (int) as values.
        """
        gate_counts = defaultdict(int)

        for cmp_res_op, counts in self.gate_types.items():
            gate_counts[cmp_res_op.name] += counts

        return gate_counts

    def __str__(self):
        """Generates a string representation of the Resources object."""

        total_wires = self.algo_wires + self.zeroed_wires + self.any_state_wires
        total_gates = sum(self.gate_counts.values())

        total_gates_str = str(total_gates) if total_gates <= 999 else f"{Decimal(total_gates):.3E}"
        total_wires_str = str(total_wires) if total_wires <= 9999 else f"{Decimal(total_wires):.3E}"

        items = "--- Resources: ---\n"
        items += f" Total wires: {total_wires_str}\n"

        qubit_breakdown_str = (
            f"   algorithmic wires: {self.algo_wires}\n"
            f"   allocated wires: {self.zeroed_wires + self.any_state_wires}\n"
            f"     zero state: {self.zeroed_wires}\n"
            f"     any state: {self.any_state_wires}\n"
        )
        items += qubit_breakdown_str

        items += f" Total gates : {total_gates_str}\n"

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
        gate_type_str = ",\n".join(
            [
                (
                    f"   '{gate_name}': {Decimal(count):.3E}"
                    if count > 999
                    else f"   '{gate_name}': {count}"
                )
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

        >>> import pennylane.estimator as qre
        >>> def circ():
        ...     qre.SemiAdder(10)
        ...     qre.Toffoli()
        ...     qre.RX(precision=1e-5)
        ...     qre.RX(precision=1e-7)
        >>> res1 = qre.estimate(circ, gate_set=['Toffoli', 'RX', 'CNOT', 'Hadamard'])()
        >>> print(res1.gate_breakdown())
        RX total: 2
            RX {'precision': 1e-05}: 1
            RX {'precision': 1e-07}: 1
        Toffoli total: 10
            Toffoli {'elbow': 'left'}: 9
            Toffoli {'elbow': None}: 1
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
        return f"Resources(zeroed={self.zeroed_wires}, any_state={self.any_state_wires}, algo_wires={self.algo_wires}, gate_types={self.gate_types})"
