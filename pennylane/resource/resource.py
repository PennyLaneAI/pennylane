# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Stores classes and logic to aggregate all the resource information from a quantum workflow.
"""
from __future__ import annotations

import copy
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field

from pennylane.measurements import Shots, add_shots
from pennylane.operation import Operation


@dataclass(frozen=True)
class Resources:
    r"""Contains attributes which store key resources such as number of gates, number of wires, shots,
    depth and gate types.

    Args:
        num_wires (int): number of qubits
        num_gates (int): number of gates
        gate_types (dict): dictionary storing operation names (str) as keys
            and the number of times they are used in the circuit (int) as values
        gate_sizes (dict): dictionary storing the number of :math:`n` qubit gates in the circuit
            as a key-value pair where :math:`n` is the key and the number of occurances is the value
        depth (int): the depth of the circuit defined as the maximum number of non-parallel operations
        shots (Shots): number of samples to generate

    .. details::

        The resources being tracked can be accessed as class attributes.
        Additionally, the :code:`Resources` instance can be nicely displayed in the console.

        **Example**

        >>> from pennylane.resource import Resources
        >>> r = Resources(num_wires=2, num_gates=2, gate_types={'Hadamard': 1, 'CNOT':1}, gate_sizes={1: 1, 2: 1}, depth=2)
        >>> print(r)
        num_wires: 2
        num_gates: 2
        depth: 2
        shots: Shots(total=None)
        gate_types:
        {'Hadamard': 1, 'CNOT': 1}
        gate_sizes:
        {1: 1, 2: 1}

        :class:`~.Resources` objects can be added together or multiplied by a scalar.

        >>> from pennylane.resource import Resources
        >>> r1 = Resources(num_wires=2, num_gates=2, gate_types={'Hadamard': 1, 'CNOT':1}, gate_sizes={1: 1, 2: 1}, depth=2)
        >>> r2 = Resources(num_wires=2, num_gates=2, gate_types={'RX': 1, 'CNOT':1}, gate_sizes={1: 1, 2: 1}, depth=2)
        >>> print(r1 + r2)
        wires: 2
        gates: 4
        depth: 4
        shots: Shots(total=None)
        gate_types:
        {'Hadamard': 1, 'CNOT': 2, 'RX': 1}
        gate_sizes:
        {1: 2, 2: 2}
        >>> print(r1 * 2)
        wires: 2
        gates: 4
        depth: 4
        shots: Shots(total=None)
        gate_types:
        {'Hadamard': 2, 'CNOT': 2}
        gate_sizes:
        {1: 2, 2: 2}
    """

    num_wires: int = 0
    num_gates: int = 0
    gate_types: dict = field(default_factory=dict)
    gate_sizes: dict = field(default_factory=dict)
    depth: int = 0
    shots: Shots = field(default_factory=Shots)

    def __add__(self, other: Resources):
        r"""Adds two :class:`~resource.Resources` objects together as if the circuits were executed in series.

        Args:
            other (Resources): the resource object to add

        Returns:
            Resources: the combined resources

        .. details::

            **Example**

            First we build two :class:`~.resource.Resources` objects.

            .. code-block:: python3

                from pennylane.measurements import Shots
                from pennylane.resource import Resources

                r1 = Resources(
                    num_wires = 2,
                    num_gates = 2,
                    gate_types = {"Hadamard": 1, "CNOT": 1},
                    gate_sizes = {1: 1, 2: 1},
                    depth = 2,
                    shots = Shots(10)
                )

                r2 = Resources(
                    num_wires = 3,
                    num_gates = 2,
                    gate_types = {"RX": 1, "CNOT": 1},
                    gate_sizes = {1: 1, 2: 1},
                    depth = 1,
                    shots = Shots((5, (2, 10)))
                )

            Now we print their sum.

            >>> print(r1 + r2)
            wires: 3
            gates: 4
            depth: 3
            shots: Shots(total=35, vector=[10 shots, 5 shots, 2 shots x 10])
            gate_types:
            {'Hadamard': 1, 'CNOT': 2, 'RX': 1}
            gate_sizes:
            {1: 2, 2: 2}
        """
        return add_in_series(self, other)

    def __mul__(self, scalar: int):
        r"""Multiply the :class:`~resource.Resources` object by a scalar as if that many copies of the circuit were executed in series

        Args:
            scalar (int): the scalar to multiply the resource object by

        Returns:
            Resources: the combined resources

        .. details::

            **Example**

            First we build a :class:`~.resource.Resources` object.

            .. code-block:: python3

                from pennylane.measurements import Shots
                from pennylane.resource import Resources

                resources = Resources(
                    num_wires = 2,
                    num_gates = 2,
                    gate_types = {"Hadamard": 1, "CNOT": 1},
                    gate_sizes = {1: 1, 2: 1},
                    depth = 2,
                    shots = Shots(10)
                )

            Now we print the product.

            >>> print(resources * 2)
            wires: 2
            gates: 4
            depth: 4
            shots: Shots(total=20)
            gate_types:
            {'Hadamard': 2, 'CNOT': 2}
            gate_sizes:
            {1: 2, 2: 2}
        """
        return mul_in_series(self, scalar)

    __rmul__ = __mul__

    def __str__(self):
        keys = ["num_wires", "num_gates", "depth"]
        vals = [self.num_wires, self.num_gates, self.depth]
        items = "\n".join([str(i) for i in zip(keys, vals)])
        items = items.replace("('", "")
        items = items.replace("',", ":")
        items = items.replace(")", "")

        items += f"\nshots: {str(self.shots)}"

        gate_type_str = ", ".join(
            [f"'{gate_name}': {count}" for gate_name, count in self.gate_types.items()]
        )
        items += "\ngate_types:\n{" + gate_type_str + "}"

        gate_size_str = ", ".join(
            [f"{n_gate}: {count}" for n_gate, count in self.gate_sizes.items()]
        )
        items += "\ngate_sizes:\n{" + gate_size_str + "}"
        return items

    def _ipython_display_(self):
        """Displays __str__ in ipython instead of __repr__"""
        print(str(self))


class ResourcesOperation(Operation):
    r"""Base class that represents quantum gates or channels applied to quantum
    states and stores the resource requirements of the quantum gate.

    .. note::
        Child classes must implement the :func:`~.ResourcesOperation.resources` method which computes
        the resource requirements of the operation.
    """

    @abstractmethod
    def resources(self) -> Resources:
        r"""Compute the resources required for this operation.

        Returns:
            Resources: The resources required by this operation.

        **Examples**

        >>> class CustomOp(ResourcesOperation):
        ...     num_wires = 2
        ...     def resources(self):
        ...         return Resources(num_wires=self.num_wires, num_gates=3, depth=2)
        ...
        >>> op = CustomOp(wires=[0, 1])
        >>> print(op.resources())
        num_wires: 2
        num_gates: 3
        depth: 2
        shots: Shots(total=None)
        gate_types:
        {}
        gate_sizes:
        {}
        """


def add_in_series(r1: Resources, r2: Resources) -> Resources:
    r"""
    Add two :class:`~.resource.Resources` objects assuming the circuits are executed in series.

    The gates in ``r1`` and ``r2`` are assumed to act on the same qubits. The resulting circuit
    depth is the sum of the depths of ``r1`` and ``r2``. To add resources as if they were executed
    in parallel see :func:`~.resource.add_in_parallel`.

    Args:
        r1 (Resources): a :class:`~resource.Resources` to add
        r2 (Resources): a :class:`~resource.Resources` to add

    Returns:
        Resources: the combined resources

    .. details::

        **Example**

        First we build two :class:`~.resource.Resources` objects.

        .. code-block:: python3

            from pennylane.measurements import Shots
            from pennylane.resource import Resources

            r1 = Resources(
                num_wires = 2,
                num_gates = 2,
                gate_types = {"Hadamard": 1, "CNOT": 1},
                gate_sizes = {1: 1, 2: 1},
                depth = 2,
                shots = Shots(10)
            )

            r2 = Resources(
                num_wires = 3,
                num_gates = 2,
                gate_types = {"RX": 1, "CNOT": 1},
                gate_sizes = {1: 1, 2: 1},
                depth = 1,
                shots = Shots((5, (2, 10)))
            )

        Now we print their sum.

        >>> print(qml.resource.add_in_series(r1, r2))
        wires: 3
        gates: 4
        depth: 3
        shots: Shots(total=35, vector=[10 shots, 5 shots, 2 shots x 10])
        gate_types:
        {'Hadamard': 1, 'CNOT': 2, 'RX': 1}
        gate_sizes:
        {1: 2, 2: 2}
    """

    new_wires = max(r1.num_wires, r2.num_wires)
    new_gates = r1.num_gates + r2.num_gates
    new_gate_types = _combine_dict(r1.gate_types, r2.gate_types)
    new_gate_sizes = _combine_dict(r1.gate_sizes, r2.gate_sizes)
    new_shots = add_shots(r1.shots, r2.shots)
    new_depth = r1.depth + r2.depth

    return Resources(new_wires, new_gates, new_gate_types, new_gate_sizes, new_depth, new_shots)


def add_in_parallel(r1: Resources, r2: Resources) -> Resources:
    r"""
    Add two :class:`~.resource.Resources` objects assuming the circuits are executed in parallel.

    The gates in ``r2`` and ``r2`` are assumed to act on disjoint sets of qubits. The resulting
    circuit depth is the max depth of ``r1`` and ``r2``. To add resources as if they were executed
    in series see :func:`~.resource.add_in_series`.

    Args:
        r1 (Resources): a :class:`~.resource.Resources` object to add
        r2 (Resources): a :class:`~.resource.Resources` object to add

    Returns:
        Resources: the combined resources

    .. details::

        **Example**

        First we build two :class:`~.resource.Resources` objects.

        .. code-block:: python3

            from pennylane.measurements import Shots
            from pennylane.resource import Resources

            r1 = Resources(
                num_wires = 2,
                num_gates = 2,
                gate_types = {"Hadamard": 1, "CNOT": 1},
                gate_sizes = {1: 1, 2: 1},
                depth = 2,
                shots = Shots(10)
            )

            r2 = Resources(
                num_wires = 3,
                num_gates = 2,
                gate_types = {"RX": 1, "CNOT": 1},
                gate_sizes = {1: 1, 2: 1},
                depth = 1,
                shots = Shots((5, (2, 10)))
            )

        Now we print their sum.

        >>> print(qml.resource.add_in_parallel(r1, r2))
        wires: 5
        gates: 4
        depth: 2
        shots: Shots(total=35, vector=[10 shots, 5 shots, 2 shots x 10])
        gate_types:
        {'Hadamard': 1, 'CNOT': 2, 'RX': 1}
        gate_sizes:
        {1: 2, 2: 2}
    """

    new_wires = r1.num_wires + r2.num_wires
    new_gates = r1.num_gates + r2.num_gates
    new_gate_types = _combine_dict(r1.gate_types, r2.gate_types)
    new_gate_sizes = _combine_dict(r1.gate_sizes, r2.gate_sizes)
    new_shots = add_shots(r1.shots, r2.shots)
    new_depth = max(r1.depth, r2.depth)

    return Resources(new_wires, new_gates, new_gate_types, new_gate_sizes, new_depth, new_shots)


def mul_in_series(resources: Resources, scalar: int) -> Resources:
    """
    Multiply the :class:`~resource.Resources` object by a scalar as if the circuit was repeated that many times in series.

    The repeated copies of ``resources`` are assumed to act on the same
    wires as ``resources``. The resulting circuit depth is the depth of ``resources`` multiplied by
    ``scalar``. To multiply as if the circuit was repeated in parallel see
    :func:`~.resource.mul_in_parallel`.

    Args:
        resources (Resources): a :class:`~resource.Resources` to be scaled
        scalar (int): the scalar to multiply the :class:`~resource.Resources` by

    Returns:
        Resources: the combined resources

    .. details::

        **Example**

        First we build a :class:`~.resource.Resources` object.

        .. code-block:: python3

            from pennylane.measurements import Shots
            from pennylane.resource import Resources

            resources = Resources(
                num_wires = 2,
                num_gates = 2,
                gate_types = {"Hadamard": 1, "CNOT": 1},
                gate_sizes = {1: 1, 2: 1},
                depth = 2,
                shots = Shots(10)
            )

        Now we print the product.

        >>> print(qml.resource.mul_in_series(resources, 2))
        wires: 2
        gates: 4
        depth: 4
        shots: Shots(total=20)
        gate_types:
        {'Hadamard': 2, 'CNOT': 2}
        gate_sizes:
        {1: 2, 2: 2}
    """

    new_wires = resources.num_wires
    new_gates = scalar * resources.num_gates
    new_gate_types = _scale_dict(resources.gate_types, scalar)
    new_gate_sizes = _scale_dict(resources.gate_sizes, scalar)
    new_shots = scalar * resources.shots
    new_depth = scalar * resources.depth

    return Resources(new_wires, new_gates, new_gate_types, new_gate_sizes, new_depth, new_shots)


def mul_in_parallel(resources: Resources, scalar: int) -> Resources:
    """
    Multiply the :class:`~resource.Resources` object by a scalar as if the circuit was repeated that many times in parallel.

    The repeated copies of ``resources`` are assumed to act on disjoint qubits. The resulting circuit
    depth is equal to the depth of ``resources``. To multiply as if the repeated copies were
    executed in series see :func:`~.resource.mul_in_series`.

    Args:
        resources (Resources): a :class:`~resource.Resources` to be scaled
        scalar (int): the scalar to multiply the :class:`~resource.Resources` by

    Returns:
        Resources: The combined resources

    .. details::

        **Example**

        First we build a :class:`~.resource.Resources` object.

        .. code-block:: python3

            from pennylane.measurements import Shots
            from pennylane.resource import Resources

            resources = Resources(
                num_wires = 2,
                num_gates = 2,
                gate_types = {"Hadamard": 1, "CNOT": 1},
                gate_sizes = {1: 1, 2: 1},
                depth = 2,
                shots = Shots(10)
            )

        Now we print the product.

        >>> print(qml.resource.mul_in_parallel(resources, 2))
        wires: 4
        gates: 4
        depth: 2
        shots: Shots(total=20)
        gate_types:
        {'Hadamard': 2, 'CNOT': 2}
        gate_sizes:
        {1: 2, 2: 2}
    """

    new_wires = scalar * resources.num_wires
    new_gates = scalar * resources.num_gates
    new_gate_types = _scale_dict(resources.gate_types, scalar)
    new_gate_sizes = _scale_dict(resources.gate_sizes, scalar)
    new_shots = scalar * resources.shots

    return Resources(
        new_wires, new_gates, new_gate_types, new_gate_sizes, resources.depth, new_shots
    )


def _combine_dict(dict1: dict, dict2: dict):
    r"""Combines two dictionaries and adds values of common keys."""
    combined_dict = copy.copy(dict1)

    for k, v in dict2.items():
        try:
            combined_dict[k] += v
        except KeyError:
            combined_dict[k] = v

    return combined_dict


def _scale_dict(dict1: dict, scalar: int):
    r"""Scales the values in a dictionary with a scalar."""

    combined_dict = copy.copy(dict1)

    for k in combined_dict:
        combined_dict[k] *= scalar

    return combined_dict


def _count_resources(tape) -> Resources:
    """Given a quantum circuit (tape), this function
     counts the resources used by standard PennyLane operations.

    Args:
        tape (.QuantumTape): The quantum circuit for which we count resources

    Returns:
        (.Resources): The total resources used in the workflow
    """
    num_wires = len(tape.wires)
    shots = tape.shots
    depth = tape.graph.get_depth()

    num_gates = 0
    gate_types = defaultdict(int)
    gate_sizes = defaultdict(int)
    for op in tape.operations:
        if isinstance(op, ResourcesOperation):
            op_resource = op.resources()
            for d in op_resource.gate_types:
                gate_types[d] += op_resource.gate_types[d]

            for n in op_resource.gate_sizes:
                gate_sizes[n] += op_resource.gate_sizes[n]

            num_gates += sum(op_resource.gate_types.values())

        else:
            gate_types[op.name] += 1
            gate_sizes[len(op.wires)] += 1
            num_gates += 1

    return Resources(num_wires, num_gates, gate_types, gate_sizes, depth, shots)
