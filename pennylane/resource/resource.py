# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from pennylane.measurements import MeasurementProcess, Shots, add_shots
from pennylane.operation import Operation
from pennylane.ops.op_math import Controlled, ControlledOp
from pennylane.tape import QuantumScript

from .error.error import _compute_algo_error


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
        num_wires: 2
        num_gates: 4
        depth: 4
        shots: Shots(total=None)
        gate_types:
        {'Hadamard': 1, 'CNOT': 2, 'RX': 1}
        gate_sizes:
        {1: 2, 2: 2}
        >>> print(r1 * 2)
        num_wires: 2
        num_gates: 4
        depth: 4
        shots: Shots(total=None)
        gate_types:
        {'Hadamard': 2, 'CNOT': 2}
        gate_sizes:
        {1: 2, 2: 2}
    """

    num_wires: int = 0
    num_gates: int = 0
    gate_types: dict[str, int] = field(default_factory=dict)
    gate_sizes: dict[int, int] = field(default_factory=dict)
    depth: int | None = 0
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

            .. code-block:: python

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
            num_wires: 3
            num_gates: 4
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

            .. code-block:: python

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
            num_wires: 2
            num_gates: 4
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
        # See https://ipython.readthedocs.io/en/stable/config/integrating.html#custom-methods
        print(str(self))


# TODO: Would be better to have SpecsResources inherit from Resources directly, but there are too
# many extra fields that are unwanted. Would be worth refactoring in the future.
@dataclass(frozen=True)
class SpecsResources:
    """
    Class for storing resource information for a quantum circuit. Contains attributes which store
    key resources such as gate counts, number of wire allocations, measurements, and circuit depth.

    Args:
        gate_types (dict[str, int]): A dictionary mapping gate names to their counts.
        gate_sizes (dict[int, int]): A dictionary mapping gate sizes to their counts.
        measurements (dict[str, int]): A dictionary mapping measurements to their counts.
        num_allocs (int): The number of unique wire allocations. For circuits that do not use
          dynamic wires, this should be equal to the number of device wires.
        depth (int | None): The depth of the circuit, or None if not computed.

    Properties:
        num_gates (int): The total number of gates in the circuit (computed from `gate_types`).

    .. details::

        Methods have been provided to allow pretty-printing, as well as
        indexing into it as a dictionary. See examples below.

        **Example**

        >>> from pennylane.resource import SpecsResources
        >>> res = SpecsResources(
        ...     gate_types={'Hadamard': 1, 'CNOT': 1},
        ...     gate_sizes={1: 1, 2: 1},
        ...     measurements={'expval(PauliZ)': 1},
        ...     num_allocs=2,
        ...     depth=2
        ... )

        >>> print(res.num_gates)
        2

        >>> print(res["num_gates"])
        2

        >>> print(res)
        Total wire allocations: 2
        Total gates: 2
        Circuit depth: 2
        <BLANKLINE>
        Gate types:
          Hadamard: 1
          CNOT: 1
        <BLANKLINE>
        Measurements:
          expval(PauliZ): 1
    """

    gate_types: dict[str, int]
    gate_sizes: dict[int, int]
    measurements: dict[str, int]
    num_allocs: int
    depth: int | None = None

    def __post_init__(self):
        if sum(self.gate_types.values()) != sum(self.gate_sizes.values()):
            raise ValueError(
                "Inconsistent gate counts: `gate_types` and `gate_sizes` describe different amounts of gates."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert the SpecsResources to a dictionary."""

        # Need to explicitly include properties
        d = asdict(self)
        d["num_gates"] = self.num_gates

        return d

    def __getitem__(self, key):
        if key in (field.name for field in fields(self)):
            return getattr(self, key)

        match key:
            # Fields that used to be included in specs output prior to PL version 0.44
            case "shots":
                raise KeyError(
                    "shots is no longer included within specs's resources, check the top-level object instead."
                )
            case "num_wires":
                raise KeyError(
                    "num_wires has been renamed to num_allocs to more accurate describe what it measures."
                )
            case "num_gates":
                # As a property, this needs to be handled differently to the true fields
                return self.num_gates

        raise KeyError(
            f"key '{key}' not available. Options are {[field.name for field in fields(self)]}"
        )

    @property
    def num_gates(self) -> int:
        """Total number of gates in the circuit."""
        return sum(self.gate_types.values())

    def to_pretty_str(self, preindent: int = 0) -> str:
        """
        Pretty string representation of the SpecsResources object.

        Args:
            preindent (int): Number of spaces to prepend to each line.

        Returns:
            str: A pretty representation of this object.
        """
        prefix = " " * preindent
        lines = []

        lines.append(f"{prefix}Total wire allocations: {self.num_allocs}")
        lines.append(f"{prefix}Total gates: {self.num_gates}")
        lines.append(
            f"{prefix}Circuit depth: {self.depth if self.depth is not None else 'Not computed'}"
        )

        lines.append("")  # Blank line

        lines.append(f"{prefix}Gate types:")
        if not self.gate_types:
            lines.append(prefix + "  No gates.")
        else:
            for gate, count in self.gate_types.items():
                lines.append(f"{prefix}  {gate}: {count}")

        lines.append("")  # Blank line

        lines.append(f"{prefix}Measurements:")
        if not self.measurements:
            lines.append(prefix + "  No measurements.")
        else:
            for meas, count in self.measurements.items():
                lines.append(f"{prefix}  {meas}: {count}")

        return "\n".join(lines)

    # Leave repr and str methods separate for simple and pretty printing
    def __str__(self) -> str:
        return self.to_pretty_str()

    def _ipython_display_(self):  # pragma: no cover
        """Displays __str__ in ipython instead of __repr__"""
        # See https://ipython.readthedocs.io/en/stable/config/integrating.html#custom-methods
        print(str(self))


@dataclass(frozen=True)
class CircuitSpecs:
    """
    Class for storing specifications of a qnode. Contains resource information as well as additional
    data such as the device, number of shots, and level of the requested specs.

    Args:
        device_name (str): The name of the device used.
        num_device_wires (int): The number of wires on the device.
        shots (Shots): The shots configuration used.
        level (Any): The level of the specs (see :func:`~pennylane.specs` for more details).
        resources (SpecsResources | list[SpecsResources] |
            dict[int | str, SpecsResources | list[SpecsResources]]): The resource specifications.
            Depending on the ``level`` chosen, this may be a single :class:`.SpecsResources` object,
            a list of :class:`.SpecsResources` objects, or a dictionary mapping levels to their
            corresponding outputs.

    .. details::

        Some helpful methods have been added to this data class to allow pretty-printing, as well as
        indexing into it as a dictionary. See examples below.

        **Example**

        >>> from pennylane.resource import SpecsResources, CircuitSpecs
        >>> specs = CircuitSpecs(
        ...     device_name="default.qubit",
        ...     num_device_wires=2,
        ...     shots=Shots(1000),
        ...     level="device",
        ...     resources=SpecsResources(
        ...         gate_types={"RX": 2, "CNOT": 1},
        ...         gate_sizes={1: 2, 2: 1},
        ...         measurements={"expval(PauliZ)": 1},
        ...         num_allocs=2,
        ...         depth=3,
        ...     ),
        ... )

        >>> print(specs.num_device_wires)
        2

        >>> print(specs["num_device_wires"])
        2

        >>> print(specs)
        Device: default.qubit
        Device wires: 2
        Shots: Shots(total=1000)
        Level: device
        <BLANKLINE>
        Resource specifications:
          Total wire allocations: 2
          Total gates: 3
          Circuit depth: 3
        <BLANKLINE>
          Gate types:
            RX: 2
            CNOT: 1
        <BLANKLINE>
          Measurements:
            expval(PauliZ): 1
    """

    device_name: str | None = None
    num_device_wires: int | None = None
    shots: Shots | None = None
    level: Any = None
    resources: (
        SpecsResources
        | list[SpecsResources]
        | dict[int | str, SpecsResources | list[SpecsResources]]
        | None
    ) = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the CircuitSpecs to a dictionary."""
        d = asdict(self)

        # Replace Resources objects with their dict representations
        if isinstance(self.resources, SpecsResources):
            d["resources"] = self.resources.to_dict()
        elif isinstance(self.resources, list):
            d["resources"] = [r.to_dict() for r in self.resources]
        elif isinstance(self.resources, dict):
            d["resources"] = {
                k: (v.to_dict() if isinstance(v, SpecsResources) else [r.to_dict() for r in v])
                for k, v in self.resources.items()
            }

        return d

    def __getitem__(self, key):
        if key in (field.name for field in fields(self)):
            return getattr(self, key)

        match key:
            # Fields that used to be included in specs output prior to PL version 0.44
            case "num_observables":
                raise KeyError(
                    "num_observables is no longer in top-level specs and has instead been absorbed into the 'measurements' attribute of the specs's resources."
                )
            case "interface" | "diff_method" | "errors" | "num_tape_wires":
                raise KeyError(f"key '{key}' is no longer included in specs.")
            case (
                "gradient_fn"
                | "gradient_options"
                | "num_gradient_executions"
                | "num_trainable_params"
            ):
                raise KeyError(
                    f"key '{key}' is no longer included in specs, as specs no longer gathers gradient information."
                )
        raise KeyError(
            f"key '{key}' not available. Options are {[field.name for field in fields(self)]}"
        )

    def _resources_to_str(self, res) -> str:
        """Helper for printing resources, prints list or single SpecsResources."""
        lines = []
        if isinstance(res, SpecsResources):
            lines.append(res.to_pretty_str(preindent=2))
        elif isinstance(res, list):
            for i, r in enumerate(res):
                lines.append(f"  Batched tape {i}:")
                lines.append(r.to_pretty_str(preindent=4))
                lines.append("")  # Blank line
        else:
            raise ValueError(
                "Resources must be either a SpecsResources object or a list of SpecsResources objects."
            )  # pragma: no cover

        return "\n".join(lines)

    # Separate str and repr methods for simple and pretty printing
    def __str__(self):
        lines = []

        lines.append(f"Device: {self.device_name}")
        lines.append(f"Device wires: {self.num_device_wires}")
        lines.append(f"Shots: {self.shots}")
        lines.append(f"Level: {self.level}")

        lines.append("")  # Blank line

        lines.append("Resource specifications:")
        if isinstance(self.resources, dict):
            for level, res in self.resources.items():
                lines.append(f"Level = {level}:")
                lines.append(self._resources_to_str(res))
                lines.append("\n" + "-" * 60 + "\n")  # Separator between levels
        else:
            lines.append(self._resources_to_str(self.resources))

        return "\n".join(lines).rstrip("\n-")

    def _ipython_display_(self):  # pragma: no cover
        """Displays __str__ in ipython instead of __repr__"""
        # See https://ipython.readthedocs.io/en/stable/config/integrating.html#custom-methods
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

        .. code-block:: python

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
        num_wires: 3
        num_gates: 4
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

        .. code-block:: python

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
        num_wires: 5
        num_gates: 4
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

        .. code-block:: python

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
        num_wires: 2
        num_gates: 4
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

        .. code-block:: python

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
        num_wires: 4
        num_gates: 4
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


def substitute(initial_resources: Resources, gate_info: tuple[str, int], replacement: Resources):
    """Replaces a specified gate in a :class:`~.resource.Resources` object with the contents of another :class:`~.resource.Resources` object.

    Args:
        initial_resources (Resources): the :class:`~resource.Resources` object to be modified
        gate_info (Iterable(str, int)): sequence containing the name of the gate to be replaced and the number of wires it acts on
        replacement (Resources): the :class:`~resource.Resources` containing the resources that will replace the gate

    Returns:
        Resources: the updated :class:`~resource.Resources` after substitution

    .. details::

        **Example**

        First we build the :class:`~.resource.Resources`.

        .. code-block:: python

            from pennylane.measurements import Shots
            from pennylane.resource import Resources

            initial_resources = Resources(
                num_wires = 2,
                num_gates = 3,
                gate_types = {"RX": 2, "CNOT": 1},
                gate_sizes = {1: 2, 2: 1},
                depth = 2,
                shots = Shots(10)
            )

            # the RX gates will be replaced by the substitution
            gate_info = ("RX", 1)

            replacement = Resources(
                num_wires = 1,
                num_gates = 7,
                gate_types = {"Hadamard": 3, "S": 4},
                gate_sizes = {1: 7},
                depth = 7
            )


        Now we print the result of the substitution.

        >>> res = qml.resource.substitute(initial_resources, gate_info, replacement)
        >>> print(res)
        num_wires: 2
        num_gates: 15
        depth: 9
        shots: Shots(total=10)
        gate_types:
        {'CNOT': 1, 'Hadamard': 6, 'S': 8}
        gate_sizes:
        {1: 14, 2: 1}
    """

    gate_name, num_wires = gate_info

    if not num_wires in initial_resources.gate_sizes:
        raise ValueError(f"initial_resources does not contain a gate acting on {num_wires} wires.")

    gate_count = initial_resources.gate_types.get(gate_name, 0)

    if gate_count > initial_resources.gate_sizes[num_wires]:
        raise ValueError(
            f"Found {gate_count} gates of type {gate_name}, but only {initial_resources.gate_sizes[num_wires]} gates act on {num_wires} wires in initial_resources."
        )

    if gate_count > 0:
        new_wires = initial_resources.num_wires
        new_gates = initial_resources.num_gates - gate_count + (gate_count * replacement.num_gates)
        replacement_gate_types = _scale_dict(replacement.gate_types, gate_count)
        replacement_gate_sizes = _scale_dict(replacement.gate_sizes, gate_count)

        new_gate_types = _combine_dict(initial_resources.gate_types, replacement_gate_types)
        new_gate_types.pop(gate_name)

        new_gate_sizes = copy.copy(initial_resources.gate_sizes)
        new_gate_sizes[num_wires] -= gate_count
        new_gate_sizes = _combine_dict(new_gate_sizes, replacement_gate_sizes)

        new_depth = initial_resources.depth + replacement.depth

        wire_diff = num_wires - replacement.num_wires
        if wire_diff < 0:
            new_wires = initial_resources.num_wires + abs(wire_diff)
        else:
            new_wires = initial_resources.num_wires

        return Resources(
            new_wires, new_gates, new_gate_types, new_gate_sizes, new_depth, initial_resources.shots
        )

    return initial_resources


# The reason why this function is not a method of the QuantumScript class is
# because we don't want a core module (QuantumScript) to depend on an auxiliary module (Resource).
# The `QuantumScript.specs` property will eventually be deprecated in favor of this function.
def resources_from_tape(
    tape: QuantumScript, compute_depth: bool = True, compute_errors: bool = False
) -> SpecsResources | tuple[SpecsResources, dict[str, Any]]:
    """
    Extracts the resource information from a quantum circuit (tape).

    The depth of the circuit is computed by default, but can be set to None
    by setting the `compute_depth` argument to False.
    This is useful when the depth is not needed, for example, in some
    resource counting scenarios or heavy circuits where computing depth is expensive.

    Args:
        tape (.QuantumScript): The quantum circuit for which we extract resources
        compute_depth (bool): If True, the depth of the circuit is computed and included in the resources.
            If False, the depth is set to None.
        compute_errors (bool): If True, algorithmic errors are computed and returned alongside the resources.
            Defaults to False.
    Returns:
        (SpecsResources | tuple[SpecsResources, dict[str, Any]]): The resources associated with this tape, optionally
        with algorithmic errors if `compute_errors` is set to True.
    """
    resources = _count_resources(tape, compute_depth=compute_depth)

    if compute_errors:
        algo_errors = _compute_algo_error(tape)
        return resources, algo_errors

    return resources


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


def _obs_to_str(obs) -> str:
    """Convert an Observable to a string representation for resource counting."""
    name = obs.name
    match name:
        case "Hamiltonian" | "LinearCombination" | "Sum" | "Prod":
            if name == "LinearCombination":
                name = "Hamiltonian"
            return f"{name}(num_wires={obs.num_wires}, num_terms={len(obs.operands)})"
        case "SProd":
            return _obs_to_str(obs.base)
        case "Exp":
            return f"Exp({_obs_to_str(obs.base)})"
        case _:
            return name


def _mp_to_str(mp: MeasurementProcess, num_wires: int) -> str:
    """Convert a MeasurementProcess to a string representation for resource counting."""
    meas_name = mp._shortname  # pylint: disable=protected-access
    if mp.mv is not None:
        meas_name += "(mcm)"
    elif mp.obs is None:
        meas_wires = len(mp.wires)
        if meas_wires in (None, 0, num_wires):
            meas_name += "(all wires)"
        else:
            meas_name += f"({meas_wires} wires)"
    else:
        meas_name += f"({_obs_to_str(mp.obs)})"
    return meas_name


def _count_resources(tape: QuantumScript, compute_depth: bool = True) -> SpecsResources:
    """Given a quantum tape, this function counts the resources used by standard PennyLane operations.

    Args:
        tape (.QuantumScript): The quantum circuit for which we count resources
        compute_depth (bool): If True, the depth of the circuit is computed and included in the resources.
            If False, the depth is set to None.

    Returns:
        (.SpecsResources): The total resources used in the workflow
    """

    num_wires = len(tape.wires)
    depth = tape.graph.get_depth() if compute_depth else None

    gate_types = defaultdict(int)
    measurements = defaultdict(int)
    gate_sizes = defaultdict(int)
    for op in tape.operations:
        if isinstance(op, ResourcesOperation):
            op_resource = op.resources()
            for d in op_resource.gate_types:
                gate_types[d] += op_resource.gate_types[d]

            for n in op_resource.gate_sizes:
                gate_sizes[n] += op_resource.gate_sizes[n]

        else:
            gate_name = op.name
            # pylint: disable=unidiomatic-typecheck
            if type(op) in (Controlled, ControlledOp):
                n_ctrls = len(op.control_wires)
                if n_ctrls > 1:
                    gate_name = f"{n_ctrls}{gate_name}"

            gate_types[gate_name] += 1
            gate_sizes[len(op.wires)] += 1

    for meas in tape.measurements:
        measurements[_mp_to_str(meas, num_wires)] += 1

    return SpecsResources(
        gate_types=dict(gate_types),
        gate_sizes=dict(gate_sizes),
        measurements=dict(measurements),
        num_allocs=num_wires,
        depth=depth,
    )
