# Copyright 2018-2026 Xanadu Quantum Technologies Inc.

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

from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields
from decimal import Decimal
from functools import lru_cache
from string import ascii_lowercase
from typing import Any

from pennylane.core.measurements import MeasurementProcess
from pennylane.core.qscript import QuantumScript
from pennylane.core.shots import Shots
from pennylane.ops.op_math import Controlled, ControlledOp

from .expression import Expression, convert_int_vals_to_expression


def _count_to_str(
    count: int | Expression, extra_compact: bool = False, markdown_safe: bool = False
) -> str:
    """
    Helper for printing counts, converts large counts to scientific notation and standardizes printing of expressions.

    Args:
        count (int | Expression): the count to convert to a string
        extra_compact (bool): whether to remove spaces from expressions for compactness
        markdown_safe (bool): whether to escape asterisks for markdown tables
    """
    if isinstance(count, Expression):
        if count.vars:
            retval = str(count)
            if markdown_safe:
                retval = retval.replace("*", "\\*")  # Escape asterisks for markdown tables
            if extra_compact:
                retval = retval.replace(" ", "")  # Remove spaces from expressions for compactness
            return retval
        count = int(count)
    return f"{count:,}" if count < 100_000 else f"{Decimal(count):.3E}"


@lru_cache
def num_to_letters(num: int) -> str:
    """Helper for assigning labels to numbered data, such as batches for circuit resources.

    Converts 0 to 'a', 1 to 'b', etc.

    Example:
    >>> num_to_letters(0)
    'a'

    >>> num_to_letters(25)
    'z'

    >>> num_to_letters(26)
    'aa'

    >>> num_to_letters(27)
    'ab'
    """
    if num < 26:
        return ascii_lowercase[num]
    return num_to_letters(num // 26 - 1) + ascii_lowercase[num % 26]


# TODO: Would be better to have SpecsResources inherit from Resources directly, but there are too
# many extra fields that are unwanted. Would be worth refactoring in the future.
@dataclass(frozen=True)
class SpecsResources:
    """
    Class for storing resource information for a quantum circuit. Contains attributes which store
    key resources such as gate counts, number of wire allocations, measurements, and circuit depth.

    Note that this class is intended to be immutable. Modifying the attributes after creation may
    lead to unexpected behavior.

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
        Wire allocations: 2
        Total gates: 2
        Gate counts:
        - Hadamard: 1
        - CNOT: 1
        Measurements:
        - expval(PauliZ): 1
        Depth: 2
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
            case "gate_counts":
                return self.gate_counts

        raise KeyError(
            f"key '{key}' not available. Options are {[field.name for field in fields(self)]}"
        )

    @property
    def num_gates(self) -> int:
        """Total number of gates in the circuit."""
        return sum(self.gate_types.values())

    @property
    def gate_counts(self) -> dict[str, int]:
        """Alias for ``gate_types``"""
        return self.gate_types

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

        lines.append(f"{prefix}Wire allocations: {_count_to_str(self.num_allocs)}")
        lines.append(f"{prefix}Total gates: {_count_to_str(self.num_gates)}")

        lines.append(f"{prefix}Gate counts:")
        if not self.gate_types:
            lines.append(prefix + "- No gates.")
        else:
            for gate, count in self.gate_types.items():
                lines.append(f"{prefix}- {gate}: {_count_to_str(count)}")

        lines.append(f"{prefix}Measurements:")
        if not self.measurements:
            lines.append(prefix + "- No measurements.")
        else:
            for meas, count in self.measurements.items():
                lines.append(f"{prefix}- {meas}: {_count_to_str(count)}")

        lines.append(
            f"{prefix}Depth: {_count_to_str(self.depth) if self.depth is not None else 'Not computed'}"
        )

        return "\n".join(lines)

    # Leave repr and str methods separate for simple and pretty printing
    def __str__(self) -> str:
        return self.to_pretty_str()

    def _repr_markdown_(self) -> str:
        """
        Return a Markdown table representation of the :class:`SpecsResources` for Jupyter notebook display.

        .. seealso::

            https://ipython.readthedocs.io/en/stable/config/integrating.html#custom-methods
        """
        lines = []
        lines.append("| **Metric** | **Value** |")
        lines.append("| :--- | ---: |")
        lines.append(
            f"| **Wire allocations** | {_count_to_str(self.num_allocs, markdown_safe=True)} |"
        )
        lines.append(f"| **Total gates** | {_count_to_str(self.num_gates, markdown_safe=True)} |")
        lines.append("| **Gate counts:** | |")
        if not self.gate_types:
            lines.append("| *No gates* | |")
        else:
            for gate, count in self.gate_types.items():
                lines.append(f"| {gate} | {_count_to_str(count, markdown_safe=True)} |")
        lines.append("| **Measurements:** | |")
        if not self.measurements:
            lines.append("| *No measurements* | |")
        else:
            for meas, count in self.measurements.items():
                lines.append(f"| {meas} | {_count_to_str(count, markdown_safe=True)} |")
        depth_str = (
            _count_to_str(self.depth, markdown_safe=True)
            if self.depth is not None
            else "Not computed"
        )
        lines.append(f"| **Depth** | {depth_str} |")
        return "\n".join(lines)


@dataclass(frozen=True)
class SymbolicSpecsResources(SpecsResources):
    """
    Class for storing symbolic resource information for a quantum circuit. Contains attributes
    which store expressions representing the resources, allowing for symbolic manipulation and
    substitution of variables.

    .. warning::

        This class is intended to be immutable. Modifying the attributes after creation may
        lead to unexpected behavior.

    .. note::

        Some of the attributes from the parent class, :class:`SpecsResources`, are overridden
        here to be of type :class:`Expression` instead of `int`.
    """

    # gate_types: dict[str, Expression]
    # gate_sizes: dict[int, Expression]
    # measurements: dict[str, Expression]
    # num_allocs: Expression
    # depth: Expression | None = None
    vars: set[str] = field(init=False)

    def __post_init__(self):
        # Make sure that all fields use expressions, (converting ints to constant expressions where necessary)
        if self.depth is not None and isinstance(self.depth, int):
            object.__setattr__(
                self,
                "depth",
                Expression(self.depth),
            )
        if isinstance(self.num_allocs, int):
            object.__setattr__(
                self,
                "num_allocs",
                Expression(self.num_allocs),
            )

        convert_int_vals_to_expression(self.gate_types)
        convert_int_vals_to_expression(self.gate_sizes)
        convert_int_vals_to_expression(self.measurements)

        vars = set()

        # Need to disable this since the type checker still thinks that many of these members are
        # `int` and therefore do not contain a `var` member
        # pylint: disable=no-member

        # Need to take a union over all variables across the different expressions to
        # ensure the top-level objects has the full set of variables
        if self.depth is not None:
            vars |= self.depth.vars
        vars |= self.num_allocs.vars

        # Union over all expressions
        for expr in self.gate_types.values():
            vars |= expr.vars
        for expr in self.gate_sizes.values():
            vars |= expr.vars
        for expr in self.measurements.values():
            vars |= expr.vars

        object.__setattr__(self, "vars", vars)

    def subs(self, substitutions: dict[str, int] | None = None, **kwargs) -> SpecsResources:
        """
        Substitute variables in the symbolic resources with concrete integer values.
        If all variables are substituted, this will return a :class:`SpecsResources` object with
        integer values instead of another :class:`SymbolicSpecsResources` object.
        """
        if substitutions is None:
            substitutions = {}
        substitutions.update(kwargs)

        subs_vars = set(substitutions.keys())
        if subs_vars - self.vars:  # If substitutions contain variables not in the expression
            raise ValueError(
                f"Substitutions contain variables {subs_vars - self.vars} which are not in the expression's variables {self.vars}."
            )

        # Need to disable this since the type checker still thinks that many of these members are
        # `int` and therefore do not contain a `var` member
        # pylint: disable=no-member

        num_allocs = self.num_allocs.subs(substitutions)
        depth = self.depth.subs(substitutions) if self.depth is not None else None

        gate_types = {k: v.subs(substitutions) for k, v in self.gate_types.items()}
        gate_sizes = {k: v.subs(substitutions) for k, v in self.gate_sizes.items()}
        measurements = {k: v.subs(substitutions) for k, v in self.measurements.items()}

        if len(self.vars - subs_vars) == 0:
            # There are no variables remaining, so this can be resolved down to a `SpecsResources`
            return SpecsResources(
                gate_types={k: int(v) for k, v in gate_types.items()},
                gate_sizes={k: int(v) for k, v in gate_sizes.items()},
                measurements={k: int(v) for k, v in measurements.items()},
                num_allocs=int(num_allocs),
                depth=int(depth) if depth is not None else None,
            )

        return SymbolicSpecsResources(
            gate_types=gate_types,
            gate_sizes=gate_sizes,
            measurements=measurements,
            num_allocs=num_allocs,
            depth=depth,
        )

    def __eq__(self, other):
        if not isinstance(other, (SpecsResources, SymbolicSpecsResources)):
            return NotImplemented
        if not isinstance(other, SymbolicSpecsResources):
            if self.vars:
                return False
            return self.subs() == other

        return (
            self.vars == other.vars
            and self.num_allocs == other.num_allocs
            and self.depth == other.depth
            and self.gate_types == other.gate_types
            and self.gate_sizes == other.gate_sizes
            and self.measurements == other.measurements
        )

    def __call__(self, **kwargs):
        return self.subs(kwargs)

    def to_pretty_str(self, preindent: int = 0) -> str:
        prefix = " " * preindent
        return (
            f"{prefix}Symbolic Variables: {', '.join(sorted(self.vars)) if self.vars else 'None'}\n"
            + super().to_pretty_str(preindent)
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the SymbolicSpecsResources to a dictionary, including the variables."""
        d = super().to_dict()
        d["vars"] = sorted(self.vars)
        return d


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
        resources (SpecsResources | list[SpecsResources] | \
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
        Wire allocations: 2
        Total gates: 3
        Gate counts:
        - RX: 2
        - CNOT: 1
        Measurements:
        - expval(PauliZ): 1
        Depth: 3
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

    def _get_specs_header(self) -> list[str]:
        """Helper for main ``to_pretty_str`` method, gathers the header information about the specs such as device and level."""
        lines = []

        lines.append(f"Device: {self.device_name}")
        lines.append(f"Device wires: {self.num_device_wires}")
        lines.append(f"Shots: {self.shots}")
        if isinstance(self.level, dict):
            lines.append("Levels:")
            for level, level_name in self.level.items():
                lines.append(f"- {level}: {level_name}")
        else:
            lines.append(f"Level: {self.level}")

        lines.append("")  # Blank line

        return lines

    def _resources_to_str(self, res, preindent=0) -> str:
        """Helper for printing resources, prints list or single SpecsResources."""
        lines = []
        if isinstance(res, SpecsResources):
            lines.append(res.to_pretty_str(preindent))
        elif isinstance(res, list):
            prefix = preindent * " "
            for i, r in enumerate(res):
                lines.append(f"{prefix}Batched tape {num_to_letters(i)}:")
                lines.append(r.to_pretty_str(preindent=preindent + 4))
                lines.append("")  # Blank line
        else:
            raise ValueError(
                "Resources must be either a SpecsResources object or a list of SpecsResources objects."
            )  # pragma: no cover

        return "\n".join(lines)

    def _flattened_resources(self) -> dict[str, SpecsResources]:
        """Helper for printing tabular format, flattens all resources across levels into a single
        dictionary with string keys."""
        flat_resources = {}
        for level, res in zip(self.level.keys(), self.resources.values(), strict=True):
            if isinstance(res, SpecsResources):
                flat_resources[str(level)] = res
            elif isinstance(res, list):
                for i, r in enumerate(res):
                    flat_resources[f"{level}-{num_to_letters(i)}"] = r
            else:
                raise ValueError(
                    "Resources must be either a SpecsResources object or a list of SpecsResources objects."
                )  # pragma: no cover
        return flat_resources

    def _get_table_format(
        self, flat_resources: dict[str, SpecsResources]
    ) -> tuple[int, int, dict[str, None], dict[str, None]]:
        """Helper for printing tabular format, determines column widths and all gate and measurement
        types across levels."""
        # This is the length of the longest metric name (currently "Wire allocations") plus padding
        max_metric_length = 16
        max_column_size = max(len(level) for level in flat_resources) + 2

        # Use dict for these since they are sorted by default unlike a set
        all_gate_types = {}
        all_meas_types = {}

        # This iteration order will present the gates in the order in which they appear
        for res in flat_resources.values():
            for gate, count in res.gate_types.items():
                all_gate_types[gate] = True
                max_metric_length = max(max_metric_length, len(gate) + 2)
                max_column_size = max(
                    max_column_size, len(_count_to_str(count, extra_compact=True)) + 1
                )
            for meas, count in res.measurements.items():
                all_meas_types[meas] = True
                max_metric_length = max(max_metric_length, len(meas) + 2)
                max_column_size = max(
                    max_column_size, len(_count_to_str(count, extra_compact=True)) + 1
                )
            max_column_size = max(
                max_column_size,
                len(_count_to_str(res.num_allocs, extra_compact=True)) + 1,
                len(_count_to_str(res.num_gates, extra_compact=True)) + 1,
            )

        return max_metric_length, max_column_size, all_gate_types, all_meas_types

    def _to_pretty_str_tabular(self) -> str:
        """Helper for main ``to_pretty_str`` for tabular format, which is more compact when there
        are many levels to display."""
        lines = self._get_specs_header()

        flat_resources = self._flattened_resources()
        max_metric_length, max_column_size, all_gate_types, all_meas_types = self._get_table_format(
            flat_resources
        )

        num_cols = len(flat_resources)
        lines.append(
            "↓Metric".ljust(max_metric_length - 6)
            + "Level→"
            + " |"
            + " |".join(level.rjust(max_column_size) for level in flat_resources)
        )
        lines.append("-" * (max_metric_length + num_cols * (max_column_size + 2)))
        lines.append(
            "Wire allocations".ljust(max_metric_length)
            + " |"
            + " |".join(
                _count_to_str(res.num_allocs, extra_compact=True).rjust(max_column_size)
                for res in flat_resources.values()
            )
        )
        lines.append(
            "Total gates".ljust(max_metric_length)
            + " |"
            + " |".join(
                _count_to_str(res.num_gates, extra_compact=True).rjust(max_column_size)
                for res in flat_resources.values()
            )
        )

        lines.append("Gate counts:".ljust(max_metric_length) + " |")
        for gate in all_gate_types:
            lines.append(
                f"- {gate}".ljust(max_metric_length)
                + " |"
                + " |".join(
                    _count_to_str(res.gate_types.get(gate, 0), extra_compact=True).rjust(
                        max_column_size
                    )
                    for res in flat_resources.values()
                )
            )
        lines.append("Measurements:".ljust(max_metric_length) + " |")
        for meas in all_meas_types:
            lines.append(
                f"- {meas}".ljust(max_metric_length)
                + " |"
                + " |".join(
                    _count_to_str(res.measurements.get(meas, 0), extra_compact=True).rjust(
                        max_column_size
                    )
                    for res in flat_resources.values()
                )
            )

        return "\n".join(lines).rstrip("\n")

    def to_pretty_str(self, tabular: bool = True) -> str:
        """
        Pretty string representation of the :class:`CircuitSpecs` object.

        Args:
            tabular (bool): Whether to display the resources in a tabular format.

        Returns:
            str: A pretty representation of this object.
        """
        if tabular and isinstance(self.resources, dict):
            return self._to_pretty_str_tabular()

        lines = self._get_specs_header()

        if isinstance(self.resources, dict):
            lines.append("")  # Blank line before levels
            for level, res in self.resources.items():
                lines.append(f"Level = {level}:")
                lines.append(self._resources_to_str(res, preindent=4))
                lines.append("\n" + "-" * 60 + "\n")  # Separator between levels
        else:
            lines.append(self._resources_to_str(self.resources))

        return "\n".join(lines).rstrip("\n-")

    # Separate str and repr methods for simple and pretty printing
    def __str__(self) -> str:
        return self.to_pretty_str()

    def _to_markdown_tabular(self) -> str:
        """Return a Markdown table for dict-type resources."""
        flat_resources = self._flattened_resources()
        levels = list(flat_resources.keys())

        all_gate_types: dict[str, None] = {}
        all_meas_types: dict[str, None] = {}
        for res in flat_resources.values():
            for gate in res.gate_types:
                all_gate_types[gate] = None
            for meas in res.measurements:
                all_meas_types[meas] = None

        def data_row(label, values):
            return f"| {label} | " + " | ".join(str(v) for v in values) + " |"

        lines = []
        lines.append("| ↓Metric / Level→ | " + " | ".join(str(lvl) for lvl in levels) + " |")
        lines.append("| :--- |" + " ---: |" * len(levels))
        lines.append(
            data_row(
                "**Wire allocations**",
                [_count_to_str(r.num_allocs, markdown_safe=True) for r in flat_resources.values()],
            )
        )
        lines.append(
            data_row(
                "**Total gates**",
                [_count_to_str(r.num_gates, markdown_safe=True) for r in flat_resources.values()],
            )
        )
        lines.append(data_row("**Gate counts**", [""] * len(levels)))
        for gate in all_gate_types:
            lines.append(
                data_row(
                    gate,
                    [
                        _count_to_str(r.gate_types.get(gate, 0), markdown_safe=True)
                        for r in flat_resources.values()
                    ],
                )
            )
        lines.append(data_row("**Measurements**", [""] * len(levels)))
        for meas in all_meas_types:
            lines.append(
                data_row(
                    meas,
                    [
                        _count_to_str(r.measurements.get(meas, 0), markdown_safe=True)
                        for r in flat_resources.values()
                    ],
                )
            )
        return "\n".join(lines)

    def _repr_markdown_(self, collapsible: bool = True) -> str:
        """
        Return a Markdown representation of the :class:`CircuitSpecs` for Jupyter notebook display.

        Args:
            collapsible (bool): Whether to display the resources in collapsible sections.

        Returns:
            str: A Markdown representation of this object for Jupyter notebooks.

        .. seealso::

            https://ipython.readthedocs.io/en/stable/config/integrating.html#custom-methods
        """
        # pylint: disable=too-many-branches
        # Ignore pylint on this one, this is not better served by splitting into even
        # smaller functions than it already has
        lines = []
        if collapsible:
            lines.append("<details open>")
            lines.append("<summary>Circuit Specs</summary>")
        else:
            lines.append("**Circuit Specs:**")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| :--- | ---: |")
        lines.append(f"| **Device** | {self.device_name} |")
        lines.append(f"| **Device wires** | {self.num_device_wires} |")
        lines.append(f"| **Shots** | {self.shots} |")
        if isinstance(self.level, dict):
            lines.append("| **Levels** | |")
            for k, v in self.level.items():
                lines.append(f"| {k} | {v} |")
        else:
            lines.append(f"| **Level** | {self.level} |")

        lines.append("")

        if collapsible:
            lines.append("</details>")
            lines.append("<details open>")
            lines.append("<summary>Resources</summary>")
        else:
            lines.append("**Resources:**")
        lines.append("")

        if isinstance(self.resources, SpecsResources):
            lines.append(self.resources._repr_markdown_())  # pylint: disable=protected-access
        elif isinstance(self.resources, list):
            for i, r in enumerate(self.resources):
                if collapsible:
                    lines.append("<details open>")
                    lines.append(f"<summary>Batched tape {num_to_letters(i)}</summary>")
                else:
                    lines.append(f"**Batched tape {num_to_letters(i)}:**")
                lines.append("")
                lines.append(r._repr_markdown_())  # pylint: disable=protected-access
                lines.append("")
                if collapsible:
                    lines.append("</details>")
        elif isinstance(self.resources, dict):
            lines.append(self._to_markdown_tabular())

        if collapsible:
            lines.append("")
            lines.append("</details>")

        return "\n".join(lines)


# The reason why this function is not a method of the QuantumScript class is
# because we don't want a core module (QuantumScript) to depend on an auxiliary module (Resource).
# The `QuantumScript.specs` property will eventually be deprecated in favor of this function.
def resources_from_tape(tape: QuantumScript, compute_depth: bool = True) -> SpecsResources:
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
    Returns:
        SpecsResources: The resources associated with this tape.
    """
    resources = _count_resources(tape, compute_depth=compute_depth)

    return resources


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
