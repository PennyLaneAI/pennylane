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
"""
Defines the DeviceCapabilities class, and tools to load it from a TOML file.
"""

import sys
from dataclasses import dataclass, field
from enum import Enum
from itertools import repeat

import pennylane as qml

if sys.version_info >= (3, 11):
    import tomllib as toml

    TOMLDocument = dict  # used for type hinting purposes
else:
    import tomlkit as toml
    from tomlkit import TOMLDocument


class InvalidCapabilitiesError(Exception):
    """Exception raised from invalid TOML files."""


def load_toml_file(file_path: str) -> TOMLDocument:
    """Loads a TOML file and returns the parsed dictionary/TOMLDocument."""
    with open(file_path, "rb") as f:
        return toml.load(f)


class ExecutionCondition(Enum):
    """The constraint on the support of something."""

    FINITE_SHOTS_ONLY = "finiteshots"
    """If the operator or measurement process is only supported with finite shots."""

    ANALYTIC_MODE_ONLY = "analytic"
    """If the operator or measurement process is only supported in analytic execution."""

    TERMS_MUST_COMMUTE = "terms-commute"
    """If the composite operator is supported only when its terms commute."""


VALID_CONDITION_STRINGS = {condition.value for condition in ExecutionCondition}


@dataclass
class OperatorProperties:
    """Information about support for each operation.

    Attributes:
        invertible (bool): Whether the adjoint of the operation is also supported.
        controllable (bool): Whether the operation can be controlled.
        differentiable (bool): Whether the operation is supported for device gradients.

    """

    invertible: bool = False
    controllable: bool = False
    differentiable: bool = False
    conditions: list[ExecutionCondition] = field(default_factory=list)


@dataclass
class DeviceCapabilities:  # pylint: disable=too-many-instance-attributes
    """Capabilities of a quantum device.

    Attributes:
        operations: Operations natively supported by the backend device.
        observables: Observables that the device can measure.
        measurement_processes: List of measurement processes supported by the backend device.
        qjit_compatible (bool): Whether the device is compatible with qjit.
        mid_circuit_measurements (bool): Whether the device supports mid-circuit measurements natively.
        runtime_code_generation (bool): Whether the device requires run time generation of the quantum circuit.
        dynamic_qubit_management (bool): Whether the device supports dynamic qubit allocation/deallocation.
        overlapping_observables (bool): Whether the device supports measuring overlapping observables on the same tape.
        non_commuting_observables (bool): Whether the device supports measuring non-commuting observables on the same tape.
        initial_state_prep (bool): Whether the device supports initial state preparation.

    """

    operations: dict[str, OperatorProperties] = field(default_factory=dict)
    observables: dict[str, OperatorProperties] = field(default_factory=dict)
    measurement_processes: dict[str, list[ExecutionCondition]] = field(default_factory=dict)
    qjit_compatible: bool = False
    mid_circuit_measurements: bool = False
    runtime_code_generation: bool = False
    dynamic_qubit_management: bool = False
    overlapping_observables: bool = True
    non_commuting_observables: bool = False
    initial_state_prep: bool = False
    options: dict[str, any] = field(default_factory=dict)


VALID_COMPILATION_FLAGS = {
    "qjit_compatible",
    "mid_circuit_measurements",
    "runtime_code_generation",
    "dynamic_qubit_management",
    "overlapping_observables",
    "non_commuting_observable",
    "initial_state_prep",
}


def _get_toml_section(document: TOMLDocument, path: str, prefix: str = "") -> TOMLDocument:
    """Retrieves a section from a TOML document using a given path."""

    if prefix:
        path = f"{prefix}.{path}"

    for k in path.split("."):
        if not isinstance(document, TOMLDocument) or k not in document:
            return {}
        document = document[k]
    return document


def _validate_conditions(conditions: list[ExecutionCondition], target=None) -> None:
    """Validates the execution conditions."""

    if (
        ExecutionCondition.ANALYTIC_MODE_ONLY in conditions
        and ExecutionCondition.FINITE_SHOTS_ONLY in conditions
    ):
        raise InvalidCapabilitiesError(
            "Conditions cannot contain both 'analytic' and 'finiteshots'"
        )

    if conditions == ExecutionCondition.TERMS_MUST_COMMUTE and not isinstance(
        target, (qml.ops.Prod, qml.ops.Sum, qml.ops.SProd)
    ):
        raise InvalidCapabilitiesError(
            "'terms-commute' is only applicable to Prod, SProd, Sum, and LinearCombination."
        )


def _get_operators(section: TOMLDocument) -> dict[str, OperatorProperties]:
    """Parses an operator section into a dictionary of OperatorProperties."""

    operators = {}
    iterator = section.items() if hasattr(section, "items") else zip(section, repeat({}))
    for o, attributes in iterator:
        if unknowns := set(attributes) - {"properties", "conditions"}:
            raise InvalidCapabilitiesError(
                f"Operator '{o}' has unknown attributes: {list(unknowns)}"
            )
        properties = attributes.get("properties", {})
        if unknowns := set(properties) - {"invertible", "controllable", "differentiable"}:
            raise InvalidCapabilitiesError(
                f"Operator '{o}' has unknown properties: {list(unknowns)}"
            )
        condition_strs = attributes.get("conditions", [])
        if unknowns := set(condition_strs) - VALID_CONDITION_STRINGS:
            raise InvalidCapabilitiesError(
                f"Operator '{o}' has unknown conditions: {list(unknowns)}"
            )
        conditions = [ExecutionCondition(c) for c in condition_strs]
        _validate_conditions(conditions, o)
        operators[o] = OperatorProperties(
            invertible="invertible" in properties,
            controllable="controllable" in properties,
            differentiable="differentiable" in properties,
            conditions=conditions,
        )
    return operators


def _get_operations(document: TOMLDocument, prefix: str = "") -> dict[str, OperatorProperties]:
    """Gets the supported operations from a TOML document."""
    section = _get_toml_section(document, "operators.gates", prefix)
    return _get_operators(section)


def _get_observables(document: TOMLDocument, prefix: str = "") -> dict[str, OperatorProperties]:
    """Gets the supported observables from a TOML document."""
    section = _get_toml_section(document, "operators.observables", prefix)
    return _get_operators(section)


def _get_measurement_processes(
    document: TOMLDocument, prefix: str = ""
) -> dict[str, list[ExecutionCondition]]:
    """Gets the supported measurement processes from a TOML document."""

    section = _get_toml_section(document, "measurement_processes", prefix)
    measurement_processes = {}
    iterator = section.items() if hasattr(section, "items") else zip(section, repeat({}))
    for mp, attributes in iterator:
        if unknowns := set(attributes) - {"conditions"}:
            raise InvalidCapabilitiesError(
                f"Measurement '{mp}' has unknown attributes: {list(unknowns)}"
            )
        conditions = [ExecutionCondition(c) for c in attributes.get("conditions", [])]
        _validate_conditions(conditions)
        measurement_processes[mp] = conditions
    return measurement_processes


def _get_compilation_flags(document: TOMLDocument, prefix: str = "") -> dict[str, bool]:
    """Gets the boolean capabilities in the compilation section."""
    section = _get_toml_section(document, "compilation", prefix)
    if unknowns := set(section) - VALID_COMPILATION_FLAGS:
        raise InvalidCapabilitiesError(
            f"The compilation section has unknown options: {list(unknowns)}"
        )
    flags = {flag: section.get(flag, False) for flag in VALID_COMPILATION_FLAGS}
    if not flags["overlapping_observables"] and flags["non_commuting_observables"]:
        raise InvalidCapabilitiesError(
            "When overlapping_observables is False, non_commuting_observables cannot be True."
        )
    return flags


def _get_options(document: TOMLDocument) -> dict[str, str]:
    """Get custom options"""
    return {str(k): v for k, v in document.get("options", {}).items()}


def parse_toml_document(document: TOMLDocument) -> DeviceCapabilities:
    """Parses a TOML document into a DeviceCapabilities object."""

    operations = _get_operations(document)
    observables = _get_observables(document)
    measurement_processes = _get_measurement_processes(document)
    compilation_flags = _get_compilation_flags(document)
    return DeviceCapabilities(
        operations=operations,
        observables=observables,
        measurement_processes=measurement_processes,
        **compilation_flags,
        options=_get_options(document),
    )


def update_device_capabilities(
    capabilities: DeviceCapabilities, document: TOMLDocument, runtime_interface: str
):
    """Updates the device capabilities objects with additions specific to the runtime interface."""

    if runtime_interface not in {"pennylane", "qjit"}:
        raise ValueError(f"Invalid runtime interface: {runtime_interface}")

    operations = _get_operations(document, runtime_interface)
    capabilities.operations.update(operations)

    observables = _get_observables(document, runtime_interface)
    capabilities.observables.update(observables)

    measurement_processes = _get_measurement_processes(document, runtime_interface)
    capabilities.measurement_processes.update(measurement_processes)

    compilation_flags = _get_compilation_flags(document, runtime_interface)
    for flag, value in compilation_flags.items():
        setattr(capabilities, flag, value)
