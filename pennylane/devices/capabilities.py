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
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum
from itertools import repeat

import tomlkit as toml

import pennylane as qml
from pennylane.exceptions import InvalidCapabilitiesError, QuantumFunctionError
from pennylane.operation import Operator

ALL_SUPPORTED_SCHEMAS = [3]


def load_toml_file(file_path: str) -> dict:
    """Loads a TOML file and returns the parsed dict."""
    with open(file_path, encoding="utf-8") as file:
        return toml.load(file)


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
    """Information about support for each operation."""

    invertible: bool = False
    """Whether the adjoint of the operation is also supported."""

    controllable: bool = False
    """Whether the operation can be controlled."""

    differentiable: bool = False
    """Whether the operation is supported for device gradients."""

    conditions: list[ExecutionCondition] = field(default_factory=list)
    """Execution conditions that the operation must meet."""

    def __and__(self, other: "OperatorProperties") -> "OperatorProperties":
        # Take the intersection of support but the union of constraints (conditions)
        return OperatorProperties(
            invertible=self.invertible and other.invertible,
            controllable=self.controllable and other.controllable,
            differentiable=self.differentiable and other.differentiable,
            conditions=list(set(self.conditions) | set(other.conditions)),
        )


def _get_supported_base_op(op_name: str, op_dict: dict[str, OperatorProperties]) -> str | None:
    """Checks if the given operator is supported by name, returns the base op for nested ops"""

    if op_name in op_dict:
        return op_name

    if match := re.match(r"Adjoint\((.*)\)", op_name):
        base_op_name = match.group(1)
        deep_supported_base = _get_supported_base_op(base_op_name, op_dict)
        if deep_supported_base and op_dict[deep_supported_base].invertible:
            return deep_supported_base

    if match := re.match(r"C\((.*)\)", op_name):
        base_op_name = match.group(1)
        deep_supported_base = _get_supported_base_op(base_op_name, op_dict)
        if deep_supported_base and op_dict[deep_supported_base].controllable:
            return deep_supported_base

    return None


@dataclass
class DeviceCapabilities:  # pylint: disable=too-many-instance-attributes
    """Capabilities of a quantum device."""

    operations: dict[str, OperatorProperties] = field(default_factory=dict)
    """Operations natively supported by the backend device."""

    observables: dict[str, OperatorProperties] = field(default_factory=dict)
    """Observables that the device can measure."""

    measurement_processes: dict[str, list[ExecutionCondition]] = field(default_factory=dict)
    """List of measurement processes supported by the backend device."""

    qjit_compatible: bool = False
    """Whether the device is compatible with qjit."""

    runtime_code_generation: bool = False
    """Whether the device requires run time generation of the quantum circuit."""

    dynamic_qubit_management: bool = False
    """Whether the device supports dynamic qubit allocation/deallocation."""

    overlapping_observables: bool = True
    """Whether the device supports measuring overlapping observables on the same tape."""

    non_commuting_observables: bool = False
    """Whether the device supports measuring non-commuting observables on the same tape."""

    initial_state_prep: bool = False
    """Whether the device supports initial state preparation."""

    supported_mcm_methods: list[str] = field(default_factory=list)
    """List of supported methods of mid-circuit measurements."""

    def filter(self, finite_shots: bool) -> "DeviceCapabilities":
        """Returns the device capabilities conditioned on the given program features."""

        return (
            self._exclude_entries_with_condition(ExecutionCondition.ANALYTIC_MODE_ONLY)
            if finite_shots
            else self._exclude_entries_with_condition(ExecutionCondition.FINITE_SHOTS_ONLY)
        )

    def _exclude_entries_with_condition(
        self, condition: ExecutionCondition
    ) -> "DeviceCapabilities":
        """Removes entries from the capabilities that has the given condition."""

        operations = {k: v for k, v in self.operations.items() if condition not in v.conditions}
        observables = {k: v for k, v in self.observables.items() if condition not in v.conditions}
        measurement_processes = {
            k: v for k, v in self.measurement_processes.items() if condition not in v
        }
        return replace(
            self,
            operations=operations,
            observables=observables,
            measurement_processes=measurement_processes,
        )

    @classmethod
    def from_toml_file(cls, file_path: str, runtime_interface="pennylane") -> "DeviceCapabilities":
        """Loads a DeviceCapabilities object from a TOML file.

        Args:
            file_path (str): The path to the TOML file.
            runtime_interface (str): The runtime execution interface to get the capabilities for.
                Acceptable values are ``"pennylane"`` and ``"qjit"``. Use ``"pennylane"`` for capabilities of
                the device's implementation of `Device.execute`, and ``"qjit"`` for capabilities of
                the runtime execution function used by a qjit-compiled workflow.

        """
        document = load_toml_file(file_path)
        capabilities = parse_toml_document(document)
        update_device_capabilities(capabilities, document, runtime_interface)
        return capabilities

    def supports_operation(self, operation: str | Operator) -> bool:
        """Checks if the given operation is supported by name."""
        operation_name = operation if isinstance(operation, str) else operation.name
        return bool(_get_supported_base_op(operation_name, self.operations))

    def supports_observable(self, observable: str | Operator) -> bool:
        """Checks if the given observable is supported by name."""
        observable_name = observable if isinstance(observable, str) else observable.name
        return bool(_get_supported_base_op(observable_name, self.observables))


VALID_COMPILATION_OPTIONS = {
    "qjit_compatible",
    "runtime_code_generation",
    "dynamic_qubit_management",
    "overlapping_observables",
    "non_commuting_observables",
    "initial_state_prep",
    "supported_mcm_methods",
}


def _get_toml_section(document: dict, path: str, prefix: str = "") -> dict:
    """Retrieves a section from a TOML document using a given path.

    Args:
        document (dict): The TOML document loaded from a file.
        path (str): The title of the section to retrieve, typically in dot-separated format.
        prefix (str): Optional prefix to the path. For example, if `path` is "operators.gates"
            and `prefix` is "qjit", the "qjit.operators.gates" section will be retrieved.

    Returns:
        dict: the requested section from the TOML document.

    """

    if prefix:
        path = f"{prefix}.{path}"

    for k in path.split("."):
        if not isinstance(document, dict) or k not in document:
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

    if ExecutionCondition.TERMS_MUST_COMMUTE in conditions and target not in (
        "Prod",
        "SProd",
        "Sum",
        "LinearCombination",
        "Hamiltonian",
    ):
        raise InvalidCapabilitiesError(
            "'terms-commute' is only applicable to Prod, SProd, Sum, and LinearCombination."
        )


def _get_operators(section: dict) -> dict[str, OperatorProperties]:
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


def _get_operations(document: dict, prefix: str = "") -> dict[str, OperatorProperties]:
    """Gets the supported operations from a TOML document.

    Args:
        document (dict): The TOML document loaded from a file.
        prefix (str): Optional prefix corresponding to the runtime interface.

    """
    section = _get_toml_section(document, "operators.gates", prefix)
    return _get_operators(section)


def _get_observables(document: dict, prefix: str = "") -> dict[str, OperatorProperties]:
    """Gets the supported observables from a TOML document.

    Args:
        document (dict): The TOML document loaded from a file.
        prefix (str): Optional prefix corresponding to the runtime interface.

    """
    section = _get_toml_section(document, "operators.observables", prefix)
    return _get_operators(section)


def _get_measurement_processes(
    document: dict, prefix: str = ""
) -> dict[str, list[ExecutionCondition]]:
    """Gets the supported measurement processes from a TOML document.

    Args:
        document (dict): The TOML document loaded from a file.
        prefix (str): Optional prefix corresponding to the runtime interface.

    """

    section = _get_toml_section(document, "measurement_processes", prefix)
    measurement_processes = {}
    iterator = section.items() if hasattr(section, "items") else zip(section, repeat({}))
    for mp, attributes in iterator:
        if unknowns := set(attributes) - {"conditions"}:
            raise InvalidCapabilitiesError(
                f"Measurement '{mp}' has unknown attributes: {list(unknowns)}"
            )
        condition_strs = attributes.get("conditions", [])
        if unknowns := set(condition_strs) - VALID_CONDITION_STRINGS:
            raise InvalidCapabilitiesError(
                f"Measurement '{mp}' has unknown conditions: {list(unknowns)}"
            )
        conditions = [ExecutionCondition(c) for c in condition_strs]
        _validate_conditions(conditions)
        measurement_processes[mp] = conditions
    return measurement_processes


def _get_compilation_options(document: dict, prefix: str = "") -> dict[str, bool]:
    """Gets the capabilities in the compilation section.

    Args:
        document (dict): The TOML document loaded from a file.
        prefix (str): Optional prefix corresponding to the runtime interface.

    """

    section = _get_toml_section(document, "compilation", prefix)

    if unknowns := set(section) - VALID_COMPILATION_OPTIONS:
        raise InvalidCapabilitiesError(
            f"The compilation section has unknown options: {list(unknowns)}"
        )

    if not section.get("overlapping_observables", True) and section.get(
        "non_commuting_observables", False
    ):
        raise InvalidCapabilitiesError(
            "When overlapping_observables is False, non_commuting_observables cannot be True."
        )

    return section


def parse_toml_document(document: dict) -> DeviceCapabilities:
    """Parses a TOML document into a DeviceCapabilities object.

    This function will ignore sections that are specific to either runtime interface, such as
    ``"qjit.operators.gates"``. To include these sections, use :func:`update_device_capabilities`
    on the capabilities object returned from this function.

    """

    schema = int(document["schema"])
    assert schema in ALL_SUPPORTED_SCHEMAS, f"Unsupported config TOML schema {schema}"
    operations = _get_operations(document)
    observables = _get_observables(document)
    measurement_processes = _get_measurement_processes(document)
    compilation_options = _get_compilation_options(document)
    return DeviceCapabilities(
        operations=operations,
        observables=observables,
        measurement_processes=measurement_processes,
        **compilation_options,
    )


def update_device_capabilities(
    capabilities: DeviceCapabilities, document: dict, runtime_interface: str
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

    compilation_options = _get_compilation_options(document, runtime_interface)
    for option, value in compilation_options.items():
        setattr(capabilities, option, value)

    if runtime_interface == "qjit" and "qjit" in document and not capabilities.qjit_compatible:
        raise InvalidCapabilitiesError(
            "qjit-specific sections are found but the device is not qjit-compatible."
        )


def observable_stopping_condition_factory(
    capabilities: DeviceCapabilities,
) -> Callable[[qml.operation.Operator], bool]:
    """Returns a default observable validation check from a capabilities object.

    The returned function checks if an observable is supported, for composite and nested
    observables, check that the operands are supported.

    """

    def observable_stopping_condition(obs: qml.operation.Operator) -> bool:

        if not capabilities.supports_observable(obs.name):
            return False

        if isinstance(obs, qml.ops.CompositeOp):
            return all(observable_stopping_condition(op) for op in obs.operands)

        if isinstance(obs, qml.ops.SymbolicOp):
            return observable_stopping_condition(obs.base)

        return True

    return observable_stopping_condition


def validate_mcm_method(capabilities: DeviceCapabilities, mcm_method: str, shots_present: bool):
    """Validates an MCM method against the device's capabilities."""

    if mcm_method is None or mcm_method == "deferred":
        return  # no need to validate if requested deferred or if no method is requested.

    if mcm_method == "one-shot" and not shots_present:
        raise QuantumFunctionError('The "one-shot" MCM method is only supported with finite shots.')

    if capabilities is None:
        # If the device does not declare its supported mcm methods through capabilities,
        # simply check that the requested mcm method is something we recognize.
        if mcm_method not in ("deferred", "one-shot", "tree-traversal"):
            raise QuantumFunctionError(
                f'Requested MCM method "{mcm_method}" unsupported by the device. Supported methods '
                f'are: "deferred", "one-shot", and "tree-traversal".'
            )
        return

    if mcm_method not in capabilities.supported_mcm_methods:

        supported_methods = capabilities.supported_mcm_methods + ["deferred"]
        supported_method_strings = [f'"{m}"' for m in supported_methods]
        raise QuantumFunctionError(
            f'Requested MCM method "{mcm_method}" unsupported by the device. Supported methods '
            f"are: {', '.join(supported_method_strings)}."
        )
